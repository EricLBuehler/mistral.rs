use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use std::sync::Arc;

mod expert;

use candle_core::Result;

use crate::{LoraConfig, Shard, ShardedVarBuilder};

use self::expert::{
    load_expert_site, plan_expert_site, ExpertSiteMeta, GateUpOrder, LoadedExpertProjection,
};
use super::{
    DynamicLoraWeights, LoraExpertProjection, LoraExpertProjectionWeights, LoraExpertSiteHandle,
    LoraExpertWeights, LoraGateUpOrder, LoraLayerRegistry, LoraParallelism, LoraSiteHandle,
    LoraWeights,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynamicLoraLoadPlan {
    bytes: u64,
}

impl DynamicLoraLoadPlan {
    pub fn bytes(self) -> u64 {
        self.bytes
    }
}

#[derive(Default)]
struct AdapterTensorPair {
    a: Option<String>,
    b: Option<String>,
}

struct AdapterTensorIndex {
    pairs: BTreeMap<String, BTreeMap<String, AdapterTensorPair>>,
    tensor_names: BTreeSet<String>,
}

impl AdapterTensorIndex {
    fn new(names: Vec<String>) -> Self {
        let mut pairs = BTreeMap::<String, BTreeMap<String, AdapterTensorPair>>::new();
        let tensor_names = names.iter().cloned().collect();
        for name in names {
            if split_lora_name(&name, "lora_embedding_A").is_some()
                || split_lora_name(&name, "lora_embedding_B").is_some()
            {
                continue;
            }
            let (key, is_a) = if let Some(key) = split_lora_name(&name, "lora_A") {
                (key, true)
            } else if let Some(key) = split_lora_name(&name, "lora_B") {
                (key, false)
            } else {
                continue;
            };
            let Some(key) = key else {
                continue;
            };
            let (prefix, suffix) = key;
            let pair = pairs.entry(prefix).or_default().entry(suffix).or_default();
            if is_a {
                pair.a = Some(name);
            } else {
                pair.b = Some(name);
            }
        }
        Self {
            pairs,
            tensor_names,
        }
    }

    fn pair_for_site<'a>(&'a self, path: &str) -> Result<Option<(&'a str, &'a str)>> {
        let peft_path = format!("base_model.model.{path}");
        let mut candidates = [path, peft_path.as_str()]
            .into_iter()
            .filter_map(|prefix| self.pairs.get(prefix).map(|pairs| (prefix, pairs)))
            .flat_map(|(prefix, pairs)| {
                pairs
                    .iter()
                    .map(move |(suffix, pair)| (prefix, suffix, pair))
            });
        let Some((prefix, suffix, pair)) = candidates.next() else {
            return Ok(None);
        };
        if candidates.next().is_some() {
            candle_core::bail!("multiple LoRA tensor pairs match site `{path}`");
        }
        let a = pair.a.as_deref().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "LoRA tensor pair `{prefix}` suffix `{suffix}` is missing A"
            ))
        })?;
        let b = pair.b.as_deref().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "LoRA tensor pair `{prefix}` suffix `{suffix}` is missing B"
            ))
        })?;
        Ok(Some((a, b)))
    }

    fn unconsumed(&self, consumed: &BTreeSet<String>) -> Vec<String> {
        self.tensor_names.difference(consumed).cloned().collect()
    }
}

fn split_lora_name(name: &str, component: &str) -> Option<Option<(String, String)>> {
    let marker = format!(".{component}.");
    let (prefix, tail) = if let Some(parts) = name.rsplit_once(&marker) {
        parts
    } else {
        let tail = name.strip_prefix(&format!("{component}."))?;
        ("", tail)
    };
    let suffix = if tail == "weight" {
        Some(String::new())
    } else {
        tail.strip_suffix(".weight")
            .filter(|suffix| !suffix.is_empty())
            .map(ToString::to_string)
    };
    Some(suffix.map(|suffix| (prefix.to_string(), suffix)))
}

struct SiteLoadSpec<'a> {
    a_name: &'a str,
    b_name: &'a str,
    rank: usize,
    a_shard: Shard,
    b_shard: Shard,
    scale: f64,
}

fn site_load_spec<'a>(
    site: &LoraSiteHandle,
    config: &LoraConfig,
    tensors: &'a AdapterTensorIndex,
) -> Result<Option<SiteLoadSpec<'a>>> {
    let path = site.key().path();
    let Some((a_name, b_name)) = tensors.pair_for_site(path)? else {
        return Ok(None);
    };
    if !config.try_targets_path(path)? {
        candle_core::bail!("LoRA tensors for site `{path}` are not declared by target_modules");
    }
    if config.try_excludes_path(path)? {
        candle_core::bail!("LoRA tensors for site `{path}` are excluded by exclude_modules");
    }
    if path.split('.').any(|part| part == "experts") {
        candle_core::bail!("dynamic LoRA does not support routed MoE expert site `{path}`");
    }
    if path == "lm_head" || path.ends_with(".lm_head") {
        candle_core::bail!("dynamic LoRA does not support adapters targeting lm_head");
    }

    let spec = site.spec();
    let rank = config.try_rank_for(path)?;
    let scale = config.scale_for(path)?;
    let (a_shard, b_shard) = match spec.parallelism() {
        LoraParallelism::Replicated => (Shard::default(), Shard::default()),
        LoraParallelism::Column { output_shard } => (Shard::default(), output_shard),
        LoraParallelism::Row { input_shard } => (input_shard, Shard::default()),
    };
    Ok(Some(SiteLoadSpec {
        a_name,
        b_name,
        rank,
        a_shard,
        b_shard,
        scale,
    }))
}

fn tensor_elements_after_shard(
    shape: &[usize],
    expected: [usize; 2],
    shard: Shard,
    name: &str,
) -> Result<u64> {
    if shape != expected {
        candle_core::bail!("LoRA tensor `{name}` has shape {shape:?}, expected {expected:?}");
    }
    let mut elements = shape.iter().try_fold(1u64, |elements, dim| {
        elements
            .checked_mul(u64::try_from(*dim).map_err(candle_core::Error::wrap)?)
            .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))
    })?;
    match shard {
        Shard::Simple { world_size: 1, .. } => {}
        Shard::Simple {
            dim,
            rank,
            world_size,
        } => {
            let size = *shape.get(dim).ok_or_else(|| {
                candle_core::Error::msg(format!("invalid shard dimension {dim} for `{name}`"))
            })?;
            if rank >= world_size || !size.is_multiple_of(world_size) {
                candle_core::bail!("invalid LoRA shard for tensor `{name}`");
            }
            elements /= u64::try_from(world_size).map_err(candle_core::Error::wrap)?;
        }
        Shard::Offset { dim, offset, len } => {
            let size = *shape.get(dim).ok_or_else(|| {
                candle_core::Error::msg(format!("invalid shard dimension {dim} for `{name}`"))
            })?;
            if offset.checked_add(len).is_none_or(|end| end > size) {
                candle_core::bail!("invalid LoRA shard for tensor `{name}`");
            }
            elements = elements
                .checked_div(u64::try_from(size).map_err(candle_core::Error::wrap)?)
                .and_then(|value| value.checked_mul(u64::try_from(len).ok()?))
                .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))?;
        }
    }
    Ok(elements)
}

fn validate_consumption(tensors: &AdapterTensorIndex, consumed: &BTreeSet<String>) -> Result<()> {
    let unconsumed = tensors.unconsumed(consumed);
    if !unconsumed.is_empty() {
        candle_core::bail!(
            "adapter contains tensors not consumed by registered LoRA sites: {}",
            unconsumed.join(", ")
        );
    }
    if consumed.is_empty() {
        candle_core::bail!("adapter does not contain weights for any registered LoRA site");
    }
    Ok(())
}

pub fn plan_dynamic_lora_weights(
    registry: &LoraLayerRegistry,
    config: &LoraConfig,
    weights: &ShardedVarBuilder,
) -> Result<DynamicLoraLoadPlan> {
    config.validate_dynamic()?;
    let tensors = AdapterTensorIndex::new(weights.tensor_names().ok_or_else(|| {
        candle_core::Error::msg("dynamic LoRA loading requires an indexed tensor backend")
    })?);
    let mut consumed = BTreeSet::new();
    let mut bytes = 0u64;
    for site in registry.sites() {
        let Some(load) = site_load_spec(&site, config, &tensors)? else {
            continue;
        };
        let spec = site.spec();
        let a_shape = weights.tensor_shape(load.a_name).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "LoRA tensor `{}` has no shape metadata",
                load.a_name
            ))
        })?;
        let b_shape = weights.tensor_shape(load.b_name).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "LoRA tensor `{}` has no shape metadata",
                load.b_name
            ))
        })?;
        let elements = tensor_elements_after_shard(
            a_shape,
            [load.rank, spec.in_features()],
            load.a_shard,
            load.a_name,
        )?
        .checked_add(tensor_elements_after_shard(
            b_shape,
            [spec.out_features(), load.rank],
            load.b_shard,
            load.b_name,
        )?)
        .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))?;
        let site_bytes = elements
            .checked_mul(
                u64::try_from(site.activation_dtype().size_in_bytes())
                    .map_err(candle_core::Error::wrap)?,
            )
            .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))?;
        bytes = bytes
            .checked_add(site_bytes)
            .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))?;
        consumed.insert(load.a_name.to_string());
        consumed.insert(load.b_name.to_string());
    }
    for site in registry.expert_sites() {
        let Some(plan) = plan_expert_site(expert_site_meta(&site), config, &tensors, weights)?
        else {
            continue;
        };
        bytes = bytes
            .checked_add(plan.bytes())
            .ok_or_else(|| candle_core::Error::msg("LoRA tensor size overflow"))?;
        consumed.extend(plan.consumed().iter().cloned());
    }
    validate_consumption(&tensors, &consumed)?;
    Ok(DynamicLoraLoadPlan { bytes })
}

fn expert_site_meta(site: &LoraExpertSiteHandle) -> ExpertSiteMeta<'_> {
    let spec = site.spec();
    ExpertSiteMeta {
        path: site.key().path(),
        num_experts: spec.num_experts(),
        hidden_size: spec.hidden_size(),
        intermediate_size: spec.intermediate_size(),
        gate_name: spec.name(LoraExpertProjection::Gate),
        up_name: spec.name(LoraExpertProjection::Up),
        down_name: spec.name(LoraExpertProjection::Down),
        gate_up_order: match spec.gate_up_order() {
            LoraGateUpOrder::Concatenated => GateUpOrder::Concatenated,
            LoraGateUpOrder::Interleaved => GateUpOrder::Interleaved,
        },
        gate_up_output_shard: spec.gate_up_output_shard(),
        down_input_shard: spec.down_input_shard(),
        activation_dtype: site.activation_dtype(),
        device: site.device(),
    }
}

fn projection_weights(weights: LoadedExpertProjection) -> Result<LoraExpertProjectionWeights> {
    LoraExpertProjectionWeights::new_loaded(weights.a, weights.b, weights.scales)
}

pub fn load_dynamic_lora_weights(
    registry: &LoraLayerRegistry,
    config: &LoraConfig,
    weights: &ShardedVarBuilder,
) -> Result<DynamicLoraWeights> {
    config.validate_dynamic()?;
    let tensors = AdapterTensorIndex::new(weights.tensor_names().ok_or_else(|| {
        candle_core::Error::msg("dynamic LoRA loading requires an indexed tensor backend")
    })?);

    let mut linear = Vec::new();
    let mut consumed = BTreeSet::new();
    for site in registry.sites() {
        let Some(load) = site_load_spec(&site, config, &tensors)? else {
            continue;
        };
        let site_weights = weights
            .root()
            .set_device(site.device().clone())
            .set_dtype(site.activation_dtype());
        let spec = site.spec();
        let a = site_weights.get_with_hints(
            (load.rank, spec.in_features()),
            load.a_name,
            load.a_shard,
        )?;
        let b = site_weights.get_with_hints(
            (spec.out_features(), load.rank),
            load.b_name,
            load.b_shard,
        )?;
        consumed.insert(load.a_name.to_string());
        consumed.insert(load.b_name.to_string());
        linear.push((site, LoraWeights::new(a, b, load.scale)?));
    }
    let mut experts = Vec::new();
    for site in registry.expert_sites() {
        let Some(plan) = plan_expert_site(expert_site_meta(&site), config, &tensors, weights)?
        else {
            continue;
        };
        consumed.extend(plan.consumed().iter().cloned());
        let loaded = load_expert_site(expert_site_meta(&site), plan, weights)?;
        let gate = loaded.gate.map(projection_weights).transpose()?;
        let up = loaded.up.map(projection_weights).transpose()?;
        let down = loaded.down.map(projection_weights).transpose()?;
        let expert_weights = LoraExpertWeights::new(&site, gate, up, down)?;
        experts.push((site, expert_weights));
    }
    validate_consumption(&tensors, &consumed)?;
    Ok(DynamicLoraWeights { linear, experts })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::lora::dynamic::{LoraExpertProjectionNames, LoraExpertSiteSpec};
    use crate::{LoraLinearSpec, LoraSiteKey, ShardedSafeTensors};

    fn config(target: &str, rank: usize, alpha: f64) -> LoraConfig {
        serde_json::from_value(serde_json::json!({
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": [target]
        }))
        .unwrap()
    }

    #[test]
    fn tensor_index_matches_standard_and_named_pairs() -> Result<()> {
        let q_a = "base_model.model.model.layers.0.q_proj.lora_A.weight";
        let q_b = "base_model.model.model.layers.0.q_proj.lora_B.weight";
        let k_a = "model.layers.0.k_proj.lora_A.default.weight";
        let k_b = "model.layers.0.k_proj.lora_B.default.weight";
        let index = AdapterTensorIndex::new(vec![
            q_a.to_string(),
            q_b.to_string(),
            k_a.to_string(),
            k_b.to_string(),
            "unexpected.weight".to_string(),
        ]);

        assert_eq!(
            index.pair_for_site("model.layers.0.q_proj")?,
            Some((q_a, q_b))
        );
        assert_eq!(
            index.pair_for_site("model.layers.0.k_proj")?,
            Some((k_a, k_b))
        );
        let consumed = BTreeSet::from([q_a.to_string(), q_b.to_string()]);
        assert_eq!(
            index.unconsumed(&consumed),
            vec![
                k_a.to_string(),
                k_b.to_string(),
                "unexpected.weight".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn tensor_index_rejects_incomplete_and_ambiguous_pairs() {
        let incomplete =
            AdapterTensorIndex::new(vec!["model.layers.0.q_proj.lora_A.weight".to_string()]);
        assert!(incomplete.pair_for_site("model.layers.0.q_proj").is_err());

        let ambiguous = AdapterTensorIndex::new(vec![
            "model.layers.0.q_proj.lora_A.weight".to_string(),
            "model.layers.0.q_proj.lora_B.weight".to_string(),
            "model.layers.0.q_proj.lora_A.default.weight".to_string(),
            "model.layers.0.q_proj.lora_B.default.weight".to_string(),
        ]);
        assert!(ambiguous.pair_for_site("model.layers.0.q_proj").is_err());
    }

    #[test]
    fn embedding_and_malformed_lora_tensors_are_unconsumed() {
        let names = vec![
            "base_model.model.embed_tokens.lora_embedding_A.weight".to_string(),
            "base_model.model.embed_tokens.lora_embedding_B.weight".to_string(),
            "model.layers.0.q_proj.lora_A.not_a_weight".to_string(),
        ];
        let index = AdapterTensorIndex::new(names.clone());
        assert_eq!(index.unconsumed(&BTreeSet::new()), names);
    }

    #[test]
    fn rejects_all_unconsumed_tensor_kinds() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.q_proj";
        let unexpected = [
            "base_model.model.model.layers.0.q_proj.bias",
            "base_model.model.lm_head.modules_to_save.default.weight",
            "base_model.model.model.layers.0.q_proj.lora_magnitude_vector.weight",
        ];
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 0.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(&[[1f32], [0.]], &device)?,
            ),
            (
                unexpected[0].to_string(),
                Tensor::zeros(2, DType::F32, &device)?,
            ),
            (
                unexpected[1].to_string(),
                Tensor::zeros((2, 2), DType::F32, &device)?,
            ),
            (
                unexpected[2].to_string(),
                Tensor::zeros(2, DType::F32, &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device,
        )?;
        registry.finalize()?;

        let error =
            load_dynamic_lora_weights(&registry, &config("q_proj", 1, 1.0), &weights).unwrap_err();
        let error = error.to_string();
        for name in unexpected {
            assert!(error.contains(name));
        }
        Ok(())
    }

    #[test]
    fn rejects_tensor_sites_outside_declared_targets() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.k_proj";
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 0.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(&[[1f32], [0.]], &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device,
        )?;
        registry.finalize()?;

        let error =
            load_dynamic_lora_weights(&registry, &config("q_proj", 1, 1.0), &weights).unwrap_err();
        assert!(error.to_string().contains("not declared by target_modules"));
        Ok(())
    }

    #[test]
    fn rejects_tensor_sites_declared_as_excluded() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.q_proj";
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 0.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(&[[1f32], [0.]], &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device,
        )?;
        registry.finalize()?;
        let config: LoraConfig = serde_json::from_value(serde_json::json!({
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "exclude_modules": ["q_proj"]
        }))
        .map_err(candle_core::Error::wrap)?;

        let error = load_dynamic_lora_weights(&registry, &config, &weights).unwrap_err();
        assert!(error.to_string().contains("excluded by exclude_modules"));
        Ok(())
    }

    #[test]
    fn accepts_safetensors_header_metadata() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.q_proj";
        let a_name = format!("{path}.lora_A.weight");
        let b_name = format!("{path}.lora_B.weight");
        let a = Tensor::new(&[[1f32, 0.]], &device)?;
        let b = Tensor::new(&[[1f32], [0.]], &device)?;
        let directory = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let weights_path = directory.path().join("adapter_model.safetensors");
        safetensors::serialize_to_file(
            [(a_name.as_str(), &a), (b_name.as_str(), &b)],
            Some(HashMap::from([
                ("format".to_string(), "pt".to_string()),
                ("peft".to_string(), "lora".to_string()),
            ])),
            &weights_path,
        )
        .map_err(candle_core::Error::wrap)?;
        let weights = unsafe {
            ShardedSafeTensors::sharded(
                std::slice::from_ref(&weights_path),
                DType::F32,
                &device,
                None,
                Arc::new(|_| true),
            )?
        };
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device,
        )?;
        registry.finalize()?;

        let loaded = load_dynamic_lora_weights(&registry, &config("q_proj", 1, 1.0), &weights)?;
        assert_eq!(loaded.linear.len(), 1);
        assert!(loaded.experts.is_empty());
        Ok(())
    }

    #[test]
    fn loads_compact_expert_site_from_safetensors_metadata() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.mlp.experts";
        let tensors = [
            (
                format!("base_model.model.{path}.0.gate_proj.lora_A.weight"),
                Tensor::new(&[[1f32, 2.]], &device)?,
            ),
            (
                format!("base_model.model.{path}.0.gate_proj.lora_B.weight"),
                Tensor::new(&[[3f32], [4.]], &device)?,
            ),
            (
                format!("base_model.model.{path}.1.gate_proj.lora_A.weight"),
                Tensor::new(&[[5f32, 6.]], &device)?,
            ),
            (
                format!("base_model.model.{path}.1.gate_proj.lora_B.weight"),
                Tensor::new(&[[7f32], [8.]], &device)?,
            ),
        ];
        let directory = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let weights_path = directory.path().join("adapter_model.safetensors");
        safetensors::serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            Some(HashMap::from([
                ("format".to_string(), "pt".to_string()),
                ("peft".to_string(), "lora".to_string()),
            ])),
            &weights_path,
        )
        .map_err(candle_core::Error::wrap)?;
        let weights = unsafe {
            ShardedSafeTensors::sharded(
                std::slice::from_ref(&weights_path),
                DType::F32,
                &device,
                None,
                Arc::new(|_| true),
            )?
        };
        let registry = LoraLayerRegistry::new();
        registry.register_expert(
            LoraSiteKey::new(path),
            LoraExpertSiteSpec::new(
                2,
                2,
                2,
                LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                Shard::default(),
                Shard::default(),
            )?,
            DType::F32,
            device,
        )?;
        registry.finalize()?;
        let config = config("gate_proj", 1, 2.0);

        assert_eq!(
            plan_dynamic_lora_weights(&registry, &config, &weights)?.bytes(),
            40
        );
        let loaded = load_dynamic_lora_weights(&registry, &config, &weights)?;
        assert!(loaded.linear.is_empty());
        assert_eq!(loaded.experts.len(), 1);
        let gate = loaded.experts[0].1.gate().expect("gate weights");
        assert_eq!(gate.a().dims(), &[2, 1, 2]);
        assert_eq!(gate.b().dims(), &[2, 2, 1]);
        assert_eq!(gate.scales().to_vec1::<f32>()?, vec![2., 2.]);
        Ok(())
    }

    #[test]
    fn loads_peft_weights_with_column_sharding() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.q_proj";
        let a_name = format!("base_model.model.{path}.lora_A.weight");
        let b_name = format!("base_model.model.{path}.lora_B.weight");
        let backend = HashMap::from([
            (
                a_name,
                Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &device)?,
            ),
            (
                b_name,
                Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.]], &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::column(
                4,
                4,
                Shard::Simple {
                    dim: 0,
                    rank: 1,
                    world_size: 2,
                },
            ),
            DType::F32,
            device,
        )?;
        registry.finalize()?;
        let config = config("q_proj", 2, 4.0);
        assert_eq!(
            plan_dynamic_lora_weights(&registry, &config, &weights)?.bytes(),
            48
        );

        let mut loaded = load_dynamic_lora_weights(&registry, &config, &weights)?.linear;
        let (_, weights) = loaded.pop().expect("loaded registered LoRA site");
        assert!(loaded.is_empty());
        assert_eq!(
            weights.a.to_vec2::<f32>()?,
            vec![vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]]
        );
        assert_eq!(
            weights.b.to_vec2::<f32>()?,
            vec![vec![5., 6.], vec![7., 8.]]
        );
        assert_eq!(weights.scale, 2.0);
        Ok(())
    }

    #[test]
    fn loads_peft_weights_with_row_sharding() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.down_proj";
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.]], &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::row(
                4,
                4,
                Shard::Simple {
                    dim: 1,
                    rank: 1,
                    world_size: 2,
                },
            ),
            DType::F32,
            device,
        )?;
        registry.finalize()?;

        let mut loaded =
            load_dynamic_lora_weights(&registry, &config("down_proj", 2, 2.0), &weights)?.linear;
        let (_, weights) = loaded.pop().expect("loaded registered LoRA site");
        assert!(loaded.is_empty());
        assert_eq!(
            weights.a.to_vec2::<f32>()?,
            vec![vec![3., 4.], vec![7., 8.]]
        );
        assert_eq!(
            weights.b.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.], vec![5., 6.], vec![7., 8.]]
        );
        Ok(())
    }

    #[test]
    fn loads_each_fused_projection_slice() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.mlp.gate_up_proj";
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(
                    &[
                        [1f32, 2.],
                        [3., 4.],
                        [5., 6.],
                        [7., 8.],
                        [9., 10.],
                        [11., 12.],
                        [13., 14.],
                        [15., 16.],
                    ],
                    &device,
                )?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        for index in 0..2 {
            registry.register(
                LoraSiteKey::with_slice(path, index, 2)?,
                LoraLinearSpec::column(
                    4,
                    8,
                    Shard::Simple {
                        dim: 0,
                        rank: index,
                        world_size: 2,
                    },
                ),
                DType::F32,
                device.clone(),
            )?;
        }
        registry.finalize()?;

        let loaded =
            load_dynamic_lora_weights(&registry, &config("gate_up_proj", 2, 2.0), &weights)?.linear;
        assert_eq!(loaded.len(), 2);
        let first_slice = loaded[0].0.key().slice().expect("fused slice");
        let second_slice = loaded[1].0.key().slice().expect("fused slice");
        assert_eq!((first_slice.index(), first_slice.count()), (0, 2));
        assert_eq!((second_slice.index(), second_slice.count()), (1, 2));
        assert_eq!(
            loaded[0].1.b.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.], vec![5., 6.], vec![7., 8.]]
        );
        assert_eq!(
            loaded[1].1.b.to_vec2::<f32>()?,
            vec![
                vec![9., 10.],
                vec![11., 12.],
                vec![13., 14.],
                vec![15., 16.]
            ]
        );
        assert_eq!(
            loaded[0].1.a.to_vec2::<f32>()?,
            loaded[1].1.a.to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn rejects_routed_moe_expert_sites_at_load_time() -> Result<()> {
        let device = Device::Cpu;
        let path = "model.layers.0.mlp.experts.0.gate_proj";
        let backend = HashMap::from([
            (
                format!("{path}.lora_A.weight"),
                Tensor::new(&[[1f32, 0.]], &device)?,
            ),
            (
                format!("{path}.lora_B.weight"),
                Tensor::new(&[[1f32], [0.]], &device)?,
            ),
        ]);
        let weights = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new(path),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device,
        )?;
        registry.finalize()?;
        let config = config("gate_proj", 1, 1.0);

        let error = load_dynamic_lora_weights(&registry, &config, &weights).unwrap_err();
        assert!(error.to_string().contains("routed MoE expert"));
        Ok(())
    }
}
