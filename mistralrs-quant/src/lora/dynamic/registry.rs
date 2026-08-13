use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use candle_core::{DType, Device, Result};

use crate::Shard;

use super::{LoraExpertSiteHandle, LoraExpertSiteSpec};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoraRuntimeId(u64);

impl LoraRuntimeId {
    fn next() -> Self {
        Self(NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoraSiteSlice {
    index: usize,
    count: usize,
}

impl LoraSiteSlice {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn count(self) -> usize {
        self.count
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoraSiteKey {
    path: Arc<str>,
    slice: Option<LoraSiteSlice>,
}

impl LoraSiteKey {
    pub fn new(path: impl Into<Arc<str>>) -> Self {
        Self {
            path: path.into(),
            slice: None,
        }
    }

    pub fn with_slice(path: impl Into<Arc<str>>, index: usize, count: usize) -> Result<Self> {
        if count == 0 || index >= count {
            candle_core::bail!("invalid LoRA site slice {index}/{count}");
        }
        Ok(Self {
            path: path.into(),
            slice: Some(LoraSiteSlice { index, count }),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn slice(&self) -> Option<LoraSiteSlice> {
        self.slice
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoraParallelism {
    Replicated,
    Column { output_shard: Shard },
    Row { input_shard: Shard },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraLinearSpec {
    in_features: usize,
    out_features: usize,
    parallelism: LoraParallelism,
    input_runtime_to_canonical: Option<Arc<[usize]>>,
    output_runtime_to_canonical: Option<Arc<[usize]>>,
}

impl LoraLinearSpec {
    pub fn replicated(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Replicated,
            input_runtime_to_canonical: None,
            output_runtime_to_canonical: None,
        }
    }

    pub fn column(in_features: usize, out_features: usize, output_shard: Shard) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Column { output_shard },
            input_runtime_to_canonical: None,
            output_runtime_to_canonical: None,
        }
    }

    pub fn row(in_features: usize, out_features: usize, input_shard: Shard) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Row { input_shard },
            input_runtime_to_canonical: None,
            output_runtime_to_canonical: None,
        }
    }

    pub fn with_input_runtime_to_canonical(
        mut self,
        runtime_to_canonical: impl Into<Arc<[usize]>>,
    ) -> Result<Self> {
        let runtime_to_canonical = runtime_to_canonical.into();
        validate_feature_permutation("input", &runtime_to_canonical, self.in_features)?;
        self.input_runtime_to_canonical = Some(runtime_to_canonical);
        Ok(self)
    }

    pub fn with_output_runtime_to_canonical(
        mut self,
        runtime_to_canonical: impl Into<Arc<[usize]>>,
    ) -> Result<Self> {
        let runtime_to_canonical = runtime_to_canonical.into();
        validate_feature_permutation("output", &runtime_to_canonical, self.out_features)?;
        self.output_runtime_to_canonical = Some(runtime_to_canonical);
        Ok(self)
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub(crate) fn parallelism(&self) -> LoraParallelism {
        self.parallelism
    }

    pub(crate) fn is_replicated(&self) -> bool {
        self.parallelism == LoraParallelism::Replicated
    }

    pub(crate) fn row_input_shard(&self) -> Option<Shard> {
        match self.parallelism {
            LoraParallelism::Row { input_shard } => Some(input_shard),
            _ => None,
        }
    }

    pub(crate) fn input_runtime_to_canonical(&self) -> Option<&[usize]> {
        self.input_runtime_to_canonical.as_deref()
    }

    pub(crate) fn output_runtime_to_canonical(&self) -> Option<&[usize]> {
        self.output_runtime_to_canonical.as_deref()
    }

    fn validate(&self) -> Result<()> {
        if let Some(runtime_to_canonical) = &self.input_runtime_to_canonical {
            validate_feature_permutation("input", runtime_to_canonical, self.in_features)?;
        }
        if let Some(runtime_to_canonical) = &self.output_runtime_to_canonical {
            validate_feature_permutation("output", runtime_to_canonical, self.out_features)?;
        }
        Ok(())
    }
}

fn validate_feature_permutation(
    axis: &str,
    runtime_to_canonical: &[usize],
    features: usize,
) -> Result<()> {
    if runtime_to_canonical.len() != features {
        candle_core::bail!(
            "LoRA {axis} runtime-to-canonical map has length {}, expected {features}",
            runtime_to_canonical.len()
        );
    }
    let mut seen = vec![false; features];
    for (runtime, &canonical) in runtime_to_canonical.iter().enumerate() {
        let Some(seen) = seen.get_mut(canonical) else {
            candle_core::bail!(
                "LoRA {axis} runtime-to-canonical map index {runtime} references out-of-range feature {canonical}"
            );
        };
        if std::mem::replace(seen, true) {
            candle_core::bail!(
                "LoRA {axis} runtime-to-canonical map references canonical feature {canonical} more than once"
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct LoraSiteHandle {
    runtime_id: LoraRuntimeId,
    key: LoraSiteKey,
    spec: LoraLinearSpec,
    activation_dtype: DType,
    device: Device,
    id: OnceLock<u32>,
}

impl LoraSiteHandle {
    pub(crate) fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub fn key(&self) -> &LoraSiteKey {
        &self.key
    }

    pub fn spec(&self) -> &LoraLinearSpec {
        &self.spec
    }

    pub fn activation_dtype(&self) -> DType {
        self.activation_dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn id(&self) -> Result<u32> {
        self.id
            .get()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("LoRA layer registry has not been finalized"))
    }
}

#[derive(Default)]
struct RegistryState {
    sites: BTreeMap<LoraSiteKey, Arc<LoraSiteHandle>>,
    expert_sites: BTreeMap<LoraSiteKey, Arc<LoraExpertSiteHandle>>,
    finalized: bool,
}

#[derive(Debug)]
struct LoraSitePrefixAlias {
    source: Arc<str>,
    target: Arc<str>,
}

#[derive(Debug)]
pub struct LoraLayerRegistry {
    runtime_id: LoraRuntimeId,
    state: Mutex<RegistryState>,
    site_prefix_alias: Option<LoraSitePrefixAlias>,
}

impl Default for LoraLayerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RegistryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryState")
            .field("sites", &self.sites)
            .field("expert_sites", &self.expert_sites)
            .field("finalized", &self.finalized)
            .finish()
    }
}

impl LoraLayerRegistry {
    pub fn new() -> Self {
        Self {
            runtime_id: LoraRuntimeId::next(),
            state: Mutex::new(RegistryState::default()),
            site_prefix_alias: None,
        }
    }

    pub fn new_with_site_prefix_alias(
        source: impl Into<Arc<str>>,
        target: impl Into<Arc<str>>,
    ) -> Result<Self> {
        let source = source.into();
        let target = target.into();
        if source.is_empty() || target.is_empty() {
            candle_core::bail!("LoRA site prefix aliases must not be empty");
        }
        if source.starts_with('.')
            || source.ends_with('.')
            || target.starts_with('.')
            || target.ends_with('.')
        {
            candle_core::bail!("LoRA site prefix aliases must not start or end with `.`");
        }
        Ok(Self {
            runtime_id: LoraRuntimeId::next(),
            state: Mutex::new(RegistryState::default()),
            site_prefix_alias: Some(LoraSitePrefixAlias { source, target }),
        })
    }

    pub fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub fn register(
        &self,
        key: LoraSiteKey,
        spec: LoraLinearSpec,
        activation_dtype: DType,
        device: Device,
    ) -> Result<Arc<LoraSiteHandle>> {
        spec.validate()?;
        let key = self.canonical_site_key(key);
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if state.expert_sites.contains_key(&key) {
            candle_core::bail!(
                "LoRA site `{}` was registered as both a linear and an expert group",
                key.path()
            );
        }
        if let Some(site) = state.sites.get(&key) {
            if site.spec != spec
                || site.activation_dtype != activation_dtype
                || site.device.location() != device.location()
            {
                candle_core::bail!(
                    "LoRA site `{}` was registered with incompatible specifications",
                    key.path()
                );
            }
            return Ok(site.clone());
        }
        if state.finalized {
            candle_core::bail!(
                "cannot register LoRA site `{}` after registry finalization",
                key.path()
            );
        }

        let site = Arc::new(LoraSiteHandle {
            runtime_id: self.runtime_id,
            key: key.clone(),
            spec,
            activation_dtype,
            device,
            id: OnceLock::new(),
        });
        state.sites.insert(key, site.clone());
        Ok(site)
    }

    pub fn register_expert(
        &self,
        key: LoraSiteKey,
        spec: LoraExpertSiteSpec,
        activation_dtype: DType,
        device: Device,
    ) -> Result<Arc<LoraExpertSiteHandle>> {
        let key = self.canonical_site_key(key);
        if key.slice().is_some() {
            candle_core::bail!("expert LoRA group sites cannot be sliced");
        }
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if state.sites.contains_key(&key) {
            candle_core::bail!(
                "LoRA site `{}` was registered as both a linear and an expert group",
                key.path()
            );
        }
        if let Some(site) = state.expert_sites.get(&key) {
            if site.spec() != &spec
                || site.activation_dtype() != activation_dtype
                || site.device().location() != device.location()
            {
                candle_core::bail!(
                    "LoRA expert site `{}` was registered with incompatible specifications",
                    key.path()
                );
            }
            return Ok(site.clone());
        }
        if state.finalized {
            candle_core::bail!(
                "cannot register LoRA expert site `{}` after registry finalization",
                key.path()
            );
        }

        let site = Arc::new(LoraExpertSiteHandle::new(
            self.runtime_id,
            key.clone(),
            spec,
            activation_dtype,
            device,
        ));
        state.expert_sites.insert(key, site.clone());
        Ok(site)
    }

    pub fn finalize(&self) -> Result<Vec<Arc<LoraSiteHandle>>> {
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if !state.finalized {
            for (id, site) in state.sites.values().enumerate() {
                let id = u32::try_from(id).map_err(candle_core::Error::wrap)?;
                site.id
                    .set(id)
                    .map_err(|_| candle_core::Error::msg("LoRA site ID was already assigned"))?;
            }
            let first_expert_id = state.sites.len();
            for (offset, site) in state.expert_sites.values().enumerate() {
                let id =
                    u32::try_from(first_expert_id + offset).map_err(candle_core::Error::wrap)?;
                site.assign_id(id)?;
            }
            state.finalized = true;
        }
        Ok(state.sites.values().cloned().collect())
    }

    pub fn sites(&self) -> Vec<Arc<LoraSiteHandle>> {
        self.state
            .lock()
            .expect("LoRA layer registry poisoned")
            .sites
            .values()
            .cloned()
            .collect()
    }

    pub fn expert_sites(&self) -> Vec<Arc<LoraExpertSiteHandle>> {
        self.state
            .lock()
            .expect("LoRA layer registry poisoned")
            .expert_sites
            .values()
            .cloned()
            .collect()
    }

    fn canonical_site_key(&self, key: LoraSiteKey) -> LoraSiteKey {
        let Some(alias) = &self.site_prefix_alias else {
            return key;
        };
        let path = key.path();
        let suffix = if path == alias.source.as_ref() {
            Some("")
        } else {
            path.strip_prefix(alias.source.as_ref())
                .and_then(|suffix| suffix.strip_prefix('.'))
        };
        let Some(suffix) = suffix else {
            return key;
        };
        let path = if suffix.is_empty() {
            alias.target.clone()
        } else {
            Arc::from(format!("{}.{suffix}", alias.target))
        };
        LoraSiteKey {
            path,
            slice: key.slice,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert_spec() -> Result<LoraExpertSiteSpec> {
        LoraExpertSiteSpec::new(
            2,
            4,
            8,
            super::super::LoraExpertProjectionNames::new("gate", "up", "down"),
            Shard::default(),
            Shard::default(),
        )
    }

    #[test]
    fn site_ids_are_independent_of_registration_order() -> Result<()> {
        fn ids(paths: &[&str]) -> Result<BTreeMap<String, u32>> {
            let registry = LoraLayerRegistry::new();
            for path in paths {
                registry.register(
                    LoraSiteKey::new(*path),
                    LoraLinearSpec::replicated(4, 8),
                    DType::F32,
                    Device::Cpu,
                )?;
            }
            registry.finalize()?;
            registry
                .sites()
                .into_iter()
                .map(|site| Ok((site.key().path().to_string(), site.id()?)))
                .collect()
        }

        assert_eq!(ids(&["c", "a", "b"])?, ids(&["a", "b", "c"])?);
        Ok(())
    }

    #[test]
    fn linear_and_expert_sites_share_one_id_namespace() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let expert = registry.register_expert(
            LoraSiteKey::new("experts"),
            expert_spec()?,
            DType::F32,
            Device::Cpu,
        )?;
        let linear = registry.register(
            LoraSiteKey::new("linear"),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;

        assert_eq!(linear.id()?, 0);
        assert_eq!(expert.id()?, 1);
        assert_ne!(linear.id()?, expert.id()?);
        Ok(())
    }

    #[test]
    fn duplicate_sites_must_have_the_same_specification() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let key = LoraSiteKey::new("model.layers.0.self_attn.q_proj");
        let first = registry.register(
            key.clone(),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        let second = registry.register(
            key.clone(),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        assert!(Arc::ptr_eq(&first, &second));
        assert!(registry
            .register(
                key.clone(),
                LoraLinearSpec::replicated(4, 8),
                DType::BF16,
                Device::Cpu,
            )
            .is_err());
        assert!(registry
            .register(
                key,
                LoraLinearSpec::replicated(8, 8),
                DType::F32,
                Device::Cpu,
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn site_prefix_aliases_apply_to_linear_and_expert_sites() -> Result<()> {
        let registry =
            LoraLayerRegistry::new_with_site_prefix_alias("model", "model.language_model")?;
        let linear = registry.register(
            LoraSiteKey::with_slice("model.layers.0.self_attn.q_proj", 0, 2)?,
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        let expert = registry.register_expert(
            LoraSiteKey::new("model.layers.0.mlp.experts"),
            expert_spec()?,
            DType::F32,
            Device::Cpu,
        )?;
        let unrelated = registry.register(
            LoraSiteKey::new("model_extra.proj"),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;

        assert_eq!(
            linear.key().path(),
            "model.language_model.layers.0.self_attn.q_proj"
        );
        assert_eq!(
            linear.key().slice(),
            Some(LoraSiteSlice { index: 0, count: 2 })
        );
        assert_eq!(
            expert.key().path(),
            "model.language_model.layers.0.mlp.experts"
        );
        assert_eq!(unrelated.key().path(), "model_extra.proj");
        Ok(())
    }

    #[test]
    fn linear_feature_maps_must_be_permutations() -> Result<()> {
        let spec = LoraLinearSpec::replicated(4, 3)
            .with_input_runtime_to_canonical(vec![0, 2, 1, 3])?
            .with_output_runtime_to_canonical(vec![2, 0, 1])?;
        assert_eq!(spec.input_runtime_to_canonical(), Some(&[0, 2, 1, 3][..]));
        assert_eq!(spec.output_runtime_to_canonical(), Some(&[2, 0, 1][..]));

        assert!(LoraLinearSpec::replicated(4, 3)
            .with_input_runtime_to_canonical(vec![0, 1, 2])
            .is_err());
        assert!(LoraLinearSpec::replicated(4, 3)
            .with_input_runtime_to_canonical(vec![0, 1, 2, 4])
            .is_err());
        assert!(LoraLinearSpec::replicated(4, 3)
            .with_input_runtime_to_canonical(vec![0, 1, 1, 3])
            .is_err());
        Ok(())
    }

    #[test]
    fn duplicate_sites_must_have_the_same_feature_maps() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let key = LoraSiteKey::new("linear");
        let first =
            LoraLinearSpec::replicated(4, 4).with_output_runtime_to_canonical(vec![0, 2, 1, 3])?;
        registry.register(key.clone(), first, DType::F32, Device::Cpu)?;
        let second =
            LoraLinearSpec::replicated(4, 4).with_output_runtime_to_canonical(vec![0, 1, 2, 3])?;
        assert!(registry
            .register(key, second, DType::F32, Device::Cpu)
            .is_err());
        Ok(())
    }

    #[test]
    fn new_sites_cannot_be_registered_after_finalization() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new("a"),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        assert!(registry
            .register(
                LoraSiteKey::new("b"),
                LoraLinearSpec::replicated(4, 8),
                DType::F32,
                Device::Cpu,
            )
            .is_err());
        Ok(())
    }
}
