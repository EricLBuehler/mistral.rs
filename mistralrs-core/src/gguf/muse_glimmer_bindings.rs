use anyhow::{bail, Context, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::MultimodalLoaderType;

use super::multimodal_binding_utils::{
    bind_required, bind_required_linear, bind_required_with, validate_architecture,
    validate_projector, TensorInventory,
};

const FAMILY: &str = "Muse-Glimmer";
pub(crate) const COLLAPSED_TEMPORAL_CONFIG_KEY: &str =
    "_mistralrs_muse_glimmer_gguf_collapsed_temporal";

pub(crate) fn build_muse_glimmer_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "muse-glimmer")?;
    validate_projector(archive, "muse-glimmer")?;
    build_muse_glimmer_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

pub(crate) fn normalize_muse_glimmer_config(
    loader_type: &MultimodalLoaderType,
    config: &str,
) -> Result<String> {
    if !matches!(loader_type, MultimodalLoaderType::MuseGlimmer) {
        return Ok(config.to_string());
    }
    let mut config: serde_json::Value =
        serde_json::from_str(config).context("invalid Muse-Glimmer config.json")?;
    let root = config
        .as_object_mut()
        .context("Muse-Glimmer config.json must be an object")?;
    root.insert(
        COLLAPSED_TEMPORAL_CONFIG_KEY.to_string(),
        serde_json::Value::Bool(true),
    );
    serde_json::to_string(&config).context("failed to normalize Muse-Glimmer GGUF config")
}

fn build_muse_glimmer_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_text(inventory, &mut bindings)?;
    bind_vision(inventory, &mut bindings)?;
    bind_projector(inventory, &mut bindings)?;
    Ok(bindings)
}

fn bind_text(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        "model.language_model.embed_tokens.weight",
        "token_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "model.language_model.norm.weight",
        "output_norm.weight",
    )?;
    bind_required(inventory, bindings, "lm_head.weight", "output.weight")?;

    for layer in inventory.require_layers("blk.", FAMILY)? {
        let native = format!("model.language_model.layers.{layer}");
        let source = format!("blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("self_attn.gate_proj", "attn_gate"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
            )?;
        }
        for (target, role) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            (
                "post_attention_layernorm.weight",
                "post_attention_norm.weight",
            ),
            ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
            ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
        ] {
            bind_shifted_norm(
                inventory,
                bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            )?;
        }

        // llama.cpp stores these synthetic scales, while the native model keeps Q/K RMS scaleless.
        for role in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            let name = format!("{source}.{role}");
            let shape = inventory.shape(&name)?;
            if shape.len() != 1 {
                bail!("{FAMILY} GGUF synthetic norm `{name}` must have rank 1, found {shape:?}");
            }
        }
    }
    Ok(())
}

fn bind_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let root = "model.vision_tower";
    bind_required_with(
        inventory,
        bindings,
        format!("{root}.patch_embedder.patch_embedding.weight"),
        "v.patch_embd.weight",
        collapsed_patch_embedding,
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{root}.patch_embedder.position_embedding_table.weight"),
        "v.position_embd.weight",
    )?;
    bind_required_linear(inventory, bindings, &format!("{root}.ln_pre"), "v.pre_ln")?;
    bind_required_linear(inventory, bindings, &format!("{root}.ln_post"), "v.post_ln")?;

    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let native = format!("{root}.layers.{layer}");
        let source = format!("v.blk.{layer}");
        for (target, role) in [
            ("attn.q_proj", "attn_q"),
            ("attn.k_proj", "attn_k"),
            ("attn.v_proj", "attn_v"),
            ("attn.proj", "attn_out"),
            ("mlp.fc1", "ffn_up"),
            ("mlp.fc2", "ffn_down"),
            ("norm1", "ln1"),
            ("norm2", "ln2"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
            )?;
        }
    }
    Ok(())
}

fn bind_projector(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind_required_linear(inventory, bindings, "model.vision_adapter.fc1", "mm.0")?;
    bind_required_linear(inventory, bindings, "model.vision_adapter.fc2", "mm.1")?;
    bind_required_linear(inventory, bindings, "model.vision_projection", "mm.2")?;
    Ok(())
}

fn bind_shifted_norm(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
) -> Result<()> {
    bind_required_with(inventory, bindings, native, source, |source, _| {
        Ok(GgufTensorBinding::tensor(source).affine(1.0, -1.0))
    })
}

fn collapsed_patch_embedding(source: String, shape: &[usize]) -> Result<GgufTensorBinding> {
    if shape.len() != 4 {
        bail!("{FAMILY} GGUF patch embedding `{source}` must have rank 4, found {shape:?}");
    }
    let input_size = shape[1..].iter().try_fold(1usize, |size, dimension| {
        size.checked_mul(*dimension)
            .context("Muse-Glimmer GGUF patch embedding size overflow")
    })?;
    Ok(GgufTensorBinding::tensor(source).reshape(vec![shape[0], input_size]))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::Write;

    use super::*;
    use crate::gguf::{
        multimodal_binding_utils::binding_sources,
        multimodal_vision_registry::resolve_native_multimodal_gguf, normal_registry::RopePairing,
    };
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};
    use tempfile::NamedTempFile;

    #[test]
    fn maps_complete_muse_glimmer_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_muse_glimmer_bindings_from_inventory(&inventory).unwrap();
        let ignored = BTreeSet::from([
            "blk.0.attn_q_norm.weight".to_string(),
            "blk.0.attn_k_norm.weight".to_string(),
        ]);
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .filter(|name| !ignored.contains(name))
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("model.language_model.norm.weight"),
            Some(&GgufTensorBinding::tensor("output_norm.weight"))
        );
        assert_eq!(
            bindings.get("model.language_model.layers.0.input_layernorm.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.attn_norm.weight").affine(1.0, -1.0))
        );
        assert_eq!(
            bindings.get("model.language_model.layers.0.self_attn.gate_proj.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.attn_gate.weight"))
        );
        assert_eq!(
            bindings.get("model.vision_tower.patch_embedder.patch_embedding.weight"),
            Some(&GgufTensorBinding::tensor("v.patch_embd.weight").reshape(vec![8, 12]))
        );
        assert_eq!(
            bindings.get("model.vision_projection.weight"),
            Some(&GgufTensorBinding::tensor("mm.2.weight"))
        );
    }

    #[test]
    fn rejects_uncollapsed_patch_embedding() {
        let mut tensors = tensor_inventory();
        tensors
            .iter_mut()
            .find(|(name, _)| name == "v.patch_embd.weight")
            .unwrap()
            .1 = vec![8, 2, 3, 2, 2];
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );

        let error = build_muse_glimmer_bindings_from_inventory(&inventory).unwrap_err();
        assert!(error.to_string().contains("must have rank 4"));
    }

    #[test]
    fn config_normalization_marks_collapsed_temporal_weights() {
        let normalized = normalize_muse_glimmer_config(
            &MultimodalLoaderType::MuseGlimmer,
            r#"{"text_config":{},"vision_config":{}}"#,
        )
        .unwrap();
        let normalized: serde_json::Value = serde_json::from_str(&normalized).unwrap();
        assert_eq!(normalized[COLLAPSED_TEMPORAL_CONFIG_KEY], true);
        assert_eq!(normalized["vision_config"], serde_json::json!({}));
    }

    #[test]
    fn generated_archive_resolves_native_muse_glimmer() -> Result<()> {
        let (_file, archive) = generated_archive()?;
        let resolved = resolve_native_multimodal_gguf(&archive)?.unwrap();

        assert_eq!(resolved.loader_type, MultimodalLoaderType::MuseGlimmer);
        assert_eq!(resolved.rope_pairing, RopePairing::Adjacent);
        assert_eq!(
            resolved
                .bindings
                .get("model.vision_tower.patch_embedder.patch_embedding.weight"),
            Some(&GgufTensorBinding::tensor("v.patch_embd.weight").reshape(vec![8, 12]))
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires local Muse-Glimmer main and mmproj GGUF paths"]
    fn local_archives_have_complete_native_bindings() -> Result<()> {
        let path = std::env::var("MISTRALRS_MUSE_GLIMMER_GGUF")
            .map_err(|_| anyhow::anyhow!("MISTRALRS_MUSE_GLIMMER_GGUF is not set"))?;
        let mmproj_path = std::env::var("MISTRALRS_MUSE_GLIMMER_MMPROJ_GGUF")
            .map_err(|_| anyhow::anyhow!("MISTRALRS_MUSE_GLIMMER_MMPROJ_GGUF is not set"))?;
        let mut archive = GgufArchive::open_file(path)?;
        archive.merge_component(GgufArchive::open_file(mmproj_path)?)?;
        let bindings = build_muse_glimmer_bindings(&archive)?;
        let expected = archive
            .tensors()
            .keys()
            .filter(|name| {
                !name.ends_with(".attn_q_norm.weight") && !name.ends_with(".attn_k_norm.weight")
            })
            .cloned()
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        Ok(())
    }

    fn generated_archive() -> Result<(NamedTempFile, GgufArchive)> {
        let tensors = tensor_inventory()
            .into_iter()
            .map(|(name, shape)| {
                let tensor = Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)?;
                Ok((name, QTensor::quantize(&tensor, GgmlDType::F32)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let tensor_refs = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let architecture =
            candle_core::quantized::gguf_file::Value::String("muse-glimmer".to_string());
        let projector =
            candle_core::quantized::gguf_file::Value::String("muse-glimmer".to_string());
        let metadata = [
            ("general.architecture", &architecture),
            ("clip.projector_type", &projector),
        ];
        let mut file = NamedTempFile::new()?;
        gguf_file::write(file.as_file_mut(), &metadata, &tensor_refs)?;
        file.as_file_mut().flush()?;
        let archive = GgufArchive::open_file(file.path())?;
        Ok((file, archive))
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![32, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![32, 8]),
            ("v.patch_embd.weight".to_string(), vec![8, 3, 2, 2]),
            ("v.position_embd.weight".to_string(), vec![16, 8]),
            ("v.pre_ln.weight".to_string(), vec![8]),
            ("v.pre_ln.bias".to_string(), vec![8]),
            ("v.post_ln.weight".to_string(), vec![8]),
            ("v.post_ln.bias".to_string(), vec![8]),
            ("mm.0.weight".to_string(), vec![8, 32]),
            ("mm.1.weight".to_string(), vec![8, 8]),
            ("mm.2.weight".to_string(), vec![8, 8]),
        ];
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "attn_gate",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8, 8]));
        }
        for role in [
            "attn_q_norm",
            "attn_k_norm",
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8]));
        }
        for role in [
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2",
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8, 8]));
            tensors.push((format!("v.blk.0.{role}.bias"), vec![8]));
        }
        tensors
    }
}
