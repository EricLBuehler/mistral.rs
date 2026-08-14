use anyhow::{bail, Context, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::NormalLoaderType;

use super::{
    multimodal_binding_utils::{
        bind, bind_required, bind_required_linear, bind_required_with, validate_architecture,
        validate_projector, TensorInventory,
    },
    normal_bindings::build_normal_bindings,
    normal_registry::CanonicalGgufArchitecture,
};

const FAMILY: &str = "LFM2-VL";

pub(crate) fn build_lfm2_vl_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "lfm2")?;
    validate_projector(archive, "lfm2")?;
    let text = build_normal_bindings(
        archive,
        &NormalLoaderType::Lfm2,
        CanonicalGgufArchitecture::Lfm2,
    )?;
    build_lfm2_vl_bindings_from_parts(&TensorInventory::from_archive(archive), text)
}

fn build_lfm2_vl_bindings_from_parts(
    inventory: &TensorInventory<'_>,
    text: GgufBindingMap,
) -> Result<GgufBindingMap> {
    let mut bindings = rebase_lfm2_text_bindings(&text);
    bind_lfm2_projector(inventory, &mut bindings)?;
    bind_lfm2_vision(inventory, &mut bindings)?;
    Ok(bindings)
}

fn rebase_lfm2_text_bindings(bindings: &GgufBindingMap) -> GgufBindingMap {
    bindings
        .iter()
        .map(|(native, binding)| {
            let native = native.strip_prefix("model.").map_or_else(
                || native.to_string(),
                |name| format!("model.language_model.{name}"),
            );
            (native, binding.clone())
        })
        .collect()
}

fn bind_lfm2_projector(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let root = "model.multi_modal_projector";
    bind_required_linear(inventory, bindings, &format!("{root}.linear_1"), "mm.1")?;
    bind_required_linear(inventory, bindings, &format!("{root}.linear_2"), "mm.2")?;
    if inventory.contains("mm.input_norm.weight") {
        bind_required_linear(
            inventory,
            bindings,
            &format!("{root}.layer_norm"),
            "mm.input_norm",
        )?;
    }
    Ok(())
}

fn bind_lfm2_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let root = "model.vision_tower.vision_model";
    bind_required_with(
        inventory,
        bindings,
        format!("{root}.embeddings.patch_embedding.weight"),
        "v.patch_embd.weight",
        inverse_lfm2_patch_layout,
    )?;
    bind(
        inventory,
        bindings,
        format!("{root}.embeddings.patch_embedding.bias"),
        "v.patch_embd.bias",
    );
    bind_required(
        inventory,
        bindings,
        format!("{root}.embeddings.position_embedding.weight"),
        "v.position_embd.weight",
    )?;
    bind_required_linear(
        inventory,
        bindings,
        &format!("{root}.post_layernorm"),
        "v.post_ln",
    )?;

    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let native = format!("{root}.encoder.layers.{layer}");
        let source = format!("v.blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.out_proj", "attn_out"),
            ("mlp.fc1", "ffn_up"),
            ("mlp.fc2", "ffn_down"),
            ("layer_norm1", "ln1"),
            ("layer_norm2", "ln2"),
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

fn inverse_lfm2_patch_layout(source: String, shape: &[usize]) -> Result<GgufTensorBinding> {
    if shape.len() != 4 {
        bail!("{FAMILY} GGUF patch embedding `{source}` must have rank 4, found {shape:?}");
    }
    let input_size = shape[1]
        .checked_mul(shape[2])
        .and_then(|size| size.checked_mul(shape[3]))
        .context("LFM2-VL patch embedding input size overflow")?;
    Ok(GgufTensorBinding::tensor(source)
        .permute(vec![0, 2, 3, 1])
        .reshape(vec![shape[0], input_size]))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_lfm2_vl_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let text = text_bindings();
        let bindings = build_lfm2_vl_bindings_from_parts(&inventory, text).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("model.language_model.layers.0.self_attn.q_proj.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.attn_q.weight"))
        );
        assert_eq!(
            bindings.get("model.language_model.layers.1.conv.conv.weight"),
            Some(&GgufTensorBinding::tensor("blk.1.shortconv.conv.weight").reshape(vec![8, 1, 4]))
        );
        assert_eq!(
            bindings.get("model.vision_tower.vision_model.embeddings.patch_embedding.weight"),
            Some(
                &GgufTensorBinding::tensor("v.patch_embd.weight")
                    .permute(vec![0, 2, 3, 1])
                    .reshape(vec![8, 12])
            )
        );
    }

    #[test]
    fn rejects_non_convolutional_lfm2_patch_layout() {
        let mut tensors = tensor_inventory();
        tensors
            .iter_mut()
            .find(|(name, _)| name == "v.patch_embd.weight")
            .unwrap()
            .1 = vec![8, 12, 1];
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );

        let error = build_lfm2_vl_bindings_from_parts(&inventory, text_bindings()).unwrap_err();
        assert!(error.to_string().contains("must have rank 4"));
    }

    fn text_bindings() -> GgufBindingMap {
        GgufBindingMap::new()
            .with_binding(
                "model.embed_tokens.weight",
                GgufTensorBinding::tensor("token_embd.weight"),
            )
            .with_binding(
                "model.embedding_norm.weight",
                GgufTensorBinding::tensor("token_embd_norm.weight"),
            )
            .with_binding(
                "model.layers.0.self_attn.q_proj.weight",
                GgufTensorBinding::tensor("blk.0.attn_q.weight"),
            )
            .with_binding(
                "model.layers.1.conv.conv.weight",
                GgufTensorBinding::tensor("blk.1.shortconv.conv.weight").reshape(vec![8, 1, 4]),
            )
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("token_embd_norm.weight".to_string(), vec![8]),
            ("blk.0.attn_q.weight".to_string(), vec![8, 8]),
            ("blk.1.shortconv.conv.weight".to_string(), vec![8, 4]),
            ("mm.1.weight".to_string(), vec![8, 32]),
            ("mm.1.bias".to_string(), vec![8]),
            ("mm.2.weight".to_string(), vec![8, 8]),
            ("mm.2.bias".to_string(), vec![8]),
            ("mm.input_norm.weight".to_string(), vec![32]),
            ("mm.input_norm.bias".to_string(), vec![32]),
            ("v.patch_embd.weight".to_string(), vec![8, 3, 2, 2]),
            ("v.patch_embd.bias".to_string(), vec![8]),
            ("v.position_embd.weight".to_string(), vec![16, 8]),
            ("v.post_ln.weight".to_string(), vec![8]),
            ("v.post_ln.bias".to_string(), vec![8]),
        ];
        for role in [
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2",
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8, 8]));
            tensors.push((format!("v.blk.0.{role}.bias"), vec![8]));
        }
        tensors
    }
}
