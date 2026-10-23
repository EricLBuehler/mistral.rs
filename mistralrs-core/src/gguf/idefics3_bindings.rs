use anyhow::Result;
use mistralrs_quant::{GgufArchive, GgufBindingMap};

use super::multimodal_binding_utils::{
    bind_llama_text, bind_required, bind_required_linear, validate_architecture,
    validate_projector, TensorInventory,
};

const FAMILY: &str = "Idefics3/SmolVLM";

pub(crate) fn build_idefics3_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "llama")?;
    validate_projector(archive, "idefics3")?;
    build_idefics3_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_idefics3_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(
        inventory,
        &mut bindings,
        "model.text_model",
        "lm_head",
        FAMILY,
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "model.connector.modality_projection.proj.weight",
        "mm.model.fc.weight",
    )?;
    bind_siglip_vision(inventory, &mut bindings)?;
    Ok(bindings)
}

fn bind_siglip_vision(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let root = "model.vision_model";
    bind_required_linear(
        inventory,
        bindings,
        &format!("{root}.embeddings.patch_embedding"),
        "v.patch_embd",
    )?;
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use mistralrs_quant::GgufTensorBinding;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_idefics3_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_idefics3_bindings_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("model.connector.modality_projection.proj.weight"),
            Some(&GgufTensorBinding::tensor("mm.model.fc.weight"))
        );
        assert_eq!(
            bindings.get("model.vision_model.encoder.layers.0.self_attn.q_proj.bias"),
            Some(&GgufTensorBinding::tensor("v.blk.0.attn_q.bias"))
        );
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("mm.model.fc.weight".to_string(), vec![8, 32]),
            ("v.patch_embd.weight".to_string(), vec![8, 3, 2, 2]),
            ("v.patch_embd.bias".to_string(), vec![8]),
            ("v.position_embd.weight".to_string(), vec![16, 8]),
            ("v.post_ln.weight".to_string(), vec![8]),
            ("v.post_ln.bias".to_string(), vec![8]),
        ];
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8, 8]));
        }
        for role in ["attn_norm", "ffn_norm"] {
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
