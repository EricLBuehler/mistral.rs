use anyhow::Result;
use mistralrs_quant::{GgufArchive, GgufBindingMap};

use super::multimodal_binding_utils::{
    bind_llama_text, bind_required, bind_required_linear, bind_required_with,
    inverse_llama_permute, metadata_usize, validate_architecture, validate_projector,
    TensorInventory,
};

const FAMILY: &str = "Mistral 3/Pixtral";
const VISION_HEAD_COUNT: &str = "clip.vision.attention.head_count";

pub(crate) fn build_mistral3_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "mistral3")?;
    validate_projector(archive, "pixtral")?;
    build_mistral3_bindings_from_inventory(
        &TensorInventory::from_archive(archive),
        metadata_usize(archive, VISION_HEAD_COUNT)?,
    )
}

fn build_mistral3_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
    vision_heads: usize,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(
        inventory,
        &mut bindings,
        "language_model.model",
        "language_model.lm_head",
        FAMILY,
    )?;
    bind_required_linear(
        inventory,
        &mut bindings,
        "multi_modal_projector.linear_1",
        "mm.1",
    )?;
    bind_required_linear(
        inventory,
        &mut bindings,
        "multi_modal_projector.linear_2",
        "mm.2",
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "multi_modal_projector.norm.weight",
        "mm.input_norm.weight",
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "multi_modal_projector.patch_merger.merging_layer.weight",
        "mm.patch_merger.weight",
    )?;
    bind_pixtral_vision(inventory, &mut bindings, vision_heads)?;
    Ok(bindings)
}

fn bind_pixtral_vision(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    vision_heads: usize,
) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        "vision_tower.patch_conv.weight",
        "v.patch_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_tower.ln_pre.weight",
        "v.pre_ln.weight",
    )?;

    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let native = format!("vision_tower.transformer.layers.{layer}");
        let source = format!("v.blk.{layer}");
        bind_inverse_linear(
            inventory,
            bindings,
            &format!("{native}.attention.q_proj"),
            &format!("{source}.attn_q"),
            vision_heads,
        )?;
        bind_inverse_linear(
            inventory,
            bindings,
            &format!("{native}.attention.k_proj"),
            &format!("{source}.attn_k"),
            vision_heads,
        )?;
        for (target, role) in [
            ("attention.v_proj", "attn_v"),
            ("attention.o_proj", "attn_out"),
            ("feed_forward.gate_proj", "ffn_gate"),
            ("feed_forward.up_proj", "ffn_up"),
            ("feed_forward.down_proj", "ffn_down"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
            )?;
        }
        bind_required(
            inventory,
            bindings,
            format!("{native}.attention_norm.weight"),
            format!("{source}.ln1.weight"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.ffn_norm.weight"),
            format!("{source}.ln2.weight"),
        )?;
    }
    Ok(())
}

fn bind_inverse_linear(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
    heads: usize,
) -> Result<()> {
    for suffix in ["weight", "bias"] {
        let source = format!("{source}.{suffix}");
        if suffix == "weight" || inventory.contains(&source) {
            bind_required_with(
                inventory,
                bindings,
                format!("{native}.{suffix}"),
                source,
                |source, shape| inverse_llama_permute(source, shape, heads),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_mistral3_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_mistral3_bindings_from_inventory(&inventory, 2).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("vision_tower.transformer.layers.0.attention.q_proj.weight"),
            Some(&inverse_llama_permute("v.blk.0.attn_q.weight".to_string(), &[8, 8], 2).unwrap())
        );
    }

    #[test]
    fn rejects_invalid_vision_permutation_shape() {
        let tensors = tensor_inventory()
            .into_iter()
            .map(|(name, shape)| {
                if name == "v.blk.0.attn_q.weight" {
                    (name, vec![7, 8])
                } else {
                    (name, shape)
                }
            })
            .collect::<Vec<_>>();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let error = build_mistral3_bindings_from_inventory(&inventory, 2).unwrap_err();

        assert!(error.to_string().contains("inverse-permuted"));
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("mm.1.weight".to_string(), vec![8, 8]),
            ("mm.2.weight".to_string(), vec![8, 8]),
            ("mm.input_norm.weight".to_string(), vec![8]),
            ("mm.patch_merger.weight".to_string(), vec![8, 32]),
            ("v.patch_embd.weight".to_string(), vec![8, 3, 2, 2]),
            ("v.pre_ln.weight".to_string(), vec![8]),
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
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down",
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8, 8]));
        }
        for role in ["ln1", "ln2"] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8]));
        }
        tensors
    }
}
