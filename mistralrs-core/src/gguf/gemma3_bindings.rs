use anyhow::Result;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use super::multimodal_binding_utils::{
    bind_required, bind_required_linear, bind_required_with, validate_architecture,
    validate_projector, TensorInventory,
};

const FAMILY: &str = "Gemma 3";

pub(crate) fn build_gemma3_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "gemma3")?;
    validate_projector(archive, "gemma3")?;
    build_gemma3_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

pub(crate) fn build_gemma3_text_bindings(
    archive: &GgufArchive,
    use_language_model_prefix: bool,
) -> Result<GgufBindingMap> {
    validate_architecture(archive, "gemma3")?;
    build_gemma3_text_bindings_from_inventory(
        &TensorInventory::from_archive(archive),
        use_language_model_prefix,
    )
}

fn build_gemma3_bindings_from_inventory(inventory: &TensorInventory<'_>) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();

    bind_text(
        inventory,
        &mut bindings,
        "language_model.model",
        "language_model.lm_head",
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "multi_modal_projector.mm_input_projection_weight",
        "mm.input_projection.weight",
    )?;
    bind_shifted_norm(
        inventory,
        &mut bindings,
        "multi_modal_projector.mm_soft_emb_norm.weight",
        "mm.soft_emb_norm.weight",
    )?;
    bind_siglip_vision(inventory, &mut bindings)?;

    Ok(bindings)
}

fn build_gemma3_text_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
    use_language_model_prefix: bool,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    let (model, lm_head) = if use_language_model_prefix {
        ("language_model.model", "language_model.lm_head")
    } else {
        ("model", "lm_head")
    };
    bind_text(inventory, &mut bindings, model, lm_head)?;
    Ok(bindings)
}

fn bind_text(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    model: &str,
    lm_head: &str,
) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        format!("{model}.embed_tokens.weight"),
        "token_embd.weight",
    )?;
    bind_shifted_norm(
        inventory,
        bindings,
        format!("{model}.norm.weight"),
        "output_norm.weight",
    )?;
    if inventory.contains("output.weight") {
        bind_required(
            inventory,
            bindings,
            format!("{lm_head}.weight"),
            "output.weight",
        )?;
    }

    for layer in inventory.require_layers("blk.", FAMILY)? {
        bind_text_layer(inventory, bindings, model, layer)?;
    }
    Ok(())
}

fn bind_text_layer(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    model: &str,
    layer: usize,
) -> Result<()> {
    let native = format!("{model}.layers.{layer}");
    let source = format!("blk.{layer}");
    for (target, role) in [
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
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
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
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
    Ok(())
}

fn bind_siglip_vision(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let root = "vision_tower.vision_model";
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_gemma3_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_gemma3_bindings_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("language_model.model.layers.0.input_layernorm.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.attn_norm.weight").affine(1.0, -1.0))
        );
        assert_eq!(
            bindings.get("vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.bias"),
            Some(&GgufTensorBinding::tensor("v.blk.0.attn_out.bias"))
        );
    }

    #[test]
    fn maps_projectorless_gemma3_to_text_roots() {
        let tensors = tensor_inventory()
            .into_iter()
            .filter(|(name, _)| !name.starts_with("mm.") && !name.starts_with("v."))
            .collect::<Vec<_>>();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_gemma3_text_bindings_from_inventory(&inventory, false).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert!(bindings.get("model.embed_tokens.weight").is_some());
        assert!(bindings
            .get("model.layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(bindings.get("lm_head.weight").is_some());
        assert!(!bindings
            .iter()
            .any(|(name, _)| name.starts_with("language_model.")
                || name.starts_with("multi_modal_projector.")
                || name.starts_with("vision_tower.")));
    }

    #[test]
    fn maps_conditional_generation_text_to_language_model_roots() {
        let tensors = tensor_inventory()
            .into_iter()
            .filter(|(name, _)| !name.starts_with("mm.") && !name.starts_with("v."))
            .collect::<Vec<_>>();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_gemma3_text_bindings_from_inventory(&inventory, true).unwrap();

        assert!(bindings
            .get("language_model.model.layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(bindings.get("language_model.lm_head.weight").is_some());
        assert!(!bindings.iter().any(|(name, _)| name.starts_with("model.")));
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("mm.input_projection.weight".to_string(), vec![8, 8]),
            ("mm.soft_emb_norm.weight".to_string(), vec![8]),
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
