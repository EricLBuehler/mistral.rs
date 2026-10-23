use anyhow::{bail, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use super::multimodal_binding_utils::{
    bind, bind_required, bind_required_linear, validate_architecture, validate_projector,
    TensorInventory,
};

const FAMILY: &str = "Llama 4";

pub(crate) fn build_llama4_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "llama4")?;
    validate_projector(archive, "llama4")?;
    build_llama4_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_llama4_bindings_from_inventory(inventory: &TensorInventory<'_>) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_required(
        inventory,
        &mut bindings,
        "language_model.model.embed_tokens.weight",
        "token_embd.weight",
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "language_model.model.norm.weight",
        "output_norm.weight",
    )?;
    bind(
        inventory,
        &mut bindings,
        "language_model.lm_head.weight",
        "output.weight",
    );

    for layer in inventory.require_layers("blk.", FAMILY)? {
        bind_text_layer(inventory, &mut bindings, layer)?;
    }
    bind_vision(inventory, &mut bindings)?;
    Ok(bindings)
}

fn bind_text_layer(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    layer: usize,
) -> Result<()> {
    let native = format!("language_model.model.layers.{layer}");
    let source = format!("blk.{layer}");
    for (target, role) in [
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
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
        format!("{native}.input_layernorm.weight"),
        format!("{source}.attn_norm.weight"),
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{native}.post_attention_layernorm.weight"),
        format!("{source}.ffn_norm.weight"),
    )?;

    let feed_forward = format!("{native}.feed_forward");
    if inventory.contains(&format!("{source}.ffn_gate_inp.weight")) {
        bind_moe(inventory, bindings, &feed_forward, &source)
    } else {
        bind_dense(inventory, bindings, &feed_forward, &source)
    }
}

fn bind_dense(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    for (target, role) in [
        ("gate_proj", "ffn_gate"),
        ("up_proj", "ffn_up"),
        ("down_proj", "ffn_down"),
    ] {
        bind_required_linear(
            inventory,
            bindings,
            &format!("{native}.{target}"),
            &format!("{source}.{role}"),
        )?;
    }
    Ok(())
}

fn bind_moe(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    bind_required_linear(
        inventory,
        bindings,
        &format!("{native}.router"),
        &format!("{source}.ffn_gate_inp"),
    )?;
    for (target, role) in [
        ("gate_proj", "ffn_gate_shexp"),
        ("up_proj", "ffn_up_shexp"),
        ("down_proj", "ffn_down_shexp"),
    ] {
        bind_required_linear(
            inventory,
            bindings,
            &format!("{native}.shared_expert.{target}"),
            &format!("{source}.{role}"),
        )?;
    }

    let fused = format!("{source}.ffn_gate_up_exps.weight");
    if inventory.contains(&fused) {
        let shape = inventory.shape(&fused)?;
        if shape.len() != 3 || !shape[1].is_multiple_of(2) {
            bail!(
                "Llama 4 fused expert tensor `{fused}` must have shape \
                 [experts, 2 * intermediate, hidden]"
            );
        }
        let intermediate = shape[1] / 2;
        let binding = GgufTensorBinding::tensor(fused);
        bindings.insert(
            format!("{native}.experts.gate_proj.weight"),
            binding.clone().slice(1, 0, intermediate),
        );
        bindings.insert(
            format!("{native}.experts.up_proj.weight"),
            binding.slice(1, intermediate, intermediate),
        );
    } else {
        for (target, role) in [("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps")] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.experts.{target}"),
                &format!("{source}.{role}"),
            )?;
        }
    }
    bind_required_linear(
        inventory,
        bindings,
        &format!("{native}.experts.down_proj"),
        &format!("{source}.ffn_down_exps"),
    )?;
    Ok(())
}

fn bind_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        "multi_modal_projector.linear_1.weight",
        "mm.model.fc.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_model.class_embedding",
        "v.class_embd",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_model.patch_embedding.linear.weight",
        "v.patch_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_model.positional_embedding_vlm",
        "v.position_embd.weight",
    )?;
    bind_required_linear(
        inventory,
        bindings,
        "vision_model.layernorm_pre",
        "v.pre_ln",
    )?;
    bind_required_linear(
        inventory,
        bindings,
        "vision_model.layernorm_post",
        "v.post_ln",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_model.vision_adapter.mlp.fc1.weight",
        "mm.model.mlp.1.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "vision_model.vision_adapter.mlp.fc2.weight",
        "mm.model.mlp.2.weight",
    )?;

    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let native = format!("vision_model.model.layers.{layer}");
        let source = format!("v.blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_out"),
            ("mlp.fc1", "ffn_up"),
            ("mlp.fc2", "ffn_down"),
            ("input_layernorm", "ln1"),
            ("post_attention_layernorm", "attn_post_norm"),
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

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_llama4_moe_inventory_to_fast_expert_keys() {
        let tensors = tensor_inventory(true);
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_llama4_bindings_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        for (target, source) in [
            ("gate_proj", "ffn_gate_exps"),
            ("up_proj", "ffn_up_exps"),
            ("down_proj", "ffn_down_exps"),
        ] {
            assert_eq!(
                bindings.get(&format!(
                    "language_model.model.layers.0.feed_forward.experts.{target}.weight"
                )),
                Some(&GgufTensorBinding::tensor(format!("blk.0.{source}.weight")))
            );
        }
    }

    #[test]
    fn maps_llama4_dense_feed_forward() {
        let tensors = tensor_inventory(false);
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_llama4_bindings_from_inventory(&inventory).unwrap();

        assert_eq!(
            bindings.get("language_model.model.layers.0.feed_forward.gate_proj.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.ffn_gate.weight"))
        );
    }

    fn tensor_inventory(moe: bool) -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("blk.0.attn_norm.weight".to_string(), vec![8]),
            ("blk.0.ffn_norm.weight".to_string(), vec![8]),
            ("mm.model.fc.weight".to_string(), vec![8, 8]),
            ("mm.model.mlp.1.weight".to_string(), vec![32, 8]),
            ("mm.model.mlp.2.weight".to_string(), vec![8, 32]),
            ("v.class_embd".to_string(), vec![8]),
            ("v.patch_embd.weight".to_string(), vec![8, 12]),
            ("v.position_embd.weight".to_string(), vec![17, 8]),
            ("v.pre_ln.weight".to_string(), vec![8]),
            ("v.pre_ln.bias".to_string(), vec![8]),
            ("v.post_ln.weight".to_string(), vec![8]),
            ("v.post_ln.bias".to_string(), vec![8]),
        ];
        for role in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8, 8]));
        }
        if moe {
            tensors.extend([
                ("blk.0.ffn_gate_inp.weight".to_string(), vec![4, 8]),
                ("blk.0.ffn_gate_shexp.weight".to_string(), vec![16, 8]),
                ("blk.0.ffn_up_shexp.weight".to_string(), vec![16, 8]),
                ("blk.0.ffn_down_shexp.weight".to_string(), vec![8, 16]),
                ("blk.0.ffn_gate_exps.weight".to_string(), vec![4, 16, 8]),
                ("blk.0.ffn_up_exps.weight".to_string(), vec![4, 16, 8]),
                ("blk.0.ffn_down_exps.weight".to_string(), vec![4, 8, 16]),
            ]);
        } else {
            for role in ["ffn_gate", "ffn_up", "ffn_down"] {
                tensors.push((format!("blk.0.{role}.weight"), vec![8, 8]));
            }
        }
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_out",
            "ffn_up",
            "ffn_down",
            "ln1",
            "attn_post_norm",
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8, 8]));
            tensors.push((format!("v.blk.0.{role}.bias"), vec![8]));
        }
        tensors
    }
}
