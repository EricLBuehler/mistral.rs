use anyhow::{bail, Context, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::NormalLoaderType;

use super::{
    multimodal_binding_utils::{
        bind_required, bind_required_linear, validate_architecture, validate_projector,
        TensorInventory,
    },
    normal_bindings::build_normal_bindings,
    normal_registry::CanonicalGgufArchitecture,
};

const FAMILY: &str = "LFM3-VL";
const PATCH_EMBEDDING: &str = "v.patch_embd.weight";
const SECOND_TEMPORAL_PATCH: &str = "v.patch_embd.weight.1";

pub(crate) fn build_lfm3_vl_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "lfm3")?;
    validate_projector(archive, "lfm3vl")?;
    let text = build_normal_bindings(
        archive,
        &NormalLoaderType::Lfm3,
        CanonicalGgufArchitecture::Lfm3,
    )?;
    build_lfm3_vl_bindings_from_parts(&TensorInventory::from_archive(archive), text)
}

fn build_lfm3_vl_bindings_from_parts(
    inventory: &TensorInventory<'_>,
    text: GgufBindingMap,
) -> Result<GgufBindingMap> {
    let mut bindings = text
        .iter()
        .map(|(native, binding)| {
            let native = native.strip_prefix("model.").map_or_else(
                || native.to_string(),
                |name| format!("model.language_model.{name}"),
            );
            (native, binding.clone())
        })
        .collect::<GgufBindingMap>();
    for (target, source) in [
        ("linear_1", "mm.0"),
        ("linear_2", "mm.2"),
        ("layer_norm", "mm.input_norm"),
    ] {
        bind_required_linear(
            inventory,
            &mut bindings,
            &format!("model.multi_modal_projector.{target}"),
            source,
        )?;
    }
    let root = "model.vision_tower";
    bindings.insert(
        format!("{root}.patchifier.weight"),
        patchifier_binding(inventory)?,
    );
    bind_required(
        inventory,
        &mut bindings,
        format!("{root}.embedding_norm.weight"),
        "v.embedding_norm.weight",
    )?;
    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let native = format!("{root}.layers.{layer}");
        let source = format!("v.blk.{layer}");
        let operator: &[(&str, &str)] =
            if inventory.contains(&format!("{source}.conv2d_in_proj.weight")) {
                &[
                    ("conv.in_proj", "conv2d_in_proj"),
                    ("conv.out_proj", "conv2d_out_proj"),
                    ("conv.spatial_conv", "conv2d_spatial"),
                ]
            } else {
                &[
                    ("self_attn.q_proj", "attn_q"),
                    ("self_attn.k_proj", "attn_k"),
                    ("self_attn.v_proj", "attn_v"),
                    ("self_attn.out_proj", "attn_out"),
                    ("self_attn.q_layernorm", "attn_q_norm"),
                    ("self_attn.k_layernorm", "attn_k_norm"),
                ]
            };
        for (target, role) in operator.iter().chain(
            [
                ("feed_forward.w1", "ffn_gate"),
                ("feed_forward.w3", "ffn_up"),
                ("feed_forward.w2", "ffn_down"),
                ("operator_norm", "ln1"),
                ("ffn_norm", "ln2"),
            ]
            .iter(),
        ) {
            bind_required(
                inventory,
                &mut bindings,
                format!("{native}.{target}.weight"),
                format!("{source}.{role}.weight"),
            )?;
        }
    }
    Ok(bindings)
}

fn patchifier_binding(inventory: &TensorInventory<'_>) -> Result<GgufTensorBinding> {
    let shape = inventory.shape(PATCH_EMBEDDING)?;
    if shape.len() != 4 {
        bail!("{FAMILY} GGUF patch embedding must have rank 4, found {shape:?}");
    }
    let mut patches = vec![GgufTensorBinding::tensor(PATCH_EMBEDDING)];
    if inventory.contains(SECOND_TEMPORAL_PATCH) {
        let second_shape = inventory.shape(SECOND_TEMPORAL_PATCH)?;
        if second_shape != shape {
            bail!("{FAMILY} GGUF temporal patch shapes differ: {shape:?} and {second_shape:?}");
        }
        patches.push(GgufTensorBinding::tensor(SECOND_TEMPORAL_PATCH));
    }
    let input_size = shape[1..]
        .iter()
        .try_fold(patches.len(), |size, &dim| size.checked_mul(dim))
        .context("LFM3-VL patch embedding input size overflow")?;
    // The converter splits [output, channel, time, patch_h, patch_w] along time.
    Ok(GgufTensorBinding::stack(patches, 2).reshape(vec![shape[0], input_size]))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    const HIDDEN: usize = 8;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 12;
    const PROJECTOR: usize = 16;
    const CHANNELS: usize = 3;
    const PATCH: usize = 2;
    const TEMPORAL: usize = 2;

    #[test]
    fn maps_hybrid_vision_and_rebases_text() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let text = GgufBindingMap::new()
            .with_binding(
                "model.embed_tokens.weight",
                GgufTensorBinding::tensor("token_embd.weight"),
            )
            .with_binding(
                "model.rope_freqs.weight",
                GgufTensorBinding::tensor("rope_freqs.weight"),
            )
            .with_binding("lm_head.weight", GgufTensorBinding::tensor("output.weight"));
        let bindings = build_lfm3_vl_bindings_from_parts(&inventory, text).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(binding_sources(&bindings), expected);
        for (native, source) in [
            (
                "model.language_model.embed_tokens.weight",
                "token_embd.weight",
            ),
            (
                "model.language_model.rope_freqs.weight",
                "rope_freqs.weight",
            ),
            ("lm_head.weight", "output.weight"),
            ("model.multi_modal_projector.linear_1.weight", "mm.0.weight"),
            (
                "model.vision_tower.layers.0.conv.spatial_conv.weight",
                "v.blk.0.conv2d_spatial.weight",
            ),
            (
                "model.vision_tower.layers.1.self_attn.q_layernorm.weight",
                "v.blk.1.attn_q_norm.weight",
            ),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source))
            );
        }
        assert_eq!(
            bindings.get("model.vision_tower.patchifier.weight"),
            Some(
                &GgufTensorBinding::stack(
                    vec![
                        GgufTensorBinding::tensor(PATCH_EMBEDDING),
                        GgufTensorBinding::tensor(SECOND_TEMPORAL_PATCH),
                    ],
                    2,
                )
                .reshape(vec![HIDDEN, CHANNELS * TEMPORAL * PATCH * PATCH])
            )
        );
    }

    #[test]
    fn rejects_inconsistent_temporal_patch_shapes() {
        let first = [HIDDEN, CHANNELS, PATCH, PATCH];
        let second = [HIDDEN, CHANNELS, PATCH, PATCH + 1];
        let inventory = TensorInventory::new([
            (PATCH_EMBEDDING, first.as_slice()),
            (SECOND_TEMPORAL_PATCH, second.as_slice()),
        ]);
        assert!(patchifier_binding(&inventory)
            .unwrap_err()
            .to_string()
            .contains("temporal patch shapes differ"));
    }

    #[test]
    fn rejects_non_convolutional_patch_layout() {
        let shape = [HIDDEN, CHANNELS * TEMPORAL * PATCH * PATCH];
        let inventory = TensorInventory::new([(PATCH_EMBEDDING, shape.as_slice())]);
        assert!(patchifier_binding(&inventory)
            .unwrap_err()
            .to_string()
            .contains("must have rank 4"));
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = [
            ("token_embd.weight", vec![PROJECTOR, HIDDEN]),
            ("rope_freqs.weight", vec![HEAD_DIM]),
            ("output.weight", vec![PROJECTOR, HIDDEN]),
            ("mm.0.weight", vec![PROJECTOR, HIDDEN * PATCH * PATCH]),
            ("mm.0.bias", vec![PROJECTOR]),
            ("mm.2.weight", vec![HIDDEN, PROJECTOR]),
            ("mm.2.bias", vec![HIDDEN]),
            ("mm.input_norm.weight", vec![HIDDEN]),
            ("mm.input_norm.bias", vec![HIDDEN]),
            (PATCH_EMBEDDING, vec![HIDDEN, CHANNELS, PATCH, PATCH]),
            (SECOND_TEMPORAL_PATCH, vec![HIDDEN, CHANNELS, PATCH, PATCH]),
            ("v.embedding_norm.weight", vec![HIDDEN]),
        ]
        .into_iter()
        .map(|(name, shape)| (name.to_string(), shape))
        .collect::<Vec<_>>();
        for layer in 0..2 {
            for (role, shape) in [
                ("ffn_gate", vec![INTERMEDIATE, HIDDEN]),
                ("ffn_up", vec![INTERMEDIATE, HIDDEN]),
                ("ffn_down", vec![HIDDEN, INTERMEDIATE]),
                ("ln1", vec![HIDDEN]),
                ("ln2", vec![HIDDEN]),
            ] {
                tensors.push((format!("v.blk.{layer}.{role}.weight"), shape));
            }
        }
        for (role, shape) in [
            ("conv2d_in_proj", vec![3 * HIDDEN, HIDDEN]),
            ("conv2d_out_proj", vec![HIDDEN, HIDDEN]),
            ("conv2d_spatial", vec![HIDDEN, 1, 3, 3]),
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), shape));
        }
        for (role, shape) in [
            ("attn_q", vec![HIDDEN, HIDDEN]),
            ("attn_k", vec![HEAD_DIM, HIDDEN]),
            ("attn_v", vec![HEAD_DIM, HIDDEN]),
            ("attn_out", vec![HIDDEN, HIDDEN]),
            ("attn_q_norm", vec![HEAD_DIM]),
            ("attn_k_norm", vec![HEAD_DIM]),
        ] {
            tensors.push((format!("v.blk.1.{role}.weight"), shape));
        }
        tensors
    }
}
