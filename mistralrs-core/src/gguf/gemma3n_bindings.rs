use anyhow::{bail, Context, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::vision_models::gemma3n::vision::{gemma3n_mobilenet_def, BlockType};

use super::multimodal_binding_utils::{
    bind, bind_required, bind_required_linear, bind_required_with, metadata_string, metadata_usize,
    validate_architecture, validate_projector, TensorInventory,
};

const FAMILY: &str = "Gemma 3n";
const AUDIO_PROJECTOR_TYPE: &str = "clip.audio.projector_type";
const ALTUP_NUM_INPUTS: &str = "gemma3n.altup.num_inputs";
const MIN_ALTUP_INPUTS: usize = 2;

pub(crate) fn build_gemma3n_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "gemma3n")?;
    validate_projector(archive, "gemma3nv")?;
    let audio_projector = metadata_string(archive, AUDIO_PROJECTOR_TYPE)?
        .context("GGUF audio projector metadata is required")?;
    if audio_projector != "gemma3na" {
        bail!("expected `gemma3na` audio projector, found `{audio_projector}`");
    }
    build_gemma3n_bindings_from_inventory(
        &TensorInventory::from_archive(archive),
        metadata_usize(archive, ALTUP_NUM_INPUTS)?,
    )
}

fn build_gemma3n_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
    altup_num_inputs: usize,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_multimodal_embeddings(inventory, &mut bindings)?;
    bind_text(inventory, &mut bindings, altup_num_inputs)?;
    bind_vision(inventory, &mut bindings)?;
    bind_audio(inventory, &mut bindings)?;
    Ok(bindings)
}

fn bind_multimodal_embeddings(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    for (native, source) in [
        ("model.embed_vision.embedding.weight", "mm.embedding.weight"),
        (
            "model.embed_vision.hard_embedding_norm.weight",
            "mm.hard_emb_norm.weight",
        ),
        (
            "model.embed_vision.soft_embedding_norm.weight",
            "mm.soft_emb_norm.weight",
        ),
        (
            "model.embed_vision.embedding_projection.weight",
            "mm.input_projection.weight",
        ),
        (
            "model.embed_audio.embedding.weight",
            "mm.a.embedding.weight",
        ),
        (
            "model.embed_audio.hard_embedding_norm.weight",
            "mm.a.hard_emb_norm.weight",
        ),
        (
            "model.embed_audio.soft_embedding_norm.weight",
            "mm.a.soft_emb_norm.weight",
        ),
        (
            "model.embed_audio.embedding_projection.weight",
            "mm.a.input_projection.weight",
        ),
    ] {
        bind_required(inventory, bindings, native, source)?;
    }
    Ok(())
}

fn bind_text(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    altup_num_inputs: usize,
) -> Result<()> {
    let layers = inventory.require_layers("blk.", FAMILY)?;
    let text_layers = layers.len();
    let token_shape = matrix_shape(inventory, "token_embd.weight")?;
    let vision_vocab = matrix_shape(inventory, "mm.embedding.weight")?[0];
    let audio_vocab = matrix_shape(inventory, "mm.a.embedding.weight")?[0];
    let multimodal_vocab = vision_vocab
        .checked_add(audio_vocab)
        .context("Gemma 3n multimodal vocabulary size overflow")?;
    let per_layer_vocab = token_shape[0]
        .checked_sub(multimodal_vocab)
        .context("Gemma 3n multimodal vocabulary exceeds the language vocabulary")?;

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
    bind(
        inventory,
        bindings,
        "model.language_model.lm_head.weight",
        "output.weight",
    );

    let per_layer_shape = matrix_shape(inventory, "per_layer_token_embd.weight")?;
    if per_layer_shape[0] != per_layer_vocab && per_layer_shape[0] != token_shape[0] {
        bail!(
            "GGUF tensor `per_layer_token_embd.weight` has {} rows; expected {per_layer_vocab} \
             or the converter-padded size {}",
            per_layer_shape[0],
            token_shape[0]
        );
    }
    let per_layer_binding = GgufTensorBinding::tensor("per_layer_token_embd.weight");
    bindings.insert(
        "model.language_model.embed_tokens_per_layer.weight",
        if per_layer_shape[0] == per_layer_vocab {
            per_layer_binding
        } else {
            per_layer_binding.slice(0, 0, per_layer_vocab)
        },
    );

    let model_projection_shape = matrix_shape(inventory, "per_layer_model_proj.weight")?;
    if !per_layer_shape[1].is_multiple_of(text_layers) {
        bail!(
            "GGUF tensor `per_layer_token_embd.weight` width {} is not divisible by \
             the {text_layers} text layers",
            per_layer_shape[1]
        );
    }
    if model_projection_shape != [per_layer_shape[1], token_shape[1]] {
        bail!(
            "GGUF tensor `per_layer_model_proj.weight` has shape {model_projection_shape:?}; \
             expected [{}, {}]",
            per_layer_shape[1],
            token_shape[1]
        );
    }
    let projection_norm_shape = inventory.shape("per_layer_proj_norm.weight")?;
    let per_layer_width = per_layer_shape[1] / text_layers;
    if projection_norm_shape != [per_layer_width] {
        bail!(
            "GGUF tensor `per_layer_proj_norm.weight` has shape {projection_norm_shape:?}; \
             expected [{per_layer_width}]"
        );
    }
    bind_required(
        inventory,
        bindings,
        "model.language_model.per_layer_model_projection.weight",
        "per_layer_model_proj.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "model.language_model.per_layer_projection_norm.weight",
        "per_layer_proj_norm.weight",
    )?;

    bind_altup_stack(
        inventory,
        bindings,
        "model.language_model.altup_projections",
        "altup_proj.weight",
        altup_num_inputs,
        token_shape[1],
    )?;
    bind_altup_stack(
        inventory,
        bindings,
        "model.language_model.altup_unembed_projections",
        "altup_unembd_proj.weight",
        altup_num_inputs,
        token_shape[1],
    )?;

    for layer in layers {
        bind_text_layer(inventory, bindings, layer)?;
    }
    Ok(())
}

fn bind_altup_stack(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
    altup_num_inputs: usize,
    hidden_size: usize,
) -> Result<()> {
    if altup_num_inputs < MIN_ALTUP_INPUTS {
        bail!("Gemma 3n requires at least {MIN_ALTUP_INPUTS} AltUp inputs");
    }
    let expected = altup_num_inputs - 1;
    let shape = inventory.shape(source)?;
    if shape != [expected, hidden_size, hidden_size] {
        bail!(
            "GGUF tensor `{source}` has shape {shape:?}; expected \
             [{expected}, {hidden_size}, {hidden_size}]"
        );
    }
    for index in 0..expected {
        bindings.insert(
            format!("{native}.{index}.weight"),
            GgufTensorBinding::tensor(source)
                .slice(0, index, 1)
                .reshape(vec![hidden_size, hidden_size]),
        );
    }
    Ok(())
}

fn bind_text_layer(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    layer: usize,
) -> Result<()> {
    let native = format!("model.language_model.layers.{layer}");
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
        ("per_layer_input_gate.weight", "inp_gate.weight"),
        ("per_layer_projection.weight", "proj.weight"),
        ("post_per_layer_input_norm.weight", "post_norm.weight"),
        ("altup.correct_output_scale", "altup_correct_scale.weight"),
        ("altup.correction_coefs.weight", "altup_correct_coef.weight"),
        ("altup.prediction_coefs.weight", "altup_predict_coef.weight"),
        ("altup.modality_router.weight", "altup_router.weight"),
        ("altup.router_norm.weight", "altup_router_norm.weight"),
        ("laurel.linear_left.weight", "laurel_l.weight"),
        ("laurel.linear_right.weight", "laurel_r.weight"),
        ("laurel.post_laurel_norm.weight", "laurel_post_norm.weight"),
    ] {
        bind_required(
            inventory,
            bindings,
            format!("{native}.{target}"),
            format!("{source}.{role}"),
        )?;
    }
    Ok(())
}

fn bind_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let root = "model.vision_tower.timm_model";
    bind_required(
        inventory,
        bindings,
        format!("{root}.conv_stem.conv.weight"),
        "v.conv_stem.conv.weight",
    )?;
    bind_vector(
        inventory,
        bindings,
        &format!("{root}.conv_stem.conv.bias"),
        "v.conv_stem.conv.bias",
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{root}.conv_stem.bn.weight"),
        "v.conv_stem.bn.weight",
    )?;
    for (target, source) in [
        (
            "msfa.ffn.pw_exp.conv.weight",
            "v.msfa.ffn.pw_exp.conv.weight",
        ),
        ("msfa.ffn.pw_exp.bn.weight", "v.msfa.ffn.pw_exp.bn.weight"),
        (
            "msfa.ffn.pw_proj.conv.weight",
            "v.msfa.ffn.pw_proj.conv.weight",
        ),
        ("msfa.ffn.pw_proj.bn.weight", "v.msfa.ffn.pw_proj.bn.weight"),
        ("msfa.norm.weight", "v.msfa.norm.weight"),
    ] {
        bind_required(inventory, bindings, format!("{root}.{target}"), source)?;
    }

    for (stage, blocks) in gemma3n_mobilenet_def().iter().enumerate() {
        for (block, block_type) in blocks.iter().enumerate() {
            bind_vision_block(inventory, bindings, stage, block, block_type)?;
        }
    }
    Ok(())
}

fn bind_vision_block(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    stage: usize,
    block: usize,
    block_type: &BlockType,
) -> Result<()> {
    let native = format!("model.vision_tower.timm_model.blocks.{stage}.{block}");
    let source = format!("v.blk.{stage}.{block}");
    for role in vision_block_roles(block_type) {
        if role == "layer_scale.gamma" {
            bind_vector(
                inventory,
                bindings,
                &format!("{native}.{role}"),
                &format!("{source}.{role}"),
            )?;
        } else {
            bind_required(
                inventory,
                bindings,
                format!("{native}.{role}"),
                format!("{source}.{role}"),
            )?;
        }
    }
    Ok(())
}

fn vision_block_roles(block_type: &BlockType) -> Vec<&'static str> {
    match block_type {
        BlockType::EdgeResidual { .. } => vec![
            "conv_exp.weight",
            "bn1.weight",
            "conv_pwl.weight",
            "bn2.weight",
        ],
        BlockType::UniversalInvertedResidual {
            start_kernel_size,
            mid_kernel_size,
            ..
        } => {
            let mut roles = Vec::new();
            if *start_kernel_size > 0 {
                roles.extend(["dw_start.conv.weight", "dw_start.bn.weight"]);
            }
            roles.extend([
                "pw_exp.conv.weight",
                "pw_exp.bn.weight",
                "pw_proj.conv.weight",
                "pw_proj.bn.weight",
                "layer_scale.gamma",
            ]);
            if *mid_kernel_size > 0 {
                roles.splice(
                    roles.len() - 3..roles.len() - 3,
                    ["dw_mid.conv.weight", "dw_mid.bn.weight"],
                );
            }
            roles
        }
        BlockType::MultiQueryAttention { kv_stride, .. } => {
            let mut roles = vec!["norm.weight", "attn.query.proj.weight"];
            if *kv_stride > 1 {
                roles.extend(["attn.key.down_conv.weight", "attn.key.norm.weight"]);
            }
            roles.push("attn.key.proj.weight");
            if *kv_stride > 1 {
                roles.extend(["attn.value.down_conv.weight", "attn.value.norm.weight"]);
            }
            roles.extend([
                "attn.value.proj.weight",
                "attn.output.proj.weight",
                "layer_scale.gamma",
            ]);
            roles
        }
    }
}

fn bind_audio(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    for index in 0..2 {
        bind_required(
            inventory,
            bindings,
            format!("model.audio_tower.subsample_conv_projection.conv_{index}.conv.weight"),
            format!("a.conv1d.{index}.weight"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("model.audio_tower.subsample_conv_projection.conv_{index}.norm.weight"),
            format!("a.conv1d.{index}.norm.weight"),
        )?;
    }
    bind_required(
        inventory,
        bindings,
        "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
        "a.pre_encode.out.weight",
    )?;

    for layer in inventory.require_layers("a.blk.", FAMILY)? {
        let native = format!("model.audio_tower.conformer.{layer}");
        let source = format!("a.blk.{layer}");
        for (target, role) in [
            ("norm.weight", "layer_pre_norm.weight"),
            ("attention.pre_attn_norm.weight", "ln1.weight"),
            ("attention.attn.q_proj.weight", "attn_q.weight"),
            ("attention.attn.k_proj.weight", "attn_k.weight"),
            ("attention.attn.v_proj.weight", "attn_v.weight"),
            ("attention.attn.per_dim_scale", "per_dim_scale"),
            (
                "attention.attn.relative_position_embedding.pos_proj.weight",
                "linear_pos.weight",
            ),
            ("attention.post.weight", "attn_out.weight"),
            ("attention.post_norm.weight", "ln2.weight"),
            ("ffw_layer_start.pre_layer_norm.weight", "ffn_norm.weight"),
            ("ffw_layer_start.ffw_layer_1.weight", "ffn_up.weight"),
            ("ffw_layer_start.ffw_layer_2.weight", "ffn_down.weight"),
            (
                "ffw_layer_start.post_layer_norm.weight",
                "ffn_post_norm.weight",
            ),
            ("lconv1d.pre_layer_norm.weight", "conv_norm.weight"),
            ("lconv1d.linear_start.weight", "conv_pw1.weight"),
            ("lconv1d.depthwise_conv1d.weight", "conv_dw.weight"),
            ("lconv1d.conv_norm.weight", "norm_conv.weight"),
            ("lconv1d.linear_end.weight", "conv_pw2.weight"),
            ("ffw_layer_end.pre_layer_norm.weight", "ffn_norm_1.weight"),
            ("ffw_layer_end.ffw_layer_1.weight", "ffn_up_1.weight"),
            ("ffw_layer_end.ffw_layer_2.weight", "ffn_down_1.weight"),
            (
                "ffw_layer_end.post_layer_norm.weight",
                "ffn_post_norm_1.weight",
            ),
        ] {
            bind_required(
                inventory,
                bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            )?;
        }
    }
    Ok(())
}

fn matrix_shape<'a>(inventory: &'a TensorInventory<'_>, name: &str) -> Result<&'a [usize]> {
    let shape = inventory.shape(name)?;
    if shape.len() != 2 {
        bail!("GGUF tensor `{name}` must be a matrix, found shape {shape:?}");
    }
    Ok(shape)
}

fn bind_vector(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    bind_required_with(
        inventory,
        bindings,
        native,
        source,
        |source, shape| match shape {
            [width] if *width > 0 => Ok(GgufTensorBinding::tensor(source)),
            [1, width, 1, 1] if *width > 0 => {
                Ok(GgufTensorBinding::tensor(source).reshape(vec![*width]))
            }
            _ => bail!("GGUF tensor `{source}` must be a vector, found shape {shape:?}"),
        },
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::multimodal_binding_utils::binding_sources;

    #[test]
    fn maps_complete_gemma3n_inventory() {
        let tensors = tensor_inventory();
        let inventory = inventory(&tensors);
        let bindings = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            expected
                .iter()
                .filter(|name| name.starts_with("v."))
                .count(),
            548
        );
        assert_eq!(
            expected
                .iter()
                .filter(|name| name.starts_with("mm."))
                .count(),
            8
        );
        assert_eq!(
            expected
                .iter()
                .filter(|name| name.starts_with("a."))
                .count(),
            269
        );
        assert_eq!(
            bindings.get("model.language_model.embed_tokens_per_layer.weight"),
            Some(&GgufTensorBinding::tensor("per_layer_token_embd.weight").slice(0, 0, 8))
        );
        assert_eq!(
            bindings.get("model.language_model.altup_projections.1.weight"),
            Some(
                &GgufTensorBinding::tensor("altup_proj.weight")
                    .slice(0, 1, 1)
                    .reshape(vec![8, 8])
            )
        );
        assert_eq!(
            bindings.get("model.vision_tower.timm_model.conv_stem.conv.bias"),
            Some(&GgufTensorBinding::tensor("v.conv_stem.conv.bias").reshape(vec![8]))
        );
        assert_eq!(
            bindings.get(
                "model.audio_tower.conformer.0.attention.attn.relative_position_embedding.pos_proj.weight"
            ),
            Some(&GgufTensorBinding::tensor("a.blk.0.linear_pos.weight"))
        );
    }

    #[test]
    fn accepts_unpadded_per_layer_embedding_and_vector_layer_scales() {
        let mut tensors = tensor_inventory();
        set_shape(&mut tensors, "per_layer_token_embd.weight", vec![8, 8]);
        for (name, shape) in &mut tensors {
            if name.ends_with("layer_scale.gamma") {
                *shape = vec![8];
            }
        }
        let inventory = inventory(&tensors);
        let bindings = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap();

        assert_eq!(
            bindings.get("model.language_model.embed_tokens_per_layer.weight"),
            Some(&GgufTensorBinding::tensor("per_layer_token_embd.weight"))
        );
    }

    #[test]
    fn preserves_native_audio_dtypes_and_quantized_linears() {
        let tensors = tensor_inventory();
        let inventory = inventory(&tensors);
        let bindings = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap();

        for (native, source) in [
            (
                "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
                "a.pre_encode.out.weight",
            ),
            (
                "model.audio_tower.conformer.0.attention.attn.q_proj.weight",
                "a.blk.0.attn_q.weight",
            ),
            (
                "model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight",
                "a.blk.0.conv_dw.weight",
            ),
            (
                "model.embed_audio.embedding_projection.weight",
                "mm.a.input_projection.weight",
            ),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source))
            );
        }
    }

    #[test]
    fn rejects_invalid_altup_stack() {
        let mut tensors = tensor_inventory();
        set_shape(&mut tensors, "altup_proj.weight", vec![2, 8, 8]);
        let inventory = inventory(&tensors);
        let error = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap_err();
        assert!(error.to_string().contains("expected [3, 8, 8]"));
    }

    #[test]
    fn rejects_missing_audio_convolution_norm() {
        let mut tensors = tensor_inventory();
        tensors.retain(|(name, _)| name != "a.conv1d.0.norm.weight");
        let inventory = inventory(&tensors);
        let error = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap_err();
        assert!(error.to_string().contains("a.conv1d.0.norm.weight"));
    }

    #[test]
    fn rejects_malformed_converter_padding() {
        let mut tensors = tensor_inventory();
        set_shape(&mut tensors, "per_layer_token_embd.weight", vec![11, 8]);
        let inventory = inventory(&tensors);
        let error = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap_err();
        assert!(error.to_string().contains("expected 8"));
    }

    #[test]
    fn rejects_noncontiguous_audio_layers() {
        let mut tensors = tensor_inventory();
        tensors.retain(|(name, _)| !name.starts_with("a.blk.1."));
        let inventory = inventory(&tensors);
        let error = build_gemma3n_bindings_from_inventory(&inventory, 4).unwrap_err();
        assert!(error.to_string().contains("non-contiguous"));
    }

    fn inventory<'a>(tensors: &'a [(String, Vec<usize>)]) -> TensorInventory<'a> {
        TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        )
    }

    fn set_shape(tensors: &mut [(String, Vec<usize>)], name: &str, shape: Vec<usize>) {
        tensors
            .iter_mut()
            .find(|(candidate, _)| candidate == name)
            .unwrap()
            .1 = shape;
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![12, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![12, 8]),
            ("per_layer_token_embd.weight".to_string(), vec![12, 8]),
            ("per_layer_model_proj.weight".to_string(), vec![8, 8]),
            ("per_layer_proj_norm.weight".to_string(), vec![8]),
            ("altup_proj.weight".to_string(), vec![3, 8, 8]),
            ("altup_unembd_proj.weight".to_string(), vec![3, 8, 8]),
            ("mm.embedding.weight".to_string(), vec![2, 8]),
            ("mm.hard_emb_norm.weight".to_string(), vec![8]),
            ("mm.soft_emb_norm.weight".to_string(), vec![8]),
            ("mm.input_projection.weight".to_string(), vec![8, 8]),
            ("mm.a.embedding.weight".to_string(), vec![2, 8]),
            ("mm.a.hard_emb_norm.weight".to_string(), vec![8]),
            ("mm.a.soft_emb_norm.weight".to_string(), vec![8]),
            ("mm.a.input_projection.weight".to_string(), vec![8, 8]),
        ];
        push_text_layer(&mut tensors, 0);
        push_vision(&mut tensors);
        push_audio(&mut tensors);
        tensors
    }

    fn push_text_layer(tensors: &mut Vec<(String, Vec<usize>)>, layer: usize) {
        for role in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "inp_gate.weight",
            "proj.weight",
            "post_norm.weight",
            "altup_correct_scale.weight",
            "altup_correct_coef.weight",
            "altup_predict_coef.weight",
            "altup_router.weight",
            "altup_router_norm.weight",
            "laurel_l.weight",
            "laurel_r.weight",
            "laurel_post_norm.weight",
        ] {
            tensors.push((format!("blk.{layer}.{role}"), vec![8, 8]));
        }
    }

    fn push_vision(tensors: &mut Vec<(String, Vec<usize>)>) {
        tensors.extend([
            ("v.conv_stem.conv.weight".to_string(), vec![8, 3, 3, 3]),
            ("v.conv_stem.conv.bias".to_string(), vec![1, 8, 1, 1]),
            ("v.conv_stem.bn.weight".to_string(), vec![8]),
            ("v.msfa.ffn.pw_exp.conv.weight".to_string(), vec![8, 8]),
            ("v.msfa.ffn.pw_exp.bn.weight".to_string(), vec![8]),
            ("v.msfa.ffn.pw_proj.conv.weight".to_string(), vec![8, 8]),
            ("v.msfa.ffn.pw_proj.bn.weight".to_string(), vec![8]),
            ("v.msfa.norm.weight".to_string(), vec![8]),
        ]);
        for (stage, blocks) in gemma3n_mobilenet_def().iter().enumerate() {
            for (block, block_type) in blocks.iter().enumerate() {
                for role in vision_block_roles(block_type) {
                    let shape = if role == "layer_scale.gamma" {
                        vec![1, 8, 1, 1]
                    } else {
                        vec![8, 8]
                    };
                    tensors.push((format!("v.blk.{stage}.{block}.{role}"), shape));
                }
            }
        }
    }

    fn push_audio(tensors: &mut Vec<(String, Vec<usize>)>) {
        for index in 0..2 {
            tensors.push((format!("a.conv1d.{index}.weight"), vec![8, 8, 3, 3]));
            tensors.push((format!("a.conv1d.{index}.norm.weight"), vec![8]));
        }
        tensors.push(("a.pre_encode.out.weight".to_string(), vec![8, 8]));
        for layer in 0..12 {
            for role in [
                "layer_pre_norm.weight",
                "ln1.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "per_dim_scale",
                "linear_pos.weight",
                "attn_out.weight",
                "ln2.weight",
                "ffn_norm.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "ffn_post_norm.weight",
                "conv_norm.weight",
                "conv_pw1.weight",
                "conv_dw.weight",
                "norm_conv.weight",
                "conv_pw2.weight",
                "ffn_norm_1.weight",
                "ffn_up_1.weight",
                "ffn_down_1.weight",
                "ffn_post_norm_1.weight",
            ] {
                tensors.push((format!("a.blk.{layer}.{role}"), vec![8, 8]));
            }
        }
    }
}
