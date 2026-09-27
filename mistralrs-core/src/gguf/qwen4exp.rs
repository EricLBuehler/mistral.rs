use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use serde_json::{json, Map, Value as JsonValue};

use super::qwen_multimodal_bindings::{
    bind, bind_experts, bind_gdn, bind_qwen3_vision, bind_shared_expert, metadata_bool_indices,
    metadata_usize, read_gdn_metadata, TensorInventory, DEEPSTACK_LAYERS,
};
use crate::gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY;
use crate::vision_models::qwen4_exp::Config as Qwen4ExpConfig;

pub(crate) const ARCHITECTURE: &str = "qwen4exp";
const NATIVE_ROOT: &str = "model.language_model";
const CAUSAL_LM_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const MODEL_TYPE: &str = "qwen4_exp";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const VIDEO_PAD_TOKEN: &str = "<|video_pad|>";
const VISION_START_TOKEN: &str = "<|vision_start|>";
const VISION_END_TOKEN: &str = "<|vision_end|>";
const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
// llama.cpp hardcodes the arch's GDN output gate instead of storing it
const OUTPUT_GATE_TYPE: &str = "sigmoid";
const PROJECTOR_TENSOR: &str = "v.patch_embd.weight";
const PROJECTOR_TEMPORAL_TENSOR: &str = "v.patch_embd.weight.1";
const PROJECTOR_POSITION_TENSOR: &str = "v.position_embd.weight";
const PROJECTOR_TYPE: &str = "clip.projector_type";
const QWEN3_VL_PROJECTOR: &str = "qwen3vl_merger";
// the Qwen3-VL tower the projector comes from; clip metadata only flags gelu
const VISION_HIDDEN_ACT: &str = "gelu_pytorch_tanh";

// llama.cpp stores these gammas as 1 + w; the native norms add the 1 themselves
const SHIFTED_NORMS: &[(&str, &str)] = &[
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    (
        "self_attn.indexer.q_layernorm.weight",
        "indexer.q_norm.weight",
    ),
    (
        "self_attn.indexer.k_layernorm.weight",
        "indexer.k_norm.weight",
    ),
    (
        "attn_hyper_connection.hc_norm.weight",
        "hc_attn_norm.weight",
    ),
    ("mlp_hyper_connection.hc_norm.weight", "hc_ffn_norm.weight"),
    ("ple.norm_key.weight", "ple_norm_key.weight"),
    ("ple.norm_query.weight", "ple_norm_query.weight"),
    ("ple.norm_conv.weight", "ple_norm_conv.weight"),
];

const LAYER_TENSORS: &[(&str, &str)] = &[
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("mlp.gate.weight", "ffn_gate_inp.weight"),
    (
        "attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_down.weight",
    ),
    (
        "attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_up.weight",
    ),
    (
        "attn_hyper_connection.block_inject_weight.weight",
        "hc_attn_inject.weight",
    ),
    (
        "mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_down.weight",
    ),
    (
        "mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_up.weight",
    ),
    (
        "mlp_hyper_connection.block_inject_weight.weight",
        "hc_ffn_inject.weight",
    ),
    ("ple.key_proj.weight", "ple_key.weight"),
    ("ple.value_proj.weight", "ple_value.weight"),
];

/// Whether a vision projector (`mmproj`) was merged into the archive.
pub(crate) fn has_projector(archive: &GgufArchive) -> bool {
    archive.contains_tensor(PROJECTOR_TENSOR)
}

/// Native names for a `qwen4exp` GGUF and its optional projector; the n-gram table stays out and is
/// read raw.
pub(crate) fn build_qwen4exp_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let inventory = TensorInventory::from_archive(archive);
    let gdn = read_gdn_metadata(archive)?;
    let mut bindings = GgufBindingMap::new();
    bind(
        &inventory,
        &mut bindings,
        format!("{NATIVE_ROOT}.embed_tokens.weight"),
        "token_embd.weight",
    );
    bind(&inventory, &mut bindings, "lm_head.weight", "output.weight");
    let mixer = format!("{NATIVE_ROOT}.hyper_connection_mixer");
    bind_shifted_norm(
        &inventory,
        &mut bindings,
        format!("{mixer}.hc_norm.weight"),
        "output_hc_norm.weight",
    );
    bind(
        &inventory,
        &mut bindings,
        format!("{mixer}.input_mix_weight_down.weight"),
        "output_hc_down.weight",
    );
    bind(
        &inventory,
        &mut bindings,
        format!("{mixer}.input_mix_weight_up.weight"),
        "output_hc_up.weight",
    );
    for layer in 0..metadata_usize(archive, &format!("{ARCHITECTURE}.block_count"))? {
        let native = format!("{NATIVE_ROOT}.layers.{layer}");
        let source = format!("blk.{layer}");
        for (target, role) in LAYER_TENSORS {
            bind(
                &inventory,
                &mut bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
        for (target, role) in SHIFTED_NORMS {
            bind_shifted_norm(
                &inventory,
                &mut bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
        let q = format!("{source}.indexer.q_proj.weight");
        let k = format!("{source}.indexer.k_proj.weight");
        if inventory.contains(&q) && inventory.contains(&k) {
            bindings.insert(
                format!("{native}.self_attn.indexer.index_qk_proj.weight"),
                GgufTensorBinding::concat(
                    vec![GgufTensorBinding::tensor(q), GgufTensorBinding::tensor(k)],
                    0,
                ),
            );
        }
        let conv = format!("{source}.ple_conv1d.weight");
        if inventory.contains(&conv) {
            let &[channels, kernel] = inventory.shape(&conv)? else {
                bail!("Qwen4-Exp PLE convolution `{conv}` must be rank 2");
            };
            bindings.insert(
                format!("{native}.ple.conv1d.weight"),
                GgufTensorBinding::tensor(&conv).reshape(vec![channels, 1, kernel]),
            );
        }
        if inventory.contains(&format!("{source}.attn_qkv.weight")) {
            bind_gdn(&inventory, &native, &source, gdn, &mut bindings)?;
        }
        bind_experts(&inventory, &native, &source, &mut bindings);
        bind_shared_expert(&inventory, &native, &source, &mut bindings)?;
    }
    if has_projector(archive) {
        match archive.metadata_value(PROJECTOR_TYPE) {
            Some(Value::String(projector)) if projector == QWEN3_VL_PROJECTOR => {}
            other => {
                bail!("Qwen4-Exp needs a `{QWEN3_VL_PROJECTOR}` vision projector, got {other:?}")
            }
        }
        let deepstack = metadata_bool_indices(archive, DEEPSTACK_LAYERS)?;
        bind_qwen3_vision(&inventory, deepstack.as_deref(), &mut bindings)?;
    }
    Ok(bindings)
}

fn bind_shifted_norm(
    inventory: &TensorInventory,
    bindings: &mut GgufBindingMap,
    native: String,
    source: impl Into<String>,
) {
    let source = source.into();
    if inventory.contains(&source) {
        bindings.insert(native, GgufTensorBinding::tensor(source).affine(1.0, -1.0));
    }
}

/// Model config for a `qwen4exp` GGUF: an external `config.json` is used as is, otherwise it is
/// rebuilt from metadata. The vision tower follows the projector, and the GGUF hash constants win.
pub(crate) fn prepare_qwen4exp_config(
    external: Option<&str>,
    archive: &GgufArchive,
) -> Result<String> {
    let mut config = match external {
        Some(external) => serde_json::from_str::<JsonValue>(external)
            .context("Qwen4-Exp `config.json` is not valid JSON")?,
        None => synthesize_config(archive)?,
    };
    let object = config
        .as_object_mut()
        .context("Qwen4-Exp config must be a JSON object")?;
    if !has_projector(archive) {
        object.remove("vision_config");
    } else if !object.contains_key("vision_config") {
        object.insert("vision_config".into(), vision_config(archive)?);
    }
    object.insert("quantization_config".into(), JsonValue::Null);
    let text = object
        .get_mut("text_config")
        .and_then(JsonValue::as_object_mut)
        .context("Qwen4-Exp config is missing `text_config`")?;
    text.insert("quantization_config".into(), JsonValue::Null);
    text.insert(GDN_V_HEAD_LAYOUT_CONFIG_KEY.into(), json!("tiled"));
    if archive.contains_tensor("per_layer_token_embd.weight") {
        text.insert("_mistralrs_ple_hash".into(), ple_hash(archive)?);
    }
    let config = serde_json::to_string(&config)?;
    let parsed: Qwen4ExpConfig = serde_json::from_str(&config)
        .context("Qwen4-Exp GGUF config is incompatible with the native loader")?;
    parsed.text_config.validate()?;
    Ok(config)
}

#[allow(clippy::cast_precision_loss)]
fn synthesize_config(archive: &GgufArchive) -> Result<JsonValue> {
    let key = |suffix: &str| format!("{ARCHITECTURE}.{suffix}");
    let usize_at = |suffix: &str| metadata_usize(archive, &key(suffix));
    let head_dim = usize_at("attention.key_length")?;
    let layers = usize_at("block_count")?;
    let compress_ratios = u64_array(archive, &key("attention.compress_ratios"))?;
    if compress_ratios.len() != layers {
        bail!(
            "Qwen4-Exp GGUF lists {} compress ratios for {layers} layers",
            compress_ratios.len()
        );
    }
    let layer_types = compress_ratios
        .iter()
        .map(|ratio| {
            if *ratio > 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    let compress_ratio = compress_ratios.iter().copied().max().unwrap_or(0);
    let value_heads = usize_at("ssm.time_step_rank")?;
    let mrope_section = u64_array(archive, &key("rope.dimension_sections"))?
        .into_iter()
        .take(3)
        .collect::<Vec<_>>();
    let vocab_size = archive
        .tensor_info("token_embd.weight")?
        .shape()
        .first()
        .copied()
        .context("Qwen4-Exp token embedding has no rows")?;
    let mut text = Map::new();
    text.extend([
        ("model_type".into(), json!("qwen4_exp_text")),
        ("head_dim".into(), json!(head_dim)),
        ("vocab_size".into(), json!(vocab_size)),
        ("hidden_size".into(), json!(usize_at("embedding_length")?)),
        ("num_hidden_layers".into(), json!(layers)),
        (
            "num_attention_heads".into(),
            json!(usize_at("attention.head_count")?),
        ),
        (
            "num_key_value_heads".into(),
            json!(usize_at("attention.head_count_kv")?),
        ),
        ("hidden_act".into(), json!("silu")),
        (
            "max_position_embeddings".into(),
            json!(usize_at("context_length")?),
        ),
        (
            "rms_norm_eps".into(),
            json!(metadata_f64(
                archive,
                &key("attention.layer_norm_rms_epsilon")
            )?),
        ),
        (
            "rope_parameters".into(),
            json!({
                "rope_type": "default",
                "rope_theta": metadata_f64(archive, &key("rope.freq_base"))?,
                "mrope_section": mrope_section,
                "mrope_interleaved": true,
                "partial_rotary_factor": usize_at("rope.dimension_count")? as f64 / head_dim as f64,
            }),
        ),
        (
            "moe_intermediate_size".into(),
            json!(usize_at("expert_feed_forward_length")?),
        ),
        (
            "shared_expert_intermediate_size".into(),
            json!(usize_at("expert_shared_feed_forward_length")?),
        ),
        ("num_experts".into(), json!(usize_at("expert_count")?)),
        (
            "num_experts_per_tok".into(),
            json!(usize_at("expert_used_count")?),
        ),
        (
            "full_attention_interval".into(),
            json!(usize_at("full_attention_interval")?),
        ),
        ("layer_types".into(), json!(layer_types)),
        (
            "linear_conv_kernel_dim".into(),
            json!(usize_at("ssm.conv_kernel")?),
        ),
        (
            "linear_key_head_dim".into(),
            json!(usize_at("ssm.state_size")?),
        ),
        (
            "linear_value_head_dim".into(),
            json!(usize_at("ssm.inner_size")? / value_heads.max(1)),
        ),
        (
            "linear_num_key_heads".into(),
            json!(usize_at("ssm.group_count")?),
        ),
        ("linear_num_value_heads".into(), json!(value_heads)),
        ("mamba_ssm_dtype".into(), json!("float32")),
        ("output_gate_type".into(), json!(OUTPUT_GATE_TYPE)),
        (
            "hc_count".into(),
            json!(usize_at("hyper_connection.count")?),
        ),
        (
            "hc_lowrank".into(),
            json!(usize_at("hyper_connection.low_rank")?),
        ),
        (
            "indexer_n_heads".into(),
            json!(usize_at("attention.indexer.head_count")?),
        ),
        ("indexer_kv_heads".into(), json!(1)),
        (
            "indexer_head_dim".into(),
            json!(usize_at("attention.indexer.key_length")?),
        ),
        (
            "indexer_budget".into(),
            json!(usize_at("attention.indexer.top_k")?),
        ),
        ("indexer_compress_ratio".into(), json!(compress_ratio)),
    ]);
    if let Some(ple_layers) = archive
        .metadata_value(&key("ple.layers"))
        .map(|_| u64_array(archive, &key("ple.layers")))
        .transpose()?
    {
        let ngram_size = usize_at("ple.ngram_size")?;
        let heads_per_ngram = usize_at("ple.heads_per_ngram")?;
        let row_dim = usize_at("embedding_length_per_layer_input")?;
        text.extend([
            (
                "ple_layer_ids".into(),
                json!(ple_layers.iter().map(|layer| layer + 1).collect::<Vec<_>>()),
            ),
            (
                "ple_embed_dim".into(),
                json!(row_dim * (ngram_size - 1) * heads_per_ngram),
            ),
            (
                "ple_conv_kernel_size".into(),
                json!(usize_at("ple.conv_kernel")?),
            ),
            ("ngram_size".into(), json!(ngram_size)),
            ("heads_per_ngram".into(), json!(heads_per_ngram)),
            ("eos_token_id".into(), json!(usize_at("ple.eos_token_id")?)),
        ]);
    } else {
        text.insert(
            "eos_token_id".into(),
            json!(metadata_usize(archive, "tokenizer.ggml.eos_token_id")?),
        );
    }
    Ok(json!({
        "architectures": [CAUSAL_LM_ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "text_config": text,
        "image_token_id": token_id(archive, IMAGE_PAD_TOKEN)?,
        "video_token_id": token_id(archive, VIDEO_PAD_TOKEN)?,
        "vision_start_token_id": token_id(archive, VISION_START_TOKEN)?,
        "vision_end_token_id": token_id(archive, VISION_END_TOKEN)?,
        "tie_word_embeddings": !archive.contains_tensor("output.weight"),
    }))
}

fn vision_config(archive: &GgufArchive) -> Result<JsonValue> {
    let usize_at = |suffix: &str| metadata_usize(archive, &format!("clip.vision.{suffix}"));
    let patch_shape = archive.tensor_info(PROJECTOR_TENSOR)?.shape().to_vec();
    let &[_, in_chans, _, _] = patch_shape.as_slice() else {
        bail!("Qwen4-Exp projector `{PROJECTOR_TENSOR}` must be rank 4, got {patch_shape:?}");
    };
    let temporal_patch_size = if archive.contains_tensor(PROJECTOR_TEMPORAL_TENSOR) {
        2
    } else {
        1
    };
    let num_position_embeddings = archive
        .tensor_info(PROJECTOR_POSITION_TENSOR)?
        .shape()
        .first()
        .copied()
        .context("Qwen4-Exp projector position embedding has no rows")?;
    Ok(json!({
        "depth": usize_at("block_count")?,
        "hidden_size": usize_at("embedding_length")?,
        "out_hidden_size": usize_at("projection_dim")?,
        "hidden_act": VISION_HIDDEN_ACT,
        "intermediate_size": usize_at("feed_forward_length")?,
        "num_heads": usize_at("attention.head_count")?,
        "in_chans": in_chans,
        "patch_size": usize_at("patch_size")?,
        "spatial_merge_size": usize_at("spatial_merge_size")?,
        "temporal_patch_size": temporal_patch_size,
        "num_position_embeddings": num_position_embeddings,
        "deepstack_visual_indexes": metadata_bool_indices(archive, DEEPSTACK_LAYERS)?.unwrap_or_default(),
    }))
}

/// Image preprocessing from the projector metadata; the processor defaults are the Qwen3-VL ones.
pub(crate) fn qwen4exp_preprocessor_config(archive: &GgufArchive) -> Result<Option<String>> {
    if !has_projector(archive) {
        return Ok(None);
    }
    let vision = vision_config(archive)?;
    Ok(Some(serde_json::to_string(&json!({
        "patch_size": vision["patch_size"],
        "temporal_patch_size": vision["temporal_patch_size"],
        "merge_size": vision["spatial_merge_size"],
        "image_mean": f64_array(archive, "clip.vision.image_mean")?,
        "image_std": f64_array(archive, "clip.vision.image_std")?,
    }))?))
}

fn f64_array(archive: &GgufArchive, key: &str) -> Result<Vec<f64>> {
    let Some(Value::Array(values)) = archive.metadata_value(key) else {
        bail!("GGUF metadata `{key}` must be a float array");
    };
    values
        .iter()
        .map(|value| match value {
            Value::F32(v) => Ok(f64::from(*v)),
            Value::F64(v) => Ok(*v),
            _ => bail!("GGUF metadata `{key}` must hold floats"),
        })
        .collect()
}

fn ple_hash(archive: &GgufArchive) -> Result<JsonValue> {
    let key = |suffix: &str| format!("{ARCHITECTURE}.ple.{suffix}");
    Ok(json!({
        "layer_multipliers": u64_array(archive, &key("layer_multipliers"))?,
        "head_vocab_sizes": u64_array(archive, &key("head_vocab_sizes"))?,
        "head_offsets": u64_array(archive, &key("head_offsets"))?,
    }))
}

fn u64_array(archive: &GgufArchive, key: &str) -> Result<Vec<u64>> {
    let Some(Value::Array(values)) = archive.metadata_value(key) else {
        bail!("GGUF metadata `{key}` must be an integer array");
    };
    values
        .iter()
        .map(|value| match value {
            Value::U8(v) => Ok(u64::from(*v)),
            Value::U16(v) => Ok(u64::from(*v)),
            Value::U32(v) => Ok(u64::from(*v)),
            Value::U64(v) => Ok(*v),
            Value::I32(v) if *v >= 0 => Ok(*v as u64),
            Value::I64(v) if *v >= 0 => Ok(*v as u64),
            _ => bail!("GGUF metadata `{key}` must hold nonnegative integers"),
        })
        .collect()
}

fn metadata_f64(archive: &GgufArchive, key: &str) -> Result<f64> {
    match archive.metadata_value(key) {
        Some(Value::F32(v)) => Ok(f64::from(*v)),
        Some(Value::F64(v)) => Ok(*v),
        Some(_) => bail!("GGUF metadata `{key}` must be a float"),
        None => bail!("GGUF metadata is missing `{key}`"),
    }
}

fn token_id(archive: &GgufArchive, token: &str) -> Result<usize> {
    let Some(Value::Array(tokens)) = archive.metadata_value(TOKENS_KEY) else {
        bail!("GGUF metadata is missing `{TOKENS_KEY}`");
    };
    tokens
        .iter()
        .position(|value| matches!(value, Value::String(t) if t == token))
        .with_context(|| format!("GGUF vocabulary has no `{token}` token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_GGUF_DIR: &str = "qwen4exp_work/gguf/UD-Q4_K_XL";
    const LOCAL_PROJECTOR: &str = "qwen4exp_work/gguf/mmproj-BF16.gguf";

    fn source_names<'a>(binding: &'a GgufTensorBinding, out: &mut Vec<&'a str>) {
        match binding {
            GgufTensorBinding::Tensor(name)
            | GgufTensorBinding::Mxfp4Blocks(name)
            | GgufTensorBinding::Mxfp4Scales(name) => out.push(name),
            GgufTensorBinding::Concat { inputs, .. }
            | GgufTensorBinding::Stack { inputs, .. }
            | GgufTensorBinding::Interleave { inputs, .. } => {
                inputs.iter().for_each(|input| source_names(input, out))
            }
            GgufTensorBinding::Slice { input, .. }
            | GgufTensorBinding::Transpose { input, .. }
            | GgufTensorBinding::Permute { input, .. }
            | GgufTensorBinding::Reshape { input, .. }
            | GgufTensorBinding::Affine { input, .. }
            | GgufTensorBinding::Log { input }
            | GgufTensorBinding::InverseSoftplus { input }
            | GgufTensorBinding::Cast { input, .. } => source_names(input, out),
        }
    }

    fn local_archive() -> Option<GgufArchive> {
        let dir = std::path::Path::new(&std::env::var("HOME").ok()?).join(LOCAL_GGUF_DIR);
        let mut files = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
            .collect::<Vec<_>>();
        files.sort();
        GgufArchive::open(&files).ok()
    }

    #[test]
    #[ignore = "requires the local Qwen3.8-Flash-Next GGUF and projector"]
    fn local_qwen4exp_projector_config_and_bindings_are_complete() -> Result<()> {
        let mut archive = local_archive().context("local GGUF not found")?;
        let projector = std::path::Path::new(&std::env::var("HOME")?).join(LOCAL_PROJECTOR);
        archive.merge_component(GgufArchive::open_file(projector)?)?;
        let config = prepare_qwen4exp_config(None, &archive)?;
        let parsed: Qwen4ExpConfig = serde_json::from_str(&config)?;
        let vision = parsed
            .vision_config
            .context("projector config has no vision tower")?;
        assert_eq!(
            (vision.depth, vision.hidden_size, vision.out_hidden_size),
            (27, 1152, 2560)
        );
        assert_eq!(
            (
                vision.patch_size,
                vision.temporal_patch_size,
                vision.num_position_embeddings
            ),
            (16, 2, 2304)
        );
        assert!(vision.deepstack_visual_indexes.is_empty());
        let preprocessor: JsonValue = serde_json::from_str(
            &qwen4exp_preprocessor_config(&archive)?.context("no preprocessor config")?,
        )?;
        assert_eq!(preprocessor["patch_size"], 16);
        assert_eq!(preprocessor["image_mean"], json!([0.5, 0.5, 0.5]));

        let bindings = build_qwen4exp_bindings(&archive)?;
        let mut bound = Vec::new();
        for (_, binding) in bindings.iter() {
            source_names(binding, &mut bound);
        }
        let bound = bound.into_iter().collect::<std::collections::HashSet<_>>();
        let unbound = archive
            .tensors()
            .keys()
            .filter(|name| !bound.contains(name.as_str()) && *name != "per_layer_token_embd.weight")
            .collect::<Vec<_>>();
        assert!(unbound.is_empty(), "unbound GGUF tensors: {unbound:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires the local Qwen3.8-Flash-Next GGUF"]
    fn local_qwen4exp_gguf_config_and_bindings_are_complete() -> Result<()> {
        let archive = local_archive().context("local GGUF not found")?;
        let config = prepare_qwen4exp_config(None, &archive)?;
        let parsed: Qwen4ExpConfig = serde_json::from_str(&config)?;
        let text = &parsed.text_config;
        assert!(parsed.vision_config.is_none());
        assert_eq!(text.ple()?.map(|ple| ple.layer_idx), Some(1));
        assert_eq!(text.qsa()?.map(|qsa| qsa.compress_ratio), Some(4));
        assert_eq!(parsed.image_token_id, 248056);
        assert_eq!(parsed.vision_start_token_id, 248053);

        let bindings = build_qwen4exp_bindings(&archive)?;
        let mut bound = Vec::new();
        for (_, binding) in bindings.iter() {
            source_names(binding, &mut bound);
        }
        let bound = bound.into_iter().collect::<std::collections::HashSet<_>>();
        let unbound = archive
            .tensors()
            .keys()
            .filter(|name| !bound.contains(name.as_str()) && *name != "per_layer_token_embd.weight")
            .collect::<Vec<_>>();
        assert!(unbound.is_empty(), "unbound GGUF tensors: {unbound:?}");
        for native in [
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            "model.language_model.layers.1.ple.conv1d.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.mlp.experts.down_proj.weight",
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
        ] {
            assert!(bindings.get(native).is_some(), "missing binding {native}");
        }
        Ok(())
    }
}
