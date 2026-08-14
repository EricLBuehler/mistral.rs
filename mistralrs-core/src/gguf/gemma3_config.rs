use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value as GgufValue;
use serde_json::{json, Value as JsonValue};

use crate::layers::Activation;
use crate::vision_models::gemma3::config::{Gemma3Config, Gemma3TextConfig};

const ARCHITECTURE: &str = "gemma3";
const TEXT_CAUSAL_LM_ARCHITECTURE: &str = "Gemma3ForCausalLM";
const TEXT_MODEL_TYPE: &str = "gemma3_text";
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 1_000_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;
const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;
const GEMMA3_27B_LAYER_COUNT: usize = 62;
const GEMMA3_MULTIMODAL_LAYER_COUNTS: &[usize] = &[34, 48, GEMMA3_27B_LAYER_COUNT];
const CONFIG_FLOAT_RELATIVE_TOLERANCE: f64 = 1e-9;

pub(crate) fn prepare_gemma3_text_config(
    external: Option<&str>,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<String> {
    anyhow::ensure!(
        metadata_string(metadata, "general.architecture")?.eq_ignore_ascii_case(ARCHITECTURE),
        "Gemma 3 config preparation requires `gemma3` GGUF architecture"
    );
    let mut value = match external {
        Some(config) => normalize_external_text_config(config, tokenizer_vocab_size(metadata)?)?,
        None => synthesize_text_config(metadata, tensor_names)?,
    };
    let object = value
        .as_object_mut()
        .context("Gemma 3 text configuration must be a JSON object")?;
    object.insert(
        "architectures".to_string(),
        json!([TEXT_CAUSAL_LM_ARCHITECTURE]),
    );
    object.insert("model_type".to_string(), json!(TEXT_MODEL_TYPE));
    let config = serde_json::to_string(&value)?;
    let parsed: Gemma3Config = serde_json::from_str(&config)
        .context("Gemma 3 text configuration is incompatible with the native loader")?;
    let Gemma3Config::Text(parsed) = parsed else {
        bail!("Projectorless Gemma 3 requires a text-only configuration");
    };
    validate_text_config(&parsed, metadata, tensor_names)?;
    Ok(config)
}

pub(crate) fn ensure_gemma3_vision_config(config: &str) -> Result<()> {
    let parsed: Gemma3Config = serde_json::from_str(config)
        .context("Gemma 3 multimodal configuration is incompatible with the native loader")?;
    anyhow::ensure!(
        matches!(parsed, Gemma3Config::WithVision { .. }),
        "Gemma 3 with a projector requires a multimodal config containing `text_config` and `vision_config`"
    );
    Ok(())
}

pub(crate) fn gemma3_text_uses_language_model_prefix(config: &str) -> Result<bool> {
    let parsed: Gemma3Config = serde_json::from_str(config)
        .context("Gemma 3 text configuration is incompatible with the native loader")?;
    let Gemma3Config::Text(parsed) = parsed else {
        bail!("Projectorless Gemma 3 requires a text-only configuration");
    };
    Ok(parsed.use_language_model_prefix)
}

fn normalize_external_text_config(config: &str, vocab_size: usize) -> Result<JsonValue> {
    let value: JsonValue =
        serde_json::from_str(config).context("External Gemma 3 model config is not valid JSON")?;
    let object = value
        .as_object()
        .context("External Gemma 3 model config must be a JSON object")?;
    let use_language_model_prefix = object.contains_key("text_config");
    let mut text = match object.get("text_config") {
        Some(JsonValue::Object(text)) => text.clone(),
        Some(value) => bail!("Gemma 3 `text_config` must be a JSON object, got {value}"),
        None => object.clone(),
    };
    text.insert(
        "_mistralrs_use_language_model_prefix".to_string(),
        JsonValue::Bool(use_language_model_prefix),
    );
    text.insert("vocab_size".to_string(), JsonValue::from(vocab_size));
    text.insert("quantization_config".to_string(), JsonValue::Null);
    Ok(JsonValue::Object(text))
}

fn synthesize_text_config(
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<JsonValue> {
    anyhow::ensure!(
        metadata_string(metadata, "general.architecture")?.eq_ignore_ascii_case(ARCHITECTURE),
        "Gemma 3 config synthesis requires `gemma3` GGUF architecture"
    );
    anyhow::ensure!(
        !tensor_names.iter().any(|name| name.contains("rope_freqs")),
        "Gemma 3 GGUF contains baked RoPE tensors; provide the original Hugging Face config"
    );

    let context_length = required_usize(metadata, "gemma3.context_length")?;
    let hidden_size = required_usize(metadata, "gemma3.embedding_length")?;
    let intermediate_size = required_usize(metadata, "gemma3.feed_forward_length")?;
    let num_hidden_layers = required_usize(metadata, "gemma3.block_count")?;
    let num_attention_heads = required_usize(metadata, "gemma3.attention.head_count")?;
    let num_key_value_heads = required_usize(metadata, "gemma3.attention.head_count_kv")?;
    let head_dim = required_usize(metadata, "gemma3.attention.key_length")?;
    let value_dim = required_usize(metadata, "gemma3.attention.value_length")?;
    anyhow::ensure!(
        head_dim == value_dim,
        "Gemma 3 native loader requires equal key and value head dimensions, got {head_dim} and {value_dim}"
    );
    let rms_norm_eps = required_f64(metadata, "gemma3.attention.layer_norm_rms_epsilon")?;
    let vocab_size = tokenizer_vocab_size(metadata)?;
    let attention_bias = attention_bias(tensor_names, num_hidden_layers)?;
    let tie_word_embeddings = !tensor_names.iter().any(|name| name == "output.weight");
    let rope_theta =
        optional_f64(metadata, "gemma3.rope.freq_base")?.unwrap_or(DEFAULT_GLOBAL_ROPE_THETA);
    let rope_local_base_freq =
        optional_f64(metadata, "gemma3.rope.freq_base_swa")?.unwrap_or(DEFAULT_LOCAL_ROPE_THETA);
    let (sliding_window, sliding_window_pattern) =
        expected_sliding_window(metadata, context_length)?;
    let query_pre_attn_scalar = expected_query_pre_attn_scalar(
        hidden_size,
        num_attention_heads,
        head_dim,
        num_hidden_layers,
    )?;

    Ok(json!({
        "_mistralrs_use_language_model_prefix": GEMMA3_MULTIMODAL_LAYER_COUNTS.contains(&num_hidden_layers),
        "attention_bias": attention_bias,
        "attn_logit_softcapping": optional_f64(metadata, "gemma3.attn_logit_softcapping")?,
        "final_logit_softcapping": optional_f64(metadata, "gemma3.final_logit_softcapping")?,
        "head_dim": head_dim,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": context_length,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
        "quantization_config": null,
        "query_pre_attn_scalar": query_pre_attn_scalar,
        "rms_norm_eps": rms_norm_eps,
        "rope_local_base_freq": rope_local_base_freq,
        "rope_scaling": rope_scaling(metadata)?,
        "rope_theta": rope_theta,
        "sliding_window": sliding_window,
        "sliding_window_pattern": sliding_window_pattern,
        "tie_word_embeddings": tie_word_embeddings,
        "vocab_size": vocab_size,
    }))
}

fn validate_text_config(
    config: &Gemma3TextConfig,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<()> {
    let expected = [
        (
            "hidden_size",
            config.hidden_size,
            required_usize(metadata, "gemma3.embedding_length")?,
        ),
        (
            "intermediate_size",
            config.intermediate_size,
            required_usize(metadata, "gemma3.feed_forward_length")?,
        ),
        (
            "num_hidden_layers",
            config.num_hidden_layers,
            required_usize(metadata, "gemma3.block_count")?,
        ),
        (
            "num_attention_heads",
            config.num_attention_heads,
            required_usize(metadata, "gemma3.attention.head_count")?,
        ),
        (
            "num_key_value_heads",
            config.num_key_value_heads,
            required_usize(metadata, "gemma3.attention.head_count_kv")?,
        ),
        (
            "head_dim",
            config.head_dim,
            required_usize(metadata, "gemma3.attention.key_length")?,
        ),
        (
            "vocab_size",
            config.vocab_size,
            tokenizer_vocab_size(metadata)?,
        ),
    ];
    for (field, actual, expected) in expected {
        anyhow::ensure!(
            actual == expected,
            "Gemma 3 config `{field}` is {actual}, but the GGUF model requires {expected}"
        );
    }
    let value_dim = required_usize(metadata, "gemma3.attention.value_length")?;
    anyhow::ensure!(
        config.head_dim == value_dim,
        "Gemma 3 config `head_dim` is {}, but the GGUF value head dimension is {value_dim}",
        config.head_dim
    );
    let expected_bias = attention_bias(tensor_names, config.num_hidden_layers)?;
    anyhow::ensure!(
        config.attention_bias == expected_bias,
        "Gemma 3 config `attention_bias` does not match the GGUF tensors"
    );
    let expected_tied = !tensor_names.iter().any(|name| name == "output.weight");
    anyhow::ensure!(
        config.tie_word_embeddings == expected_tied,
        "Gemma 3 config `tie_word_embeddings` does not match the GGUF tensors"
    );
    anyhow::ensure!(
        config.hidden_activation == Activation::GeluPytorchTanh,
        "Gemma 3 config `hidden_activation` must be `gelu_pytorch_tanh`"
    );

    let context_length = required_usize(metadata, "gemma3.context_length")?;
    anyhow::ensure!(
        config.max_position_embeddings == context_length,
        "Gemma 3 config `max_position_embeddings` is {}, but the GGUF model requires {context_length}",
        config.max_position_embeddings
    );
    ensure_float_matches(
        "rms_norm_eps",
        config.rms_norm_eps,
        required_f64(metadata, "gemma3.attention.layer_norm_rms_epsilon")?,
    )?;
    ensure_float_matches(
        "rope_theta",
        config.rope_theta,
        optional_f64(metadata, "gemma3.rope.freq_base")?.unwrap_or(DEFAULT_GLOBAL_ROPE_THETA),
    )?;
    ensure_float_matches(
        "rope_local_base_freq",
        config.rope_local_base_freq,
        optional_f64(metadata, "gemma3.rope.freq_base_swa")?.unwrap_or(DEFAULT_LOCAL_ROPE_THETA),
    )?;

    let (sliding_window, sliding_window_pattern) =
        expected_sliding_window(metadata, context_length)?;
    anyhow::ensure!(
        config.sliding_window == sliding_window,
        "Gemma 3 config `sliding_window` is {}, but the GGUF model requires {sliding_window}",
        config.sliding_window
    );
    anyhow::ensure!(
        config.sliding_window_pattern == sliding_window_pattern,
        "Gemma 3 config `sliding_window_pattern` is {}, but the GGUF model requires {sliding_window_pattern}",
        config.sliding_window_pattern
    );

    let query_pre_attn_scalar = expected_query_pre_attn_scalar(
        config.hidden_size,
        config.num_attention_heads,
        config.head_dim,
        config.num_hidden_layers,
    )?;
    anyhow::ensure!(
        config.query_pre_attn_scalar == query_pre_attn_scalar,
        "Gemma 3 config `query_pre_attn_scalar` is {}, but the GGUF model requires {query_pre_attn_scalar}",
        config.query_pre_attn_scalar
    );

    let config_rope_scaling = serde_json::to_value(&config.rope_scaling)?;
    let metadata_rope_scaling = rope_scaling(metadata)?;
    anyhow::ensure!(
        config_rope_scaling == metadata_rope_scaling,
        "Gemma 3 config `rope_scaling` does not match the GGUF metadata"
    );
    ensure_optional_float_matches(
        "attn_logit_softcapping",
        config.attn_logit_softcapping,
        optional_f64(metadata, "gemma3.attn_logit_softcapping")?,
    )?;
    ensure_optional_float_matches(
        "final_logit_softcapping",
        config.final_logit_softcapping,
        optional_f64(metadata, "gemma3.final_logit_softcapping")?,
    )?;
    Ok(())
}

fn expected_sliding_window(
    metadata: &HashMap<String, GgufValue>,
    context_length: usize,
) -> Result<(usize, usize)> {
    match optional_usize(metadata, "gemma3.attention.sliding_window")? {
        Some(window) if window != 0 => Ok((
            window,
            optional_usize(metadata, "gemma3.attention.sliding_window_pattern")?
                .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN),
        )),
        Some(_) | None => Ok((context_length, 1)),
    }
}

fn expected_query_pre_attn_scalar(
    hidden_size: usize,
    num_attention_heads: usize,
    head_dim: usize,
    num_hidden_layers: usize,
) -> Result<usize> {
    if num_hidden_layers == GEMMA3_27B_LAYER_COUNT {
        exact_div(
            hidden_size,
            num_attention_heads,
            "embedding length / attention head count",
        )
    } else {
        Ok(head_dim)
    }
}

fn ensure_float_matches(field: &str, actual: f64, expected: f64) -> Result<()> {
    let tolerance = actual.abs().max(expected.abs()).max(1.0) * CONFIG_FLOAT_RELATIVE_TOLERANCE;
    anyhow::ensure!(
        (actual - expected).abs() <= tolerance,
        "Gemma 3 config `{field}` is {actual}, but the GGUF model requires {expected}"
    );
    Ok(())
}

fn ensure_optional_float_matches(
    field: &str,
    actual: Option<f64>,
    expected: Option<f64>,
) -> Result<()> {
    let actual = actual.filter(|value| *value != 0.0);
    let expected = expected.filter(|value| *value != 0.0);
    match (actual, expected) {
        (Some(actual), Some(expected)) => ensure_float_matches(field, actual, expected),
        (None, None) => Ok(()),
        _ => bail!(
            "Gemma 3 config `{field}` is {actual:?}, but the GGUF model requires {expected:?}"
        ),
    }
}

fn attention_bias(tensor_names: &[String], layers: usize) -> Result<bool> {
    let mut present = 0;
    let mut total = 0;
    for layer in 0..layers {
        for projection in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            total += 1;
            if tensor_names
                .iter()
                .any(|name| name == &format!("blk.{layer}.{projection}.bias"))
            {
                present += 1;
            }
        }
    }
    anyhow::ensure!(
        present == 0 || present == total,
        "Gemma 3 GGUF has an incomplete attention bias tensor set"
    );
    Ok(present != 0)
}

fn rope_scaling(metadata: &HashMap<String, GgufValue>) -> Result<JsonValue> {
    match optional_string(metadata, "gemma3.rope.scaling.type")? {
        None | Some("none") => Ok(JsonValue::Null),
        Some("linear") => Ok(json!({
            "factor": required_f64(metadata, "gemma3.rope.scaling.factor")?,
            "rope_type": "linear",
        })),
        Some(kind) => bail!(
            "Gemma 3 GGUF RoPE scaling type `{kind}` cannot be reconstructed; provide the original Hugging Face config"
        ),
    }
}

fn tokenizer_vocab_size(metadata: &HashMap<String, GgufValue>) -> Result<usize> {
    match metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(tokens)) if !tokens.is_empty() => Ok(tokens.len()),
        Some(_) => bail!("GGUF metadata `tokenizer.ggml.tokens` must be a nonempty array"),
        None => bail!("GGUF metadata is missing `tokenizer.ggml.tokens`"),
    }
}

fn required_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<usize> {
    let value = metadata
        .get(key)
        .with_context(|| format!("GGUF metadata is missing `{key}`"))?;
    value_usize(value)
        .with_context(|| format!("GGUF metadata `{key}` must be a nonnegative integer"))
}

fn optional_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Option<usize>> {
    metadata
        .get(key)
        .map(|value| {
            value_usize(value)
                .with_context(|| format!("GGUF metadata `{key}` must be a nonnegative integer"))
        })
        .transpose()
}

fn value_usize(value: &GgufValue) -> Option<usize> {
    let value = match value {
        GgufValue::U8(value) => *value as u64,
        GgufValue::U16(value) => *value as u64,
        GgufValue::U32(value) => *value as u64,
        GgufValue::U64(value) => *value,
        GgufValue::I8(value) if *value >= 0 => *value as u64,
        GgufValue::I16(value) if *value >= 0 => *value as u64,
        GgufValue::I32(value) if *value >= 0 => *value as u64,
        GgufValue::I64(value) if *value >= 0 => *value as u64,
        _ => return None,
    };
    usize::try_from(value).ok()
}

fn required_f64(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<f64> {
    optional_f64(metadata, key)?.with_context(|| format!("GGUF metadata is missing `{key}`"))
}

fn optional_f64(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Option<f64>> {
    metadata
        .get(key)
        .map(|value| match value {
            GgufValue::F32(value) => Ok(*value as f64),
            GgufValue::F64(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` must be a floating-point number"),
        })
        .transpose()
}

fn metadata_string<'a>(metadata: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a str> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) => Ok(value),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => bail!("GGUF metadata is missing `{key}`"),
    }
}

fn optional_string<'a>(
    metadata: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Result<Option<&'a str>> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) => Ok(Some(value)),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => Ok(None),
    }
}

fn exact_div(numerator: usize, denominator: usize, label: &str) -> Result<usize> {
    anyhow::ensure!(denominator != 0, "Gemma 3 {label} has a zero denominator");
    anyhow::ensure!(
        numerator.is_multiple_of(denominator),
        "Gemma 3 {label} is not integral: {numerator} / {denominator}"
    );
    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(layers: u32, sliding_window: Option<u32>) -> HashMap<String, GgufValue> {
        let mut metadata = HashMap::from([
            (
                "general.architecture".to_string(),
                GgufValue::String("gemma3".to_string()),
            ),
            ("gemma3.context_length".to_string(), GgufValue::U32(32768)),
            ("gemma3.embedding_length".to_string(), GgufValue::U32(1152)),
            ("gemma3.block_count".to_string(), GgufValue::U32(layers)),
            (
                "gemma3.feed_forward_length".to_string(),
                GgufValue::U32(6912),
            ),
            ("gemma3.attention.head_count".to_string(), GgufValue::U32(4)),
            (
                "gemma3.attention.head_count_kv".to_string(),
                GgufValue::U32(1),
            ),
            (
                "gemma3.attention.key_length".to_string(),
                GgufValue::U32(256),
            ),
            (
                "gemma3.attention.value_length".to_string(),
                GgufValue::U32(256),
            ),
            (
                "gemma3.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::F32(1e-6),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    (0..8)
                        .map(|index| GgufValue::String(format!("token-{index}")))
                        .collect(),
                ),
            ),
        ]);
        if let Some(window) = sliding_window {
            metadata.insert(
                "gemma3.attention.sliding_window".to_string(),
                GgufValue::U32(window),
            );
        }
        metadata
    }

    fn tensor_names(layers: usize, output: bool) -> Vec<String> {
        let mut names = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        if output {
            names.push("output.weight".to_string());
        }
        for layer in 0..layers {
            for role in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "attn_q_norm",
                "attn_k_norm",
                "attn_norm",
                "post_attention_norm",
                "ffn_norm",
                "post_ffw_norm",
            ] {
                names.push(format!("blk.{layer}.{role}.weight"));
            }
        }
        names
    }

    #[test]
    fn synthesizes_projectorless_gemma3_1b_config() {
        let metadata = metadata(26, Some(512));
        let config = prepare_gemma3_text_config(None, &metadata, &tensor_names(26, false)).unwrap();
        let value: JsonValue = serde_json::from_str(&config).unwrap();

        assert_eq!(value["hidden_size"], 1152);
        assert_eq!(value["head_dim"], 256);
        assert_eq!(value["query_pre_attn_scalar"], 256);
        assert_eq!(value["sliding_window"], 512);
        assert_eq!(value["sliding_window_pattern"], 6);
        assert_eq!(value["vocab_size"], 8);
        assert_eq!(value["tie_word_embeddings"], true);
        assert_eq!(value["_mistralrs_use_language_model_prefix"], false);
        assert_eq!(value["architectures"], json!(["Gemma3ForCausalLM"]));
        assert_eq!(value["model_type"], "gemma3_text");
        assert!(matches!(
            crate::MultimodalLoaderType::from_causal_lm_name(
                value["architectures"][0].as_str().unwrap()
            ),
            Ok(crate::MultimodalLoaderType::Gemma3)
        ));
    }

    #[test]
    fn synthesizes_global_only_attention_without_sliding_window_metadata() {
        let metadata = metadata(26, None);
        let config = prepare_gemma3_text_config(None, &metadata, &tensor_names(26, false)).unwrap();
        let value: JsonValue = serde_json::from_str(&config).unwrap();

        assert_eq!(value["sliding_window"], 32768);
        assert_eq!(value["sliding_window_pattern"], 1);
    }

    #[test]
    fn derives_gemma3_27b_attention_scalar() {
        let mut metadata = metadata(62, Some(1024));
        metadata.insert("gemma3.embedding_length".to_string(), GgufValue::U32(5376));
        metadata.insert(
            "gemma3.attention.head_count".to_string(),
            GgufValue::U32(32),
        );
        metadata.insert(
            "gemma3.attention.head_count_kv".to_string(),
            GgufValue::U32(16),
        );
        metadata.insert(
            "gemma3.attention.key_length".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "gemma3.attention.value_length".to_string(),
            GgufValue::U32(128),
        );
        let config = prepare_gemma3_text_config(None, &metadata, &tensor_names(62, false)).unwrap();
        let value: JsonValue = serde_json::from_str(&config).unwrap();

        assert_eq!(value["query_pre_attn_scalar"], 168);
        assert_eq!(value["_mistralrs_use_language_model_prefix"], true);
    }

    #[test]
    fn flattens_multimodal_config_for_projectorless_loading() {
        let metadata = metadata(26, Some(512));
        let external = json!({
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 1152,
                "intermediate_size": 6912,
                "num_hidden_layers": 26,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "sliding_window": 512
            },
            "vision_config": {},
            "image_token_index": 7,
            "mm_tokens_per_image": 256
        });
        let config = prepare_gemma3_text_config(
            Some(&external.to_string()),
            &metadata,
            &tensor_names(26, false),
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&config).unwrap();

        assert!(value.get("text_config").is_none());
        assert_eq!(value["hidden_size"], 1152);
        assert_eq!(value["vocab_size"], 8);
        assert!(value["quantization_config"].is_null());
        assert_eq!(value["_mistralrs_use_language_model_prefix"], true);
        assert_eq!(value["architectures"], json!(["Gemma3ForCausalLM"]));
        assert_eq!(value["model_type"], "gemma3_text");
    }

    #[test]
    fn rejects_mismatched_value_head_dimension() {
        let mut metadata = metadata(26, Some(512));
        metadata.insert(
            "gemma3.attention.value_length".to_string(),
            GgufValue::U32(128),
        );
        let error =
            prepare_gemma3_text_config(None, &metadata, &tensor_names(26, false)).unwrap_err();

        assert!(error.to_string().contains("equal key and value"));
    }

    #[test]
    fn rejects_wrong_external_attention_semantics() {
        let metadata = metadata(26, Some(512));
        let external = json!({
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "max_position_embeddings": 131072,
            "sliding_window": 512,
            "vocab_size": 8
        });
        let error = prepare_gemma3_text_config(
            Some(&external.to_string()),
            &metadata,
            &tensor_names(26, false),
        )
        .unwrap_err();

        assert!(error.to_string().contains("max_position_embeddings"));
    }

    #[test]
    fn rejects_wrong_external_activation() {
        let metadata = metadata(26, Some(512));
        let external = json!({
            "hidden_activation": "silu",
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "max_position_embeddings": 32768,
            "sliding_window": 512,
            "vocab_size": 8
        });
        let error = prepare_gemma3_text_config(
            Some(&external.to_string()),
            &metadata,
            &tensor_names(26, false),
        )
        .unwrap_err();

        assert!(error.to_string().contains("hidden_activation"));
    }

    #[test]
    fn rejects_flat_config_for_multimodal_loading() {
        let config = json!({
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_hidden_layers": 26,
            "sliding_window": 512
        });
        let error = ensure_gemma3_vision_config(&config.to_string()).unwrap_err();

        assert!(error.to_string().contains("requires a multimodal config"));
    }
}
