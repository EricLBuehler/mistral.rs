use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Value;
use serde_json::{json, Map, Value as JsonValue};

use crate::pipeline::chat_template::GenerationConfig;

const EOG_TOKEN_IDS: &str = "tokenizer.ggml.eog_token_ids";
const EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
const BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
const TOP_K: &str = "general.sampling.top_k";
const FLINT_EOG_TOKENS: &[&str] = &[
    "<|end_of_text|>",
    "<|user|>",
    "<|observation|>",
    "<|developer|>",
];

pub(crate) fn generation_config_from_metadata(
    metadata: &HashMap<String, Value>,
) -> Result<Option<GenerationConfig>> {
    let mut fields = Map::new();
    let mut stops = Vec::new();
    if let Some(value) = metadata.get(EOS_TOKEN_ID) {
        stops.push(token_id(value, EOS_TOKEN_ID)?);
    }
    if let Some(value) = metadata.get(EOG_TOKEN_IDS) {
        let values = value
            .to_vec()
            .with_context(|| format!("GGUF `{EOG_TOKEN_IDS}` must be an array of token IDs"))?;
        for value in values {
            let id = token_id(value, EOG_TOKEN_IDS)?;
            if !stops.contains(&id) {
                stops.push(id);
            }
        }
    } else if matches!(metadata.get("general.architecture"), Some(Value::String(arch)) if arch == "flint")
    {
        // Flint's HF stop set includes role boundaries omitted by older GGUF converters.
        if let Some(value) = metadata.get("tokenizer.ggml.tokens") {
            let tokens = value
                .to_vec()
                .context("GGUF `tokenizer.ggml.tokens` must be an array")?;
            for (id, token) in tokens.iter().enumerate() {
                let token = token
                    .to_string()
                    .context("GGUF `tokenizer.ggml.tokens` must contain strings")?;
                if FLINT_EOG_TOKENS.contains(&token.as_str()) {
                    let id = u32::try_from(id).context("GGUF token ID exceeds u32")?;
                    if !stops.contains(&id) {
                        stops.push(id);
                    }
                }
            }
        }
    }
    for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
        if let Some(value) = metadata.get(key) {
            let id = token_id(value, key)?;
            if !stops.contains(&id) {
                stops.push(id);
            }
        }
    }
    if !stops.is_empty() {
        fields.insert("eos_token_id".to_string(), json!(stops));
    }
    if let Some(value) = metadata.get(BOS_TOKEN_ID) {
        fields.insert(
            "bos_token_id".to_string(),
            json!(token_id(value, BOS_TOKEN_ID)?),
        );
    }
    if let Some(value) = metadata.get(TOP_K) {
        let top_k = usize::try_from(unsigned_integer(value, TOP_K)?)
            .with_context(|| format!("GGUF `{TOP_K}` is too large"))?;
        fields.insert("top_k".to_string(), json!(top_k));
    }
    for (key, field) in [
        ("general.sampling.temp", "temperature"),
        ("general.sampling.top_p", "top_p"),
        ("general.sampling.min_p", "min_p"),
    ] {
        if let Some(value) = metadata.get(key) {
            let value = match value {
                Value::F32(value) => f64::from(*value),
                Value::F64(value) => *value,
                _ => anyhow::bail!("GGUF `{key}` must be a floating-point value"),
            };
            anyhow::ensure!(value.is_finite(), "GGUF `{key}` must be finite");
            fields.insert(field.to_string(), json!(value));
        }
    }
    if fields.is_empty() {
        return Ok(None);
    }
    serde_json::from_value(JsonValue::Object(fields))
        .map(Some)
        .context("Invalid GGUF generation metadata")
}

fn token_id(value: &Value, key: &str) -> Result<u32> {
    u32::try_from(unsigned_integer(value, key)?)
        .with_context(|| format!("GGUF `{key}` token ID exceeds u32"))
}

fn unsigned_integer(value: &Value, key: &str) -> Result<u64> {
    match value {
        Value::U8(value) => Ok(u64::from(*value)),
        Value::U16(value) => Ok(u64::from(*value)),
        Value::U32(value) => Ok(u64::from(*value)),
        Value::U64(value) => Ok(*value),
        Value::I8(value) => u64::try_from(*value).context("negative integer"),
        Value::I16(value) => u64::try_from(*value).context("negative integer"),
        Value::I32(value) => u64::try_from(*value).context("negative integer"),
        Value::I64(value) => u64::try_from(*value).context("negative integer"),
        _ => anyhow::bail!("GGUF `{key}` must be a nonnegative integer"),
    }
    .with_context(|| format!("GGUF `{key}` must be a nonnegative integer"))
}

#[cfg(test)]
mod tests {
    use super::generation_config_from_metadata;
    use crate::pipeline::chat_template::{calculate_eos_tokens, ChatTemplate};
    use ahash::AHashMap;
    use anyhow::Result;
    use candle_core::quantized::gguf_file::Value;
    use std::collections::HashMap;
    use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

    #[test]
    fn preserves_all_stop_tokens_and_deduplicates_metadata() -> Result<()> {
        let metadata = HashMap::from([
            ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(1)),
            (
                "tokenizer.ggml.eog_token_ids".to_string(),
                Value::Array([1, 3, 4, 3].into_iter().map(Value::U32).collect()),
            ),
            ("tokenizer.ggml.eot_token_id".to_string(), Value::U32(4)),
            ("tokenizer.ggml.eom_token_id".to_string(), Value::U32(5)),
            ("tokenizer.ggml.bos_token_id".to_string(), Value::U32(2)),
        ]);
        let config = generation_config_from_metadata(&metadata)?.unwrap();
        let vocab = [
            "<unk>",
            "<eos>",
            "<bos>",
            "<user>",
            "<observation>",
            "<developer>",
        ]
        .into_iter()
        .zip(0u32..)
        .map(|(token, id)| (token.to_string(), id))
        .collect::<AHashMap<_, _>>();
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .map_err(anyhow::Error::msg)?,
        );
        assert_eq!(
            calculate_eos_tokens(&ChatTemplate::default(), Some(&config), &tokenizer),
            vec![1, 3, 4, 5]
        );
        config.validate_token_ids(6)?;
        assert!(config.validate_token_ids(5).is_err());
        assert!(config.generation_defaults().is_none());
        Ok(())
    }

    #[test]
    fn retains_sampling_defaults_and_accepts_float_precisions() -> Result<()> {
        for top_k in [Value::U32(20), Value::I32(20)] {
            let metadata = HashMap::from([
                ("general.sampling.temp".to_string(), Value::F32(1.0)),
                ("general.sampling.top_k".to_string(), top_k),
                ("general.sampling.top_p".to_string(), Value::F64(0.95)),
                ("general.sampling.min_p".to_string(), Value::F32(0.05)),
            ]);
            let defaults = generation_config_from_metadata(&metadata)?
                .unwrap()
                .generation_defaults()
                .unwrap();
            assert_eq!(defaults.temperature, Some(1.0));
            assert_eq!(defaults.top_k, Some(20));
            assert_eq!(defaults.top_p, Some(0.95));
            assert_eq!(defaults.min_p, Some(f64::from(0.05f32)));
            assert_eq!(defaults.do_sample, None);
        }
        Ok(())
    }

    #[test]
    fn flint_infers_missing_stops_from_vocabulary_without_overriding_explicit_sets() -> Result<()> {
        let tokens = [
            "<unk>",
            "<|observation|>",
            "<|assistant|>",
            "<|end_of_text|>",
            "<|developer|>",
            "<|user|>",
            "<think>",
            "<tool_call>",
        ];
        let vocab = tokens
            .into_iter()
            .zip(0u32..)
            .map(|(token, id)| (token.to_string(), id))
            .collect::<AHashMap<_, _>>();
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .map_err(anyhow::Error::msg)?,
        );
        for (architecture, explicit, expected) in [
            ("flint", None, vec![3, 1, 4, 5]),
            ("qwen2", None, vec![3]),
            ("flint", Some(vec![3, 1]), vec![3, 1]),
            ("flint", Some(Vec::new()), vec![3]),
        ] {
            let mut metadata = HashMap::from([
                (
                    "general.architecture".to_string(),
                    Value::String(architecture.to_string()),
                ),
                ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(3)),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(
                        tokens
                            .into_iter()
                            .map(|token| Value::String(token.to_string()))
                            .collect(),
                    ),
                ),
            ]);
            if let Some(explicit) = explicit {
                metadata.insert(
                    "tokenizer.ggml.eog_token_ids".to_string(),
                    Value::Array(explicit.into_iter().map(Value::U32).collect()),
                );
            }
            let config = generation_config_from_metadata(&metadata)?.unwrap();
            assert_eq!(
                calculate_eos_tokens(&ChatTemplate::default(), Some(&config), &tokenizer),
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn unrelated_or_empty_metadata_has_no_generation_fallback() -> Result<()> {
        for metadata in [
            HashMap::new(),
            HashMap::from([(
                "general.name".to_string(),
                Value::String("Flint".to_string()),
            )]),
            HashMap::from([(
                "tokenizer.ggml.eog_token_ids".to_string(),
                Value::Array(Vec::new()),
            )]),
        ] {
            assert!(generation_config_from_metadata(&metadata)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn bos_only_metadata_remains_available_for_validation() -> Result<()> {
        let metadata = HashMap::from([("tokenizer.ggml.bos_token_id".to_string(), Value::U32(4))]);
        let config = generation_config_from_metadata(&metadata)?.unwrap();
        config.validate_token_ids(5)?;
        assert!(config.validate_token_ids(4).is_err());
        Ok(())
    }

    #[test]
    fn malformed_generation_metadata_returns_keyed_errors() {
        for (key, value) in [
            ("tokenizer.ggml.eog_token_ids", Value::U32(1)),
            (
                "tokenizer.ggml.eog_token_ids",
                Value::Array(vec![Value::F32(1.0)]),
            ),
            ("tokenizer.ggml.eos_token_id", Value::Bool(true)),
            ("tokenizer.ggml.eot_token_id", Value::I32(-1)),
            (
                "tokenizer.ggml.eom_token_id",
                Value::String("1".to_string()),
            ),
            ("tokenizer.ggml.bos_token_id", Value::U64(u64::MAX)),
            ("general.sampling.top_k", Value::F32(20.0)),
            ("general.sampling.top_k", Value::I32(-1)),
            ("general.sampling.temp", Value::U32(1)),
            ("general.sampling.temp", Value::F32(f32::NAN)),
            ("general.sampling.top_p", Value::F64(f64::INFINITY)),
            ("general.sampling.min_p", Value::String("0.05".to_string())),
        ] {
            let metadata = HashMap::from([(key.to_string(), value)]);
            let error = generation_config_from_metadata(&metadata).unwrap_err();
            assert!(error.to_string().contains(key), "{error}");
        }
    }
}
