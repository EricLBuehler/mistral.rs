use std::collections::HashMap;

use anyhow::{bail, Result};
use candle_core::quantized::gguf_file::Value;

const BASE_MODEL_COUNT: &str = "general.base_model.count";
const BASE_MODEL_PREFIX: &str = "general.base_model.";
const BASE_MODEL_REPO_URL_SUFFIX: &str = ".repo_url";
const HUGGING_FACE_PREFIX: &str = "https://huggingface.co/";

pub(crate) fn infer_hf_base_model_id<'a>(
    sources: impl IntoIterator<Item = (&'a str, &'a HashMap<String, Value>)>,
) -> Result<Option<String>> {
    let mut selected: Option<(&str, String)> = None;
    for (label, metadata) in sources {
        let Some(model_id) = source_hf_base_model_id(label, metadata)? else {
            continue;
        };
        if let Some((selected_label, selected_id)) = selected.as_ref() {
            if selected_id != &model_id {
                bail!(
                    "GGUF base-model metadata mismatch: `{selected_label}` references \
                     `{selected_id}`, but `{label}` references `{model_id}`"
                );
            }
        } else {
            selected = Some((label, model_id));
        }
    }
    Ok(selected.map(|(_, model_id)| model_id))
}

fn source_hf_base_model_id(
    label: &str,
    metadata: &HashMap<String, Value>,
) -> Result<Option<String>> {
    let count = metadata
        .get(BASE_MODEL_COUNT)
        .map(|value| metadata_usize(value, BASE_MODEL_COUNT))
        .transpose()?;
    if count.is_some_and(|count| count > 1) {
        bail!(
            "GGUF `{label}` identifies multiple base models; automatic Hugging Face asset \
             discovery requires exactly one base model"
        );
    }

    let mut candidates = metadata
        .iter()
        .filter_map(|(key, value)| {
            let index = key
                .strip_prefix(BASE_MODEL_PREFIX)?
                .strip_suffix(BASE_MODEL_REPO_URL_SUFFIX)?;
            Some((index, value))
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(index, _)| *index);

    if candidates.len() > 1 {
        bail!(
            "GGUF `{label}` contains multiple base-model repository URLs; automatic Hugging Face \
             asset discovery requires exactly one"
        );
    }
    let Some((index, value)) = candidates.into_iter().next() else {
        return Ok(None);
    };
    let index = index.parse::<usize>().map_err(|_| {
        anyhow::anyhow!(
            "GGUF metadata key `{BASE_MODEL_PREFIX}{index}{BASE_MODEL_REPO_URL_SUFFIX}` has a \
             nonnumeric base-model index"
        )
    })?;
    if count == Some(0) {
        bail!(
            "GGUF `{label}` declares no base models but contains \
             `{BASE_MODEL_PREFIX}{index}{BASE_MODEL_REPO_URL_SUFFIX}`"
        );
    }
    if let Some(count) = count {
        if index >= count {
            bail!(
                "GGUF `{label}` base-model index {index} is outside its declared base-model count \
                 {count}"
            );
        }
    }
    let Value::String(url) = value else {
        bail!(
            "GGUF metadata `{BASE_MODEL_PREFIX}{index}{BASE_MODEL_REPO_URL_SUFFIX}` must be a string"
        );
    };
    canonical_hf_model_id(url).map(Some)
}

fn canonical_hf_model_id(url: &str) -> Result<String> {
    let Some(path) = url.strip_prefix(HUGGING_FACE_PREFIX) else {
        bail!(
            "GGUF base-model repository URL `{url}` is not a canonical Hugging Face model URL; \
             expected `https://huggingface.co/<owner>/<repo>`"
        );
    };
    let mut components = path.split('/');
    let owner = components.next().unwrap_or_default();
    let repo = components.next().unwrap_or_default();
    if owner.is_empty()
        || repo.is_empty()
        || components.next().is_some()
        || !valid_model_id_component(owner)
        || !valid_model_id_component(repo)
    {
        bail!(
            "GGUF base-model repository URL `{url}` is not a canonical Hugging Face model URL; \
             expected `https://huggingface.co/<owner>/<repo>`"
        );
    }
    Ok(format!("{owner}/{repo}"))
}

fn valid_model_id_component(value: &str) -> bool {
    value.len() <= 96
        && value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        && value
            .bytes()
            .next_back()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        && !value.contains("--")
        && !value.contains("..")
        && !value.ends_with(".git")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn metadata_usize(value: &Value, key: &str) -> Result<usize> {
    let value = match value {
        Value::U8(value) => *value as u64,
        Value::U16(value) => *value as u64,
        Value::U32(value) => *value as u64,
        Value::U64(value) => *value,
        Value::I8(value) if *value >= 0 => *value as u64,
        Value::I16(value) if *value >= 0 => *value as u64,
        Value::I32(value) if *value >= 0 => *value as u64,
        Value::I64(value) if *value >= 0 => *value as u64,
        _ => bail!("GGUF metadata `{key}` must be a nonnegative integer"),
    };
    usize::try_from(value).map_err(|_| anyhow::anyhow!("GGUF metadata `{key}` does not fit usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(url: &str) -> HashMap<String, Value> {
        HashMap::from([
            (BASE_MODEL_COUNT.to_string(), Value::U32(1)),
            (
                "general.base_model.0.repo_url".to_string(),
                Value::String(url.to_string()),
            ),
        ])
    }

    #[test]
    fn infers_matching_hugging_face_base_model() {
        let model = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        let projector = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        assert_eq!(
            infer_hf_base_model_id([("model", &model), ("projector", &projector)]).unwrap(),
            Some("Qwen/Qwen3.5-4B".to_string())
        );
    }

    #[test]
    fn accepts_one_source_without_base_metadata() {
        let model = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        let projector = HashMap::new();
        assert_eq!(
            infer_hf_base_model_id([("model", &model), ("projector", &projector)]).unwrap(),
            Some("Qwen/Qwen3.5-4B".to_string())
        );
    }

    #[test]
    fn rejects_mismatched_component_base_models() {
        let model = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        let projector = metadata("https://huggingface.co/Qwen/Qwen3.5-9B");
        let error =
            infer_hf_base_model_id([("model", &model), ("projector", &projector)]).unwrap_err();
        assert!(error.to_string().contains("metadata mismatch"));
    }

    #[test]
    fn rejects_a_mismatch_from_any_projector_component() {
        let model = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        let vision = metadata("https://huggingface.co/Qwen/Qwen3.5-4B");
        let audio = metadata("https://huggingface.co/Qwen/Qwen3.5-9B");
        let error = infer_hf_base_model_id([
            ("model", &model),
            ("vision projector", &vision),
            ("audio projector", &audio),
        ])
        .unwrap_err();

        assert!(error.to_string().contains("audio projector"));
        assert!(error.to_string().contains("metadata mismatch"));
    }

    #[test]
    fn rejects_ambiguous_base_models() {
        let mut model = metadata("https://huggingface.co/org/first");
        model.insert(BASE_MODEL_COUNT.to_string(), Value::U32(2));
        model.insert(
            "general.base_model.1.repo_url".to_string(),
            Value::String("https://huggingface.co/org/second".to_string()),
        );
        let error = infer_hf_base_model_id([("model", &model)]).unwrap_err();
        assert!(error.to_string().contains("multiple base models"));
    }

    #[test]
    fn rejects_noncanonical_base_model_urls() {
        for url in [
            "http://huggingface.co/Qwen/Qwen3.5-4B",
            "https://example.com/Qwen/Qwen3.5-4B",
            "https://huggingface.co/Qwen/Qwen3.5-4B/tree/main",
            "https://huggingface.co/Qwen/Qwen3.5-4B?revision=main",
            "https://huggingface.co/Qwen/Qwen3%2E5-4B",
            "https://huggingface.co/.Qwen/Qwen3.5-4B",
            "https://huggingface.co/Qwen/Qwen3.5-4B-",
            "https://huggingface.co/Qwen/Qwen3..5-4B",
            "https://huggingface.co/Qwen/Qwen3--5-4B",
            "https://huggingface.co/Qwen/Qwen3.5-4B.git",
        ] {
            let model = metadata(url);
            let error = infer_hf_base_model_id([("model", &model)]).unwrap_err();
            assert!(error.to_string().contains("not a canonical"));
        }
    }

    #[test]
    fn rejects_malformed_base_model_indices() {
        let malformed = HashMap::from([
            (BASE_MODEL_COUNT.to_string(), Value::U32(1)),
            (
                "general.base_model.primary.repo_url".to_string(),
                Value::String("https://huggingface.co/Qwen/Qwen3.5-4B".to_string()),
            ),
        ]);
        assert!(infer_hf_base_model_id([("model", &malformed)])
            .unwrap_err()
            .to_string()
            .contains("nonnumeric"));

        let out_of_range = HashMap::from([
            (BASE_MODEL_COUNT.to_string(), Value::U32(1)),
            (
                "general.base_model.1.repo_url".to_string(),
                Value::String("https://huggingface.co/Qwen/Qwen3.5-4B".to_string()),
            ),
        ]);
        assert!(infer_hf_base_model_id([("model", &out_of_range)])
            .unwrap_err()
            .to_string()
            .contains("outside"));
    }
}
