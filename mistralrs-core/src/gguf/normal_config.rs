use super::normal_registry::{schema_for, CanonicalGgufArchitecture, GgufDescriptor};
#[cfg(test)]
use super::normal_registry::{NativeModelAdapter, NORMAL_MODEL_ADAPTERS};
use crate::{gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY, NormalLoaderType};
use candle_core::quantized::gguf_file::Value as GgufValue;
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::{collections::HashMap, error::Error, fmt, num::TryFromIntError};

const DEFAULT_ROPE_THETA: f64 = 10_000.0;
const GEMMA2_27B_LAYER_COUNT: usize = 46;
const GPT_OSS_ALPHA: f64 = 1.702;
const GPT_OSS_SWIGLU_LIMIT: f64 = 7.0;
const GPT_OSS_SWA_PERIOD: usize = 2;
const SMOLLM3_NO_ROPE_INTERVAL: usize = 4;

type BuilderFn = fn(&MetadataView<'_>) -> SynthesisResult<JsonValue>;
type SynthesisResult<T> = Result<T, NormalConfigSynthesisError>;

#[derive(Debug)]
pub(crate) struct NormalConfigSynthesisError {
    message: String,
}

impl NormalConfigSynthesisError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for NormalConfigSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for NormalConfigSynthesisError {}

#[derive(Debug)]
pub(crate) struct NormalConfigBuilder {
    pub(crate) loader: NormalLoaderType,
    build: BuilderFn,
}

pub(crate) const NORMAL_CONFIG_BUILDERS: &[NormalConfigBuilder; 26] = &[
    NormalConfigBuilder {
        loader: NormalLoaderType::Mistral,
        build: build_mistral,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Gemma,
        build: build_gemma,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Mixtral,
        build: build_mixtral,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Llama,
        build: build_llama,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Phi2,
        build: build_phi2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Phi3,
        build: build_phi3,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Qwen2,
        build: build_qwen2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Gemma2,
        build: build_gemma2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Starcoder2,
        build: build_starcoder2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Phi3_5MoE,
        build: build_phi3_moe,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::DeepSeekV2,
        build: build_deepseek_v2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::DeepSeekV3,
        build: build_deepseek_v3,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Qwen3,
        build: build_qwen3,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::GLM4,
        build: build_glm4,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::GLM4MoeLite,
        build: build_glm4_moe_lite,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::GLM4Moe,
        build: build_glm4_moe,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Qwen3Moe,
        build: build_qwen3_moe,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::SmolLm3,
        build: build_smollm3,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::GraniteMoeHybrid,
        build: build_granite,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::GptOss,
        build: build_gpt_oss,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::HunYuanDenseV1,
        build: build_hunyuan_dense,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::HunYuanMoEV1,
        build: build_hunyuan_moe,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Qwen3Next,
        build: build_qwen3_next,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Qwen3_5,
        build: build_qwen35,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Lfm2,
        build: build_lfm2,
    },
    NormalConfigBuilder {
        loader: NormalLoaderType::Lfm2Moe,
        build: build_lfm2_moe,
    },
];

pub(crate) fn synthesize_normal_config(
    loader: &NormalLoaderType,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> SynthesisResult<String> {
    let value = synthesize_normal_config_value(loader, metadata, tensor_names)?;
    serde_json::to_string(&value).map_err(|error| {
        NormalConfigSynthesisError::new(format!(
            "Failed to serialize synthesized `{loader}` config: {error}"
        ))
    })
}

pub(crate) fn synthesize_normal_config_value(
    loader: &NormalLoaderType,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> SynthesisResult<JsonValue> {
    let view = MetadataView::new(loader, metadata, tensor_names)?;
    let builder = NORMAL_CONFIG_BUILDERS
        .iter()
        .find(|builder| &builder.loader == loader)
        .ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "No standalone GGUF config builder is registered for `{loader}`"
            ))
        })?;
    let value = (builder.build)(&view)?;
    with_reload_identity(loader, value)
}

fn with_reload_identity(
    loader: &NormalLoaderType,
    mut value: JsonValue,
) -> SynthesisResult<JsonValue> {
    let object = value.as_object_mut().ok_or_else(|| {
        NormalConfigSynthesisError::new(format!(
            "Synthesized `{loader}` config must be a JSON object"
        ))
    })?;
    object.insert(
        "architectures".to_string(),
        json!([loader.causal_lm_name()]),
    );
    object.insert("model_type".to_string(), json!(loader.model_type_name()));
    Ok(value)
}

pub(crate) fn normal_loader_hint_from_external_config(
    config: &str,
) -> anyhow::Result<Option<NormalLoaderType>> {
    let value: JsonValue = serde_json::from_str(config)
        .map_err(|error| anyhow::anyhow!("External model config is not valid JSON: {error}"))?;
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!("External model config must be a JSON object, got {value}")
    })?;
    loader_hint_from_config_object(object)
}

pub(crate) fn normalize_external_normal_config(
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
    config: &str,
) -> anyhow::Result<String> {
    let value: JsonValue = serde_json::from_str(config)
        .map_err(|error| anyhow::anyhow!("External model config is not valid JSON: {error}"))?;
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!("External model config must be a JSON object, got {value}")
    })?;
    let hint = loader_hint_from_config_object(object)?;
    if let Some(hint) = hint.as_ref() {
        anyhow::ensure!(
            hint == loader,
            "External config selects native `{hint}` but GGUF resolution selected `{loader}`"
        );
    }

    let Some(shape) = external_config_shape(object)? else {
        let mut text_config = object.clone();
        text_config.insert("quantization_config".to_string(), JsonValue::Null);
        if matches!(
            architecture,
            CanonicalGgufArchitecture::Qwen35 | CanonicalGgufArchitecture::Qwen35Moe
        ) {
            text_config.insert(GDN_V_HEAD_LAYOUT_CONFIG_KEY.to_string(), json!("tiled"));
        }
        text_config.insert(
            "architectures".to_string(),
            json!([loader.causal_lm_name()]),
        );
        text_config.insert("model_type".to_string(), json!(loader.model_type_name()));
        return serde_json::to_string(&text_config)
            .map_err(|error| anyhow::anyhow!("Failed to serialize native text config: {error}"));
    };
    let expected_loader = shape.loader();
    anyhow::ensure!(
        &expected_loader == loader,
        "{} external config requires native `{expected_loader}`, not `{loader}`",
        shape.label()
    );

    let mut text_config = match shape {
        ExternalConfigShape::Nested(_) => object
            .get("text_config")
            .and_then(JsonValue::as_object)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} external config requires an object-valued `text_config` for native text loading",
                    shape.label()
                )
            })?,
        ExternalConfigShape::Qwen35Text | ExternalConfigShape::Qwen35MoeText => object.clone(),
    };
    if let Some(nested_hint) = loader_hint_from_config_object(&text_config)? {
        anyhow::ensure!(
            &nested_hint == loader,
            "{} text config selects native `{nested_hint}` instead of `{loader}`",
            shape.label()
        );
    }

    text_config.insert("quantization_config".to_string(), JsonValue::Null);
    if matches!(
        shape,
        ExternalConfigShape::Nested(NestedTextConfig::Qwen35Moe)
            | ExternalConfigShape::Qwen35MoeText
    ) {
        normalize_qwen35_text_config(&mut text_config)?;
    }
    if matches!(
        shape,
        ExternalConfigShape::Nested(NestedTextConfig::Qwen35) | ExternalConfigShape::Qwen35Text
    ) {
        if let Some(tie_word_embeddings) = object.get("tie_word_embeddings") {
            text_config.insert(
                "tie_word_embeddings".to_string(),
                tie_word_embeddings.clone(),
            );
        }
    }
    if matches!(
        architecture,
        CanonicalGgufArchitecture::Qwen35 | CanonicalGgufArchitecture::Qwen35Moe
    ) {
        text_config.insert(GDN_V_HEAD_LAYOUT_CONFIG_KEY.to_string(), json!("tiled"));
    }
    text_config.insert(
        "architectures".to_string(),
        json!([loader.causal_lm_name()]),
    );
    text_config.insert("model_type".to_string(), json!(loader.model_type_name()));

    serde_json::to_string(&text_config).map_err(|error| {
        anyhow::anyhow!(
            "Failed to serialize {} native text config: {error}",
            shape.label()
        )
    })
}

pub(crate) fn validate_normal_config_tensor_inventory(
    config: &str,
    tensor_names: &[String],
) -> anyhow::Result<()> {
    let value: JsonValue = serde_json::from_str(config)
        .map_err(|error| anyhow::anyhow!("Native model config is not valid JSON: {error}"))?;
    let Some(configured_tied) = value
        .get("tie_word_embeddings")
        .and_then(JsonValue::as_bool)
    else {
        return Ok(());
    };
    let expected_tied = !tensor_names.iter().any(|name| name == "output.weight");
    let output_state = if expected_tied { "absent" } else { "present" };
    anyhow::ensure!(
        configured_tied == expected_tied,
        "Model config has `tie_word_embeddings={configured_tied}`, but GGUF `output.weight` is {output_state}; expected `tie_word_embeddings={expected_tied}`"
    );
    Ok(())
}

fn loader_hint_from_config_object(
    object: &JsonMap<String, JsonValue>,
) -> anyhow::Result<Option<NormalLoaderType>> {
    if let Some(architecture) = first_config_architecture(object)? {
        return normal_loader_from_architecture(architecture).map(Some);
    }

    match config_model_type(object)? {
        Some("mistral3") => Ok(Some(NormalLoaderType::Mistral)),
        Some("lfm2_vl") => Ok(Some(NormalLoaderType::Lfm2)),
        Some("qwen3_5" | "qwen3_5_text") => Ok(Some(NormalLoaderType::Qwen3_5)),
        Some("qwen3_5_moe" | "qwen3_5_moe_text") => Ok(Some(NormalLoaderType::Qwen3Next)),
        _ => Ok(None),
    }
}

fn normal_loader_from_architecture(architecture: &str) -> anyhow::Result<NormalLoaderType> {
    match architecture {
        "Mistral3ForConditionalGeneration" => return Ok(NormalLoaderType::Mistral),
        "Lfm2VlForConditionalGeneration" => return Ok(NormalLoaderType::Lfm2),
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" => {
            return Ok(NormalLoaderType::Qwen3_5)
        }
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
            return Ok(NormalLoaderType::Qwen3Next)
        }
        _ => {}
    }
    NormalLoaderType::from_causal_lm_name(architecture).map_err(|_| {
        anyhow::anyhow!(
            "External config architecture `{architecture}` is not a supported native text model"
        )
    })
}

#[derive(Clone, Copy)]
enum NestedTextConfig {
    Mistral3,
    Lfm2Vl,
    Qwen35,
    Qwen35Moe,
}

#[derive(Clone, Copy)]
enum ExternalConfigShape {
    Nested(NestedTextConfig),
    Qwen35Text,
    Qwen35MoeText,
}

impl ExternalConfigShape {
    fn loader(self) -> NormalLoaderType {
        match self {
            Self::Nested(NestedTextConfig::Mistral3) => NormalLoaderType::Mistral,
            Self::Nested(NestedTextConfig::Lfm2Vl) => NormalLoaderType::Lfm2,
            Self::Nested(NestedTextConfig::Qwen35) | Self::Qwen35Text => NormalLoaderType::Qwen3_5,
            Self::Nested(NestedTextConfig::Qwen35Moe) | Self::Qwen35MoeText => {
                NormalLoaderType::Qwen3Next
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Nested(NestedTextConfig::Mistral3) => "Mistral3",
            Self::Nested(NestedTextConfig::Lfm2Vl) => "LFM2-VL",
            Self::Nested(NestedTextConfig::Qwen35) | Self::Qwen35Text => "Qwen3.5",
            Self::Nested(NestedTextConfig::Qwen35Moe) | Self::Qwen35MoeText => "Qwen3.5 MoE",
        }
    }
}

fn external_config_shape(
    object: &JsonMap<String, JsonValue>,
) -> anyhow::Result<Option<ExternalConfigShape>> {
    let architecture = first_config_architecture(object)?;
    let model_type = config_model_type(object)?;
    if architecture == Some("Mistral3ForConditionalGeneration") || model_type == Some("mistral3") {
        return Ok(Some(ExternalConfigShape::Nested(
            NestedTextConfig::Mistral3,
        )));
    }
    if architecture == Some("Lfm2VlForConditionalGeneration") || model_type == Some("lfm2_vl") {
        return Ok(Some(ExternalConfigShape::Nested(NestedTextConfig::Lfm2Vl)));
    }
    if architecture == Some("Qwen3_5ForConditionalGeneration") || model_type == Some("qwen3_5") {
        return Ok(Some(ExternalConfigShape::Nested(NestedTextConfig::Qwen35)));
    }
    if architecture == Some("Qwen3_5ForCausalLM") || model_type == Some("qwen3_5_text") {
        return Ok(Some(ExternalConfigShape::Qwen35Text));
    }
    if architecture == Some("Qwen3_5MoeForConditionalGeneration")
        || model_type == Some("qwen3_5_moe")
    {
        return Ok(Some(ExternalConfigShape::Nested(
            NestedTextConfig::Qwen35Moe,
        )));
    }
    if architecture == Some("Qwen3_5MoeForCausalLM") || model_type == Some("qwen3_5_moe_text") {
        return Ok(Some(ExternalConfigShape::Qwen35MoeText));
    }
    Ok(None)
}

fn first_config_architecture(object: &JsonMap<String, JsonValue>) -> anyhow::Result<Option<&str>> {
    Ok(match object.get("architectures") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Array(architectures)) => match architectures.first() {
            None => None,
            Some(JsonValue::String(architecture)) => Some(architecture.as_str()),
            Some(value) => {
                anyhow::bail!("External config `architectures[0]` must be a string, got {value}")
            }
        },
        Some(value) => {
            anyhow::bail!("External config `architectures` must be an array, got {value}")
        }
    })
}

fn config_model_type(object: &JsonMap<String, JsonValue>) -> anyhow::Result<Option<&str>> {
    Ok(match object.get("model_type") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(model_type)) => Some(model_type.as_str()),
        Some(value) => {
            anyhow::bail!("External config `model_type` must be a string, got {value}")
        }
    })
}

fn normalize_qwen35_text_config(
    text_config: &mut JsonMap<String, JsonValue>,
) -> anyhow::Result<()> {
    copy_qwen35_rope_field(text_config, "rope_theta")?;
    copy_qwen35_rope_field(text_config, "partial_rotary_factor")?;
    match text_config.get("intermediate_size") {
        Some(value) if value.is_number() => {}
        Some(value) if !value.is_null() => {
            anyhow::bail!(
                "Qwen3.5 MoE text config `intermediate_size` must be numeric, got {value}"
            )
        }
        _ => {
            let intermediate_size = text_config
                .get("shared_expert_intermediate_size")
                .filter(|value| value.is_number())
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.5 MoE text config requires numeric `shared_expert_intermediate_size` to derive native `intermediate_size`"
                    )
                })?;
            text_config.insert("intermediate_size".to_string(), intermediate_size);
        }
    }
    Ok(())
}

fn copy_qwen35_rope_field(
    text_config: &mut JsonMap<String, JsonValue>,
    field: &str,
) -> anyhow::Result<()> {
    match text_config.get(field) {
        Some(value) if value.is_number() => return Ok(()),
        Some(value) if !value.is_null() => {
            anyhow::bail!("Qwen3.5 MoE text config `{field}` must be numeric, got {value}")
        }
        _ => {}
    }
    let value = text_config
        .get("rope_parameters")
        .and_then(JsonValue::as_object)
        .and_then(|rope| rope.get(field))
        .filter(|value| value.is_number())
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("Qwen3.5 MoE text config requires numeric `rope_parameters.{field}`")
        })?;
    text_config.insert(field.to_string(), value);
    Ok(())
}

struct MetadataView<'a> {
    loader: &'a NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
    metadata: &'a HashMap<String, GgufValue>,
    tensor_names: &'a [String],
}

impl<'a> MetadataView<'a> {
    fn new(
        loader: &'a NormalLoaderType,
        metadata: &'a HashMap<String, GgufValue>,
        tensor_names: &'a [String],
    ) -> SynthesisResult<Self> {
        let architecture_value = metadata.get("general.architecture").ok_or_else(|| {
            NormalConfigSynthesisError::new(
                "GGUF metadata is missing required key `general.architecture`",
            )
        })?;
        let architecture_name = value_string(architecture_value).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `general.architecture` must be a string, got {architecture_value:?}"
            ))
        })?;
        let architecture = architecture_name.parse().map_err(
            |error: super::normal_registry::NormalGgufRegistryError| {
                NormalConfigSynthesisError::new(error.to_string())
            },
        )?;
        let schema = schema_for(architecture);
        if !schema.compatible_loaders.contains(loader) {
            return Err(NormalConfigSynthesisError::new(format!(
                "GGUF architecture `{architecture}` cannot synthesize native `{loader}` config; compatible loaders are {:?}",
                schema.compatible_loaders
            )));
        }

        let metadata_keys = metadata.keys().map(String::as_str).collect::<Vec<_>>();
        let tensor_refs = tensor_names.iter().map(String::as_str).collect::<Vec<_>>();
        let descriptor = GgufDescriptor::new(architecture.as_str(), &metadata_keys, &tensor_refs)
            .map_err(|error| NormalConfigSynthesisError::new(error.to_string()))?;
        schema
            .validate(&descriptor)
            .map_err(|error| NormalConfigSynthesisError::new(error.to_string()))?;

        Ok(Self {
            loader,
            architecture,
            metadata,
            tensor_names,
        })
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.architecture.as_str())
    }

    fn value(&self, suffix: &str) -> Option<&GgufValue> {
        self.metadata.get(&self.key(suffix))
    }

    fn required_value(&self, suffix: &str) -> SynthesisResult<&GgufValue> {
        self.value(suffix).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "Standalone `{}` config requires GGUF metadata `{}`; provide the original Hugging Face config if this GGUF omitted it",
                self.loader,
                self.key(suffix)
            ))
        })
    }

    fn required_usize(&self, suffix: &str) -> SynthesisResult<usize> {
        let key = self.key(suffix);
        value_usize(self.required_value(suffix)?).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` must be a non-negative integer fitting this platform"
            ))
        })
    }

    fn optional_usize(&self, suffix: &str) -> SynthesisResult<Option<usize>> {
        let key = self.key(suffix);
        self.value(suffix)
            .map(|value| {
                value_usize(value).ok_or_else(|| {
                    NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` must be a non-negative integer fitting this platform"
                    ))
                })
            })
            .transpose()
    }

    fn required_uniform_usize(&self, suffix: &str) -> SynthesisResult<usize> {
        let values = self.required_usize_values(suffix)?;
        uniform_value(self, suffix, &values)
    }

    fn required_uniform_nonzero_usize(&self, suffix: &str) -> SynthesisResult<usize> {
        let values = self.required_usize_values(suffix)?;
        let nonzero = values
            .into_iter()
            .filter(|value| *value != 0)
            .collect::<Vec<_>>();
        if nonzero.is_empty() {
            return Err(NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{}` has no non-zero value",
                self.key(suffix)
            )));
        }
        uniform_value(self, suffix, &nonzero)
    }

    fn required_usize_values(&self, suffix: &str) -> SynthesisResult<Vec<usize>> {
        let key = self.key(suffix);
        values_usize(self.required_value(suffix)?).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` must be an integer or integer array"
            ))
        })
    }

    fn per_layer_usize(
        &self,
        suffix: &str,
        layer_count: usize,
    ) -> SynthesisResult<Option<Vec<usize>>> {
        let Some(value) = self.value(suffix) else {
            return Ok(None);
        };
        let key = self.key(suffix);
        let values = values_usize(value).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` must be an integer or integer array"
            ))
        })?;
        if values.len() == 1 {
            return Ok(Some(vec![values[0]; layer_count]));
        }
        if values.len() != layer_count {
            return Err(NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` has {} entries for {layer_count} layers",
                values.len()
            )));
        }
        Ok(Some(values))
    }

    fn required_f64(&self, suffix: &str) -> SynthesisResult<f64> {
        let key = self.key(suffix);
        value_f64(self.required_value(suffix)?).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` must be a floating-point number"
            ))
        })
    }

    fn optional_f64(&self, suffix: &str) -> SynthesisResult<Option<f64>> {
        let key = self.key(suffix);
        self.value(suffix)
            .map(|value| {
                value_f64(value).ok_or_else(|| {
                    NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` must be a floating-point number"
                    ))
                })
            })
            .transpose()
    }

    fn optional_bool(&self, suffix: &str) -> SynthesisResult<Option<bool>> {
        let key = self.key(suffix);
        self.value(suffix)
            .map(|value| {
                value_bool(value).ok_or_else(|| {
                    NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` must be a boolean"
                    ))
                })
            })
            .transpose()
    }

    fn optional_string(&self, suffix: &str) -> SynthesisResult<Option<&str>> {
        let key = self.key(suffix);
        self.value(suffix)
            .map(|value| {
                value_string(value).ok_or_else(|| {
                    NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` must be a string"
                    ))
                })
            })
            .transpose()
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names.iter().any(|tensor| tensor == name)
    }

    fn has_tensor_marker(&self, marker: &str) -> bool {
        self.tensor_names
            .iter()
            .any(|tensor| tensor.contains(marker))
    }

    fn layer_has_tensor(&self, layer: usize, marker: &str) -> bool {
        let prefix = format!("blk.{layer}.");
        self.tensor_names
            .iter()
            .any(|tensor| tensor.starts_with(&prefix) && tensor.contains(marker))
    }

    fn vocab_size(&self) -> SynthesisResult<usize> {
        if let Some(size) = self.optional_usize("vocab_size")? {
            return Ok(size);
        }
        let tokens = self.metadata.get("tokenizer.ggml.tokens").ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "Standalone `{}` config requires either `{}` or `tokenizer.ggml.tokens`",
                self.loader,
                self.key("vocab_size")
            ))
        })?;
        match tokens {
            GgufValue::Array(tokens) => Ok(tokens.len()),
            _ => Err(NormalConfigSynthesisError::new(
                "GGUF metadata `tokenizer.ggml.tokens` must be an array",
            )),
        }
    }

    fn norm_epsilon(&self) -> SynthesisResult<f64> {
        match (
            self.optional_f64("attention.layer_norm_rms_epsilon")?,
            self.optional_f64("attention.layer_norm_epsilon")?,
        ) {
            (Some(value), _) | (None, Some(value)) => Ok(value),
            (None, None) => Err(NormalConfigSynthesisError::new(format!(
                "Standalone `{}` config requires `{}` or `{}`",
                self.loader,
                self.key("attention.layer_norm_rms_epsilon"),
                self.key("attention.layer_norm_epsilon")
            ))),
        }
    }

    fn rope_theta(&self, architecture_default: Option<f64>) -> SynthesisResult<f64> {
        match self.optional_f64("rope.freq_base")? {
            Some(value) => Ok(value),
            None => architecture_default.ok_or_else(|| {
                NormalConfigSynthesisError::new(format!(
                    "Standalone `{}` config requires GGUF metadata `{}`",
                    self.loader,
                    self.key("rope.freq_base")
                ))
            }),
        }
    }

    fn tie_word_embeddings(&self) -> bool {
        !self.has_tensor("output.weight")
    }

    fn head_dim(&self, hidden_size: usize, head_count: usize) -> SynthesisResult<usize> {
        if let Some(value) = self.optional_usize("attention.key_length")? {
            return Ok(value);
        }
        exact_div(
            self,
            "embedding length / attention head count",
            hidden_size,
            head_count,
        )
    }

    fn sliding_window(&self) -> SynthesisResult<Option<usize>> {
        Ok(self
            .optional_usize("attention.sliding_window")?
            .filter(|window| *window != 0))
    }

    fn expert_gating(&self) -> SynthesisResult<Option<ExpertGating>> {
        let Some(value) = self.optional_usize("expert_gating_func")? else {
            return Ok(None);
        };
        match value {
            1 => Ok(Some(ExpertGating::Softmax)),
            2 => Ok(Some(ExpertGating::Sigmoid)),
            _ => Err(NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{}` has unsupported expert gating value {value}",
                self.key("expert_gating_func")
            ))),
        }
    }

    fn optional_token_id(&self, key: &str) -> SynthesisResult<Option<u32>> {
        let Some(value) = self.metadata.get(key) else {
            return Ok(None);
        };
        let value = value_u64(value).ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` must be a non-negative integer"
            ))
        })?;
        u32::try_from(value).map(Some).map_err(|_| {
            NormalConfigSynthesisError::new(format!(
                "GGUF metadata `{key}` value {value} does not fit a token id"
            ))
        })
    }

    fn sliding_pattern(&self, layer_count: usize) -> SynthesisResult<Option<Vec<bool>>> {
        let Some(value) = self.value("attention.sliding_window_pattern") else {
            return Ok(None);
        };
        let key = self.key("attention.sliding_window_pattern");
        match value {
            GgufValue::Array(values) => {
                if values.len() != layer_count {
                    return Err(NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` has {} entries for {layer_count} layers",
                        values.len()
                    )));
                }
                values
                    .iter()
                    .map(|value| {
                        value_bool(value).ok_or_else(|| {
                            NormalConfigSynthesisError::new(format!(
                                "GGUF metadata `{key}` must contain only booleans"
                            ))
                        })
                    })
                    .collect::<SynthesisResult<Vec<_>>>()
                    .map(Some)
            }
            _ => {
                let period = value_usize(value).ok_or_else(|| {
                    NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` must be a positive period or boolean array"
                    ))
                })?;
                if period == 0 {
                    return Err(NormalConfigSynthesisError::new(format!(
                        "GGUF metadata `{key}` cannot use period zero"
                    )));
                }
                Ok(Some(
                    (0..layer_count)
                        .map(|layer| layer % period < period - 1)
                        .collect(),
                ))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpertGating {
    Softmax,
    Sigmoid,
}

struct StandardFields {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    norm_epsilon: f64,
    rope_theta: f64,
    head_dim: usize,
    sliding_window: Option<usize>,
    tie_word_embeddings: bool,
}

impl StandardFields {
    fn read(metadata: &MetadataView<'_>, rope_default: Option<f64>) -> SynthesisResult<Self> {
        let intermediate_size = metadata.required_uniform_usize("feed_forward_length")?;
        Self::read_with_intermediate_size(metadata, rope_default, intermediate_size)
    }

    fn read_with_intermediate_size(
        metadata: &MetadataView<'_>,
        rope_default: Option<f64>,
        intermediate_size: usize,
    ) -> SynthesisResult<Self> {
        let hidden_size = metadata.required_uniform_usize("embedding_length")?;
        let num_attention_heads = metadata.required_uniform_usize("attention.head_count")?;
        Ok(Self {
            vocab_size: metadata.vocab_size()?,
            hidden_size,
            intermediate_size,
            num_hidden_layers: transformer_block_count(metadata)?,
            num_attention_heads,
            num_key_value_heads: metadata
                .required_uniform_nonzero_usize("attention.head_count_kv")?,
            max_position_embeddings: metadata.required_usize("context_length")?,
            norm_epsilon: metadata.norm_epsilon()?,
            rope_theta: metadata.rope_theta(rope_default)?,
            head_dim: metadata.head_dim(hidden_size, num_attention_heads)?,
            sliding_window: metadata.sliding_window()?,
            tie_word_embeddings: metadata.tie_word_embeddings(),
        })
    }

    fn rms_json(&self) -> JsonMap<String, JsonValue> {
        let mut config = JsonMap::new();
        config.insert("vocab_size".into(), json!(self.vocab_size));
        config.insert("hidden_size".into(), json!(self.hidden_size));
        config.insert("intermediate_size".into(), json!(self.intermediate_size));
        config.insert("num_hidden_layers".into(), json!(self.num_hidden_layers));
        config.insert(
            "num_attention_heads".into(),
            json!(self.num_attention_heads),
        );
        config.insert(
            "num_key_value_heads".into(),
            json!(self.num_key_value_heads),
        );
        config.insert(
            "max_position_embeddings".into(),
            json!(self.max_position_embeddings),
        );
        config.insert("rms_norm_eps".into(), json!(self.norm_epsilon));
        config.insert("rope_theta".into(), json!(self.rope_theta));
        config.insert("sliding_window".into(), json!(self.sliding_window));
        config.insert(
            "tie_word_embeddings".into(),
            json!(self.tie_word_embeddings),
        );
        config.insert("quantization_config".into(), JsonValue::Null);
        config
    }
}

fn transformer_block_count(metadata: &MetadataView<'_>) -> SynthesisResult<usize> {
    let block_count = metadata.required_usize("block_count")?;
    let nextn_layers = metadata
        .optional_usize("nextn_predict_layers")?
        .unwrap_or(0);
    block_count.checked_sub(nextn_layers).ok_or_else(|| {
        NormalConfigSynthesisError::new(format!(
            "`{}` ({nextn_layers}) exceeds `{}` ({block_count})",
            metadata.key("nextn_predict_layers"),
            metadata.key("block_count")
        ))
    })
}

#[cfg(test)]
fn builder_for(loader: &NormalLoaderType) -> Option<&'static NormalConfigBuilder> {
    NORMAL_CONFIG_BUILDERS
        .iter()
        .find(|builder| &builder.loader == loader)
}

#[cfg(test)]
fn registry_adapter(loader: &NormalLoaderType) -> Option<&'static NativeModelAdapter> {
    NORMAL_MODEL_ADAPTERS
        .iter()
        .find(|adapter| &adapter.loader == loader)
}

fn uniform_value(
    metadata: &MetadataView<'_>,
    suffix: &str,
    values: &[usize],
) -> SynthesisResult<usize> {
    let Some(first) = values.first().copied() else {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` cannot be an empty array",
            metadata.key(suffix)
        )));
    };
    if values.iter().any(|value| *value != first) {
        return Err(NormalConfigSynthesisError::new(format!(
            "Native `{}` config cannot represent per-layer `{}` values {values:?}; provide the original Hugging Face config or use a compatible architecture path",
            metadata.loader,
            metadata.key(suffix)
        )));
    }
    Ok(first)
}

fn exact_div(
    metadata: &MetadataView<'_>,
    field: &str,
    numerator: usize,
    denominator: usize,
) -> SynthesisResult<usize> {
    if denominator == 0 || !numerator.is_multiple_of(denominator) {
        return Err(NormalConfigSynthesisError::new(format!(
            "Cannot derive `{field}` for native `{}` config: {numerator} is not evenly divisible by {denominator}",
            metadata.loader
        )));
    }
    Ok(numerator / denominator)
}

fn value_string(value: &GgufValue) -> Option<&str> {
    match value {
        GgufValue::String(value) => Some(value),
        _ => None,
    }
}

fn value_bool(value: &GgufValue) -> Option<bool> {
    match value {
        GgufValue::Bool(value) => Some(*value),
        _ => None,
    }
}

fn value_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U8(value) => Some(u64::from(*value)),
        GgufValue::U16(value) => Some(u64::from(*value)),
        GgufValue::U32(value) => Some(u64::from(*value)),
        GgufValue::U64(value) => Some(*value),
        GgufValue::I8(value) => u64::try_from(*value).ok(),
        GgufValue::I16(value) => u64::try_from(*value).ok(),
        GgufValue::I32(value) => u64::try_from(*value).ok(),
        GgufValue::I64(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn value_usize(value: &GgufValue) -> Option<usize> {
    value_u64(value).and_then(|value| usize::try_from(value).ok())
}

fn values_usize(value: &GgufValue) -> Option<Vec<usize>> {
    match value {
        GgufValue::Array(values) => values.iter().map(value_usize).collect(),
        _ => value_usize(value).map(|value| vec![value]),
    }
}

fn value_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F32(value) => Some(f64::from(*value)),
        GgufValue::F64(value) => Some(*value),
        _ => None,
    }
}

fn usize_to_u32(value: usize) -> Result<u32, TryFromIntError> {
    u32::try_from(value)
}

fn build_mistral(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    let rope_parameters = match metadata.optional_string("rope.scaling.type")? {
        None | Some("none") => JsonValue::Null,
        Some("yarn") => json!({
            "rope_theta": fields.rope_theta,
            "rope_type": "yarn",
            "factor": metadata.required_f64("rope.scaling.factor")?,
            "beta_fast": metadata.required_f64("rope.scaling.yarn_beta_fast")?,
            "beta_slow": metadata.required_f64("rope.scaling.yarn_beta_slow")?,
            "mscale": 1.0,
            "mscale_all_dim": metadata.required_f64("rope.scaling.yarn_log_multiplier")?,
            "original_max_position_embeddings": metadata.required_usize("rope.scaling.original_context_length")?,
            "llama_4_scaling_beta": metadata.optional_f64("attention.temperature_scale")?,
        }),
        Some(scaling_type) => {
            return Err(NormalConfigSynthesisError::new(format!(
                "Native Mistral standalone config does not implement GGUF RoPE scaling type `{scaling_type}`"
            )))
        }
    };
    config.insert("rope_parameters".into(), rope_parameters);
    Ok(JsonValue::Object(config))
}

fn build_gemma(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Gemma")?;
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("hidden_act".into(), JsonValue::Null);
    config.insert("hidden_activation".into(), json!("gelu_pytorch_tanh"));
    Ok(JsonValue::Object(config))
}

fn build_mixtral(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Mixtral")?;
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert(
        "num_local_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    Ok(JsonValue::Object(config))
}

fn build_llama(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("rope_scaling".into(), llama_rope_scaling(metadata)?);
    Ok(JsonValue::Object(config))
}

fn build_phi2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Phi-2")?;
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let rope_dim = metadata.required_usize("rope.dimension_count")?;
    let partial_rotary_factor =
        ratio_f64(metadata, "partial_rotary_factor", rope_dim, fields.head_dim)?;
    let mut config = fields.rms_json();
    config.remove("rms_norm_eps");
    config.remove("sliding_window");
    config.insert("layer_norm_eps".into(), json!(fields.norm_epsilon));
    config.insert("hidden_act".into(), json!("gelu_new"));
    config.insert("partial_rotary_factor".into(), json!(partial_rotary_factor));
    config.insert(
        "qk_layernorm".into(),
        json!(metadata.has_tensor_marker(".attn_q_norm.")),
    );
    Ok(JsonValue::Object(config))
}

fn build_phi3(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    let original_context = phi_rope_original_context(metadata, fields.max_position_embeddings)?;
    let partial_rotary_factor = ratio_f64(
        metadata,
        "partial_rotary_factor",
        metadata.required_usize("rope.dimension_count")?,
        fields.head_dim,
    )?;
    config.insert("hidden_act".into(), json!("silu"));
    config.insert(
        "original_max_position_embeddings".into(),
        json!(original_context),
    );
    config.insert("rope_scaling".into(), phi_rope_scaling(metadata)?);
    config.insert(
        "rope_scaling_attn_factor".into(),
        metadata
            .optional_f64("rope.scaling.attn_factor")?
            .map_or(JsonValue::Null, JsonValue::from),
    );
    config.insert("partial_rotary_factor".into(), json!(partial_rotary_factor));
    config.insert(
        "bos_token_id".into(),
        json!(metadata.optional_token_id("tokenizer.ggml.bos_token_id")?),
    );
    config.insert(
        "eos_token_id".into(),
        json!(metadata.optional_token_id("tokenizer.ggml.eos_token_id")?),
    );
    Ok(JsonValue::Object(config))
}

fn build_qwen2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Qwen2")?;
    let fields = StandardFields::read(metadata, None)?;
    let mut config = fields.rms_json();
    let pattern = qwen2_layer_types(metadata, &fields)?;
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("use_sliding_window".into(), json!(pattern.is_some()));
    config.insert("max_window_layers".into(), json!(fields.num_hidden_layers));
    config.insert("layer_types".into(), json!(pattern));
    Ok(JsonValue::Object(config))
}

fn build_gemma2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Gemma2")?;
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let sliding_window = fields.sliding_window.ok_or_else(|| {
        NormalConfigSynthesisError::new(format!(
            "Standalone Gemma2 config requires `{}`",
            metadata.key("attention.sliding_window")
        ))
    })?;
    let query_pre_attn_scalar = if fields.num_hidden_layers == GEMMA2_27B_LAYER_COUNT {
        exact_div(
            metadata,
            "Gemma2 27B query_pre_attn_scalar",
            fields.hidden_size,
            fields.num_attention_heads,
        )?
    } else {
        fields.head_dim
    };
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("hidden_act".into(), JsonValue::Null);
    config.insert("hidden_activation".into(), json!("gelu_pytorch_tanh"));
    config.insert("sliding_window".into(), json!(sliding_window));
    config.insert(
        "attn_logit_softcapping".into(),
        json!(metadata.optional_f64("attn_logit_softcapping")?),
    );
    config.insert(
        "final_logit_softcapping".into(),
        json!(metadata.optional_f64("final_logit_softcapping")?),
    );
    config.insert("query_pre_attn_scalar".into(), json!(query_pre_attn_scalar));
    Ok(JsonValue::Object(config))
}

fn build_starcoder2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "StarCoder2")?;
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.remove("rms_norm_eps");
    config.insert("norm_epsilon".into(), json!(fields.norm_epsilon));
    config.insert("hidden_act".into(), json!("gelu_pytorch_tanh"));
    config.insert(
        "use_bias".into(),
        json!(
            metadata.has_tensor_marker(".attn_q.bias")
                || metadata.has_tensor_marker(".ffn_up.bias")
        ),
    );
    Ok(JsonValue::Object(config))
}

fn build_phi3_moe(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let expert_used = metadata.required_usize("expert_used_count")?;
    if expert_used != 2 {
        return Err(NormalConfigSynthesisError::new(format!(
            "Native Phi-3.5 MoE routing supports exactly 2 experts per token, but `{}` is {expert_used}",
            metadata.key("expert_used_count")
        )));
    }
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert(
        "original_max_position_embeddings".into(),
        json!(phi_rope_original_context(
            metadata,
            fields.max_position_embeddings
        )?),
    );
    config.insert("rope_scaling".into(), phi_rope_scaling(metadata)?);
    config.insert(
        "rope_scaling_attn_factor".into(),
        metadata
            .optional_f64("rope.scaling.attn_factor")?
            .map_or(JsonValue::Null, JsonValue::from),
    );
    config.insert(
        "num_local_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "lm_head_bias".into(),
        json!(metadata.has_tensor("output.bias")),
    );
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert("router_jitter_noise".into(), json!(0.0));
    Ok(JsonValue::Object(config))
}

fn build_deepseek_v2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    if metadata.expert_gating()?.unwrap_or(ExpertGating::Softmax) != ExpertGating::Softmax {
        return Err(NormalConfigSynthesisError::new(format!(
            "`{}` uses sigmoid routing and is incompatible with the native DeepSeekV2 loader; select DeepSeekV3 or GLM4MoeLite explicitly",
            metadata.key("expert_gating_func")
        )));
    }
    build_deepseek(metadata, false)
}

fn build_deepseek_v3(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    build_deepseek(metadata, true)
}

fn build_qwen3(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Qwen3")?;
    let fields = StandardFields::read(metadata, None)?;
    let (use_sliding_window, max_window_layers) = qwen3_sliding_policy(metadata, &fields)?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("use_sliding_window".into(), json!(use_sliding_window));
    config.insert("max_window_layers".into(), json!(max_window_layers));
    Ok(JsonValue::Object(config))
}

fn build_glm4(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "GLM4")?;
    let fields = StandardFields::read(metadata, None)?;
    let partial_rotary_factor = ratio_f64(
        metadata,
        "partial_rotary_factor",
        metadata.required_usize("rope.dimension_count")?,
        fields.head_dim,
    )?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("partial_rotary_factor".into(), json!(partial_rotary_factor));
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    Ok(JsonValue::Object(config))
}

fn build_glm4_moe_lite(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let q_lora_rank = metadata.required_usize("attention.q_lora_rank")?;
    let mut config = deepseek_core_json(metadata, true)?;
    config.remove("q_lora_rank");
    config.insert("q_lora_rank".into(), json!(q_lora_rank));
    config.insert(
        "num_key_value_heads".into(),
        json!(metadata
            .optional_usize("attention.head_count_kv")?
            .unwrap_or(1)),
    );
    config.insert(
        "n_routed_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "n_shared_experts".into(),
        json!(metadata.required_usize("expert_shared_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    Ok(JsonValue::Object(config))
}

fn build_glm4_moe(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "GLM4 MoE")?;
    let fields = StandardFields::read(metadata, None)?;
    let partial_rotary_factor = ratio_f64(
        metadata,
        "partial_rotary_factor",
        metadata.required_usize("rope.dimension_count")?,
        fields.head_dim,
    )?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("partial_rotary_factor".into(), json!(partial_rotary_factor));
    config.insert(
        "use_qk_norm".into(),
        json!(metadata.has_tensor_marker(".attn_q_norm.")),
    );
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    insert_common_moe(metadata, &mut config)?;
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(true)),
    );
    Ok(JsonValue::Object(config))
}

fn build_qwen3_moe(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Qwen3 MoE")?;
    let fields = StandardFields::read(metadata, None)?;
    let (use_sliding_window, max_window_layers) = qwen3_sliding_policy(metadata, &fields)?;
    let mut config = fields.rms_json();
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("use_sliding_window".into(), json!(use_sliding_window));
    config.insert("max_window_layers".into(), json!(max_window_layers));
    config.insert(
        "moe_intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert(
        "num_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(true)),
    );
    config.insert("decoder_sparse_step".into(), json!(1));
    config.insert(
        "mlp_only_layers".into(),
        json!(dense_layer_indices(metadata, fields.num_hidden_layers)),
    );
    Ok(JsonValue::Object(config))
}

fn build_smollm3(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, Some(DEFAULT_ROPE_THETA))?;
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("rope_scaling".into(), smollm3_rope_scaling(metadata)?);
    config.insert("no_rope_layers".into(), JsonValue::Null);
    config.insert(
        "no_rope_layer_interval".into(),
        json!(SMOLLM3_NO_ROPE_INTERVAL),
    );
    Ok(JsonValue::Object(config))
}

fn build_granite(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Granite")?;
    let hidden_size = metadata.required_uniform_usize("embedding_length")?;
    let num_hidden_layers = transformer_block_count(metadata)?;
    let num_attention_heads = metadata.required_uniform_usize("attention.head_count")?;
    let kv_heads = metadata
        .per_layer_usize("attention.head_count_kv", num_hidden_layers)?
        .ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "Standalone Granite config requires `{}`",
                metadata.key("attention.head_count_kv")
            ))
        })?;
    let num_key_value_heads =
        uniform_nonzero_values(metadata, "attention.head_count_kv", &kv_heads)?;
    let layer_types = granite_layer_types(metadata, &kv_heads, num_hidden_layers);
    let is_hybrid = layer_types.contains(&"mamba");
    let mut config = JsonMap::new();
    config.insert("vocab_size".into(), json!(metadata.vocab_size()?));
    config.insert("hidden_size".into(), json!(hidden_size));
    config.insert(
        "intermediate_size".into(),
        json!(metadata.required_uniform_usize("feed_forward_length")?),
    );
    config.insert(
        "shared_intermediate_size".into(),
        json!(metadata.optional_usize("expert_shared_feed_forward_length")?),
    );
    config.insert("num_hidden_layers".into(), json!(num_hidden_layers));
    config.insert("num_attention_heads".into(), json!(num_attention_heads));
    config.insert("num_key_value_heads".into(), json!(num_key_value_heads));
    config.insert("rms_norm_eps".into(), json!(metadata.norm_epsilon()?));
    config.insert(
        "rope_theta".into(),
        json!(metadata.rope_theta(Some(DEFAULT_ROPE_THETA))?),
    );
    config.insert(
        "max_position_embeddings".into(),
        json!(metadata.required_usize("context_length")?),
    );
    config.insert("rope_scaling".into(), JsonValue::Null);
    config.insert("quantization_config".into(), JsonValue::Null);
    config.insert(
        "tie_word_embeddings".into(),
        json!(metadata.tie_word_embeddings()),
    );
    config.insert("layer_types".into(), json!(layer_types));
    config.insert(
        "attention_multiplier".into(),
        json!(metadata.optional_f64("attention.scale")?.unwrap_or(1.0)),
    );
    config.insert(
        "embedding_multiplier".into(),
        json!(metadata.optional_f64("embedding_scale")?.unwrap_or(1.0)),
    );
    config.insert(
        "residual_multiplier".into(),
        json!(metadata.optional_f64("residual_scale")?.unwrap_or(1.0)),
    );
    config.insert(
        "logits_scaling".into(),
        json!(metadata.optional_f64("logit_scale")?.unwrap_or(1.0)),
    );
    config.insert(
        "num_local_experts".into(),
        json!(metadata.optional_usize("expert_count")?.unwrap_or(0)),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.optional_usize("expert_used_count")?.unwrap_or(1)),
    );
    let use_rope = metadata
        .optional_bool("rope.scaling.finetuned")?
        .unwrap_or(!is_hybrid);
    config.insert(
        "position_embedding_type".into(),
        json!(if use_rope { "rope" } else { "nope" }),
    );

    if is_hybrid {
        let inner_size = metadata.required_usize("ssm.inner_size")?;
        let mamba_n_heads = metadata.required_usize("ssm.time_step_rank")?;
        config.insert(
            "mamba_n_groups".into(),
            json!(metadata.required_usize("ssm.group_count")?),
        );
        config.insert(
            "mamba_d_state".into(),
            json!(metadata.required_usize("ssm.state_size")?),
        );
        config.insert(
            "mamba_d_conv".into(),
            json!(metadata.required_usize("ssm.conv_kernel")?),
        );
        config.insert("mamba_n_heads".into(), json!(mamba_n_heads));
        config.insert(
            "mamba_d_head".into(),
            json!(exact_div(
                metadata,
                "mamba_d_head",
                inner_size,
                mamba_n_heads
            )?),
        );
        config.insert(
            "mamba_expand".into(),
            json!(exact_div(
                metadata,
                "mamba_expand",
                inner_size,
                hidden_size
            )?),
        );
        config.insert(
            "mamba_conv_bias".into(),
            json!(metadata.has_tensor_marker(".ssm_conv1d.bias")),
        );
        config.insert(
            "mamba_proj_bias".into(),
            json!(
                metadata.has_tensor_marker(".ssm_in.bias")
                    || metadata.has_tensor_marker(".ssm_out.bias")
            ),
        );
    } else {
        config.insert("mamba_n_heads".into(), JsonValue::Null);
        config.insert("mamba_d_head".into(), JsonValue::Null);
    }

    Ok(JsonValue::Object(config))
}

fn build_gpt_oss(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, None)?;
    let sliding_window = fields.sliding_window.ok_or_else(|| {
        NormalConfigSynthesisError::new(format!(
            "Standalone GPT-OSS config requires `{}`",
            metadata.key("attention.sliding_window")
        ))
    })?;
    let mut config = fields.rms_json();
    config.insert(
        "intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert("sliding_window".into(), json!(sliding_window));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert(
        "num_local_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert(
        "layer_types".into(),
        json!(gpt_oss_layer_types(metadata, fields.num_hidden_layers)?),
    );
    config.insert("alpha".into(), json!(GPT_OSS_ALPHA));
    config.insert("swiglu_limit".into(), json!(GPT_OSS_SWIGLU_LIMIT));
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert("rope_scaling".into(), gpt_oss_rope_scaling(metadata)?);
    Ok(JsonValue::Object(config))
}

fn build_hunyuan_dense(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let fields = StandardFields::read(metadata, None)?;
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("rope_scaling".into(), JsonValue::Null);
    config.insert("use_cla".into(), json!(false));
    config.insert("cla_share_factor".into(), JsonValue::Null);
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert(
        "mlp_bias".into(),
        json!(metadata.has_tensor_marker(".ffn_gate.bias")),
    );
    config.insert("pretraining_tp".into(), json!(1));
    config.insert("add_classification_head".into(), json!(false));
    Ok(JsonValue::Object(config))
}

fn build_hunyuan_moe(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    let leading_dense = metadata
        .optional_usize("leading_dense_block_count")?
        .unwrap_or(0);
    let fields = StandardFields::read(metadata, None)?;
    let gating = metadata.expert_gating()?.unwrap_or(ExpertGating::Softmax);
    let has_shared_expert = metadata.has_tensor_marker(".ffn_gate_shexp.");
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert("rope_scaling".into(), JsonValue::Null);
    config.insert(
        "num_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "num_shared_expert".into(),
        json!(metadata.required_usize("expert_shared_count")?),
    );
    config.insert(
        "moe_topk".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert(
        "moe_intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert("use_mixed_mlp_moe".into(), json!(has_shared_expert));
    config.insert("moe_layer_num_skipped".into(), json!(leading_dense));
    config.insert("moe_drop_tokens".into(), json!(false));
    config.insert("moe_random_routing_dropped_token".into(), json!(false));
    config.insert(
        "routed_scaling_factor".into(),
        json!(metadata
            .optional_f64("expert_weights_scale")?
            .unwrap_or(1.0)),
    );
    config.insert(
        "moe_router_enable_expert_bias".into(),
        json!(metadata.has_tensor_marker(".exp_probs_b")),
    );
    config.insert(
        "moe_router_use_sigmoid".into(),
        json!(gating == ExpertGating::Sigmoid),
    );
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(true)),
    );
    config.insert("use_cla".into(), json!(false));
    config.insert("cla_share_factor".into(), JsonValue::Null);
    config.insert(
        "use_qk_norm".into(),
        json!(metadata.has_tensor_marker(".attn_q_norm.")),
    );
    config.insert(
        "attention_bias".into(),
        json!(metadata.has_tensor_marker(".attn_q.bias")),
    );
    config.insert(
        "mlp_bias".into(),
        json!(metadata.has_tensor_marker(".ffn_gate.bias")),
    );
    config.insert("pretraining_tp".into(), json!(1));
    config.insert("add_classification_head".into(), json!(false));
    Ok(JsonValue::Object(config))
}

fn build_qwen3_next(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Qwen3Next")?;
    let intermediate_size = match metadata.optional_usize("feed_forward_length")? {
        Some(intermediate_size) => intermediate_size,
        None => metadata.required_usize("expert_shared_feed_forward_length")?,
    };
    let fields = StandardFields::read_with_intermediate_size(metadata, None, intermediate_size)?;
    let key_head_dim = metadata.required_usize("ssm.state_size")?;
    let key_head_count = metadata.required_usize("ssm.group_count")?;
    let value_head_count = metadata.required_usize("ssm.time_step_rank")?;
    let value_head_dim = exact_div(
        metadata,
        "linear_value_head_dim",
        metadata.required_usize("ssm.inner_size")?,
        value_head_count,
    )?;
    let mut config = fields.rms_json();
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert(
        "partial_rotary_factor".into(),
        json!(ratio_f64(
            metadata,
            "partial_rotary_factor",
            metadata.required_usize("rope.dimension_count")?,
            fields.head_dim
        )?),
    );
    config.insert(
        "linear_conv_kernel_dim".into(),
        json!(metadata.required_usize("ssm.conv_kernel")?),
    );
    config.insert("linear_key_head_dim".into(), json!(key_head_dim));
    config.insert("linear_value_head_dim".into(), json!(value_head_dim));
    config.insert("linear_num_key_heads".into(), json!(key_head_count));
    config.insert("linear_num_value_heads".into(), json!(value_head_count));
    config.insert("decoder_sparse_step".into(), json!(1));
    config.insert(
        "moe_intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert(
        "shared_expert_intermediate_size".into(),
        json!(metadata.required_usize("expert_shared_feed_forward_length")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert(
        "num_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(true)),
    );
    config.insert(
        "mlp_only_layers".into(),
        json!(dense_layer_indices(metadata, fields.num_hidden_layers)),
    );
    config.insert(
        "full_attention_interval".into(),
        json!(metadata.required_usize("full_attention_interval")?),
    );
    if metadata.architecture == CanonicalGgufArchitecture::Qwen35Moe {
        config.insert(GDN_V_HEAD_LAYOUT_CONFIG_KEY.into(), json!("tiled"));
    }
    Ok(JsonValue::Object(config))
}

fn build_qwen35(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "Qwen3.5")?;
    let fields = StandardFields::read(metadata, None)?;
    let rotary_dim = metadata.required_usize("rope.dimension_count")?;
    let partial_rotary_factor = ratio_f64(
        metadata,
        "partial_rotary_factor",
        rotary_dim,
        fields.head_dim,
    )?;
    if rotary_dim == 0 || rotary_dim > fields.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` must be positive, even, and no larger than head dimension {}",
            metadata.key("rope.dimension_count"),
            fields.head_dim
        )));
    }
    if !fields.rope_theta.is_finite() || fields.rope_theta <= 0.0 {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` must be finite and positive",
            metadata.key("rope.freq_base")
        )));
    }
    if let Some(value_head_dim) = metadata.optional_usize("attention.value_length")? {
        if value_head_dim != fields.head_dim {
            return Err(NormalConfigSynthesisError::new(format!(
                "Qwen3.5 attention key length {} differs from value length {value_head_dim}",
                fields.head_dim
            )));
        }
    }
    let mut mrope_section = metadata.required_usize_values("rope.dimension_sections")?;
    while mrope_section.last() == Some(&0) {
        mrope_section.pop();
    }
    if mrope_section.len() != 3 || mrope_section.contains(&0) {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` must contain three non-zero MRoPE sections",
            metadata.key("rope.dimension_sections")
        )));
    }
    let section_width = mrope_section.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width)
            .ok_or_else(|| NormalConfigSynthesisError::new("Qwen3.5 MRoPE section width overflow"))
    })?;
    let represented_rotary_dim = section_width.checked_mul(2).ok_or_else(|| {
        NormalConfigSynthesisError::new("Qwen3.5 MRoPE rotary dimension overflow")
    })?;
    if represented_rotary_dim != rotary_dim {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` spans {} rotary dimensions, expected {rotary_dim}",
            metadata.key("rope.dimension_sections"),
            represented_rotary_dim
        )));
    }
    let value_head_count = metadata.required_usize("ssm.time_step_rank")?;
    let key_head_count = metadata.required_usize("ssm.group_count")?;
    if key_head_count == 0
        || value_head_count == 0
        || !value_head_count.is_multiple_of(key_head_count)
    {
        return Err(NormalConfigSynthesisError::new(format!(
            "Qwen3.5 has incompatible GDN head counts: {key_head_count} key and {value_head_count} value"
        )));
    }
    let value_head_dim = exact_div(
        metadata,
        "linear_value_head_dim",
        metadata.required_usize("ssm.inner_size")?,
        value_head_count,
    )?;
    let mut config = fields.rms_json();
    config.remove("rope_theta");
    config.remove("sliding_window");
    config.insert("hidden_act".into(), json!("silu"));
    config.insert("head_dim".into(), json!(fields.head_dim));
    config.insert(
        "rope_parameters".into(),
        json!({
            "rope_theta": fields.rope_theta,
            "mrope_section": mrope_section,
            "partial_rotary_factor": partial_rotary_factor,
        }),
    );
    let full_attention_interval = metadata.required_usize("full_attention_interval")?;
    if full_attention_interval == 0 || full_attention_interval > fields.num_hidden_layers {
        return Err(NormalConfigSynthesisError::new(format!(
            "Qwen3.5 full attention interval {full_attention_interval} is invalid for {} layers",
            fields.num_hidden_layers
        )));
    }
    config.insert(
        "full_attention_interval".into(),
        json!(full_attention_interval),
    );
    config.insert(
        "linear_conv_kernel_dim".into(),
        json!(metadata.required_usize("ssm.conv_kernel")?),
    );
    config.insert(
        "linear_key_head_dim".into(),
        json!(metadata.required_usize("ssm.state_size")?),
    );
    config.insert("linear_value_head_dim".into(), json!(value_head_dim));
    config.insert("linear_num_key_heads".into(), json!(key_head_count));
    config.insert("linear_num_value_heads".into(), json!(value_head_count));
    config.insert(GDN_V_HEAD_LAYOUT_CONFIG_KEY.into(), json!("tiled"));
    Ok(JsonValue::Object(config))
}

fn build_lfm2(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    build_lfm(metadata, false)
}

fn build_lfm2_moe(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    build_lfm(metadata, true)
}

fn build_lfm(metadata: &MetadataView<'_>, is_moe: bool) -> SynthesisResult<JsonValue> {
    reject_unsupported_rope_scaling(metadata, "LFM2")?;
    let hidden_size = metadata.required_uniform_usize("embedding_length")?;
    let num_hidden_layers = transformer_block_count(metadata)?;
    let num_attention_heads = metadata.required_uniform_usize("attention.head_count")?;
    let kv_heads = metadata
        .per_layer_usize("attention.head_count_kv", num_hidden_layers)?
        .ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "Standalone LFM2 config requires `{}`",
                metadata.key("attention.head_count_kv")
            ))
        })?;
    let num_key_value_heads =
        uniform_nonzero_values(metadata, "attention.head_count_kv", &kv_heads)?;
    let layer_types = lfm_layer_types(metadata, &kv_heads, num_hidden_layers);
    let feed_forward = metadata.required_uniform_usize("feed_forward_length")?;
    let mut config = JsonMap::new();
    config.insert(
        "model_type".into(),
        json!(if is_moe { "lfm2_moe" } else { "lfm2" }),
    );
    config.insert(
        "architectures".into(),
        json!(if is_moe {
            vec!["Lfm2MoeForCausalLM"]
        } else {
            vec!["Lfm2ForCausalLM"]
        }),
    );
    config.insert("vocab_size".into(), json!(metadata.vocab_size()?));
    config.insert("hidden_size".into(), json!(hidden_size));
    config.insert("intermediate_size".into(), json!(feed_forward));
    config.insert("block_ff_dim".into(), json!(feed_forward));
    config.insert("num_hidden_layers".into(), json!(num_hidden_layers));
    config.insert("num_attention_heads".into(), json!(num_attention_heads));
    config.insert("num_key_value_heads".into(), json!(num_key_value_heads));
    config.insert(
        "max_position_embeddings".into(),
        json!(metadata.required_usize("context_length")?),
    );
    config.insert("norm_eps".into(), json!(metadata.norm_epsilon()?));
    config.insert(
        "rope_parameters".into(),
        json!({
            "rope_theta": metadata.rope_theta(None)?,
            "rope_type": "default",
        }),
    );
    config.insert(
        "conv_bias".into(),
        json!(metadata.has_tensor_marker(".shortconv.conv.bias")),
    );
    config.insert(
        "conv_L_cache".into(),
        json!(metadata.required_usize("shortconv.l_cache")?),
    );
    config.insert("block_auto_adjust_ff_dim".into(), json!(false));
    config.insert(
        "tie_word_embeddings".into(),
        json!(metadata.tie_word_embeddings()),
    );
    config.insert("tie_embedding".into(), JsonValue::Null);
    config.insert("layer_types".into(), json!(layer_types));
    config.insert("quantization_config".into(), JsonValue::Null);

    if is_moe {
        let gating = metadata.expert_gating()?.unwrap_or(ExpertGating::Sigmoid);
        if gating != ExpertGating::Sigmoid {
            return Err(NormalConfigSynthesisError::new(format!(
                "Native LFM2 MoE expects sigmoid routing, but `{}` is not sigmoid",
                metadata.key("expert_gating_func")
            )));
        }
        config.insert(
            "moe_intermediate_size".into(),
            json!(metadata.required_usize("expert_feed_forward_length")?),
        );
        config.insert(
            "num_dense_layers".into(),
            json!(metadata
                .optional_usize("leading_dense_block_count")?
                .unwrap_or(0)),
        );
        config.insert(
            "num_experts".into(),
            json!(metadata.required_usize("expert_count")?),
        );
        config.insert(
            "num_experts_per_tok".into(),
            json!(metadata.required_usize("expert_used_count")?),
        );
        config.insert(
            "use_expert_bias".into(),
            json!(metadata.has_tensor_marker(".exp_probs_b")),
        );
        config.insert(
            "norm_topk_prob".into(),
            json!(metadata
                .optional_bool("expert_weights_norm")?
                .unwrap_or(true)),
        );
        config.insert(
            "routed_scaling_factor".into(),
            json!(metadata
                .optional_f64("expert_weights_scale")?
                .unwrap_or(1.0)),
        );
    } else {
        config.insert("moe_intermediate_size".into(), json!(0));
        config.insert("num_dense_layers".into(), json!(num_hidden_layers));
        config.insert("num_experts".into(), json!(0));
        config.insert("num_experts_per_tok".into(), json!(0));
        config.insert("use_expert_bias".into(), json!(false));
        config.insert("norm_topk_prob".into(), json!(false));
        config.insert("routed_scaling_factor".into(), json!(1.0));
    }

    Ok(JsonValue::Object(config))
}

fn build_deepseek(metadata: &MetadataView<'_>, is_v3: bool) -> SynthesisResult<JsonValue> {
    let mut config = deepseek_core_json(metadata, false)?;
    let gating = metadata.expert_gating()?.unwrap_or(ExpertGating::Softmax);
    let group_count = metadata.optional_usize("expert_group_count")?.unwrap_or(1);
    let has_router_bias = metadata.has_tensor_marker(".exp_probs_b");
    let topk_method = if is_v3 && gating == ExpertGating::Sigmoid && has_router_bias {
        "noaux_tc"
    } else if group_count > 1 {
        "group_limited_greedy"
    } else {
        "greedy"
    };
    config.insert(
        "n_shared_experts".into(),
        json!(metadata.required_usize("expert_shared_count")?),
    );
    config.insert(
        "n_routed_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert("topk_method".into(), json!(topk_method));
    config.insert(
        "scoring_func".into(),
        json!(if gating == ExpertGating::Sigmoid {
            "sigmoid"
        } else {
            "softmax"
        }),
    );
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(false)),
    );
    Ok(JsonValue::Object(config))
}

fn deepseek_core_json(
    metadata: &MetadataView<'_>,
    q_lora_required: bool,
) -> SynthesisResult<JsonMap<String, JsonValue>> {
    let hidden_size = metadata.required_uniform_usize("embedding_length")?;
    let num_attention_heads = metadata.required_uniform_usize("attention.head_count")?;
    let qk_rope_head_dim = metadata.required_usize("rope.dimension_count")?;
    let q_head_dim = metadata.required_usize("attention.key_length_mla")?;
    let qk_nope_head_dim = q_head_dim.checked_sub(qk_rope_head_dim).ok_or_else(|| {
        NormalConfigSynthesisError::new(format!(
            "`{}` ({q_head_dim}) is smaller than `{}` ({qk_rope_head_dim})",
            metadata.key("attention.key_length_mla"),
            metadata.key("rope.dimension_count")
        ))
    })?;
    let q_lora_rank = if q_lora_required {
        Some(metadata.required_usize("attention.q_lora_rank")?)
    } else {
        metadata.optional_usize("attention.q_lora_rank")?
    };
    let mut config = JsonMap::new();
    config.insert("vocab_size".into(), json!(metadata.vocab_size()?));
    config.insert("hidden_size".into(), json!(hidden_size));
    config.insert(
        "intermediate_size".into(),
        json!(metadata.required_uniform_usize("feed_forward_length")?),
    );
    config.insert(
        "moe_intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert(
        "num_hidden_layers".into(),
        json!(transformer_block_count(metadata)?),
    );
    config.insert("num_attention_heads".into(), json!(num_attention_heads));
    config.insert(
        "routed_scaling_factor".into(),
        json!(metadata
            .optional_f64("expert_weights_scale")?
            .unwrap_or(1.0)),
    );
    config.insert(
        "moe_layer_freq".into(),
        json!(metadata.optional_usize("moe_every_n_layers")?.unwrap_or(1)),
    );
    config.insert(
        "first_k_dense_replace".into(),
        json!(metadata
            .optional_usize("leading_dense_block_count")?
            .unwrap_or(0)),
    );
    config.insert("hidden_act".into(), json!("silu"));
    config.insert(
        "max_position_embeddings".into(),
        json!(metadata.required_usize("context_length")?),
    );
    config.insert("rms_norm_eps".into(), json!(metadata.norm_epsilon()?));
    config.insert(
        "tie_word_embeddings".into(),
        json!(metadata.tie_word_embeddings()),
    );
    config.insert("rope_theta".into(), json!(metadata.rope_theta(None)?));
    config.insert("rope_scaling".into(), deepseek_rope_scaling(metadata)?);
    config.insert(
        "attention_bias".into(),
        json!(
            metadata.has_tensor_marker(".attn_q_a.bias")
                || metadata.has_tensor_marker(".attn_kv_a_mqa.bias")
        ),
    );
    config.insert("q_lora_rank".into(), json!(q_lora_rank));
    config.insert("qk_rope_head_dim".into(), json!(qk_rope_head_dim));
    config.insert(
        "kv_lora_rank".into(),
        json!(metadata.required_usize("attention.kv_lora_rank")?),
    );
    config.insert(
        "v_head_dim".into(),
        json!(metadata.required_usize("attention.value_length_mla")?),
    );
    config.insert("qk_nope_head_dim".into(), json!(qk_nope_head_dim));
    config.insert("quantization_config".into(), JsonValue::Null);
    config.insert(
        "n_group".into(),
        json!(metadata.optional_usize("expert_group_count")?.unwrap_or(1)),
    );
    config.insert(
        "topk_group".into(),
        json!(metadata
            .optional_usize("expert_group_used_count")?
            .unwrap_or(1)),
    );
    Ok(config)
}

fn insert_common_moe(
    metadata: &MetadataView<'_>,
    config: &mut JsonMap<String, JsonValue>,
) -> SynthesisResult<()> {
    config.insert(
        "moe_intermediate_size".into(),
        json!(metadata.required_usize("expert_feed_forward_length")?),
    );
    config.insert(
        "n_routed_experts".into(),
        json!(metadata.required_usize("expert_count")?),
    );
    config.insert(
        "n_shared_experts".into(),
        json!(metadata.required_usize("expert_shared_count")?),
    );
    config.insert(
        "num_experts_per_tok".into(),
        json!(metadata.required_usize("expert_used_count")?),
    );
    config.insert(
        "first_k_dense_replace".into(),
        json!(metadata
            .optional_usize("leading_dense_block_count")?
            .unwrap_or(0)),
    );
    config.insert(
        "routed_scaling_factor".into(),
        json!(metadata
            .optional_f64("expert_weights_scale")?
            .unwrap_or(1.0)),
    );
    config.insert(
        "n_group".into(),
        json!(metadata.optional_usize("expert_group_count")?.unwrap_or(1)),
    );
    config.insert(
        "topk_group".into(),
        json!(metadata
            .optional_usize("expert_group_used_count")?
            .unwrap_or(1)),
    );
    config.insert(
        "norm_topk_prob".into(),
        json!(metadata
            .optional_bool("expert_weights_norm")?
            .unwrap_or(false)),
    );
    Ok(())
}

fn reject_unsupported_rope_scaling(
    metadata: &MetadataView<'_>,
    family: &str,
) -> SynthesisResult<()> {
    let scaling_type = metadata.optional_string("rope.scaling.type")?;
    let has_baked_factors = metadata.has_tensor_marker("rope_freqs")
        || metadata.has_tensor_marker("rope_factors_long")
        || metadata.has_tensor_marker("rope_factors_short");
    if has_baked_factors || scaling_type.is_some_and(|scaling_type| scaling_type != "none") {
        return Err(NormalConfigSynthesisError::new(format!(
            "Standalone {family} config cannot reconstruct this GGUF's RoPE scaling from metadata alone; provide the original Hugging Face config"
        )));
    }
    Ok(())
}

fn llama_rope_scaling(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    rope_scaling_for_llama_family(metadata, "Llama")
}

fn smollm3_rope_scaling(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    rope_scaling_for_llama_family(metadata, "SmolLM3")
}

fn rope_scaling_for_llama_family(
    metadata: &MetadataView<'_>,
    family: &str,
) -> SynthesisResult<JsonValue> {
    if metadata.has_tensor_marker("rope_freqs") {
        return Ok(JsonValue::Null);
    }
    match metadata.optional_string("rope.scaling.type")? {
        None | Some("none") => Ok(JsonValue::Null),
        Some("linear") => Ok(json!({
            "factor": metadata.required_f64("rope.scaling.factor")?,
            "low_freq_factor": null,
            "high_freq_factor": null,
            "original_max_position_embeddings": null,
            "rope_type": "linear",
        })),
        Some(scaling_type) => Err(NormalConfigSynthesisError::new(format!(
            "Native {family} standalone config cannot represent GGUF RoPE scaling type `{scaling_type}` without its original config"
        ))),
    }
}

fn phi_rope_original_context(
    metadata: &MetadataView<'_>,
    context_length: usize,
) -> SynthesisResult<usize> {
    Ok(metadata
        .optional_usize("rope.scaling.original_context_length")?
        .unwrap_or(context_length))
}

fn phi_rope_scaling(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    if metadata.has_tensor_marker("rope_factors_long")
        || metadata.has_tensor_marker("rope_factors_short")
    {
        if !metadata.has_tensor_marker("rope_factors_long")
            || !metadata.has_tensor_marker("rope_factors_short")
        {
            return Err(NormalConfigSynthesisError::new(
                "Standalone Phi config requires both LongRoPE factor tensors",
            ));
        }
        metadata.required_f64("rope.scaling.attn_factor")?;
        return Ok(JsonValue::Null);
    }
    match metadata.optional_string("rope.scaling.type")? {
        None | Some("none") => Ok(JsonValue::Null),
        Some(scaling_type) => Err(NormalConfigSynthesisError::new(format!(
            "Standalone Phi config cannot reconstruct RoPE scaling type `{scaling_type}` without its factor tensors; provide the original Hugging Face config"
        ))),
    }
}

fn deepseek_rope_scaling(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    match metadata.optional_string("rope.scaling.type")? {
        None | Some("none") => Ok(JsonValue::Null),
        Some("yarn") => {
            let log_multiplier =
                metadata.required_f64("rope.scaling.yarn_log_multiplier")?;
            Ok(json!({
                "type": "yarn",
                "factor": metadata.required_f64("rope.scaling.factor")?,
                "original_max_position_embeddings": metadata.required_usize("rope.scaling.original_context_length")?,
                "beta_fast": metadata.required_f64("rope.scaling.yarn_beta_fast")?,
                "beta_slow": metadata.required_f64("rope.scaling.yarn_beta_slow")?,
                "mscale": 1.0,
                "mscale_all_dim": log_multiplier * 10.0,
            }))
        }
        Some(scaling_type) => Err(NormalConfigSynthesisError::new(format!(
            "Native DeepSeek standalone config does not implement GGUF RoPE scaling type `{scaling_type}`"
        ))),
    }
}

fn gpt_oss_rope_scaling(metadata: &MetadataView<'_>) -> SynthesisResult<JsonValue> {
    match metadata.optional_string("rope.scaling.type")? {
        None | Some("none") => Ok(JsonValue::Null),
        Some("yarn") => Ok(json!({
            "rope_type": "yarn",
            "factor": metadata.required_f64("rope.scaling.factor")?,
            "original_max_position_embeddings": metadata.required_usize("rope.scaling.original_context_length")?,
            "beta_fast": metadata.required_f64("rope.scaling.yarn_beta_fast")?,
            "beta_slow": metadata.required_f64("rope.scaling.yarn_beta_slow")?,
            "truncate": false,
        })),
        Some(scaling_type) => Err(NormalConfigSynthesisError::new(format!(
            "Native GPT-OSS standalone config cannot represent GGUF RoPE scaling type `{scaling_type}`"
        ))),
    }
}

fn qwen2_layer_types(
    metadata: &MetadataView<'_>,
    fields: &StandardFields,
) -> SynthesisResult<Option<Vec<&'static str>>> {
    if fields.sliding_window.is_none() {
        return Ok(None);
    }
    let pattern = metadata
        .sliding_pattern(fields.num_hidden_layers)?
        .unwrap_or_else(|| vec![true; fields.num_hidden_layers]);
    Ok(Some(
        pattern
            .into_iter()
            .map(|is_sliding| {
                if is_sliding {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect(),
    ))
}

fn qwen3_sliding_policy(
    metadata: &MetadataView<'_>,
    fields: &StandardFields,
) -> SynthesisResult<(bool, usize)> {
    if fields.sliding_window.is_none() {
        return Ok((false, fields.num_hidden_layers));
    }
    let pattern = metadata
        .sliding_pattern(fields.num_hidden_layers)?
        .ok_or_else(|| {
            NormalConfigSynthesisError::new(format!(
                "`{}` is present but `{}` is absent; native Qwen3 needs the layer policy, so provide the original Hugging Face config",
                metadata.key("attention.sliding_window"),
                metadata.key("attention.sliding_window_pattern")
            ))
        })?;
    let first_sliding = pattern.iter().position(|is_sliding| *is_sliding);
    let Some(first_sliding) = first_sliding else {
        return Ok((false, fields.num_hidden_layers));
    };
    if pattern[..first_sliding]
        .iter()
        .any(|is_sliding| *is_sliding)
        || pattern[first_sliding..]
            .iter()
            .any(|is_sliding| !*is_sliding)
    {
        return Err(NormalConfigSynthesisError::new(
            "Native Qwen3 config can represent only a contiguous sliding-attention suffix",
        ));
    }
    Ok((true, first_sliding))
}

fn gpt_oss_layer_types(
    metadata: &MetadataView<'_>,
    layer_count: usize,
) -> SynthesisResult<Vec<&'static str>> {
    let pattern = metadata.sliding_pattern(layer_count)?.unwrap_or_else(|| {
        (0..layer_count)
            .map(|layer| layer % GPT_OSS_SWA_PERIOD == 0)
            .collect()
    });
    Ok(pattern
        .into_iter()
        .map(|is_sliding| {
            if is_sliding {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect())
}

fn dense_layer_indices(metadata: &MetadataView<'_>, layer_count: usize) -> Vec<usize> {
    (0..layer_count)
        .filter(|layer| {
            metadata.layer_has_tensor(*layer, ".ffn_gate.")
                && !metadata.layer_has_tensor(*layer, ".ffn_gate_inp.")
        })
        .collect()
}

fn granite_layer_types(
    metadata: &MetadataView<'_>,
    kv_heads: &[usize],
    layer_count: usize,
) -> Vec<&'static str> {
    (0..layer_count)
        .map(|layer| {
            if kv_heads[layer] == 0 || metadata.layer_has_tensor(layer, ".ssm_") {
                "mamba"
            } else {
                "attention"
            }
        })
        .collect()
}

fn lfm_layer_types(
    metadata: &MetadataView<'_>,
    kv_heads: &[usize],
    layer_count: usize,
) -> Vec<&'static str> {
    (0..layer_count)
        .map(|layer| {
            if kv_heads[layer] == 0 || metadata.layer_has_tensor(layer, ".shortconv.") {
                "conv"
            } else {
                "full_attention"
            }
        })
        .collect()
}

fn uniform_nonzero_values(
    metadata: &MetadataView<'_>,
    suffix: &str,
    values: &[usize],
) -> SynthesisResult<usize> {
    let nonzero = values
        .iter()
        .copied()
        .filter(|value| *value != 0)
        .collect::<Vec<_>>();
    if nonzero.is_empty() {
        return Err(NormalConfigSynthesisError::new(format!(
            "GGUF metadata `{}` has no attention layer with non-zero KV heads",
            metadata.key(suffix)
        )));
    }
    uniform_value(metadata, suffix, &nonzero)
}

fn ratio_f64(
    metadata: &MetadataView<'_>,
    field: &str,
    numerator: usize,
    denominator: usize,
) -> SynthesisResult<f64> {
    if denominator == 0 || numerator > denominator {
        return Err(NormalConfigSynthesisError::new(format!(
            "Cannot derive `{field}` for native `{}` config from {numerator}/{denominator}",
            metadata.loader
        )));
    }
    let numerator = usize_to_u32(numerator).map_err(|_| {
        NormalConfigSynthesisError::new(format!("`{field}` numerator {numerator} is too large"))
    })?;
    let denominator = usize_to_u32(denominator).map_err(|_| {
        NormalConfigSynthesisError::new(format!("`{field}` denominator {denominator} is too large"))
    })?;
    Ok(f64::from(numerator) / f64::from(denominator))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;
    use serde::de::DeserializeOwned;
    use std::collections::HashSet;
    use strum::IntoEnumIterator;

    const LAYER_COUNT: usize = 4;

    fn insert_u32(
        metadata: &mut HashMap<String, GgufValue>,
        architecture: CanonicalGgufArchitecture,
        suffix: &str,
        value: u32,
    ) {
        metadata.insert(
            format!("{}.{suffix}", architecture.as_str()),
            GgufValue::U32(value),
        );
    }

    fn insert_f32(
        metadata: &mut HashMap<String, GgufValue>,
        architecture: CanonicalGgufArchitecture,
        suffix: &str,
        value: f32,
    ) {
        metadata.insert(
            format!("{}.{suffix}", architecture.as_str()),
            GgufValue::F32(value),
        );
    }

    fn insert_bool(
        metadata: &mut HashMap<String, GgufValue>,
        architecture: CanonicalGgufArchitecture,
        suffix: &str,
        value: bool,
    ) {
        metadata.insert(
            format!("{}.{suffix}", architecture.as_str()),
            GgufValue::Bool(value),
        );
    }

    fn insert_string(
        metadata: &mut HashMap<String, GgufValue>,
        architecture: CanonicalGgufArchitecture,
        suffix: &str,
        value: &str,
    ) {
        metadata.insert(
            format!("{}.{suffix}", architecture.as_str()),
            GgufValue::String(value.to_string()),
        );
    }

    fn insert_u32_array(
        metadata: &mut HashMap<String, GgufValue>,
        architecture: CanonicalGgufArchitecture,
        suffix: &str,
        values: &[u32],
    ) {
        metadata.insert(
            format!("{}.{suffix}", architecture.as_str()),
            GgufValue::Array(values.iter().copied().map(GgufValue::U32).collect()),
        );
    }

    fn is_moe(loader: &NormalLoaderType) -> bool {
        matches!(
            loader,
            NormalLoaderType::Mixtral
                | NormalLoaderType::Phi3_5MoE
                | NormalLoaderType::DeepSeekV2
                | NormalLoaderType::DeepSeekV3
                | NormalLoaderType::GLM4MoeLite
                | NormalLoaderType::GLM4Moe
                | NormalLoaderType::Qwen3Moe
                | NormalLoaderType::GptOss
                | NormalLoaderType::HunYuanMoEV1
                | NormalLoaderType::Qwen3Next
                | NormalLoaderType::Lfm2Moe
        )
    }

    fn fixture(
        loader: &NormalLoaderType,
        architecture: CanonicalGgufArchitecture,
    ) -> (HashMap<String, GgufValue>, Vec<String>) {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String(architecture.as_str().to_string()),
        );
        insert_u32(&mut metadata, architecture, "vocab_size", 32_000);
        insert_u32(&mut metadata, architecture, "context_length", 4_096);
        insert_u32(&mut metadata, architecture, "embedding_length", 512);
        insert_u32(
            &mut metadata,
            architecture,
            "block_count",
            u32::try_from(LAYER_COUNT).expect("test layer count fits u32"),
        );
        insert_u32(&mut metadata, architecture, "feed_forward_length", 1_024);
        insert_u32(&mut metadata, architecture, "attention.head_count", 8);
        insert_u32(&mut metadata, architecture, "attention.head_count_kv", 4);
        insert_u32(&mut metadata, architecture, "attention.key_length", 64);
        insert_f32(
            &mut metadata,
            architecture,
            "attention.layer_norm_rms_epsilon",
            1e-5,
        );
        insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
        insert_f32(&mut metadata, architecture, "rope.freq_base", 10_000.0);

        let mut tensors = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
        ];

        if is_moe(loader) {
            insert_u32(&mut metadata, architecture, "expert_count", 8);
            insert_u32(&mut metadata, architecture, "expert_used_count", 2);
            insert_u32(
                &mut metadata,
                architecture,
                "expert_feed_forward_length",
                1_024,
            );
            insert_u32(&mut metadata, architecture, "expert_shared_count", 1);
            insert_u32(
                &mut metadata,
                architecture,
                "expert_shared_feed_forward_length",
                512,
            );
            insert_u32(&mut metadata, architecture, "expert_group_count", 1);
            insert_u32(&mut metadata, architecture, "expert_group_used_count", 1);
            insert_u32(&mut metadata, architecture, "leading_dense_block_count", 0);
            insert_f32(&mut metadata, architecture, "expert_weights_scale", 1.0);
            insert_bool(&mut metadata, architecture, "expert_weights_norm", true);
            insert_u32(
                &mut metadata,
                architecture,
                "expert_gating_func",
                if matches!(loader, NormalLoaderType::Lfm2Moe) {
                    2
                } else {
                    1
                },
            );
            tensors.extend(
                [
                    "blk.1.ffn_gate_inp.weight",
                    "blk.1.ffn_gate_exps.weight",
                    "blk.1.ffn_up_exps.weight",
                    "blk.1.ffn_down_exps.weight",
                ]
                .into_iter()
                .map(str::to_string),
            );
        }

        if matches!(
            loader,
            NormalLoaderType::DeepSeekV2
                | NormalLoaderType::DeepSeekV3
                | NormalLoaderType::GLM4MoeLite
        ) {
            insert_u32(&mut metadata, architecture, "rope.dimension_count", 32);
            insert_u32(&mut metadata, architecture, "attention.q_lora_rank", 128);
            insert_u32(&mut metadata, architecture, "attention.kv_lora_rank", 64);
            insert_u32(&mut metadata, architecture, "attention.key_length_mla", 64);
            insert_u32(
                &mut metadata,
                architecture,
                "attention.value_length_mla",
                32,
            );
            tensors.extend(
                ["blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"]
                    .into_iter()
                    .map(str::to_string),
            );
        }

        if matches!(
            loader,
            NormalLoaderType::Qwen3Next
                | NormalLoaderType::Qwen3_5
                | NormalLoaderType::GraniteMoeHybrid
        ) {
            insert_u32(&mut metadata, architecture, "ssm.conv_kernel", 4);
            insert_u32(&mut metadata, architecture, "ssm.inner_size", 1_024);
            insert_u32(&mut metadata, architecture, "ssm.state_size", 64);
            insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 8);
            insert_u32(&mut metadata, architecture, "ssm.group_count", 4);
        }

        if matches!(
            loader,
            NormalLoaderType::Qwen3Next | NormalLoaderType::Qwen3_5
        ) {
            insert_u32(&mut metadata, architecture, "full_attention_interval", 4);
            tensors.extend(
                ["blk.1.ssm_a.weight", "blk.1.ssm_conv1d.weight"]
                    .into_iter()
                    .map(str::to_string),
            );
            if architecture == CanonicalGgufArchitecture::Qwen35Moe {
                tensors.extend(
                    ["blk.1.ssm_alpha.weight", "blk.1.ssm_beta.weight"]
                        .into_iter()
                        .map(str::to_string),
                );
            }
            if architecture == CanonicalGgufArchitecture::Qwen35 {
                insert_u32_array(
                    &mut metadata,
                    architecture,
                    "rope.dimension_sections",
                    &[11, 11, 10, 0],
                );
                tensors.extend(
                    ["blk.1.ssm_alpha.weight", "blk.1.ssm_beta.weight"]
                        .into_iter()
                        .map(str::to_string),
                );
            }
        }

        if matches!(loader, NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe) {
            tensors.retain(|tensor| tensor != "output_norm.weight");
            insert_u32_array(
                &mut metadata,
                architecture,
                "attention.head_count_kv",
                &[4, 0, 4, 0],
            );
            insert_u32(&mut metadata, architecture, "shortconv.l_cache", 3);
            tensors.extend(
                [
                    "blk.1.shortconv.conv.weight",
                    "blk.1.shortconv.in_proj.weight",
                ]
                .into_iter()
                .map(str::to_string),
            );
        }

        if matches!(loader, NormalLoaderType::Gemma2 | NormalLoaderType::GptOss) {
            insert_u32(
                &mut metadata,
                architecture,
                "attention.sliding_window",
                1_024,
            );
        }

        if matches!(loader, NormalLoaderType::HunYuanMoEV1) {
            tensors.push("blk.1.ffn_gate_shexp.weight".to_string());
        }

        (metadata, tensors)
    }

    fn default_fixture(
        loader: &NormalLoaderType,
    ) -> (
        CanonicalGgufArchitecture,
        HashMap<String, GgufValue>,
        Vec<String>,
    ) {
        let architecture = registry_adapter(loader).unwrap().architectures[0];
        let (metadata, tensors) = fixture(loader, architecture);
        (architecture, metadata, tensors)
    }

    fn assert_deserializes<T: DeserializeOwned>(loader: &NormalLoaderType, config: JsonValue) {
        if let Err(error) = serde_json::from_value::<T>(config.clone()) {
            panic!("synthesized `{loader}` config did not deserialize: {error}\n{config:#}");
        }
    }

    fn assert_native_config_deserializes(loader: &NormalLoaderType, config: JsonValue) {
        match loader {
            NormalLoaderType::Mistral => {
                assert_deserializes::<models::mistral::Config>(loader, config)
            }
            NormalLoaderType::Gemma => assert_deserializes::<models::gemma::Config>(loader, config),
            NormalLoaderType::Mixtral => {
                assert_deserializes::<models::mixtral::Config>(loader, config)
            }
            NormalLoaderType::Llama => assert_deserializes::<models::llama::Config>(loader, config),
            NormalLoaderType::Phi2 => assert_deserializes::<models::phi2::Config>(loader, config),
            NormalLoaderType::Phi3 => assert_deserializes::<models::phi3::Config>(loader, config),
            NormalLoaderType::Qwen2 => assert_deserializes::<models::qwen2::Config>(loader, config),
            NormalLoaderType::Gemma2 => {
                assert_deserializes::<models::gemma2::Config>(loader, config)
            }
            NormalLoaderType::Starcoder2 => {
                assert_deserializes::<models::starcoder2::Config>(loader, config)
            }
            NormalLoaderType::Phi3_5MoE => {
                assert_deserializes::<models::phi3_5_moe::Config>(loader, config)
            }
            NormalLoaderType::DeepSeekV2 => {
                assert_deserializes::<models::deepseek2::DeepSeekV2Config>(loader, config)
            }
            NormalLoaderType::DeepSeekV3 => {
                assert_deserializes::<models::deepseek3::DeepSeekV3Config>(loader, config)
            }
            NormalLoaderType::Qwen3 => assert_deserializes::<models::qwen3::Config>(loader, config),
            NormalLoaderType::GLM4 => assert_deserializes::<models::glm4::Config>(loader, config),
            NormalLoaderType::GLM4MoeLite => {
                assert_deserializes::<models::glm4_moe_lite::Glm4MoeLiteConfig>(loader, config)
            }
            NormalLoaderType::GLM4Moe => {
                assert_deserializes::<models::glm4_moe::Glm4MoeConfig>(loader, config)
            }
            NormalLoaderType::Qwen3Moe => {
                assert_deserializes::<models::qwen3_moe::Config>(loader, config)
            }
            NormalLoaderType::SmolLm3 => {
                assert_deserializes::<models::smollm3::Config>(loader, config)
            }
            NormalLoaderType::GraniteMoeHybrid => {
                assert_deserializes::<models::granite::Config>(loader, config)
            }
            NormalLoaderType::GptOss => {
                assert_deserializes::<models::gpt_oss::Config>(loader, config)
            }
            NormalLoaderType::HunYuanDenseV1 => {
                assert_deserializes::<models::hunyuan_v1_dense::Config>(loader, config)
            }
            NormalLoaderType::HunYuanMoEV1 => {
                assert_deserializes::<models::hunyuan_v1_moe::Config>(loader, config)
            }
            NormalLoaderType::Qwen3Next => {
                assert_deserializes::<models::qwen3_next::Config>(loader, config)
            }
            NormalLoaderType::Qwen3_5 => {
                assert_deserializes::<crate::vision_models::qwen3_5::TextConfig>(loader, config)
            }
            NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
                assert_deserializes::<models::lfm2::Config>(loader, config)
            }
        }
    }

    #[test]
    fn config_builder_registry_is_exhaustive() {
        assert_eq!(
            NormalLoaderType::iter().count(),
            NORMAL_CONFIG_BUILDERS.len()
        );
        assert_eq!(NORMAL_MODEL_ADAPTERS.len(), NORMAL_CONFIG_BUILDERS.len());

        let mut builders = HashSet::new();
        for builder in NORMAL_CONFIG_BUILDERS {
            assert!(builders.insert(builder.loader.to_string()));
            assert!(registry_adapter(&builder.loader).is_some());
        }
        for loader in NormalLoaderType::iter() {
            assert!(builder_for(&loader).is_some(), "{loader}");
            assert!(builders.contains(&loader.to_string()), "{loader}");
        }
    }

    #[test]
    fn every_loader_synthesizes_a_native_config() {
        for loader in NormalLoaderType::iter() {
            let (_, metadata, tensors) = default_fixture(&loader);
            let config = synthesize_normal_config_value(&loader, &metadata, &tensors)
                .unwrap_or_else(|error| panic!("failed to synthesize `{loader}` fixture: {error}"));
            assert_eq!(
                config["architectures"],
                json!([loader.causal_lm_name()]),
                "synthesized `{loader}` config is not auto-loadable"
            );
            assert_eq!(config["model_type"], loader.model_type_name());
            assert_eq!(
                normal_loader_hint_from_external_config(&config.to_string()).unwrap(),
                Some(loader.clone())
            );
            assert_native_config_deserializes(&loader, config);
        }
    }

    #[test]
    fn qwen35moe_standalone_metadata_synthesizes_tiled_native_config() {
        let loader = NormalLoaderType::Qwen3Next;
        let architecture = CanonicalGgufArchitecture::Qwen35Moe;
        let (mut metadata, tensors) = fixture(&loader, architecture);
        metadata.remove(&format!("{}.feed_forward_length", architecture.as_str()));
        insert_u32(&mut metadata, architecture, "attention.key_length", 256);
        insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
        insert_u32(&mut metadata, architecture, "ssm.state_size", 128);
        insert_u32(&mut metadata, architecture, "ssm.group_count", 16);
        insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 32);
        insert_u32(&mut metadata, architecture, "ssm.inner_size", 4_096);

        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(config["intermediate_size"], 512);
        assert_eq!(config["head_dim"], 256);
        assert_eq!(config["partial_rotary_factor"], 0.25);
        assert_eq!(config[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
        assert_eq!(config["architectures"][0], "Qwen3NextForCausalLM");
        let native: models::qwen3_next::Config = serde_json::from_value(config.clone()).unwrap();
        assert_eq!(
            crate::gdn::GdnConfig::v_head_layout(&native),
            crate::gdn::GdnVHeadLayout::Tiled
        );
        assert_native_config_deserializes(&loader, config);
    }

    #[test]
    fn qwen35_standalone_metadata_synthesizes_dense_text_config() {
        let loader = NormalLoaderType::Qwen3_5;
        let architecture = CanonicalGgufArchitecture::Qwen35;
        let (mut metadata, tensors) = fixture(&loader, architecture);
        insert_u32(&mut metadata, architecture, "vocab_size", 248_320);
        insert_u32(&mut metadata, architecture, "context_length", 262_144);
        insert_u32(&mut metadata, architecture, "embedding_length", 2_560);
        insert_u32(&mut metadata, architecture, "feed_forward_length", 9_216);
        insert_u32(&mut metadata, architecture, "block_count", 32);
        insert_u32(&mut metadata, architecture, "attention.head_count", 16);
        insert_u32(&mut metadata, architecture, "attention.head_count_kv", 4);
        insert_u32(&mut metadata, architecture, "attention.key_length", 256);
        insert_u32(&mut metadata, architecture, "attention.value_length", 256);
        insert_f32(
            &mut metadata,
            architecture,
            "attention.layer_norm_rms_epsilon",
            1e-6,
        );
        insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
        insert_f32(&mut metadata, architecture, "rope.freq_base", 10_000_000.0);
        insert_u32(&mut metadata, architecture, "ssm.state_size", 128);
        insert_u32(&mut metadata, architecture, "ssm.group_count", 16);
        insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 32);
        insert_u32(&mut metadata, architecture, "ssm.inner_size", 4_096);

        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(config["vocab_size"], 248_320);
        assert_eq!(config["hidden_size"], 2_560);
        assert_eq!(config["intermediate_size"], 9_216);
        assert_eq!(config["num_hidden_layers"], 32);
        assert_eq!(config["head_dim"], 256);
        assert_eq!(config["rope_parameters"]["partial_rotary_factor"], 0.25);
        assert_eq!(
            config["rope_parameters"]["mrope_section"],
            json!([11, 11, 10])
        );
        assert_eq!(config[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
        let native: crate::vision_models::qwen3_5::TextConfig =
            serde_json::from_value(config.clone()).unwrap();
        assert_eq!(
            crate::gdn::GdnConfig::v_head_layout(&native),
            crate::gdn::GdnVHeadLayout::Tiled
        );
        assert_native_config_deserializes(&loader, config);
    }

    #[test]
    fn qwen35_rejects_invalid_hybrid_and_mrope_metadata() {
        let loader = NormalLoaderType::Qwen3_5;
        let architecture = CanonicalGgufArchitecture::Qwen35;
        let (mut metadata, tensors) = fixture(&loader, architecture);
        insert_u32(&mut metadata, architecture, "full_attention_interval", 0);
        let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("full attention interval"));

        insert_u32(&mut metadata, architecture, "full_attention_interval", 4);
        insert_u32_array(
            &mut metadata,
            architecture,
            "rope.dimension_sections",
            &[11, 0, 21, 0],
        );
        let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("three non-zero MRoPE sections"));

        insert_u32_array(
            &mut metadata,
            architecture,
            "rope.dimension_sections",
            &[11, 11, 10, 0],
        );
        insert_u32(&mut metadata, architecture, "ssm.group_count", 3);
        insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 8);
        let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("incompatible GDN head counts"));
    }

    #[test]
    fn standalone_moe_normalization_defaults_match_native_models() {
        for (loader, architecture) in [
            (
                NormalLoaderType::Qwen3Moe,
                CanonicalGgufArchitecture::Qwen3Moe,
            ),
            (
                NormalLoaderType::Qwen3Next,
                CanonicalGgufArchitecture::Qwen3Next,
            ),
            (
                NormalLoaderType::Qwen3Next,
                CanonicalGgufArchitecture::Qwen35Moe,
            ),
            (
                NormalLoaderType::GLM4Moe,
                CanonicalGgufArchitecture::Glm4Moe,
            ),
        ] {
            let (mut metadata, tensors) = fixture(&loader, architecture);
            metadata.remove(&format!("{}.expert_weights_norm", architecture.as_str()));

            let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
            assert_eq!(
                config["norm_topk_prob"], true,
                "{loader} with {architecture}"
            );
            assert_native_config_deserializes(&loader, config);
        }
    }

    #[test]
    fn llama_linear_rope_and_baked_factor_gate_are_explicit() {
        let loader = NormalLoaderType::Llama;
        let (architecture, mut metadata, mut tensors) = default_fixture(&loader);
        insert_string(&mut metadata, architecture, "rope.scaling.type", "linear");
        insert_f32(&mut metadata, architecture, "rope.scaling.factor", 2.0);
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(config["rope_scaling"]["rope_type"], "linear");
        assert_eq!(config["rope_scaling"]["factor"], 2.0);
        assert_native_config_deserializes(&loader, config);

        tensors.push("rope_freqs.weight".to_string());
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert!(config["rope_scaling"].is_null());
        assert_native_config_deserializes(&loader, config);
    }

    #[test]
    fn qwen_sliding_policies_preserve_layer_semantics() {
        let loader = NormalLoaderType::Qwen2;
        let (architecture, mut metadata, tensors) = default_fixture(&loader);
        insert_u32(
            &mut metadata,
            architecture,
            "attention.sliding_window",
            1_024,
        );
        metadata.insert(
            format!("{}.attention.sliding_window_pattern", architecture.as_str()),
            GgufValue::Array(
                [true, false, true, false]
                    .into_iter()
                    .map(GgufValue::Bool)
                    .collect(),
            ),
        );
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(
            config["layer_types"],
            json!([
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ])
        );
        assert_native_config_deserializes(&loader, config);

        let qwen3 = NormalLoaderType::Qwen3;
        let (architecture, mut metadata, tensors) = default_fixture(&qwen3);
        insert_u32(
            &mut metadata,
            architecture,
            "attention.sliding_window",
            1_024,
        );
        let error = synthesize_normal_config_value(&qwen3, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("layer policy"));
    }

    #[test]
    fn qwen_moe_inventory_marks_dense_layers() {
        let loader = NormalLoaderType::Qwen3Moe;
        let (_, metadata, tensors) = default_fixture(&loader);
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(config["mlp_only_layers"], json!([0]));
        assert_eq!(config["num_experts"], 8);
        assert_eq!(config["num_experts_per_tok"], 2);
        assert_native_config_deserializes(&loader, config);
    }

    #[test]
    fn hybrid_ssm_and_shortconv_configs_preserve_layer_inventory() {
        let loader = NormalLoaderType::GraniteMoeHybrid;
        let architecture = CanonicalGgufArchitecture::GraniteHybrid;
        let (mut metadata, mut tensors) = fixture(&loader, architecture);
        insert_u32_array(
            &mut metadata,
            architecture,
            "attention.head_count_kv",
            &[4, 0, 4, 0],
        );
        insert_u32(&mut metadata, architecture, "expert_count", 8);
        insert_u32(&mut metadata, architecture, "expert_used_count", 2);
        tensors.extend(
            ["blk.1.ssm_a.weight", "blk.1.ssm_conv1d.weight"]
                .into_iter()
                .map(str::to_string),
        );
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(
            config["layer_types"],
            json!(["attention", "mamba", "attention", "mamba"])
        );
        assert_eq!(config["mamba_d_head"], 128);
        assert_eq!(config["mamba_expand"], 2);
        assert_native_config_deserializes(&loader, config);

        let loader = NormalLoaderType::Lfm2;
        let (_, metadata, tensors) = default_fixture(&loader);
        assert!(!tensors.iter().any(|tensor| tensor == "output_norm.weight"));
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(
            config["layer_types"],
            json!(["full_attention", "conv", "full_attention", "conv"])
        );
        assert_eq!(config["conv_L_cache"], 3);
        assert_native_config_deserializes(&loader, config);
    }

    #[test]
    fn missing_ssm_and_shortconv_semantics_are_actionable() {
        let loader = NormalLoaderType::Qwen3Next;
        let (architecture, mut metadata, tensors) = default_fixture(&loader);
        metadata.remove(&format!("{}.ssm.time_step_rank", architecture.as_str()));
        let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("ssm.time_step_rank"));
        assert!(error.to_string().contains("original Hugging Face config"));

        let loader = NormalLoaderType::Lfm2;
        let (architecture, mut metadata, tensors) = default_fixture(&loader);
        metadata.remove(&format!("{}.shortconv.l_cache", architecture.as_str()));
        let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
        assert!(error.to_string().contains("shortconv.l_cache"));
    }

    #[test]
    fn external_flat_configs_are_preserved_for_every_loader() {
        let mut architectures = HashSet::new();
        for (index, loader) in NormalLoaderType::iter().enumerate() {
            let model_architecture = loader.causal_lm_name();
            assert!(architectures.insert(model_architecture));
            let raw_value = json!({
                "architectures": [model_architecture],
                "sentinel": index,
            });
            let raw = raw_value.to_string();
            assert_eq!(
                normal_loader_hint_from_external_config(&raw).unwrap(),
                Some(loader.clone())
            );
            let gguf_architecture = registry_adapter(&loader).unwrap().architectures[0];
            let normalized: JsonValue = serde_json::from_str(
                &normalize_external_normal_config(&loader, gguf_architecture, &raw).unwrap(),
            )
            .unwrap();
            let mut expected = raw_value;
            expected["quantization_config"] = JsonValue::Null;
            expected["model_type"] = json!(loader.model_type_name());
            if matches!(
                gguf_architecture,
                CanonicalGgufArchitecture::Qwen35 | CanonicalGgufArchitecture::Qwen35Moe
            ) {
                expected[GDN_V_HEAD_LAYOUT_CONFIG_KEY] = json!("tiled");
            }
            assert_eq!(
                normalized, expected,
                "external config fields changed for native `{loader}`"
            );
        }
    }

    #[test]
    fn mistral3_external_config_unwraps_native_text_config() {
        let raw = json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000.0,
                "sliding_window": null,
                "head_dim": 64,
                "quantization_config": null
            },
            "vision_config": {
                "sentinel": true
            }
        })
        .to_string();
        assert_eq!(
            normal_loader_hint_from_external_config(&raw).unwrap(),
            Some(NormalLoaderType::Mistral)
        );
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Mistral,
            CanonicalGgufArchitecture::Mistral3,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert!(value.get("vision_config").is_none());
        assert_eq!(value["hidden_size"], 512);
        assert_deserializes::<models::mistral::Config>(&NormalLoaderType::Mistral, value);
    }

    #[test]
    fn mistral3_external_config_uses_gguf_quantization() {
        let raw = json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 512,
                "quantization_config": null
            },
            "quantization_config": {
                "quant_method": "sentinel"
            }
        })
        .to_string();
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Mistral,
            CanonicalGgufArchitecture::Mistral3,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert!(value["quantization_config"].is_null());
    }

    #[test]
    fn ordinary_external_config_uses_gguf_quantization() {
        let raw = json!({
            "architectures": ["Qwen3ForCausalLM"],
            "quantization_config": {
                "quant_method": "awq"
            }
        })
        .to_string();
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Qwen3,
            CanonicalGgufArchitecture::Qwen3,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert!(value["quantization_config"].is_null());
    }

    #[test]
    fn lfm2_vl_external_config_unwraps_native_text_config() {
        let raw = json!({
            "architectures": ["Lfm2VlForConditionalGeneration"],
            "model_type": "lfm2_vl",
            "text_config": {
                "architectures": ["Lfm2ForCausalLM"],
                "model_type": "lfm2",
                "vocab_size": 32000,
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "max_position_embeddings": 4096,
                "norm_eps": 0.00001,
                "rope_parameters": {
                    "rope_theta": 1000000.0,
                    "rope_type": "default"
                },
                "conv_L_cache": 3,
                "layer_types": ["conv", "full_attention", "conv", "full_attention"],
                "quantization_config": null
            },
            "vision_config": {
                "sentinel": true
            }
        })
        .to_string();
        assert_eq!(
            normal_loader_hint_from_external_config(&raw).unwrap(),
            Some(NormalLoaderType::Lfm2)
        );
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Lfm2,
            CanonicalGgufArchitecture::Lfm2,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert!(value.get("vision_config").is_none());
        assert_native_config_deserializes(&NormalLoaderType::Lfm2, value);
    }

    #[test]
    fn qwen35_moe_external_config_maps_to_qwen3_next() {
        let raw = json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 32000,
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.00001,
                "head_dim": 64,
                "rope_parameters": {
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25
                },
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 64,
                "linear_value_head_dim": 64,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 8,
                "moe_intermediate_size": 256,
                "shared_expert_intermediate_size": 512,
                "num_experts_per_tok": 2,
                "num_experts": 8,
                "full_attention_interval": 4,
                "quantization_config": null
            },
            "vision_config": {
                "sentinel": true
            }
        })
        .to_string();
        assert_eq!(
            normal_loader_hint_from_external_config(&raw).unwrap(),
            Some(NormalLoaderType::Qwen3Next)
        );
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Qwen3Next,
            CanonicalGgufArchitecture::Qwen35Moe,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert_eq!(value["rope_theta"], 10000000.0);
        assert_eq!(value["partial_rotary_factor"], 0.25);
        assert_eq!(value["intermediate_size"], 512);
        assert_eq!(value[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
        assert_eq!(value["architectures"][0], "Qwen3NextForCausalLM");
        assert_native_config_deserializes(&NormalLoaderType::Qwen3Next, value);
    }

    #[test]
    fn qwen35_external_config_unwraps_dense_text_config() {
        let raw = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 32000,
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.00001,
                "head_dim": 64,
                "rope_parameters": {
                    "rope_theta": 10000000.0,
                    "mrope_section": [3, 3, 2],
                    "partial_rotary_factor": 0.25
                },
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 64,
                "linear_value_head_dim": 64,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 8,
                "full_attention_interval": 4,
                "tie_word_embeddings": true
            },
            "vision_config": {
                "sentinel": true
            }
        })
        .to_string();
        assert_eq!(
            normal_loader_hint_from_external_config(&raw).unwrap(),
            Some(NormalLoaderType::Qwen3_5)
        );
        let normalized = normalize_external_normal_config(
            &NormalLoaderType::Qwen3_5,
            CanonicalGgufArchitecture::Qwen35,
            &raw,
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&normalized).unwrap();
        assert!(value.get("vision_config").is_none());
        assert_eq!(value["tie_word_embeddings"], false);
        assert_eq!(value[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
        assert_native_config_deserializes(&NormalLoaderType::Qwen3_5, value);
    }

    #[test]
    fn external_config_tying_matches_the_gguf_output_tensor() {
        let tied = r#"{"tie_word_embeddings":true}"#;
        let untied = r#"{"tie_word_embeddings":false}"#;
        let no_output = vec!["token_embd.weight".to_string()];
        let with_output = vec!["token_embd.weight".to_string(), "output.weight".to_string()];

        validate_normal_config_tensor_inventory(tied, &no_output).unwrap();
        validate_normal_config_tensor_inventory(untied, &with_output).unwrap();
        assert!(validate_normal_config_tensor_inventory(tied, &with_output)
            .unwrap_err()
            .to_string()
            .contains("tie_word_embeddings"));
        assert!(validate_normal_config_tensor_inventory(untied, &no_output)
            .unwrap_err()
            .to_string()
            .contains("tie_word_embeddings"));
        validate_normal_config_tensor_inventory("{}", &with_output).unwrap();
    }

    #[test]
    fn external_config_architecture_errors_are_actionable() {
        let mismatched = r#"{"architectures":["Qwen3ForCausalLM"]}"#;
        let error = normalize_external_normal_config(
            &NormalLoaderType::Llama,
            CanonicalGgufArchitecture::Llama,
            mismatched,
        )
        .unwrap_err();
        assert!(error.to_string().contains("qwen3"));
        assert!(error.to_string().contains("llama"));

        let unknown = r#"{"architectures":["FutureForCausalLM"]}"#;
        let error = normal_loader_hint_from_external_config(unknown).unwrap_err();
        assert!(error.to_string().contains("FutureForCausalLM"));
        assert!(error.to_string().contains("supported native text model"));

        let missing_text = r#"{
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3"
        }"#;
        let error = normalize_external_normal_config(
            &NormalLoaderType::Mistral,
            CanonicalGgufArchitecture::Mistral3,
            missing_text,
        )
        .unwrap_err();
        assert!(error.to_string().contains("text_config"));
    }
}
