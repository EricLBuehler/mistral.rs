mod dynamic;
mod static_lora;

use std::collections::HashSet;

use candle_core::Result;
pub(crate) use dynamic::maybe_wrap_dynamic_lora_with_key;
pub use dynamic::{
    load_dynamic_lora_weights, maybe_wrap_dynamic_lora, plan_dynamic_lora_weights,
    with_lora_execution, DynamicLoraLoadPlan, LoraAdapterWeights, LoraExecution, LoraLayerRegistry,
    LoraLinearSpec, LoraRuntimeId, LoraSiteHandle, LoraSiteKey, LoraSiteSlice, LoraSlotId,
    LoraWeights,
};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
pub use static_lora::linear_no_bias_static_lora;

pub const MULTI_LORA_DELIMITER: &str = ";";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StaticLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LoraTargetModules {
    Pattern(String),
    Modules(HashSet<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoraConfig {
    #[serde(default)]
    pub peft_type: Option<String>,
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
    #[serde(rename = "r")]
    pub rank: usize,
    #[serde(rename = "lora_alpha")]
    pub alpha: f64,
    #[serde(default)]
    pub target_modules: Option<LoraTargetModules>,
    #[serde(default)]
    pub exclude_modules: Option<LoraTargetModules>,
    #[serde(default)]
    pub rank_pattern: IndexMap<String, usize>,
    #[serde(default)]
    pub alpha_pattern: IndexMap<String, f64>,
    #[serde(default)]
    pub use_rslora: bool,
    #[serde(default)]
    pub use_dora: bool,
    #[serde(default)]
    pub lora_bias: bool,
    #[serde(default = "default_lora_bias_mode")]
    pub bias: String,
    #[serde(default)]
    pub modules_to_save: Option<Vec<String>>,
    #[serde(default = "default_init_lora_weights")]
    pub init_lora_weights: serde_json::Value,
    #[serde(default)]
    pub alora_invocation_tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub use_qalora: bool,
    #[serde(default = "default_qalora_group_size")]
    pub qalora_group_size: usize,
    #[serde(default)]
    pub layer_replication: Option<Vec<[usize; 2]>>,
    #[serde(default)]
    pub target_parameters: Option<Vec<String>>,
    #[serde(default)]
    pub use_bdlora: Option<serde_json::Value>,
    #[serde(default)]
    pub arrow_config: Option<serde_json::Value>,
    #[serde(default)]
    pub monteclora_config: Option<serde_json::Value>,
    #[serde(default)]
    pub ensure_weight_tying: bool,
    #[serde(default)]
    pub trainable_token_indices: Option<serde_json::Value>,
    #[serde(default)]
    pub megatron_config: Option<serde_json::Value>,
}

fn default_lora_bias_mode() -> String {
    "none".to_string()
}

fn default_init_lora_weights() -> serde_json::Value {
    serde_json::Value::Bool(true)
}

fn default_qalora_group_size() -> usize {
    16
}

fn target_regex(pattern: &str) -> std::result::Result<Regex, regex::Error> {
    Regex::new(&format!(r"\A(?:{pattern})\z"))
}

fn pattern_key_regex(pattern: &str) -> std::result::Result<Regex, regex::Error> {
    Regex::new(&format!(r"\A(?:.*\.)?(?:{pattern})\z"))
}

fn validate_pattern_keys<T>(name: &str, values: &IndexMap<String, T>) -> Result<()> {
    for key in values.keys() {
        pattern_key_regex(key).map_err(|error| {
            candle_core::Error::msg(format!("invalid LoRA {name} regex `{key}`: {error}"))
        })?;
    }
    Ok(())
}

impl LoraTargetModules {
    fn matches(&self, path: &str) -> bool {
        match self {
            Self::Pattern(pattern) => target_regex(pattern).is_ok_and(|regex| regex.is_match(path)),
            Self::Modules(modules) => modules.iter().any(|target| {
                path == target
                    || path
                        .strip_suffix(target)
                        .is_some_and(|prefix| prefix.ends_with('.'))
            }),
        }
    }

    fn validate(&self, name: &str, require_nonempty: bool) -> Result<()> {
        match self {
            Self::Pattern(pattern) if require_nonempty && pattern.is_empty() => {
                candle_core::bail!("LoRA {name} must not be empty");
            }
            Self::Pattern(pattern) => target_regex(pattern).map(|_| ()).map_err(|error| {
                candle_core::Error::msg(format!("invalid LoRA {name} regex `{pattern}`: {error}"))
            }),
            Self::Modules(modules) if require_nonempty && modules.is_empty() => {
                candle_core::bail!("LoRA {name} must not be empty");
            }
            Self::Modules(modules) if modules.contains("") => {
                candle_core::bail!("LoRA {name} must not contain an empty module");
            }
            Self::Modules(_) => Ok(()),
        }
    }
}

impl LoraConfig {
    fn pattern_value<T: Copy>(path: &str, values: &IndexMap<String, T>) -> Option<T> {
        values
            .iter()
            .find(|(key, _)| pattern_key_regex(key).is_ok_and(|regex| regex.is_match(path)))
            .map(|(_, value)| *value)
    }

    pub fn rank_for(&self, path: &str) -> usize {
        Self::pattern_value(path, &self.rank_pattern).unwrap_or(self.rank)
    }

    pub fn alpha_for(&self, path: &str) -> f64 {
        Self::pattern_value(path, &self.alpha_pattern).unwrap_or(self.alpha)
    }

    pub fn scale_for(&self, path: &str) -> Result<f64> {
        let rank = self.rank_for(path);
        if rank == 0 {
            candle_core::bail!("LoRA rank for `{path}` must be nonzero");
        }
        let alpha = self.alpha_for(path);
        if !alpha.is_finite() {
            candle_core::bail!("LoRA alpha for `{path}` must be finite");
        }
        Ok(if self.use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        })
    }

    pub fn targets_path(&self, path: &str) -> bool {
        self.target_modules
            .as_ref()
            .is_some_and(|targets| {
                matches!(targets, LoraTargetModules::Pattern(pattern) if pattern.eq_ignore_ascii_case("all-linear"))
                    || targets.matches(path)
            })
    }

    pub fn excludes_path(&self, path: &str) -> bool {
        self.exclude_modules
            .as_ref()
            .is_some_and(|excludes| excludes.matches(path))
    }

    pub fn validate_dynamic(&self) -> Result<()> {
        if self
            .peft_type
            .as_deref()
            .is_some_and(|peft_type| !peft_type.eq_ignore_ascii_case("LORA"))
        {
            candle_core::bail!("dynamic LoRA requires peft_type LORA");
        }
        if self.use_dora {
            candle_core::bail!("dynamic LoRA does not support DoRA adapters");
        }
        if self.lora_bias || self.bias != "none" {
            candle_core::bail!("dynamic LoRA does not support adapter-specific bias parameters");
        }
        if self
            .modules_to_save
            .as_ref()
            .is_some_and(|modules| !modules.is_empty())
        {
            candle_core::bail!("dynamic LoRA does not support modules_to_save");
        }
        if self.alora_invocation_tokens.is_some() {
            candle_core::bail!("dynamic LoRA does not support aLoRA alora_invocation_tokens");
        }
        if self.use_qalora {
            candle_core::bail!("dynamic LoRA does not support use_qalora");
        }
        if self
            .layer_replication
            .as_ref()
            .is_some_and(|layers| !layers.is_empty())
        {
            candle_core::bail!("dynamic LoRA does not support layer_replication");
        }
        if self
            .target_parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_empty())
        {
            candle_core::bail!("dynamic LoRA does not support target_parameters");
        }
        for (name, config) in [
            ("use_bdlora", &self.use_bdlora),
            ("arrow_config", &self.arrow_config),
            ("monteclora_config", &self.monteclora_config),
        ] {
            if config
                .as_ref()
                .is_some_and(|value| !matches!(value, serde_json::Value::Bool(false)))
            {
                candle_core::bail!("dynamic LoRA does not support {name}");
            }
        }
        if self.ensure_weight_tying {
            candle_core::bail!("dynamic LoRA does not support ensure_weight_tying");
        }
        for (name, config) in [
            ("trainable_token_indices", &self.trainable_token_indices),
            ("megatron_config", &self.megatron_config),
        ] {
            if config.as_ref().is_some_and(|value| match value {
                serde_json::Value::Null | serde_json::Value::Bool(false) => false,
                serde_json::Value::Array(values) => !values.is_empty(),
                serde_json::Value::Object(values) => !values.is_empty(),
                _ => true,
            }) {
                candle_core::bail!("dynamic LoRA does not support {name}");
            }
        }
        match &self.init_lora_weights {
            serde_json::Value::Null | serde_json::Value::Bool(_) => {}
            serde_json::Value::String(strategy) => {
                let strategy = strategy.to_ascii_lowercase();
                if strategy == "pissa"
                    || strategy.starts_with("pissa_niter_")
                    || matches!(strategy.as_str(), "olora" | "corda" | "loftq" | "lora_ga")
                {
                    candle_core::bail!(
                        "dynamic LoRA does not support init_lora_weights `{strategy}` because it requires transformed base weights"
                    );
                }
            }
            _ => candle_core::bail!("invalid LoRA init_lora_weights value"),
        }
        if self.rank == 0 {
            candle_core::bail!("LoRA rank must be nonzero");
        }
        self.target_modules
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("LoRA target_modules must not be empty"))?
            .validate("target_modules", true)?;
        if let Some(exclude_modules) = &self.exclude_modules {
            exclude_modules.validate("exclude_modules", false)?;
        }
        if let Some((path, _)) = self.rank_pattern.iter().find(|(_, rank)| **rank == 0) {
            candle_core::bail!("LoRA rank for `{path}` must be nonzero");
        }
        validate_pattern_keys("rank_pattern", &self.rank_pattern)?;
        if !self.alpha.is_finite() {
            candle_core::bail!("LoRA alpha must be finite");
        }
        if let Some((path, _)) = self
            .alpha_pattern
            .iter()
            .find(|(_, alpha)| !alpha.is_finite())
        {
            candle_core::bail!("LoRA alpha for `{path}` must be finite");
        }
        validate_pattern_keys("alpha_pattern", &self.alpha_pattern)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_accepts_single_target_and_rs_lora_patterns() -> Result<()> {
        let config: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "q_proj",
                "rank_pattern": {"layers.0.self_attn.q_proj": 4},
                "alpha_pattern": {"q_proj": 8},
                "use_rslora": true
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;

        let path = "model.layers.0.self_attn.q_proj";
        assert_eq!(
            config.target_modules,
            Some(LoraTargetModules::Pattern("q_proj".to_string()))
        );
        assert!(config.targets_path("q_proj"));
        assert!(!config.targets_path(path));
        assert_eq!(config.rank_for(path), 4);
        assert_eq!(config.alpha_for(path), 8.0);
        assert_eq!(config.scale_for(path)?, 4.0);
        Ok(())
    }

    #[test]
    fn target_module_forms_follow_peft_matching() -> Result<()> {
        let regex: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "model\\.layers\\.[0-9]+\\.(q|v)_proj"
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        regex.validate_dynamic()?;
        assert!(regex.targets_path("model.layers.12.q_proj"));
        assert!(regex.targets_path("model.layers.3.v_proj"));
        assert!(!regex.targets_path("base.model.layers.3.v_proj"));
        assert!(!regex.targets_path("model.layers.3.k_proj"));

        let all_linear: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "ALL-LINEAR"
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        all_linear.validate_dynamic()?;
        assert!(all_linear.targets_path("model.layers.3.self_attn.q_proj"));
        assert!(all_linear.targets_path("model.layers.3.mlp.down_proj"));

        let modules: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "layers.0.self_attn.v_proj"]
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        modules.validate_dynamic()?;
        assert!(modules.targets_path("model.layers.0.self_attn.q_proj"));
        assert!(modules.targets_path("model.layers.0.self_attn.v_proj"));
        assert!(!modules.targets_path("model.layers.0.self_attn.not_q_proj"));
        assert!(!modules.targets_path("model.layers.0.self_attn.q_proj_extra"));

        let regex_exclude: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj"],
                "exclude_modules": "model\\.layers\\.0\\.self_attn\\.k_proj"
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        regex_exclude.validate_dynamic()?;
        assert!(regex_exclude.excludes_path("model.layers.0.self_attn.k_proj"));
        assert!(!regex_exclude.excludes_path("model.layers.1.self_attn.k_proj"));

        let list_exclude: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "exclude_modules": ["k_proj"]
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        list_exclude.validate_dynamic()?;
        assert!(list_exclude.excludes_path("model.layers.0.self_attn.k_proj"));
        assert!(!list_exclude.excludes_path("model.layers.0.self_attn.q_proj"));
        assert!(!list_exclude.excludes_path("all.linear"));

        let all_linear_exclude: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "exclude_modules": "all-linear"
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        all_linear_exclude.validate_dynamic()?;
        assert!(!all_linear_exclude.excludes_path("model.layers.0.self_attn.q_proj"));
        assert!(all_linear_exclude.excludes_path("all-linear"));
        Ok(())
    }

    #[test]
    fn rank_and_alpha_patterns_follow_peft_matching() -> Result<()> {
        let config: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
                "rank_pattern": {
                    "^model\\.layers\\.0\\.self_attn\\.q_proj": 2,
                    "q_proj": 4,
                    "layers\\.[0-9]+\\.self_attn\\.v_proj": 6
                },
                "alpha_pattern": {"self_attn\\.(q|v)_proj": 4}
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;
        config.validate_dynamic()?;

        assert_eq!(config.rank_for("model.layers.0.self_attn.q_proj"), 2);
        assert_eq!(config.rank_for("base.model.layers.0.self_attn.q_proj"), 4);
        assert_eq!(config.rank_for("model.layers.12.self_attn.v_proj"), 6);
        assert_eq!(config.rank_for("model.layers.0.self_attn.k_proj"), 8);
        assert_eq!(config.alpha_for("model.layers.0.self_attn.q_proj"), 4.0);
        assert_eq!(config.scale_for("model.layers.0.self_attn.q_proj")?, 2.0);
        Ok(())
    }

    #[test]
    fn invalid_peft_regexes_are_rejected() {
        let target: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "["
            }"#,
        )
        .unwrap();
        assert!(target
            .validate_dynamic()
            .unwrap_err()
            .to_string()
            .contains("target_modules regex"));

        let rank: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "rank_pattern": {"[": 4}
            }"#,
        )
        .unwrap();
        assert!(rank
            .validate_dynamic()
            .unwrap_err()
            .to_string()
            .contains("rank_pattern regex"));

        let alpha: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "alpha_pattern": {"(": 4}
            }"#,
        )
        .unwrap();
        assert!(alpha
            .validate_dynamic()
            .unwrap_err()
            .to_string()
            .contains("alpha_pattern regex"));
    }

    #[test]
    fn inactive_peft_variant_fields_are_accepted() -> Result<()> {
        let config: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "exclude_modules": [],
                "init_lora_weights": true,
                "alora_invocation_tokens": null,
                "use_qalora": false,
                "qalora_group_size": 16,
                "layer_replication": [],
                "target_parameters": [],
                "use_bdlora": false,
                "arrow_config": false,
                "monteclora_config": false,
                "ensure_weight_tying": false,
                "trainable_token_indices": [],
                "megatron_config": {}
            }"#,
        )
        .map_err(candle_core::Error::wrap)?;

        config.validate_dynamic()
    }

    #[test]
    fn active_peft_variants_are_rejected() {
        let base = serde_json::json!({
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"]
        });
        let variants = [
            ("alora_invocation_tokens", serde_json::json!([1, 2])),
            ("use_qalora", serde_json::json!(true)),
            ("layer_replication", serde_json::json!([[0, 1]])),
            (
                "target_parameters",
                serde_json::json!(["experts.gate_up_proj"]),
            ),
            ("use_bdlora", serde_json::json!({"nblocks": 2})),
            ("arrow_config", serde_json::json!({})),
            ("monteclora_config", serde_json::json!({})),
            ("ensure_weight_tying", serde_json::json!(true)),
            ("trainable_token_indices", serde_json::json!([1])),
            ("megatron_config", serde_json::json!({"tp": 2})),
        ];

        for (name, value) in variants {
            let mut input = base.clone();
            input
                .as_object_mut()
                .unwrap()
                .insert(name.to_string(), value);
            let config: LoraConfig = serde_json::from_value(input).unwrap();
            assert!(config
                .validate_dynamic()
                .unwrap_err()
                .to_string()
                .contains(name));
        }
    }

    #[test]
    fn base_transforming_initializers_are_rejected() {
        for strategy in [
            "pissa",
            "pissa_niter_16",
            "olora",
            "corda",
            "loftq",
            "lora_ga",
        ] {
            let config: LoraConfig = serde_json::from_value(serde_json::json!({
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "init_lora_weights": strategy
            }))
            .unwrap();
            let error = config.validate_dynamic().unwrap_err().to_string();
            assert!(error.contains("init_lora_weights"));
            assert!(error.contains(strategy));
        }
    }

    #[test]
    fn unsupported_dynamic_features_are_rejected() {
        let config: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "use_dora": true
            }"#,
        )
        .unwrap();
        assert!(config.validate_dynamic().is_err());

        let config: LoraConfig = serde_json::from_str(
            r#"{
                "peft_type": "IA3",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"]
            }"#,
        )
        .unwrap();
        assert!(config.validate_dynamic().is_err());

        let config: LoraConfig = serde_json::from_str(
            r#"{
                "r": 8,
                "lora_alpha": 16,
                "target_modules": []
            }"#,
        )
        .unwrap();
        assert!(config.validate_dynamic().is_err());
    }
}
