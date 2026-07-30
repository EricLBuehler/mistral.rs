//! Configuration file loading for mistralrs-cli
//!
//! Supports a full TOML configuration that mirrors the CLI options while
//! allowing multiple models without aliases.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::args::{
    AdapterOptions, CacheOptions, DeviceOptions, FormatOptions, GlobalOptions, ModelSourceOptions,
    ModelType, MultimodalOptions, PagedAttentionOptions, QuantizationOptions, RuntimeOptions,
    SandboxOptions, ServerOptions,
};
use mistralrs_core::{ModelDType, NormalLoaderType, TokenSource};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum CliConfig {
    Serve(ServeConfig),
    Run(RunConfig),
}

#[derive(Deserialize, Default)]
pub struct ServeConfig {
    #[serde(default)]
    pub global: GlobalOptionsToml,
    #[serde(default)]
    pub runtime: RuntimeOptions,
    #[serde(default)]
    pub server: ServerOptions,
    #[serde(default)]
    pub paged_attn: PagedAttentionOptions,
    #[serde(default)]
    pub sandbox: SandboxOptions,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub default_model_id: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct RunConfig {
    #[serde(default)]
    pub global: GlobalOptionsToml,
    #[serde(default)]
    pub runtime: RuntimeOptions,
    #[serde(default)]
    pub paged_attn: PagedAttentionOptions,
    #[serde(default)]
    pub sandbox: SandboxOptions,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default, alias = "enable_thinking")]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
pub struct GlobalOptionsToml {
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub log: Option<PathBuf>,
    #[serde(default)]
    pub token_source: Option<String>,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    #[default]
    Auto,
    Text,
    Multimodal,
    Diffusion,
    Speech,
    Embedding,
}

#[derive(Deserialize, Clone)]
pub struct ModelEntry {
    #[serde(default)]
    pub kind: ModelKind,
    pub model_id: String,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub arch: Option<NormalLoaderType>,
    #[serde(default)]
    pub dtype: ModelDType,
    #[serde(default)]
    pub format: FormatOptions,
    #[serde(default)]
    pub adapter: AdapterOptions,
    #[serde(default)]
    pub quantization: QuantizationOptions,
    #[serde(default)]
    pub device: DeviceOptionsToml,
    #[serde(default)]
    pub multimodal: MultimodalOptions,
    #[serde(default)]
    pub chat_template: Option<PathBuf>,
    #[serde(default)]
    pub jinja_explicit: Option<PathBuf>,
    /// Path to a MatFormer slice config. Only meaningful for MatFormer-trained models like Gemma 3n.
    #[serde(default)]
    pub matformer_config_path: Option<PathBuf>,
    /// Named slice to load from the MatFormer config.
    #[serde(default)]
    pub matformer_slice_name: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
pub struct DeviceOptionsToml {
    #[serde(default)]
    pub cpu: Option<bool>,
    #[serde(default)]
    pub device_layers: Option<Vec<String>>,
    #[serde(default)]
    pub topology: Option<PathBuf>,
    #[serde(default)]
    pub hf_cache: Option<PathBuf>,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
}

pub fn load_cli_config(path: &Path) -> Result<CliConfig> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("toml"))
        != Some(true)
    {
        anyhow::bail!("mistralrs-cli config files must be .toml");
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file {}", path.to_string_lossy()))?;

    let config: CliConfig =
        toml::from_str(&contents).context("Failed to parse TOML config file")?;
    validate_config(&config)?;
    Ok(config)
}

fn validate_config(config: &CliConfig) -> Result<()> {
    let (models, default_model_id) = match config {
        CliConfig::Serve(cfg) => (&cfg.models, cfg.default_model_id.as_ref()),
        CliConfig::Run(cfg) => (&cfg.models, None),
    };

    if models.is_empty() {
        anyhow::bail!("Config must define at least one model in [[models]]");
    }

    if let CliConfig::Run(cfg) = config {
        let _ = crate::commands::normalize_requested_adapter(
            &models[0].to_model_type(false),
            cfg.adapter.as_deref(),
        )?;
    }

    if let Some(default_id) = default_model_id {
        let has_model = models.iter().any(|model| model.model_id == *default_id);
        if !has_model {
            anyhow::bail!(
                "default_model_id '{}' does not match any model_id in [[models]]",
                default_id
            );
        }
    }

    let mut cpu_setting: Option<bool> = None;
    for model in models {
        model
            .adapter
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid adapter configuration: {error}"))?;
        if let Some(cpu) = model.device.cpu {
            match cpu_setting {
                None => cpu_setting = Some(cpu),
                Some(existing) if existing != cpu => {
                    anyhow::bail!(
                        "cpu must be consistent across all models (found both true and false)"
                    );
                }
                _ => {}
            }
        }
    }

    Ok(())
}

impl GlobalOptionsToml {
    pub fn to_global_options(&self) -> Result<GlobalOptions> {
        let token_source = match &self.token_source {
            Some(value) => value
                .parse()
                .map_err(|err| anyhow::anyhow!("Invalid token_source: {err}"))?,
            None => TokenSource::CacheToken,
        };

        Ok(GlobalOptions {
            seed: self.seed,
            log: self.log.clone(),
            token_source,
            verbose: 0,
        })
    }
}

impl DeviceOptionsToml {
    pub fn to_device_options(&self, cpu: bool) -> DeviceOptions {
        let defaults = DeviceOptions::default();
        DeviceOptions {
            cpu,
            device_layers: self.device_layers.clone(),
            topology: self.topology.clone(),
            hf_cache: self.hf_cache.clone(),
            max_seq_len: self.max_seq_len.unwrap_or(defaults.max_seq_len),
            max_batch_size: self.max_batch_size.unwrap_or(defaults.max_batch_size),
        }
    }
}

impl ModelEntry {
    pub fn to_model_type(&self, cpu: bool) -> ModelType {
        let model = ModelSourceOptions {
            model_id: self.model_id.clone(),
            tokenizer: self.tokenizer.clone(),
            arch: self.arch.clone(),
            dtype: self.dtype,
        };

        let device = self.device.to_device_options(cpu);
        let cache = CacheOptions::default();

        match self.kind {
            ModelKind::Auto => ModelType::Auto {
                model,
                format: self.format.clone(),
                adapter: self.adapter.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
                multimodal: self.multimodal.clone(),
            },
            ModelKind::Text => ModelType::Text {
                model,
                format: self.format.clone(),
                adapter: self.adapter.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
            },
            ModelKind::Multimodal => ModelType::Multimodal {
                model,
                format: self.format.clone(),
                adapter: self.adapter.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
                multimodal: self.multimodal.clone(),
            },
            ModelKind::Diffusion => ModelType::Diffusion { model, device },
            ModelKind::Speech => ModelType::Speech { model, device },
            ModelKind::Embedding => ModelType::Embedding {
                model,
                format: self.format.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_model_kind_defaults_to_auto() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
model_id = "google/gemma-4-E4B-it"
"#,
        )
        .unwrap();

        let cfg = match config {
            CliConfig::Serve(cfg) => cfg,
            CliConfig::Run(_) => panic!("expected serve config"),
        };

        assert!(matches!(cfg.models[0].kind, ModelKind::Auto));
        assert!(matches!(
            cfg.models[0].to_model_type(false),
            ModelType::Auto { .. }
        ));
    }

    #[test]
    fn explicit_model_kind_is_preserved() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
kind = "multimodal"
model_id = "google/gemma-4-E4B-it"
"#,
        )
        .unwrap();

        let cfg = match config {
            CliConfig::Serve(cfg) => cfg,
            CliConfig::Run(_) => panic!("expected serve config"),
        };

        assert!(matches!(cfg.models[0].kind, ModelKind::Multimodal));
        assert!(matches!(
            cfg.models[0].to_model_type(false),
            ModelType::Multimodal { .. }
        ));
    }

    #[test]
    fn run_config_uses_structured_lora_preloads_and_explicit_selection() {
        let config: CliConfig = toml::from_str(
            r#"
command = "run"
adapter = "code"

[[models]]
model_id = "org/base"

[models.adapter]
enable_lora = true
lora = [{ alias = "code", source = "org/code-lora", revision = "refs/pr/7" }]
lora_max_adapters = 4
lora_max_rank = 64
lora_max_bytes = 1048576
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let cfg = match config {
            CliConfig::Run(cfg) => cfg,
            CliConfig::Serve(_) => panic!("expected run config"),
        };
        assert_eq!(cfg.adapter.as_deref(), Some("code"));
        assert_eq!(cfg.models[0].adapter.lora[0].alias, "code");
        assert_eq!(cfg.models[0].adapter.lora[0].source, "org/code-lora");
        assert_eq!(cfg.models[0].adapter.lora[0].revision(), "refs/pr/7");
        assert_eq!(cfg.models[0].adapter.lora_runtime_config().max_rank, 64);
    }

    #[test]
    fn duplicate_toml_lora_aliases_are_rejected() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
model_id = "org/base"

[models.adapter]
lora = [
    { alias = "code", source = "org/first" },
    { alias = "code", source = "org/second" },
]
"#,
        )
        .unwrap();

        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("more than once"));
    }

    #[test]
    fn run_config_rejects_unconfigured_adapter_selection() {
        let config: CliConfig = toml::from_str(
            r#"
command = "run"
adapter = "math"

[[models]]
model_id = "org/base"

[models.adapter]
lora = [{ alias = "code", source = "org/code-lora" }]
"#,
        )
        .unwrap();

        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("not configured"));
    }
}
