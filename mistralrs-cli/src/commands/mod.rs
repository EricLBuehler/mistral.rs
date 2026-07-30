//! Command implementations for mistralrs-cli

mod bench;
mod cache;
mod config;
mod doctor;
mod login;
mod manage;
pub(crate) mod quant;
mod quantize;
mod run;
pub(crate) mod serve;
mod tune;
mod uqff;

pub use bench::{run_bench, BenchRunConfig};
pub use cache::{run_cache_delete, run_cache_list};
pub use config::run_from_config;
pub use doctor::run_doctor;
pub use login::run_login;
pub use manage::{run_uninstall, run_update};
pub use quantize::run_quantize;
pub use run::run_interactive;
pub use serve::run_server;
pub use tune::run_tune;
pub use uqff::run_uqff;

use crate::args::{AdapterOptions, ModelType};
use mistralrs_core::MAX_LORA_ALIAS_BYTES;

fn dynamic_adapter_options(model_type: &ModelType) -> Option<&AdapterOptions> {
    match model_type {
        ModelType::Auto { adapter, .. }
        | ModelType::Text { adapter, .. }
        | ModelType::Multimodal { adapter, .. } => Some(adapter),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } | ModelType::Embedding { .. } => {
            None
        }
    }
}

pub(crate) fn normalize_requested_adapter(
    model_type: &ModelType,
    request_adapter: Option<&str>,
) -> anyhow::Result<Option<String>> {
    let Some(alias) = request_adapter.map(str::trim) else {
        return Ok(None);
    };
    if alias.is_empty() {
        anyhow::bail!("--adapter must not be empty");
    }
    if alias.len() > MAX_LORA_ALIAS_BYTES {
        anyhow::bail!("--adapter must not exceed {MAX_LORA_ALIAS_BYTES} bytes");
    }

    let Some(options) = dynamic_adapter_options(model_type) else {
        anyhow::bail!(
            "--adapter `{alias}` requires a dynamic LoRA runtime and a matching --lora preload"
        );
    };
    if !options.dynamic_lora_enabled() {
        anyhow::bail!(
            "--adapter `{alias}` requires a dynamic LoRA runtime and a matching --lora preload"
        );
    }
    if !options
        .lora
        .iter()
        .any(|adapter| adapter.alias.trim() == alias)
    {
        anyhow::bail!(
            "LoRA adapter alias `{alias}` is not configured; preload it with --lora {alias}=SOURCE"
        );
    }
    Ok(Some(alias.to_string()))
}

#[cfg(test)]
mod tests {
    use mistralrs_core::{LoraAdapterSpec, ModelDType};

    use super::*;
    use crate::args::{
        CacheOptions, DeviceOptions, FormatOptions, ModelSourceOptions, MultimodalOptions,
        QuantizationOptions,
    };

    fn auto_model(adapter: AdapterOptions) -> ModelType {
        ModelType::Auto {
            model: ModelSourceOptions {
                model_id: "org/base".to_string(),
                tokenizer: None,
                arch: None,
                dtype: ModelDType::Auto,
            },
            format: FormatOptions::default(),
            adapter,
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        }
    }

    #[test]
    fn requested_adapter_must_be_preloaded() {
        let model = auto_model(AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        });
        let error = normalize_requested_adapter(&model, Some("code")).unwrap_err();
        assert!(error.to_string().contains("not configured"));
    }

    #[test]
    fn requested_adapter_is_accepted_when_preloaded() {
        let model = auto_model(AdapterOptions {
            lora: vec![LoraAdapterSpec::new("code", "org/code-lora")],
            ..AdapterOptions::default()
        });
        assert_eq!(
            normalize_requested_adapter(&model, Some(" code ")).unwrap(),
            Some("code".to_string())
        );
    }

    #[test]
    fn requested_adapter_requires_dynamic_lora() {
        let model = auto_model(AdapterOptions::default());
        let error = normalize_requested_adapter(&model, Some("code")).unwrap_err();
        assert!(error
            .to_string()
            .contains("requires a dynamic LoRA runtime"));
    }

    #[test]
    fn requested_adapter_rejects_oversized_alias_before_loading() {
        let model = auto_model(AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        });
        let alias = "a".repeat(MAX_LORA_ALIAS_BYTES + 1);
        let error = normalize_requested_adapter(&model, Some(&alias)).unwrap_err();
        assert!(error.to_string().contains("must not exceed"));
    }
}
