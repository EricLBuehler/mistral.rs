use mistralrs_core::{
    LoraAdapterSpec, LoraRuntimeConfig, NormalLoaderBuilder, NormalSpecificConfig,
};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    Model, TextModelBuilder,
};

/// Wrapper of [`TextModelBuilder`] for LoRA models.
pub struct LoraModelBuilder {
    text_model: TextModelBuilder,
    adapters: Vec<LoraAdapterSpec>,
    runtime_config: LoraRuntimeConfig,
}

impl LoraModelBuilder {
    /// Create a dynamic LoRA builder from a base text model.
    pub fn from_text_model_builder(text_model: TextModelBuilder) -> Self {
        Self {
            text_model,
            adapters: Vec::new(),
            runtime_config: LoraRuntimeConfig::default(),
        }
    }

    /// Preload an adapter under a request-facing alias.
    pub fn with_adapter(mut self, alias: impl Into<String>, source: impl Into<String>) -> Self {
        self.adapters.push(LoraAdapterSpec::new(alias, source));
        self
    }

    /// Preload an adapter repository at a specific Hugging Face revision.
    pub fn with_adapter_revision(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.adapters
            .push(LoraAdapterSpec::new(alias, source).with_revision(revision));
        self
    }

    /// Preload several typed adapter specifications.
    pub fn with_adapters(mut self, adapters: impl IntoIterator<Item = LoraAdapterSpec>) -> Self {
        self.adapters.extend(adapters);
        self
    }

    /// Set adapter residency and rank limits.
    pub fn with_runtime_config(mut self, runtime_config: LoraRuntimeConfig) -> Self {
        self.runtime_config = runtime_config;
        self
    }

    /// Build the base model and its dynamic LoRA runtime.
    pub async fn build(self) -> anyhow::Result<Model> {
        let text_model = self.text_model.clone();
        let config = NormalSpecificConfig {
            topology: self.text_model.topology,
            organization: self.text_model.organization,
            write_uqff: self.text_model.write_uqff,
            from_uqff: self.text_model.from_uqff,
            imatrix: self.text_model.imatrix,
            calibration_file: self.text_model.calibration_file,
            hf_cache_path: self.text_model.hf_cache_path,
            matformer_config_path: self.text_model.matformer_config_path,
            matformer_slice_name: self.text_model.matformer_slice_name,
        };

        maybe_initialize_logging(self.text_model.with_logging);

        let loader = NormalLoaderBuilder::new(
            config,
            self.text_model.chat_template,
            self.text_model.tokenizer_json,
            Some(self.text_model.model_id),
            self.text_model.no_kv_cache,
            self.text_model.jinja_explicit,
        )
        .with_lora(self.adapters, self.runtime_config)
        .build(self.text_model.loader_type)?;

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_text_loader(text_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
