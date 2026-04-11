use mistralrs_core::{GGUFLoaderBuilder, GGUFSpecificConfig, Ordering};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_gguf_loader, maybe_initialize_logging,
    },
    GgufModelBuilder, Model,
};

/// Wrapper of [`GgufModelBuilder`] for LoRA models.
pub struct GgufLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    lora_model_id: String,
    ordering: Ordering,
}

impl GgufLoraModelBuilder {
    /// Create a GGUF LoRA builder from a [`GgufModelBuilder`], LoRA model ID, and ordering.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        lora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            gguf_model,
            lora_model_id: lora_model_id.to_string(),
            ordering,
        }
    }

    /// Load the GGUF LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let gguf_model = self.gguf_model.clone();
        let config = GGUFSpecificConfig {
            topology: self.gguf_model.topology,
        };

        maybe_initialize_logging(self.gguf_model.with_logging);

        let loader = GGUFLoaderBuilder::new(
            self.gguf_model.chat_template,
            self.gguf_model.tok_model_id,
            self.gguf_model.model_id,
            self.gguf_model.files,
            config,
            self.gguf_model.no_kv_cache,
            self.gguf_model.jinja_explicit,
        )
        .with_lora(self.lora_model_id, self.ordering)
        .build();

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_gguf_loader(gguf_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
