use mistralrs_core::{GGUFLoaderBuilder, GGUFSpecificConfig, Ordering};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_gguf_loader, maybe_initialize_logging,
    },
    GgufModelBuilder, Model,
};

/// Wrapper of [`GgufModelBuilder`] for X-LoRA models.
pub struct GgufXLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    xlora_model_id: String,
    ordering: Ordering,
    tgt_non_granular_index: Option<usize>,
}

impl GgufXLoraModelBuilder {
    /// Create a GGUF X-LoRA builder from a [`GgufModelBuilder`], X-LoRA model ID, and ordering.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        xlora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            gguf_model,
            xlora_model_id: xlora_model_id.to_string(),
            ordering,
            tgt_non_granular_index: None,
        }
    }

    /// Set the target non-granular index for X-LoRA scaling.
    pub fn tgt_non_granular_index(mut self, tgt_non_granular_idx: usize) -> Self {
        self.tgt_non_granular_index = Some(tgt_non_granular_idx);
        self
    }

    /// Load the GGUF X-LoRA model and return a ready-to-use [`Model`].
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
        .with_xlora(
            self.xlora_model_id,
            self.ordering,
            self.gguf_model.no_kv_cache,
            self.tgt_non_granular_index,
        )
        .build();

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_gguf_loader(gguf_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
