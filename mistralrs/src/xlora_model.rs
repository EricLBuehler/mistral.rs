use mistralrs_core::{NormalLoaderBuilder, NormalSpecificConfig, Ordering};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    Model, TextModelBuilder,
};

/// Wrapper of [`TextModelBuilder`] for X-LoRA models.
pub struct XLoraModelBuilder {
    text_model: TextModelBuilder,
    xlora_model_id: String,
    ordering: Ordering,
    tgt_non_granular_index: Option<usize>,
}

impl XLoraModelBuilder {
    /// Create an X-LoRA builder from a [`TextModelBuilder`], X-LoRA model ID, and ordering.
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        xlora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            text_model,
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

    /// Load the X-LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let text_model = self.text_model.clone();
        let config = NormalSpecificConfig {
            topology: self.text_model.topology,
            organization: self.text_model.organization,
            write_uqff: self.text_model.write_uqff,
            from_uqff: self.text_model.from_uqff,
            imatrix: None,
            calibration_file: None,
            hf_cache_path: self.text_model.hf_cache_path,
            matformer_config_path: None,
            matformer_slice_name: None,
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
        .with_xlora(
            self.xlora_model_id,
            self.ordering,
            self.text_model.no_kv_cache,
            self.tgt_non_granular_index,
        )
        .build(self.text_model.loader_type)?;

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_text_loader(text_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
