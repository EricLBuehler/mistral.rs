use mistralrs_core::{NormalLoaderBuilder, NormalSpecificConfig};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    Model, TextModelBuilder,
};

/// Wrapper of [`TextModelBuilder`] for LoRA models.
pub struct LoraModelBuilder {
    text_model: TextModelBuilder,
    lora_adapter_ids: Vec<String>,
}

impl LoraModelBuilder {
    /// Create a LoRA builder from a [`TextModelBuilder`] and LoRA adapter IDs.
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        lora_adapter_ids: impl IntoIterator<Item = impl ToString>,
    ) -> Self {
        Self {
            text_model,
            lora_adapter_ids: lora_adapter_ids
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        }
    }

    /// Load the LoRA model and return a ready-to-use [`Model`].
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
        .with_lora(self.lora_adapter_ids)
        .build(self.text_model.loader_type)?;

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_text_loader(text_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
