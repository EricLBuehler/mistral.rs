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
        if self.gguf_model.lora_adapters.is_some() {
            anyhow::bail!(
                "`GgufLoraModelBuilder` cannot combine legacy static LoRA with dynamic LoRA; use \
                 `GgufModelBuilder` directly for dynamic adapters"
            );
        }
        if self.gguf_model.mmproj_files.is_some() {
            anyhow::bail!(
                "`GgufLoraModelBuilder` provides legacy static LoRA, which is not supported for \
                 multimodal GGUF; use `GgufModelBuilder::with_lora_adapter`"
            );
        }
        let gguf_model = self.gguf_model.clone();
        let config = GGUFSpecificConfig {
            topology: self.gguf_model.topology,
            organization: self.gguf_model.organization,
            write_uqff: self.gguf_model.write_uqff,
            imatrix: self.gguf_model.imatrix,
            calibration_file: self.gguf_model.calibration_file,
            max_edge: self.gguf_model.max_edge,
            max_model_len: self.gguf_model.max_model_len,
            hf_cache_path: self.gguf_model.hf_cache_path,
            matformer_config_path: self.gguf_model.matformer_config_path,
            matformer_slice_name: self.gguf_model.matformer_slice_name,
        };

        maybe_initialize_logging(self.gguf_model.with_logging);

        let mut loader = GGUFLoaderBuilder::new(
            self.gguf_model.chat_template,
            self.gguf_model.tok_model_id,
            self.gguf_model.model_id,
            self.gguf_model.files,
            config,
            self.gguf_model.no_kv_cache,
            self.gguf_model.jinja_explicit,
        )
        .with_lora(self.lora_model_id, self.ordering);
        if let Some(tokenizer_json) = self.gguf_model.tokenizer_json {
            loader = loader.with_tokenizer_json(tokenizer_json);
        }
        let loader = loader.build();

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_gguf_loader(gguf_model, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn legacy_lora_rejects_a_dynamic_gguf_builder() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"]).with_lora();
        let ordering = Ordering {
            adapters: None,
            layers: None,
            base_model_id: "repo".to_string(),
            preload_adapters: None,
        };
        let error = GgufLoraModelBuilder::from_gguf_model_builder(builder, "legacy", ordering)
            .build()
            .await
            .err()
            .expect("mixed adapter modes should fail");

        assert!(error.to_string().contains("cannot combine"));
    }
}
