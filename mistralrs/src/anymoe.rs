use mistralrs_core::{
    AnyMoeConfig, AnyMoeLoader, GGUFLoaderBuilder, GGUFSpecificConfig, Loader, NormalLoaderBuilder,
    NormalSpecificConfig,
};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_gguf_loader,
        build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    GgufModelBuilder, Model, TextModelBuilder,
};

enum AnyMoeBase {
    Text(TextModelBuilder),
    Gguf(GgufModelBuilder),
}

/// Configure and build an AnyMoE (Mixture of Experts) model on top of a text model.
pub struct AnyMoeModelBuilder {
    base: AnyMoeBase,
    config: AnyMoeConfig,
    path: String,
    prefix: String,
    mlp: String,
    model_ids: Vec<String>,
    layers: Vec<usize>,
}

impl AnyMoeModelBuilder {
    /// Create from a base [`TextModelBuilder`] with AnyMoE config, gating model path, prefix,
    /// MLP name, expert model IDs, and target layers.
    pub fn from_text_builder(
        base: TextModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        Self {
            base: AnyMoeBase::Text(base),
            config,
            path: path.to_string(),
            prefix: prefix.to_string(),
            mlp: mlp.to_string(),
            model_ids: model_ids
                .into_iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>(),
            layers,
        }
    }

    /// Create an AnyMoE model from a GGUF base.
    pub fn from_gguf_builder(
        base: GgufModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        Self {
            base: AnyMoeBase::Gguf(base),
            config,
            path: path.to_string(),
            prefix: prefix.to_string(),
            mlp: mlp.to_string(),
            model_ids: model_ids
                .into_iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>(),
            layers,
        }
    }

    /// Load the AnyMoE model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = match &self.base {
            AnyMoeBase::Text(base) => {
                let base = base.clone();
                let builder = base.clone();
                let config = NormalSpecificConfig {
                    topology: base.topology,
                    organization: base.organization,
                    write_uqff: base.write_uqff,
                    from_uqff: base.from_uqff,
                    imatrix: None,
                    calibration_file: None,
                    hf_cache_path: base.hf_cache_path,
                    hf_config_overrides: base.hf_config_overrides,
                    max_model_len: base.max_model_len,
                    matformer_config_path: None,
                    matformer_slice_name: None,
                };

                maybe_initialize_logging(base.with_logging);

                let loader = NormalLoaderBuilder::new(
                    config,
                    base.chat_template,
                    base.tokenizer_json,
                    Some(base.model_id),
                    base.no_kv_cache,
                    base.jinja_explicit,
                )
                .build(base.loader_type)?;

                let loader = self.wrap_loader(loader);
                build_pipeline_from_text_loader(builder, loader).await?
            }
            AnyMoeBase::Gguf(base) => {
                let base = base.clone();
                let builder = base.clone();
                let config = GGUFSpecificConfig {
                    topology: base.topology.clone(),
                    organization: base.organization,
                    write_uqff: base.write_uqff.clone(),
                    imatrix: base.imatrix.clone(),
                    calibration_file: base.calibration_file.clone(),
                    max_edge: base.max_edge,
                    max_model_len: base.max_model_len,
                    hf_cache_path: base.hf_cache_path.clone(),
                    matformer_config_path: base.matformer_config_path.clone(),
                    matformer_slice_name: base.matformer_slice_name.clone(),
                };

                maybe_initialize_logging(base.with_logging);

                let mut loader = GGUFLoaderBuilder::new(
                    base.chat_template.clone(),
                    base.tok_model_id.clone(),
                    base.model_id.clone(),
                    base.files.clone(),
                    config,
                    base.no_kv_cache,
                    base.jinja_explicit.clone(),
                );
                if let Some(mmproj_files) = base.mmproj_files.clone() {
                    loader = loader.with_mmproj_files(mmproj_files);
                }
                if let Some(tokenizer_json) = base.tokenizer_json.clone() {
                    loader = loader.with_tokenizer_json(tokenizer_json);
                }
                if let Some(adapters) = base.lora_adapters.clone() {
                    loader = loader.with_dynamic_lora(adapters, base.lora_runtime_config);
                }

                let loader = self.wrap_loader(loader.build());
                build_pipeline_from_gguf_loader(builder, loader).await?
            }
        };

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }

    fn wrap_loader(&self, target: Box<dyn Loader>) -> Box<dyn Loader> {
        Box::new(AnyMoeLoader {
            target,
            config: self.config.clone(),
            prefix: self.prefix.clone(),
            mlp: self.mlp.clone(),
            path: self.path.clone(),
            model_ids: self.model_ids.clone(),
            layers: self.layers.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use mistralrs_core::{AnyMoeConfig, AnyMoeExpertType};

    use super::{AnyMoeBase, AnyMoeModelBuilder};
    use crate::GgufModelBuilder;

    #[test]
    fn accepts_a_gguf_base() {
        let builder = AnyMoeModelBuilder::from_gguf_builder(
            GgufModelBuilder::new("repo", vec!["model.gguf"]),
            AnyMoeConfig {
                hidden_size: 128,
                lr: 1e-3,
                epochs: 1,
                batch_size: 1,
                expert_type: AnyMoeExpertType::FineTuned,
                gate_model_id: None,
                training: false,
                loss_csv_path: None,
            },
            "train.json",
            "model.layers",
            "mlp",
            vec!["expert"],
            vec![0],
        );

        assert!(matches!(builder.base, AnyMoeBase::Gguf(_)));
    }
}
