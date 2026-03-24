use mistralrs_core::{
    AnyMoeConfig, AnyMoeLoader, Loader, NormalLoaderBuilder, NormalSpecificConfig,
};

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    Model, TextModelBuilder,
};

/// Configure and build an AnyMoE (Mixture of Experts) model on top of a text model.
pub struct AnyMoeModelBuilder {
    base: TextModelBuilder,
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
            base,
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
        let base = self.base.clone();
        let config = NormalSpecificConfig {
            topology: self.base.topology,
            organization: self.base.organization,
            write_uqff: self.base.write_uqff,
            from_uqff: self.base.from_uqff,
            imatrix: None,
            calibration_file: None,
            hf_cache_path: self.base.hf_cache_path,
            matformer_config_path: None,
            matformer_slice_name: None,
        };

        maybe_initialize_logging(self.base.with_logging);

        let loader = NormalLoaderBuilder::new(
            config,
            self.base.chat_template,
            self.base.tokenizer_json,
            Some(self.base.model_id),
            self.base.no_kv_cache,
            self.base.jinja_explicit,
        )
        .build(self.base.loader_type)?;

        let loader: Box<dyn Loader> = Box::new(AnyMoeLoader {
            target: loader,
            config: self.config,
            prefix: self.prefix,
            mlp: self.mlp,
            path: self.path,
            model_ids: self.model_ids,
            layers: self.layers,
        });

        let (pipeline, scheduler_config, add_model_config) =
            build_pipeline_from_text_loader(base, loader).await?;

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
