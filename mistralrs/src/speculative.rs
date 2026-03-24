use std::sync::Arc;

use mistralrs_core::{
    DefaultSchedulerMethod, NormalLoaderBuilder, NormalSpecificConfig, Pipeline, SchedulerConfig,
    SpeculativeConfig, SpeculativePipeline,
};
use tokio::sync::Mutex;

use crate::{
    model_builder_trait::{
        build_model_from_pipeline, build_pipeline_from_text_loader, maybe_initialize_logging,
    },
    Model, TextModelBuilder,
};

/// Configure speculative decoding with a target and draft text model.
pub struct TextSpeculativeBuilder {
    target: TextModelBuilder,
    draft: TextModelBuilder,
    speculative_config: SpeculativeConfig,
}

impl TextSpeculativeBuilder {
    /// Create a builder for a speculative decoding pipeline.
    ///
    /// - PagedAttention settings are ignored as our impl of speculative decoding does not support this yet.
    /// - Prefix caching settings are ignored as our impl of speculative decoding does not support this yet.
    ///
    /// Otherwise, scheduling parameters such as `max_num_seqs` are sourced from the target model.
    pub fn new(
        target: TextModelBuilder,
        draft: TextModelBuilder,
        speculative_config: SpeculativeConfig,
    ) -> anyhow::Result<Self> {
        if target.no_kv_cache || draft.no_kv_cache {
            anyhow::bail!("Both target and draft must have KV cache enabled.");
        }

        Ok(Self {
            target,
            draft,
            speculative_config,
        })
    }

    async fn build_pipeline(
        builder: TextModelBuilder,
    ) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, mistralrs_core::AddModelConfig)> {
        let model = builder.clone();
        let config = NormalSpecificConfig {
            topology: builder.topology,
            organization: builder.organization,
            write_uqff: builder.write_uqff,
            from_uqff: builder.from_uqff,
            imatrix: builder.imatrix,
            calibration_file: builder.calibration_file,
            hf_cache_path: builder.hf_cache_path,
            matformer_config_path: None,
            matformer_slice_name: None,
        };

        maybe_initialize_logging(builder.with_logging);

        let loader = NormalLoaderBuilder::new(
            config,
            builder.chat_template,
            builder.tokenizer_json,
            Some(builder.model_id),
            builder.no_kv_cache,
            builder.jinja_explicit,
        )
        .build(builder.loader_type)?;

        let (pipeline, _, add_model_config) =
            build_pipeline_from_text_loader(model, loader).await?;
        Ok((pipeline, add_model_config))
    }

    /// Build target and draft pipelines and return a speculative-decoding [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (target, mut add_model_config) = Self::build_pipeline(self.target.clone()).await?;
        let (draft, _) = Self::build_pipeline(self.draft.clone()).await?;

        let scheduler_method = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(self.target.max_num_seqs.try_into()?),
        };

        let pipeline = Arc::new(Mutex::new(SpeculativePipeline::new(
            target,
            draft,
            self.speculative_config,
        )?));

        // Speculative decoding still does not expose its own prefix-cache tuning surface.
        add_model_config.engine_config.no_prefix_cache = false;
        add_model_config.engine_config.prefix_cache_n = 16;

        Ok(build_model_from_pipeline(pipeline, scheduler_method, add_model_config).await)
    }
}
