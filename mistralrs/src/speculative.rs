use std::sync::Arc;

use mistralrs_core::{
    initialize_logging, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting, IsqType,
    MistralRsBuilder, NormalLoaderBuilder, NormalSpecificConfig, Pipeline, SchedulerConfig,
    SpeculativeConfig, SpeculativePipeline,
};
use tokio::sync::Mutex;

use crate::{best_device, resolve_isq, Model, TextModelBuilder};

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

    fn build_pipeline(builder: TextModelBuilder) -> anyhow::Result<Arc<Mutex<dyn Pipeline>>> {
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

        if builder.with_logging {
            initialize_logging();
        }

        let loader = NormalLoaderBuilder::new(
            config,
            builder.chat_template,
            builder.tokenizer_json,
            Some(builder.model_id),
            builder.no_kv_cache,
            builder.jinja_explicit,
        )
        .build(builder.loader_type)?;

        // Load, into a Pipeline
        let device = best_device(builder.force_cpu)?;
        let isq_type: Option<IsqType> = builder
            .isq
            .as_ref()
            .map(|s| resolve_isq(s, &device))
            .transpose()?;

        let pipeline = loader.load_model_from_hf(
            builder.hf_revision,
            builder.token_source,
            &builder.dtype,
            &device,
            !builder.with_logging,
            builder
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            isq_type,
            builder.paged_attn_cfg,
        )?;
        Ok(pipeline)
    }

    /// Build target and draft pipelines and return a speculative-decoding [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let target = Self::build_pipeline(self.target.clone())?;
        let draft = Self::build_pipeline(self.draft.clone())?;

        let scheduler_method = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(self.target.max_num_seqs.try_into()?),
        };

        let pipeline = Arc::new(Mutex::new(SpeculativePipeline::new(
            target,
            draft,
            self.speculative_config,
        )?));

        let mut runner = MistralRsBuilder::new(
            pipeline,
            scheduler_method,
            self.target.throughput_logging,
            self.target.search_embedding_model,
        );
        if let Some(cb) = self.target.search_callback.clone() {
            runner = runner.with_search_callback(cb);
        }
        for (name, cb) in &self.target.tool_callbacks {
            runner = runner.with_tool_callback(name.clone(), cb.clone());
        }

        Ok(Model::new(runner.build().await))
    }
}
