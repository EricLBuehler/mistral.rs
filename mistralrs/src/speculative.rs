use std::sync::Arc;

use mistralrs_core::{
    initialize_logging, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting,
    MistralRsBuilder, NormalLoaderBuilder, NormalSpecificConfig, Pipeline, SchedulerConfig,
    SpeculativeConfig, SpeculativePipeline,
};
use tokio::sync::Mutex;

use crate::{best_device, Model, TextModelBuilder};

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
            use_flash_attn: builder.use_flash_attn,
            prompt_chunksize: builder.prompt_chunksize,
            topology: builder.topology,
            organization: builder.organization,
            write_uqff: builder.write_uqff,
            from_uqff: builder.from_uqff,
            imatrix: builder.imatrix,
            calibration_file: builder.calibration_file,
        };

        if builder.with_logging {
            initialize_logging();
        }

        let loader = NormalLoaderBuilder::new(
            config,
            builder.chat_template,
            builder.tokenizer_json,
            Some(builder.model_id),
        )
        .build(builder.loader_type)?;

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            builder.hf_revision,
            builder.token_source,
            &builder.dtype,
            &best_device(builder.force_cpu)?,
            !builder.with_logging,
            builder
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            builder.isq,
            builder.paged_attn_cfg,
        )?;
        Ok(pipeline)
    }

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

        let runner =
            MistralRsBuilder::new(pipeline, scheduler_method).with_gemm_full_precision_f16(true);

        Ok(Model::new(runner.build()))
    }
}
