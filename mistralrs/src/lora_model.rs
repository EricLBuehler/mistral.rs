use mistralrs_core::*;

use crate::{best_device, Model, TextModelBuilder};

/// Wrapper of [`TextModelBuilder`] for LoRA models.
pub struct LoraModelBuilder {
    text_model: TextModelBuilder,
    lora_model_id: String,
    ordering: Ordering,
}

impl LoraModelBuilder {
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        lora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            text_model,
            lora_model_id: lora_model_id.to_string(),
            ordering,
        }
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        let config = NormalSpecificConfig {
            use_flash_attn: self.text_model.use_flash_attn,
            prompt_chunksize: self.text_model.prompt_chunksize,
            topology: self.text_model.topology,
            organization: self.text_model.organization,
            write_uqff: self.text_model.write_uqff,
            from_uqff: self.text_model.from_uqff,
            imatrix: None,
            calibration_file: None,
        };

        if self.text_model.with_logging {
            initialize_logging();
        }

        let loader = NormalLoaderBuilder::new(
            config,
            self.text_model.chat_template,
            self.text_model.tokenizer_json,
            Some(self.text_model.model_id),
        )
        .with_lora(self.lora_model_id, self.ordering)
        .with_no_kv_cache(self.text_model.no_kv_cache)
        .build(self.text_model.loader_type)?;

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.text_model.hf_revision,
            self.text_model.token_source,
            &self.text_model.dtype,
            &best_device(self.text_model.force_cpu)?,
            !self.text_model.with_logging,
            self.text_model
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            self.text_model.isq,
            self.text_model.paged_attn_cfg,
        )?;

        let scheduler_method = match self.text_model.paged_attn_cfg {
            Some(_) => {
                let config = pipeline
                    .lock()
                    .await
                    .get_metadata()
                    .cache_config
                    .as_ref()
                    .unwrap()
                    .clone();

                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: self.text_model.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.text_model.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(self.text_model.no_kv_cache)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(self.text_model.prefix_cache_n.is_none());

        if let Some(n) = self.text_model.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build()))
    }
}
