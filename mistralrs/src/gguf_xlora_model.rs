use mistralrs_core::*;

use crate::{best_device, GgufModelBuilder, Model};

/// Wrapper of [`GgufModelBuilder`] for X-LoRA models.
pub struct GgufXLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    xlora_model_id: String,
    ordering: Ordering,
    tgt_non_granular_index: Option<usize>,
}

impl GgufXLoraModelBuilder {
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

    pub fn tgt_non_granular_index(mut self, tgt_non_granular_idx: usize) -> Self {
        self.tgt_non_granular_index = Some(tgt_non_granular_idx);
        self
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        let config = GGUFSpecificConfig {
            prompt_chunksize: self.gguf_model.prompt_chunksize,
            topology: self.gguf_model.topology,
        };

        if self.gguf_model.with_logging {
            initialize_logging();
        }

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

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.gguf_model.hf_revision,
            self.gguf_model.token_source,
            &ModelDType::Auto,
            &best_device(self.gguf_model.force_cpu)?,
            !self.gguf_model.with_logging,
            self.gguf_model
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            None,
            self.gguf_model.paged_attn_cfg,
        )?;

        let scheduler_method = match self.gguf_model.paged_attn_cfg {
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
                    max_num_seqs: self.gguf_model.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.gguf_model.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(self.gguf_model.no_kv_cache)
            .with_no_prefix_cache(self.gguf_model.prefix_cache_n.is_none());

        if let Some(n) = self.gguf_model.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build()))
    }
}
