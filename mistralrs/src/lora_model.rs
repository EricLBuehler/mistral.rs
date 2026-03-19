use mistralrs_core::*;

use crate::{best_device, resolve_isq, Model, TextModelBuilder};

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

        if self.text_model.with_logging {
            initialize_logging();
        }

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

        // Load, into a Pipeline
        let device = best_device(self.text_model.force_cpu)?;
        let isq_type: Option<IsqType> = self
            .text_model
            .isq
            .as_ref()
            .map(|s| resolve_isq(s, &device))
            .transpose()?;

        let pipeline = loader.load_model_from_hf(
            self.text_model.hf_revision,
            self.text_model.token_source,
            &self.text_model.dtype,
            &device,
            !self.text_model.with_logging,
            self.text_model
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            isq_type,
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

        let mut runner = MistralRsBuilder::new(
            pipeline,
            scheduler_method,
            self.text_model.throughput_logging,
            self.text_model.search_embedding_model,
        );
        if let Some(cb) = self.text_model.search_callback.clone() {
            runner = runner.with_search_callback(cb);
        }
        for (name, cb) in &self.text_model.tool_callbacks {
            runner = runner.with_tool_callback(name.clone(), cb.clone());
        }
        runner = runner
            .with_no_kv_cache(self.text_model.no_kv_cache)
            .with_no_prefix_cache(self.text_model.prefix_cache_n.is_none());

        if let Some(n) = self.text_model.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build().await))
    }
}
