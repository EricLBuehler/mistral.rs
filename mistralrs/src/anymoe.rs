use mistralrs_core::{
    initialize_logging, AnyMoeConfig, AnyMoeLoader, AutoDeviceMapParams, DefaultSchedulerMethod,
    DeviceMapSetting, IsqType, Loader, MistralRsBuilder, NormalLoaderBuilder, NormalSpecificConfig,
    SchedulerConfig,
};

use crate::{best_device, resolve_isq, Model, TextModelBuilder};

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

        if self.base.with_logging {
            initialize_logging();
        }

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

        // Load, into a Pipeline
        let device = best_device(self.base.force_cpu)?;
        let isq_type: Option<IsqType> = self
            .base
            .isq
            .as_ref()
            .map(|s| resolve_isq(s, &device))
            .transpose()?;

        let pipeline = loader.load_model_from_hf(
            self.base.hf_revision,
            self.base.token_source,
            &self.base.dtype,
            &device,
            !self.base.with_logging,
            self.base
                .device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            isq_type,
            self.base.paged_attn_cfg,
        )?;

        let scheduler_method = match self.base.paged_attn_cfg {
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
                    max_num_seqs: self.base.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.base.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(
            pipeline,
            scheduler_method,
            self.base.throughput_logging,
            self.base.search_embedding_model,
        );
        if let Some(cb) = self.base.search_callback.clone() {
            runner = runner.with_search_callback(cb);
        }
        for (name, cb) in &self.base.tool_callbacks {
            runner = runner.with_tool_callback(name.clone(), cb.clone());
        }
        runner = runner
            .with_no_kv_cache(self.base.no_kv_cache)
            .with_no_prefix_cache(self.base.prefix_cache_n.is_none());

        if let Some(n) = self.base.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build().await))
    }
}
