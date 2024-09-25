use mistralrs_core::{
    initialize_logging, AnyMoeConfig, AnyMoeLoader, DefaultSchedulerMethod, DeviceMapMetadata,
    Loader, MistralRsBuilder, NormalLoaderBuilder, NormalSpecificConfig, SchedulerConfig,
};

use crate::{best_device, Model, TextModelBuilder};

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

    pub async fn build(self) -> anyhow::Result<Model> {
        let config = NormalSpecificConfig {
            use_flash_attn: self.base.use_flash_attn,
            prompt_batchsize: self.base.prompt_batchsize,
            topology: self.base.topology,
            organization: self.base.organization,
            write_uqff: self.base.write_uqff,
            from_uqff: self.base.from_uqff,
        };

        if self.base.with_logging {
            initialize_logging();
        }

        let loader = NormalLoaderBuilder::new(
            config,
            self.base.chat_template,
            self.base.tokenizer_json,
            Some(self.base.model_id),
        )
        .with_no_kv_cache(self.base.no_kv_cache)
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
        let pipeline = loader.load_model_from_hf(
            self.base.hf_revision,
            self.base.token_source,
            &self.base.dtype,
            &best_device(self.base.force_cpu)?,
            !self.base.with_logging,
            self.base
                .device_mapping
                .unwrap_or(DeviceMapMetadata::dummy()),
            self.base.isq,
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

        let mut runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(self.base.no_kv_cache)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(self.base.prefix_cache_n.is_none());

        if let Some(n) = self.base.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build()))
    }
}
