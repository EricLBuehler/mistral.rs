use std::sync::Arc;

use anyhow::Context;
use mistralrs_core::*;
use tokio::sync::mpsc::channel;

use crate::{text_model::best_device, RequestLike, TextModelBuilder};

pub struct LoraModel {
    runner: Arc<MistralRs>,
}

pub struct LoraModelBuilder {
    text_model: TextModelBuilder,
    lora_model_id: String,
    ordering: Ordering,
}

impl LoraModelBuilder {
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        lora_model_id: String,
        ordering: Ordering,
    ) -> Self {
        Self {
            text_model,
            lora_model_id,
            ordering,
        }
    }

    pub async fn build(self) -> anyhow::Result<LoraModel> {
        let config = NormalSpecificConfig {
            use_flash_attn: self.text_model.use_flash_attn,
            prompt_batchsize: self.text_model.prompt_batchsize,
            topology: self.text_model.topology,
            organization: self.text_model.organization,
            write_uqff: self.text_model.write_uqff,
            from_uqff: self.text_model.from_uqff,
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
            DeviceMapMetadata::dummy(),
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

        Ok(LoraModel::new(runner.build()))
    }
}

impl LoraModel {
    pub fn new(runner: Arc<MistralRs>) -> Self {
        Self { runner }
    }

    /// Generate with the model.
    pub async fn send_chat_request<R: RequestLike>(
        &self,
        mut request: R,
    ) -> anyhow::Result<ChatCompletionResponse> {
        let (tx, mut rx) = channel(1);

        let (tools, tool_choice) = if let Some((a, b)) = request.take_tools() {
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Chat(request.take_messages()),
            sampling_params: SamplingParams::default(),
            response: tx,
            return_logprobs: request.return_logprobs(),
            is_streaming: false,
            id: 0,
            constraint: request.take_constraint(),
            suffix: None,
            adapters: request.take_adapters(),
            tools,
            tool_choice,
            logits_processors: request.take_logits_processors(),
        });

        self.runner.get_sender()?.send(request).await?;

        let ResponseOk::Done(response) = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok(response)
    }
}
