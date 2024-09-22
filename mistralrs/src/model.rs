use anyhow::Context;
use candle_core::{Device, Result};
use mistralrs_core::*;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use crate::RequestLike;

/// Gets the best device, cpu, cuda if compiled with CUDA, or Metal
pub fn best_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

pub struct Model {
    runner: Arc<MistralRs>,
}

impl Model {
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
            messages: request.take_messages(),
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

    /// Activate certain adapters on the model, they will be used for requests which do not specify unique adapters.
    pub async fn activate_adapters<A: ToString>(&self, adapters: Vec<A>) -> anyhow::Result<()> {
        let request = Request::ActivateAdapters(
            adapters
                .into_iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>(),
        );

        Ok(self.runner.get_sender()?.send(request).await?)
    }

    /// Reapply ISQ to the model. This will be done on whatever device the model is already on.
    pub async fn re_isq_model(&self, isq_type: IsqType) -> anyhow::Result<()> {
        let request = Request::ReIsq(isq_type);

        Ok(self.runner.get_sender()?.send(request).await?)
    }
}
