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

/// The object used to interact with the model. This can be used with many varietes of models, \
/// and as such may be created with one of:
/// - [`TextModelBuilder`]
/// - [`LoraModelBuilder`]
/// - [`XLoraModelBuilder`]
/// - [`GgufModelBuilder`]
/// - [`GgufLoraModelBuilder`]
/// - [`GgufXLoraModelBuilder`]
/// - [`VisionModelBuilder`]
/// - [`AnyMoeModelBuilder`]
///
/// [`TextModelBuilder`]: crate::TextModelBuilder
/// [`LoraModelBuilder`]: crate::LoraModelBuilder
/// [`XLoraModelBuilder`]: crate::XLoraModelBuilder
/// [`GgufModelBuilder`]: crate::GgufModelBuilder
/// [`GgufModelBuilder`]: crate::GgufModelBuilder
/// [`GgufLoraModelBuilder`]: crate::GgufLoraModelBuilder
/// [`GgufXLoraModelBuilder`]: crate::GgufXLoraModelBuilder
/// [`VisionModelBuilder`]: crate::VisionModelBuilder
/// [`AnyMoeModelBuilder`]: crate::AnyMoeModelBuilder
///
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
            sampling_params: request.take_sampling_params(),
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

    pub async fn generate_image(
        &self,
        prompt: impl ToString,
        response_format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(NormalRequest {
            id: 0,
            messages: RequestMessage::ImageGeneration {
                prompt: prompt.to_string(),
                format: response_format,
                generation_params,
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            suffix: None,
            constraint: Constraint::None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
        });

        self.runner.get_sender()?.send(request).await?;

        let ResponseOk::ImageGeneration(response) = rx
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
