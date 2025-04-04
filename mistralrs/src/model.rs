use anyhow::Context;
use candle_core::{Device, Result, Tensor};
use either::Either;
use mistralrs_core::*;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver};

use crate::{RequestLike, TextMessages};

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

pub struct Stream<'a> {
    _server: &'a Model,
    rx: Receiver<Response>,
}

impl Stream<'_> {
    pub async fn next(&mut self) -> Option<Response> {
        self.rx.recv().await
    }
}

impl Model {
    pub fn new(runner: Arc<MistralRs>) -> Self {
        Self { runner }
    }

    /// Generate with the model.
    pub async fn stream_chat_request<R: RequestLike>(
        &self,
        mut request: R,
    ) -> anyhow::Result<Stream> {
        let (tx, rx) = channel(1);

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
            is_streaming: true,
            id: 0,
            constraint: request.take_constraint(),
            suffix: None,
            tools,
            tool_choice,
            logits_processors: request.take_logits_processors(),
            return_raw_logits: false,
            web_search_options: request.take_web_search_options(),
        });

        self.runner.get_sender()?.send(request).await?;

        let stream = Stream { _server: self, rx };

        Ok(stream)
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
            tools,
            tool_choice,
            logits_processors: request.take_logits_processors(),
            return_raw_logits: false,
            web_search_options: request.take_web_search_options(),
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

    /// Generate with the model, returning raw logits of the first token generated.
    ///
    /// Returns the chunks of the logits (1 or more, determined by prompt batchsize) and the tokens.
    pub async fn send_raw_chat_request<R: RequestLike>(
        &self,
        mut request: R,
    ) -> anyhow::Result<(Vec<Tensor>, Vec<u32>)> {
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
            tools,
            tool_choice,
            logits_processors: request.take_logits_processors(),
            return_raw_logits: true,
            web_search_options: request.take_web_search_options(),
        });

        self.runner.get_sender()?.send(request).await?;

        let ResponseOk::Raw {
            logits_chunks,
            tokens,
        } = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok((logits_chunks, tokens))
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
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
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

    /// Reapply ISQ to the model. This will be done on whatever device the model is already on.
    pub async fn re_isq_model(&self, isq_type: IsqType) -> anyhow::Result<()> {
        let request = Request::ReIsq(isq_type);

        Ok(self.runner.get_sender()?.send(request).await?)
    }

    /// Tokenize some text or messages.
    /// - `tools` is only used if messages are provided.
    pub async fn tokenize(
        &self,
        text: Either<TextMessages, String>,
        tools: Option<Vec<Tool>>,
        add_special_tokens: bool,
        add_generation_prompt: bool,
    ) -> anyhow::Result<Vec<u32>> {
        let (tx, mut rx) = channel(1);
        let request = Request::Tokenize(TokenizationRequest {
            text: text.map_left(Into::into),
            tools,
            add_special_tokens,
            add_generation_prompt,
            response: tx,
        });
        self.runner.get_sender()?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    /// Detokenize some tokens.
    pub async fn detokenize(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
    ) -> anyhow::Result<String> {
        let (tx, mut rx) = channel(1);
        let request = Request::Detokenize(DetokenizationRequest {
            tokens,
            skip_special_tokens,
            response: tx,
        });
        self.runner.get_sender()?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    /// Retrieve some information about this model.
    pub fn config(&self) -> &MistralRsConfig {
        self.runner.config()
    }

    pub fn inner(&self) -> &MistralRs {
        &self.runner
    }
}
