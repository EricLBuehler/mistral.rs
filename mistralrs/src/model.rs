use anyhow::Context;
use candle_core::{Device, Result, Tensor};
use either::Either;
use futures::future::join_all;
use mistralrs_core::*;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver};

use crate::{EmbeddingRequest, EmbeddingRequestBuilder, RequestLike, TextMessages};

// Re-export for convenience
pub use mistralrs_core::{AddModelConfig, ModelStatus, Pipeline, SchedulerConfig};

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
    pub(crate) runner: Arc<MistralRs>,
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

    // ========================================================================
    // Chat Request Methods
    // ========================================================================

    /// Generate with the model (streaming).
    pub async fn stream_chat_request<R: RequestLike>(
        &self,
        request: R,
    ) -> anyhow::Result<Stream<'_>> {
        self.stream_chat_request_with_model(request, None).await
    }

    /// Generate with a specific model (streaming).
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn stream_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> anyhow::Result<Stream<'_>> {
        let (tx, rx) = channel(1);

        let truncate_sequence = request.truncate_sequence();
        let (tools, tool_choice) = if let Some((a, b)) = request.take_tools() {
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let request = Request::Normal(Box::new(NormalRequest {
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
            model_id: model_id.map(|s| s.to_string()),
            truncate_sequence,
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

        let stream = Stream { _server: self, rx };

        Ok(stream)
    }

    /// Generate with the model (non-streaming).
    pub async fn send_chat_request<R: RequestLike>(
        &self,
        request: R,
    ) -> anyhow::Result<ChatCompletionResponse> {
        self.send_chat_request_with_model(request, None).await
    }

    /// Send a chat request to a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn send_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> anyhow::Result<ChatCompletionResponse> {
        let (tx, mut rx) = channel(1);

        let truncate_sequence = request.truncate_sequence();
        let (tools, tool_choice) = if let Some((a, b)) = request.take_tools() {
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let request = Request::Normal(Box::new(NormalRequest {
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
            model_id: model_id.map(|s| s.to_string()),
            truncate_sequence,
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

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
        request: R,
    ) -> anyhow::Result<(Vec<Tensor>, Vec<u32>)> {
        self.send_raw_chat_request_with_model(request, None).await
    }

    /// Generate with a specific model, returning raw logits of the first token generated.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn send_raw_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> anyhow::Result<(Vec<Tensor>, Vec<u32>)> {
        let (tx, mut rx) = channel(1);

        let truncate_sequence = request.truncate_sequence();
        let (tools, tool_choice) = if let Some((a, b)) = request.take_tools() {
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let request = Request::Normal(Box::new(NormalRequest {
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
            model_id: model_id.map(|s| s.to_string()),
            truncate_sequence,
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

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

    // ========================================================================
    // Image Generation Methods
    // ========================================================================

    /// Generate an image using the default model.
    pub async fn generate_image(
        &self,
        prompt: impl ToString,
        response_format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
    ) -> anyhow::Result<ImageGenerationResponse> {
        self.generate_image_with_model(prompt, response_format, generation_params, None)
            .await
    }

    /// Generate an image using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn generate_image_with_model(
        &self,
        prompt: impl ToString,
        response_format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
        model_id: Option<&str>,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
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
            model_id: model_id.map(|s| s.to_string()),
            truncate_sequence: false,
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

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

    // ========================================================================
    // Speech Generation Methods
    // ========================================================================

    /// Generate audio given a (model specific) prompt.
    ///
    /// This returns: (pcm, sampling rate, channels)
    pub async fn generate_speech(
        &self,
        prompt: impl ToString,
    ) -> anyhow::Result<(Arc<Vec<f32>>, usize, usize)> {
        self.generate_speech_with_model(prompt, None).await
    }

    /// Generate audio given a (model specific) prompt using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    ///
    /// This returns: (pcm, sampling rate, channels)
    pub async fn generate_speech_with_model(
        &self,
        prompt: impl ToString,
        model_id: Option<&str>,
    ) -> anyhow::Result<(Arc<Vec<f32>>, usize, usize)> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: 0,
            messages: RequestMessage::SpeechGeneration {
                prompt: prompt.to_string(),
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
            model_id: model_id.map(|s| s.to_string()),
            truncate_sequence: false,
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

        let ResponseOk::Speech {
            pcm,
            rate,
            channels,
        } = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok((pcm, rate, channels))
    }

    // ========================================================================
    // Embedding Methods
    // ========================================================================

    /// Generate embeddings for one or more inputs configured via an [`EmbeddingRequestBuilder`].
    ///
    /// Returns one embedding vector per input in the same order they were added.
    pub async fn generate_embeddings(
        &self,
        request: EmbeddingRequestBuilder,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.generate_embeddings_with_model(request, None).await
    }

    /// Generate embeddings for one or more inputs using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    ///
    /// Returns one embedding vector per input in the same order they were added.
    pub async fn generate_embeddings_with_model(
        &self,
        request: EmbeddingRequestBuilder,
        model_id: Option<&str>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let request = request.build()?;
        let EmbeddingRequest {
            inputs,
            truncate_sequence,
        } = request;

        let runner = self.runner.clone();
        let model_id_owned = model_id.map(|s| s.to_string());
        let futures = inputs.into_iter().map(|input| {
            let runner = runner.clone();
            let model_id_owned = model_id_owned.clone();
            async move {
                let message = input.into_request_message();
                let (tx, mut rx) = channel(1);

                let request = Request::Normal(Box::new(NormalRequest {
                    id: 0,
                    messages: message,
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
                    model_id: model_id_owned.clone(),
                    truncate_sequence,
                }));

                runner
                    .get_sender(model_id_owned.as_deref())?
                    .send(request)
                    .await
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

                let ResponseOk::Embeddings { embeddings, .. } = rx
                    .recv()
                    .await
                    .context("Channel was erroneously closed!")?
                    .as_result()?
                else {
                    anyhow::bail!("Got unexpected response type.")
                };

                Ok::<Vec<f32>, anyhow::Error>(embeddings)
            }
        });

        let results = join_all(futures).await;
        let mut embeddings = Vec::with_capacity(results.len());
        for result in results {
            embeddings.push(result?);
        }
        Ok(embeddings)
    }

    /// Convenience wrapper for generating a single embedding.
    pub async fn generate_embedding(&self, prompt: impl ToString) -> anyhow::Result<Vec<f32>> {
        self.generate_embedding_with_model(prompt, None).await
    }

    /// Convenience wrapper for generating a single embedding using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn generate_embedding_with_model(
        &self,
        prompt: impl ToString,
        model_id: Option<&str>,
    ) -> anyhow::Result<Vec<f32>> {
        let mut embeddings = self
            .generate_embeddings_with_model(
                EmbeddingRequest::builder().add_prompt(prompt.to_string()),
                model_id,
            )
            .await?;

        Ok(embeddings
            .pop()
            .expect("EmbeddingRequestBuilder should guarantee at least one input"))
    }

    // ========================================================================
    // Model Management Methods
    // ========================================================================

    /// Reapply ISQ to the model. This will be done on whatever device the model is already on.
    pub async fn re_isq_model(&self, isq_type: IsqType) -> anyhow::Result<()> {
        self.re_isq_model_with_model(isq_type, None).await
    }

    /// Reapply ISQ to a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn re_isq_model_with_model(
        &self,
        isq_type: IsqType,
        model_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let request = Request::ReIsq(isq_type);

        Ok(self.runner.get_sender(model_id)?.send(request).await?)
    }

    // ========================================================================
    // Tokenization Methods
    // ========================================================================

    /// Tokenize some text or messages.
    /// - `tools` is only used if messages are provided.
    pub async fn tokenize(
        &self,
        text: Either<TextMessages, String>,
        tools: Option<Vec<Tool>>,
        add_special_tokens: bool,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<Vec<u32>> {
        self.tokenize_with_model(
            text,
            tools,
            add_special_tokens,
            add_generation_prompt,
            enable_thinking,
            None,
        )
        .await
    }

    /// Tokenize some text or messages using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    /// - `tools` is only used if messages are provided.
    pub async fn tokenize_with_model(
        &self,
        text: Either<TextMessages, String>,
        tools: Option<Vec<Tool>>,
        add_special_tokens: bool,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        model_id: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        let (tx, mut rx) = channel(1);
        let request = Request::Tokenize(TokenizationRequest {
            text: text.map_left(Into::into),
            tools,
            add_special_tokens,
            add_generation_prompt,
            response: tx,
            enable_thinking,
            reasoning_effort: None,
        });
        self.runner.get_sender(model_id)?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    /// Detokenize some tokens.
    pub async fn detokenize(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
    ) -> anyhow::Result<String> {
        self.detokenize_with_model(tokens, skip_special_tokens, None)
            .await
    }

    /// Detokenize some tokens using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn detokenize_with_model(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
        model_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let (tx, mut rx) = channel(1);
        let request = Request::Detokenize(DetokenizationRequest {
            tokens,
            skip_special_tokens,
            response: tx,
        });
        self.runner.get_sender(model_id)?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    // ========================================================================
    // Configuration Methods
    // ========================================================================

    /// Retrieve some information about this model.
    pub fn config(&self) -> std::result::Result<MistralRsConfig, String> {
        self.config_with_model(None)
    }

    /// Retrieve some information about a specific model.
    /// If `model_id` is `None`, returns config for the default model.
    pub fn config_with_model(
        &self,
        model_id: Option<&str>,
    ) -> std::result::Result<MistralRsConfig, String> {
        self.runner.config(model_id)
    }

    /// Returns the maximum supported sequence length for this model, if applicable.
    pub fn max_sequence_length(&self) -> std::result::Result<Option<usize>, MistralRsError> {
        self.max_sequence_length_with_model(None)
    }

    /// Returns the maximum supported sequence length for a specific model, if applicable.
    /// If `model_id` is `None`, returns for the default model.
    pub fn max_sequence_length_with_model(
        &self,
        model_id: Option<&str>,
    ) -> std::result::Result<Option<usize>, MistralRsError> {
        self.runner.max_sequence_length(model_id)
    }

    // ========================================================================
    // Multi-Model Management Methods
    // ========================================================================

    /// List all available model IDs.
    pub fn list_models(&self) -> std::result::Result<Vec<String>, String> {
        self.runner.list_models()
    }

    /// Get the current default model ID.
    pub fn get_default_model_id(&self) -> std::result::Result<Option<String>, String> {
        self.runner.get_default_model_id()
    }

    /// Set the default model ID.
    pub fn set_default_model_id(&self, model_id: &str) -> std::result::Result<(), String> {
        self.runner.set_default_model_id(model_id)
    }

    /// Add a new model dynamically.
    pub async fn add_model(
        &self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: AddModelConfig,
    ) -> std::result::Result<(), String> {
        self.runner
            .add_model(model_id, pipeline, method, config)
            .await
    }

    /// Remove a model by ID.
    pub fn remove_model(&self, model_id: &str) -> std::result::Result<(), String> {
        self.runner.remove_model(model_id)
    }

    /// Unload a model from memory (can be reloaded later).
    pub fn unload_model(&self, model_id: &str) -> std::result::Result<(), MistralRsError> {
        self.runner.unload_model(model_id)
    }

    /// Reload a previously unloaded model.
    pub async fn reload_model(&self, model_id: &str) -> std::result::Result<(), MistralRsError> {
        self.runner.reload_model(model_id).await
    }

    /// Check if a model is currently loaded.
    pub fn is_model_loaded(&self, model_id: &str) -> std::result::Result<bool, MistralRsError> {
        self.runner.is_model_loaded(model_id)
    }

    /// List all models with their status (Loaded, Unloaded, Reloading).
    pub fn list_models_with_status(
        &self,
    ) -> std::result::Result<Vec<(String, ModelStatus)>, MistralRsError> {
        self.runner.list_models_with_status()
    }

    /// Get the underlying MistralRs instance.
    pub fn inner(&self) -> &MistralRs {
        &self.runner
    }
}
