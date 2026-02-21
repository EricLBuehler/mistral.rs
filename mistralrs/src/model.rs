use candle_core::{Device, Result, Tensor};
use either::Either;
use futures::future::join_all;
use mistralrs_core::*;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::mpsc::{channel, Receiver};

use crate::error::Error as SdkError;
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

/// The object used to interact with the model. This can be used with many varieties of models, \
/// and as such may be created with one of:
/// - [`ModelBuilder`] (auto-detecting)
/// - [`TextModelBuilder`]
/// - [`VisionModelBuilder`]
/// - [`GgufModelBuilder`]
/// - [`EmbeddingModelBuilder`]
/// - [`DiffusionModelBuilder`]
/// - [`SpeechModelBuilder`]
/// - [`LoraModelBuilder`]
/// - [`XLoraModelBuilder`]
/// - [`GgufLoraModelBuilder`]
/// - [`GgufXLoraModelBuilder`]
/// - [`AnyMoeModelBuilder`]
/// - [`TextSpeculativeBuilder`]
///
/// [`ModelBuilder`]: crate::ModelBuilder
/// [`TextModelBuilder`]: crate::TextModelBuilder
/// [`VisionModelBuilder`]: crate::VisionModelBuilder
/// [`GgufModelBuilder`]: crate::GgufModelBuilder
/// [`EmbeddingModelBuilder`]: crate::EmbeddingModelBuilder
/// [`DiffusionModelBuilder`]: crate::DiffusionModelBuilder
/// [`SpeechModelBuilder`]: crate::SpeechModelBuilder
/// [`LoraModelBuilder`]: crate::LoraModelBuilder
/// [`XLoraModelBuilder`]: crate::XLoraModelBuilder
/// [`GgufLoraModelBuilder`]: crate::GgufLoraModelBuilder
/// [`GgufXLoraModelBuilder`]: crate::GgufXLoraModelBuilder
/// [`AnyMoeModelBuilder`]: crate::AnyMoeModelBuilder
/// [`TextSpeculativeBuilder`]: crate::TextSpeculativeBuilder
///
pub struct Model {
    pub(crate) runner: Arc<MistralRs>,
}

/// Token-by-token stream returned by [`Model::stream_chat_request`].
///
/// Implements [`futures::Stream`], so you can use `StreamExt` combinators
/// (e.g., `stream.next().await`).
pub struct Stream<'a> {
    _server: &'a Model,
    rx: Receiver<Response>,
}

impl Stream<'_> {
    /// Receive the next response chunk, or `None` when the stream is exhausted.
    pub async fn next(&mut self) -> Option<Response> {
        self.rx.recv().await
    }

    /// Consume this stream, returning the underlying receiver.
    pub(crate) fn into_receiver(self) -> Receiver<Response> {
        self.rx
    }
}

impl futures::Stream for Stream<'_> {
    type Item = Response;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Model {
    /// Wrap an existing [`MistralRs`] engine instance.
    /// Prefer using a builder (e.g., [`ModelBuilder`](crate::ModelBuilder)) instead.
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
    ) -> crate::error::Result<Stream<'_>> {
        self.stream_chat_request_with_model(request, None).await
    }

    /// Generate with a specific model (streaming).
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn stream_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> crate::error::Result<Stream<'_>> {
        let (tx, rx) = channel(1);

        if let Ok(config) = self.config_with_model(model_id) {
            request.resolve_pending_prefixes(&config.category);
        }
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
    ) -> crate::error::Result<ChatCompletionResponse> {
        self.send_chat_request_with_model(request, None).await
    }

    /// Send a chat request to a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn send_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> crate::error::Result<ChatCompletionResponse> {
        let (tx, mut rx) = channel(1);

        if let Ok(config) = self.config_with_model(model_id) {
            request.resolve_pending_prefixes(&config.category);
        }
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
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .as_result()?
        else {
            return Err(SdkError::UnexpectedResponse { expected: "Done" });
        };

        Ok(response)
    }

    /// Generate with the model, returning raw logits of the first token generated.
    ///
    /// Returns the chunks of the logits (1 or more, determined by prompt batchsize) and the tokens.
    pub async fn send_raw_chat_request<R: RequestLike>(
        &self,
        request: R,
    ) -> crate::error::Result<(Vec<Tensor>, Vec<u32>)> {
        self.send_raw_chat_request_with_model(request, None).await
    }

    /// Generate with a specific model, returning raw logits of the first token generated.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn send_raw_chat_request_with_model<R: RequestLike>(
        &self,
        mut request: R,
        model_id: Option<&str>,
    ) -> crate::error::Result<(Vec<Tensor>, Vec<u32>)> {
        let (tx, mut rx) = channel(1);

        if let Ok(config) = self.config_with_model(model_id) {
            request.resolve_pending_prefixes(&config.category);
        }
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
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .as_result()?
        else {
            return Err(SdkError::UnexpectedResponse { expected: "Raw" });
        };

        Ok((logits_chunks, tokens))
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Quick chat: send a single user message and get the assistant's text reply.
    ///
    /// For more control (system prompt, sampling, tools, etc.), use
    /// [`send_chat_request`](Self::send_chat_request) with a [`RequestBuilder`](crate::RequestBuilder).
    pub async fn chat(&self, message: impl ToString) -> crate::error::Result<String> {
        let messages = TextMessages::new().add_message(crate::TextMessageRole::User, message);
        let response = self.send_chat_request(messages).await?;
        response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or(SdkError::UnexpectedResponse {
                expected: "content",
            })
    }

    /// Send a chat request constrained to a JSON schema derived from `T`, then
    /// deserialize the response into the target type.
    ///
    /// `T` must implement both [`serde::de::DeserializeOwned`] and
    /// [`schemars::JsonSchema`]. The JSON schema is automatically derived from
    /// `T` and used to constrain the model's output.
    ///
    /// # Example
    /// ```no_run
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    /// # use mistralrs::*;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Address {
    ///     street: String,
    ///     city: String,
    ///     state: String,
    ///     zip: u32,
    /// }
    ///
    /// # async fn example(model: Model) -> anyhow::Result<()> {
    /// let address: Address = model
    ///     .generate_structured(
    ///         TextMessages::new()
    ///             .add_message(TextMessageRole::User, "Give me a sample US address."),
    ///     )
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_structured<T>(
        &self,
        messages: impl Into<crate::RequestBuilder>,
    ) -> crate::error::Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        self.generate_structured_with_model::<T>(messages, None)
            .await
    }

    /// Send a structured request to a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn generate_structured_with_model<T>(
        &self,
        messages: impl Into<crate::RequestBuilder>,
        model_id: Option<&str>,
    ) -> crate::error::Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let schema_value = serde_json::to_value(schemars::schema_for!(T))?;
        let request: crate::RequestBuilder = messages.into();
        let request = request.set_constraint(Constraint::JsonSchema(schema_value));
        let response = self.send_chat_request_with_model(request, model_id).await?;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or(SdkError::UnexpectedResponse {
                expected: "content",
            })?;
        Ok(serde_json::from_str(&content)?)
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
        save_file: Option<PathBuf>,
    ) -> crate::error::Result<ImageGenerationResponse> {
        self.generate_image_with_model(prompt, response_format, generation_params, None, save_file)
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
        save_file: Option<PathBuf>,
    ) -> crate::error::Result<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: 0,
            messages: RequestMessage::ImageGeneration {
                prompt: prompt.to_string(),
                format: response_format,
                generation_params,
                save_file,
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
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .as_result()?
        else {
            return Err(SdkError::UnexpectedResponse {
                expected: "ImageGeneration",
            });
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
    ) -> crate::error::Result<(Arc<Vec<f32>>, usize, usize)> {
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
    ) -> crate::error::Result<(Arc<Vec<f32>>, usize, usize)> {
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
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .as_result()?
        else {
            return Err(SdkError::UnexpectedResponse { expected: "Speech" });
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
    ) -> crate::error::Result<Vec<Vec<f32>>> {
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
    ) -> crate::error::Result<Vec<Vec<f32>>> {
        let request = request.build().map_err(|e| SdkError::Inference(e.into()))?;
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
                    .ok_or_else(|| anyhow::anyhow!("channel closed unexpectedly"))?
                    .as_result()
                    .map_err(|e| anyhow::anyhow!(e))?
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
    pub async fn generate_embedding(
        &self,
        prompt: impl ToString,
    ) -> crate::error::Result<Vec<f32>> {
        self.generate_embedding_with_model(prompt, None).await
    }

    /// Convenience wrapper for generating a single embedding using a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn generate_embedding_with_model(
        &self,
        prompt: impl ToString,
        model_id: Option<&str>,
    ) -> crate::error::Result<Vec<f32>> {
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
    pub async fn re_isq_model(&self, isq_type: IsqType) -> crate::error::Result<()> {
        self.re_isq_model_with_model(isq_type, None).await
    }

    /// Reapply ISQ to a specific model.
    /// If `model_id` is `None`, the request is sent to the default model.
    pub async fn re_isq_model_with_model(
        &self,
        isq_type: IsqType,
        model_id: Option<&str>,
    ) -> crate::error::Result<()> {
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
    ) -> crate::error::Result<Vec<u32>> {
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
    ) -> crate::error::Result<Vec<u32>> {
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

        rx.recv()
            .await
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Detokenize some tokens.
    pub async fn detokenize(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
    ) -> crate::error::Result<String> {
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
    ) -> crate::error::Result<String> {
        let (tx, mut rx) = channel(1);
        let request = Request::Detokenize(DetokenizationRequest {
            tokens,
            skip_special_tokens,
            response: tx,
        });
        self.runner.get_sender(model_id)?.send(request).await?;

        rx.recv()
            .await
            .ok_or(SdkError::Channel("channel closed unexpectedly".into()))?
            .map_err(|e| SdkError::Inference(e.into()))
    }

    // ========================================================================
    // Configuration Methods
    // ========================================================================

    /// Retrieve some information about this model.
    pub fn config(&self) -> crate::error::Result<MistralRsConfig> {
        self.config_with_model(None)
    }

    /// Retrieve some information about a specific model.
    /// If `model_id` is `None`, returns config for the default model.
    pub fn config_with_model(
        &self,
        model_id: Option<&str>,
    ) -> crate::error::Result<MistralRsConfig> {
        self.runner
            .config(model_id)
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Returns the maximum supported sequence length for this model, if applicable.
    pub fn max_sequence_length(&self) -> crate::error::Result<Option<usize>> {
        self.max_sequence_length_with_model(None)
    }

    /// Returns the maximum supported sequence length for a specific model, if applicable.
    /// If `model_id` is `None`, returns for the default model.
    pub fn max_sequence_length_with_model(
        &self,
        model_id: Option<&str>,
    ) -> crate::error::Result<Option<usize>> {
        Ok(self.runner.max_sequence_length(model_id)?)
    }

    // ========================================================================
    // Multi-Model Management Methods
    // ========================================================================

    /// List all available model IDs (aliases if configured).
    pub fn list_models(&self) -> crate::error::Result<Vec<String>> {
        self.runner
            .list_models()
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Get the current default model ID.
    pub fn get_default_model_id(&self) -> crate::error::Result<Option<String>> {
        self.runner
            .get_default_model_id()
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Set the default model ID.
    pub fn set_default_model_id(&self, model_id: &str) -> crate::error::Result<()> {
        self.runner
            .set_default_model_id(model_id)
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Add a new model dynamically.
    pub async fn add_model(
        &self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: AddModelConfig,
    ) -> crate::error::Result<()> {
        self.runner
            .add_model(model_id, pipeline, method, config)
            .await
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Remove a model by ID.
    pub fn remove_model(&self, model_id: &str) -> crate::error::Result<()> {
        self.runner
            .remove_model(model_id)
            .map_err(|e| SdkError::Inference(e.into()))
    }

    /// Unload a model from memory (can be reloaded later).
    pub fn unload_model(&self, model_id: &str) -> crate::error::Result<()> {
        Ok(self.runner.unload_model(model_id)?)
    }

    /// Reload a previously unloaded model.
    pub async fn reload_model(&self, model_id: &str) -> crate::error::Result<()> {
        Ok(self.runner.reload_model(model_id).await?)
    }

    /// Check if a model is currently loaded.
    pub fn is_model_loaded(&self, model_id: &str) -> crate::error::Result<bool> {
        Ok(self.runner.is_model_loaded(model_id)?)
    }

    /// List all models with their status (Loaded, Unloaded, Reloading).
    pub fn list_models_with_status(&self) -> crate::error::Result<Vec<(String, ModelStatus)>> {
        Ok(self.runner.list_models_with_status()?)
    }

    /// Get the underlying MistralRs instance.
    pub fn inner(&self) -> &MistralRs {
        &self.runner
    }
}
