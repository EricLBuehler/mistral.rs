use anyhow::Context;
use mistralrs_core::*;
use std::sync::Arc;

use crate::Model;

/// A simpler multi-model interface that wraps an existing MistralRs instance
/// and provides methods to interact with multiple loaded models.
pub struct MultiModel {
    runner: Arc<MistralRs>,
}

impl MultiModel {
    /// Create a MultiModel from an existing Model that has multiple models loaded.
    /// This is useful when you've created a Model using regular builders and then
    /// added more models to it using the add_model method.
    pub fn from_model(model: Model) -> Self {
        Self {
            runner: model.runner,
        }
    }

    /// Create a MultiModel directly from a MistralRs instance.
    pub fn from_mistralrs(mistralrs: Arc<MistralRs>) -> Self {
        Self { runner: mistralrs }
    }

    /// List all available model IDs.
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        self.runner.list_models()
    }

    /// Get the default model ID.
    pub fn get_default_model_id(&self) -> Result<Option<String>, String> {
        self.runner.get_default_model_id()
    }

    /// Set the default model ID.
    pub fn set_default_model_id(&self, model_id: &str) -> Result<(), String> {
        self.runner.set_default_model_id(model_id)
    }

    /// Remove a model by ID.
    pub fn remove_model(&self, model_id: &str) -> Result<(), String> {
        self.runner.remove_model(model_id)
    }

    /// Add a new model to the multi-model instance.
    pub async fn add_model(
        &self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: mistralrs_core::AddModelConfig,
    ) -> Result<(), String> {
        self.runner
            .add_model(model_id, pipeline, method, config)
            .await
    }

    /// Send a chat request to a specific model.
    pub async fn send_chat_request_to_model(
        &self,
        mut request: impl crate::RequestLike,
        model_id: Option<&str>,
    ) -> anyhow::Result<ChatCompletionResponse> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

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
        }));

        self.runner.get_sender(model_id)?.send(request).await?;

        let Response::Done(response) =
            rx.recv().await.context("Channel was erroneously closed!")?
        else {
            anyhow::bail!("Got unexpected response, expected `Response::Done`");
        };

        Ok(response)
    }

    /// Get the underlying MistralRs instance.
    pub fn inner(&self) -> &MistralRs {
        &self.runner
    }

    /// Get configuration for a specific model.
    pub fn config(&self, model_id: Option<&str>) -> Result<MistralRsConfig, String> {
        self.runner.config(model_id)
    }
}
