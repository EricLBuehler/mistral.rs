//! Task payload and result types for LLM inference jobs.
//!
//! These types implement the prometheus_parking_lot requirements
//! and define the contract between the scheduler and inference workers.

use crate::{
    response::{ChatCompletionResponse, CompletionResponse},
    sampler::SamplingParams,
    request::{Request, RequestMessage, NormalRequest},
    Constraint, Tool, ToolChoice,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

/// A serializable description of an LLM inference request.
///
/// This struct contains all data needed to process one inference request.
#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceJob {
    /// Unique request ID for tracing and result correlation
    pub request_id: usize,

    /// Whether this is a streaming request
    pub is_streaming: bool,

    /// Request message (chat, completion, etc.)
    #[serde(skip)] // Skip serialization for complex types
    pub messages: Option<RequestMessage>,

    /// Sampling parameters for generation
    #[serde(skip)] // SamplingParams doesn't implement Serialize
    pub sampling_params: Option<SamplingParams>,

    /// Constraint for guided generation
    #[serde(skip)] // Skip serialization for complex types
    pub constraint: Option<Constraint>,

    /// Return logprobs
    pub return_logprobs: bool,

    /// Truncate sequence to fit in max context length
    pub truncate_sequence: bool,

    /// Tools available
    #[serde(skip)]
    pub tools: Option<Vec<Tool>>,

    /// Tool choice
    #[serde(skip)]
    pub tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for InferenceJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceJob")
            .field("request_id", &self.request_id)
            .field("is_streaming", &self.is_streaming)
            .field("return_logprobs", &self.return_logprobs)
            .field("truncate_sequence", &self.truncate_sequence)
            .finish()
    }
}

impl InferenceJob {
    /// Create a new inference job from a NormalRequest.
    #[must_use]
    pub fn from_normal_request(request: &NormalRequest) -> Self {
        Self {
            request_id: request.id,
            is_streaming: request.is_streaming,
            messages: Some(request.messages.clone()),
            sampling_params: Some(request.sampling_params.clone()),
            constraint: Some(request.constraint.clone()),
            return_logprobs: request.return_logprobs,
            truncate_sequence: request.truncate_sequence,
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
        }
    }

    /// Convert back to a Request::Normal for processing.
    pub fn to_request(&self, response_sender: Sender<crate::Response>) -> Request {
        Request::Normal(Box::new(NormalRequest {
            id: self.request_id,
            messages: self.messages.clone().unwrap_or_else(|| {
                RequestMessage::Completion {
                    text: String::new(),
                    echo_prompt: false,
                    best_of: None,
                }
            }),
            sampling_params: self.sampling_params.clone().unwrap_or_default(),
            response: response_sender,
            return_logprobs: self.return_logprobs,
            is_streaming: self.is_streaming,
            constraint: self.constraint.clone().unwrap_or(Constraint::None),
            suffix: None,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            model_id: None,
            truncate_sequence: self.truncate_sequence,
        }))
    }
}

/// A single token in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingTokenResult {
    /// The generated token text
    pub text: String,

    /// Token ID (if available)
    pub token_id: Option<u32>,

    /// Whether this is the final token
    pub is_finished: bool,

    /// Finish reason if finished (e.g., "stop", "length")
    pub finish_reason: Option<String>,

    /// Model identifier
    pub model: String,

    /// Request ID  
    pub id: String,

    /// Created timestamp
    pub created: u64,

    /// Choice index
    pub index: usize,
}

/// Result of an inference job execution.
#[derive(Debug)]
pub enum InferenceResult {
    /// Complete chat completion response
    ChatCompletion {
        response: ChatCompletionResponse,
    },

    /// Complete text completion response  
    Completion {
        response: CompletionResponse,
    },

    /// Streaming response - contains a receiver for tokens
    Streaming {
        /// Request ID for correlation
        request_id: String,
        /// Receiver for streaming chunks
        token_rx: flume::Receiver<Result<StreamingTokenResult, String>>,
    },

    /// Error during inference
    Error {
        /// Error message
        message: String,
    },
}

impl InferenceResult {
    /// Create a chat completion result
    #[must_use]
    pub fn chat_completion(response: ChatCompletionResponse) -> Self {
        Self::ChatCompletion { response }
    }

    /// Create a completion result
    #[must_use]
    pub fn completion(response: CompletionResponse) -> Self {
        Self::Completion { response }
    }

    /// Create a streaming result with the given chunk receiver
    #[must_use]
    pub fn streaming(
        request_id: String,
        token_rx: flume::Receiver<Result<StreamingTokenResult, String>>,
    ) -> Self {
        Self::Streaming {
            request_id,
            token_rx,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Check if this is an error result
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if this is an error result
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

/// Serializable wrapper for InferenceResult (for mailbox storage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableInferenceResult {
    /// Complete chat completion
    ChatCompletion(ChatCompletionResponse),

    /// Complete text completion
    Completion(CompletionResponse),

    /// Streaming response - contains a mailbox key to retrieve the channel
    StreamingChannel {
        /// Request ID for correlation
        request_id: String,
        /// Channel key to retrieve from StreamingRegistry
        channel_key: String,
    },

    /// Error during inference
    Error {
        /// Error message
        message: String,
    },
}

impl SerializableInferenceResult {
    /// Create a chat completion result
    #[must_use]
    pub fn chat_completion(response: ChatCompletionResponse) -> Self {
        Self::ChatCompletion(response)
    }

    /// Create a completion result
    #[must_use]
    pub fn completion(response: CompletionResponse) -> Self {
        Self::Completion(response)
    }

    /// Create a streaming channel result
    #[must_use]
    pub fn streaming_channel(request_id: String, channel_key: String) -> Self {
        Self::StreamingChannel {
            request_id,
            channel_key,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Check if this is an error result
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if this is an error result
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}
