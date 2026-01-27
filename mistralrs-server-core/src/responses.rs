//! ## Responses API functionality and route handlers.
//!
//! This module implements the OpenResponses API specification.
//! See: <https://www.openresponses.org/>

use std::{
    collections::HashMap,
    pin::Pin,
    task::Poll,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use axum::{
    extract::{Json, Path, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
};
use either::Either;
use mistralrs_core::{ChatCompletionResponse, MistralRs, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{Receiver, Sender};
use utoipa::{
    openapi::{schema::SchemaType, ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};
use uuid::Uuid;

use crate::{
    background_tasks::get_background_task_manager,
    cached_responses::get_response_cache,
    chat_completion::parse_request as parse_chat_request,
    completion_core::{handle_completion_error, BaseCompletionResponder},
    handler_core::{
        create_response_channel, send_request_with_model, BaseJsonModelError, ErrorToResponse,
        JsonError, ModelErrorMessage,
    },
    openai::{ChatCompletionRequest, Message, MessageContent, ToolCall},
    responses_types::{
        content::OutputContent,
        enums::{ItemStatus, ResponseStatus},
        events::StreamingState,
        items::{InputItem, MessageContentParam, OutputItem},
        resource::{ResponseError, ResponseResource, ResponseUsage},
    },
    streaming::{get_keep_alive_interval, DoneState},
    types::{ExtractedMistralRsState, OnDoneCallback, SharedMistralRsState},
    util::sanitize_error_message,
};

/// Input type for OpenResponses API requests
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenResponsesInput {
    /// Simple string input
    Text(String),
    /// Array of input items (OpenResponses format)
    Items(Vec<InputItem>),
}

impl PartialSchema for OpenResponsesInput {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::Type(Type::String))
                        .description(Some("Simple text input"))
                        .build(),
                ))
                .item(Schema::Array(
                    ArrayBuilder::new()
                        .items(InputItem::schema())
                        .description(Some("Array of input items (OpenResponses format)"))
                        .build(),
                ))
                .build(),
        ))
    }
}

impl ToSchema for OpenResponsesInput {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            OpenResponsesInput::name().into(),
            OpenResponsesInput::schema(),
        ));
    }
}

impl OpenResponsesInput {
    /// Convert to Either for internal processing
    pub fn into_either(self) -> Either<Vec<Message>, String> {
        match self {
            OpenResponsesInput::Text(s) => Either::Right(s),
            OpenResponsesInput::Items(items) => {
                let messages = convert_input_items_to_messages(items);
                Either::Left(messages)
            }
        }
    }
}

/// Convert InputItem types to legacy Message format.
///
/// This function handles multimodal content including text, images, audio, and files.
fn convert_input_items_to_messages(items: Vec<InputItem>) -> Vec<Message> {
    use crate::responses_types::content::NormalizedInputContent;
    use crate::responses_types::items::TaggedInputItem;

    let mut messages = Vec::new();

    for item in items {
        // Normalize to TaggedInputItem for uniform processing
        match item.into_tagged() {
            TaggedInputItem::Message(msg_param) => {
                let content = match msg_param.content {
                    MessageContentParam::Text(text) => Some(MessageContent::from_text(text)),
                    MessageContentParam::Parts(parts) => {
                        // Handle multimodal content parts
                        let mut content_parts = Vec::new();
                        let mut has_non_text_content = false;

                        for part in parts {
                            // Normalize the content part to handle both OpenAI and OpenResponses formats
                            match part.into_normalized() {
                                NormalizedInputContent::Text { text } => {
                                    content_parts.push(MessageContent::text_part(text));
                                }
                                NormalizedInputContent::Image {
                                    image_url,
                                    image_data,
                                    detail,
                                } => {
                                    has_non_text_content = true;
                                    // Prefer image_url over image_data
                                    let url = if let Some(url) = image_url {
                                        url
                                    } else if let Some(data) = image_data {
                                        // Convert base64 data to data URL
                                        format!("data:image/png;base64,{}", data)
                                    } else {
                                        continue; // Skip if no image source
                                    };

                                    let image_part = if let Some(detail_level) = detail {
                                        let detail_str = match detail_level {
                                            crate::responses_types::enums::ImageDetail::Auto => {
                                                "auto"
                                            }
                                            crate::responses_types::enums::ImageDetail::Low => {
                                                "low"
                                            }
                                            crate::responses_types::enums::ImageDetail::High => {
                                                "high"
                                            }
                                        };
                                        MessageContent::image_url_part_with_detail(
                                            url,
                                            detail_str.to_string(),
                                        )
                                    } else {
                                        MessageContent::image_url_part(url)
                                    };
                                    content_parts.push(image_part);
                                }
                                NormalizedInputContent::Audio { data, format } => {
                                    has_non_text_content = true;
                                    // Convert audio to data URL format
                                    let mime_type = match format.as_str() {
                                        "wav" => "audio/wav",
                                        "mp3" => "audio/mpeg",
                                        "flac" => "audio/flac",
                                        "ogg" => "audio/ogg",
                                        _ => "audio/wav", // Default to wav
                                    };
                                    let audio_url = format!("data:{};base64,{}", mime_type, data);
                                    // Audio is represented as a special content part
                                    // Note: Not all models support audio input
                                    let mut audio_part = std::collections::HashMap::new();
                                    audio_part.insert(
                                        "type".to_string(),
                                        crate::openai::MessageInnerContent(Either::Left(
                                            "input_audio".to_string(),
                                        )),
                                    );
                                    let mut audio_obj = std::collections::HashMap::new();
                                    audio_obj.insert("data".to_string(), data);
                                    audio_obj.insert("format".to_string(), format);
                                    audio_part.insert(
                                        "input_audio".to_string(),
                                        crate::openai::MessageInnerContent(Either::Right(
                                            audio_obj,
                                        )),
                                    );
                                    content_parts.push(audio_part);
                                    // Also add as text reference for models that don't support audio
                                    content_parts.push(MessageContent::text_part(format!(
                                        "[Audio content: {}]",
                                        audio_url
                                    )));
                                }
                                NormalizedInputContent::File {
                                    file_id,
                                    file_data,
                                    filename,
                                } => {
                                    has_non_text_content = true;
                                    // Files are typically handled as text descriptions or references
                                    let file_ref = if let Some(id) = file_id {
                                        format!("[File reference: {}]", id)
                                    } else if let Some(data) = file_data {
                                        let name =
                                            filename.unwrap_or_else(|| "unnamed_file".to_string());
                                        format!(
                                            "[File: {} (base64 data: {} bytes)]",
                                            name,
                                            data.len()
                                        )
                                    } else if let Some(name) = filename {
                                        format!("[File: {}]", name)
                                    } else {
                                        "[File reference]".to_string()
                                    };
                                    content_parts.push(MessageContent::text_part(file_ref));
                                }
                            }
                        }

                        if content_parts.is_empty() {
                            None
                        } else if !has_non_text_content && content_parts.len() == 1 {
                            // Optimization: if only one text part, use simple text format
                            // Extract text from the first part
                            let first = &content_parts[0];
                            if let Some(text_value) = first.get("text") {
                                if let Either::Left(text) = &**text_value {
                                    Some(MessageContent::from_text(text.clone()))
                                } else {
                                    Some(MessageContent::from_parts(content_parts))
                                }
                            } else {
                                Some(MessageContent::from_parts(content_parts))
                            }
                        } else {
                            Some(MessageContent::from_parts(content_parts))
                        }
                    }
                };

                messages.push(Message {
                    content,
                    role: msg_param.role,
                    name: msg_param.name,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            TaggedInputItem::ItemReference { id: _ } => {
                // Item references should be resolved before this point
                // Skip for now - they'll be handled in parse_responses_request
            }
            TaggedInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                // Convert to assistant message with tool_calls
                messages.push(Message {
                    content: None,
                    role: "assistant".to_string(),
                    name: None,
                    tool_calls: Some(vec![ToolCall {
                        id: Some(call_id),
                        tp: mistralrs_core::ToolType::Function,
                        function: crate::openai::FunctionCalled { name, arguments },
                    }]),
                    tool_call_id: None,
                });
            }
            TaggedInputItem::FunctionCallOutput { call_id, output } => {
                // Convert to tool message
                messages.push(Message {
                    content: Some(MessageContent::from_text(output)),
                    role: "tool".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                });
            }
        }
    }

    messages
}

/// Reasoning configuration for models that support extended thinking
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReasoningConfig {
    /// Effort level for reasoning (low, medium, high)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<crate::responses_types::enums::ReasoningEffort>,
    /// Whether to generate a summary of reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// Reasoning summary configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// Generate a concise summary
    Concise,
    /// Generate a detailed summary
    Detailed,
    /// Auto-select summary level
    Auto,
}

/// Text output configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TextConfig {
    /// Format for text output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

/// Text format configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum TextFormat {
    /// Plain text output
    #[serde(rename = "text")]
    Text,
    /// JSON output with optional schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Name for the schema
        name: String,
        /// JSON Schema definition
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<Value>,
        /// Whether to use strict schema validation
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// JSON object output
    #[serde(rename = "json_object")]
    JsonObject,
}

/// Stream options configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct StreamOptions {
    /// Include usage statistics in stream
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Request context carrying parameters to echo back in the response.
///
/// This struct captures relevant request parameters that should be
/// echoed back in the ResponseResource per the OpenResponses spec.
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// Tool definitions from the request
    pub tools: Option<Vec<mistralrs_core::Tool>>,
    /// Tool choice configuration from the request
    pub tool_choice: Option<mistralrs_core::ToolChoice>,
    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: Option<bool>,
    /// Text configuration from the request
    pub text: Option<TextConfig>,
    /// Temperature from the request
    pub temperature: Option<f64>,
    /// Top-p from the request
    pub top_p: Option<f64>,
    /// Presence penalty from the request
    pub presence_penalty: Option<f32>,
    /// Frequency penalty from the request
    pub frequency_penalty: Option<f32>,
    /// Top logprobs from the request
    pub top_logprobs: Option<usize>,
    /// Max output tokens from the request
    pub max_output_tokens: Option<usize>,
    /// Max tool calls from the request (even if unsupported)
    pub max_tool_calls: Option<usize>,
    /// Whether to store the response
    pub store: Option<bool>,
    /// Whether request runs in background
    pub background: Option<bool>,
}

/// Include options for response content.
///
/// This enum specifies additional content to include in the response.
/// By default, certain content may be omitted for efficiency.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncludeOption {
    /// Include file search results (not currently supported by mistral.rs)
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include message input image URLs in the response
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    /// Include computer call output image URLs (not currently supported by mistral.rs)
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    /// Include reasoning encrypted content
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// Configuration for what to include in the response
#[derive(Debug, Clone, Default)]
pub struct IncludeConfig {
    /// The raw include options from the request
    pub options: Vec<IncludeOption>,
}

impl IncludeConfig {
    /// Create a new IncludeConfig from the request options
    pub fn new(options: Option<Vec<IncludeOption>>) -> Self {
        Self {
            options: options.unwrap_or_default(),
        }
    }

    /// Check if a specific option is included
    pub fn has(&self, option: &IncludeOption) -> bool {
        self.options.contains(option)
    }

    /// Check if reasoning content should be included
    pub fn include_reasoning(&self) -> bool {
        // Reasoning is included by default unless explicitly filtered
        // The ReasoningEncryptedContent option is for encrypted reasoning,
        // which we don't support - regular reasoning is always included
        true
    }
}

/// OpenResponses API create request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct OpenResponsesCreateRequest {
    // ===== Core OpenResponses Fields =====
    /// The model to use for this request
    #[serde(default = "default_model")]
    pub model: String,

    /// The input for the response - can be a string or array of input items
    pub input: OpenResponsesInput,

    /// Additional instructions that guide the model's behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// ID of a previous response for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Whether to stream the response using server-sent events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stream options for controlling streaming behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Whether to run the request in background (async)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Whether to store the response for later retrieval
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// User-provided metadata (up to 16 key-value pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// Specifies additional content to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeOption>>,

    // ===== Generation Parameters =====
    /// Maximum number of output tokens to generate
    #[serde(
        alias = "max_tokens",
        alias = "max_completion_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_output_tokens: Option<usize>,

    /// Temperature for sampling (0-2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p (nucleus) sampling parameter (0-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Number of top log probabilities to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,

    // ===== Tool Calling =====
    /// Tool definitions available for the model to call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<mistralrs_core::Tool>>,

    /// Controls how the model uses tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<mistralrs_core::ToolChoice>,

    /// Whether to allow parallel tool calls.
    ///
    /// NOTE: Only `true` (default) or `None` is supported. Setting this to `false`
    /// will return an error as mistral.rs does not support disabling parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Maximum number of tool calls allowed.
    ///
    /// NOTE: This parameter is not supported. Setting any value will return an error
    /// as mistral.rs does not support limiting the number of tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<usize>,

    // ===== Reasoning =====
    /// Configuration for reasoning/thinking behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    // ===== Output Format =====
    /// Text output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// Truncation strategy when input exceeds context window
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<crate::responses_types::enums::TruncationStrategy>,

    // ===== mistral.rs Extensions (non-standard) =====
    /// Stop sequences to end generation
    #[serde(rename = "stop", skip_serializing_if = "Option::is_none")]
    pub stop_seqs: Option<crate::openai::StopTokens>,

    /// Response format (legacy, prefer `text` field)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::openai::ResponseFormat>,

    /// Logit bias for token manipulation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<u32, f32>>,

    /// Whether to return log probabilities
    #[serde(default)]
    pub logprobs: bool,

    /// Number of completions to generate
    #[serde(rename = "n", default = "default_1usize")]
    pub n_choices: usize,

    /// Repetition penalty (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Top-k sampling (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Grammar for constrained generation (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<crate::openai::Grammar>,

    /// Min-p sampling (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,

    /// DRY multiplier (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,

    /// DRY base (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,

    /// DRY allowed length (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<usize>,

    /// DRY sequence breakers (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_sequence_breakers: Option<Vec<String>>,

    /// Web search options (mistral.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<mistralrs_core::WebSearchOptions>,
}

fn default_model() -> String {
    "default".to_string()
}

fn default_1usize() -> usize {
    1
}

/// OpenResponses streaming event format
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenResponsesStreamEvent {
    /// Response created event
    #[serde(rename = "response.created")]
    ResponseCreated {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response in progress event
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Output item added event
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: OutputItem,
    },
    /// Content part added event
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    /// Text delta event
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    /// Content part done event
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    /// Output item done event
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: OutputItem,
    },
    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        output_index: usize,
        call_id: String,
        delta: String,
    },
    /// Function call arguments done
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        output_index: usize,
        call_id: String,
        arguments: String,
    },
    /// Response completed event
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response failed event
    #[serde(rename = "response.failed")]
    ResponseFailed {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response incomplete event
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Error event
    #[serde(rename = "error")]
    Error {
        sequence_number: u64,
        error: ResponseError,
    },
}

/// OpenResponses streamer that emits proper event types
pub struct OpenResponsesStreamer {
    /// Receiver for responses from the core
    rx: Receiver<Response>,
    /// Done state
    done_state: DoneState,
    /// Shared state
    state: SharedMistralRsState,
    /// Streaming state for tracking events
    streaming_state: StreamingState,
    /// Metadata from the request
    metadata: Option<Value>,
    /// Pending events to emit
    pending_events: Vec<OpenResponsesStreamEvent>,
    /// Accumulated text for the current output
    accumulated_text: String,
    /// Accumulated reasoning for the current output
    accumulated_reasoning: String,
    /// Whether content part has been added
    content_part_added: bool,
    /// Whether output item has been added
    output_item_added: bool,
    /// Store flag
    store: bool,
    /// Conversation history for storage
    conversation_history: Option<Vec<Message>>,
    /// Callback when done
    on_done: Option<OnDoneCallback<OpenResponsesStreamEvent>>,
    /// Collected events for storage
    events: Vec<OpenResponsesStreamEvent>,
    /// Request context for echoing back request parameters
    request_context: RequestContext,
}

impl OpenResponsesStreamer {
    /// Create a new OpenResponses streamer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: Receiver<Response>,
        state: SharedMistralRsState,
        response_id: String,
        model: String,
        metadata: Option<Value>,
        store: bool,
        conversation_history: Option<Vec<Message>>,
        request_context: RequestContext,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            rx,
            done_state: DoneState::Running,
            state,
            streaming_state: StreamingState::new(response_id, model, created_at),
            metadata,
            pending_events: Vec::new(),
            accumulated_text: String::new(),
            accumulated_reasoning: String::new(),
            content_part_added: false,
            output_item_added: false,
            store,
            conversation_history,
            on_done: None,
            events: Vec::new(),
            request_context,
        }
    }

    /// Build initial response resource
    fn build_response_resource(&self, status: ResponseStatus) -> ResponseResource {
        let mut resource = ResponseResource::new(
            self.streaming_state.response_id.clone(),
            self.streaming_state.model.clone(),
            self.streaming_state.created_at,
        );
        resource.status = status;
        resource.metadata = self.metadata.clone();

        // Populate request parameters from context
        resource.tools = self.request_context.tools.clone();
        resource.tool_choice = self.request_context.tool_choice.clone();
        resource.parallel_tool_calls = self.request_context.parallel_tool_calls;
        resource.text = self.request_context.text.clone();
        resource.temperature = self.request_context.temperature;
        resource.top_p = self.request_context.top_p;
        resource.presence_penalty = self.request_context.presence_penalty;
        resource.frequency_penalty = self.request_context.frequency_penalty;
        resource.top_logprobs = self.request_context.top_logprobs;
        resource.max_output_tokens = self.request_context.max_output_tokens;
        resource.max_tool_calls = self.request_context.max_tool_calls;
        resource.store = self.request_context.store;
        resource.background = self.request_context.background;

        resource
    }

    /// Build current response resource with output
    fn build_current_response(&self, status: ResponseStatus) -> ResponseResource {
        let mut resource = self.build_response_resource(status);

        // Build output items from accumulated state
        if !self.accumulated_text.is_empty() {
            let content = vec![OutputContent::text(self.accumulated_text.clone())];
            let item = OutputItem::message(
                format!("msg_{}", Uuid::new_v4()),
                content,
                if status == ResponseStatus::Completed {
                    ItemStatus::Completed
                } else {
                    ItemStatus::InProgress
                },
            );
            resource.output = vec![item];
            resource.output_text = Some(self.accumulated_text.clone());
        }

        // Include reasoning if available
        if !self.accumulated_reasoning.is_empty() {
            resource.reasoning = Some(self.accumulated_reasoning.clone());
        }

        resource
    }
}

impl futures::Stream for OpenResponsesStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Check for pending events first
        if !self.pending_events.is_empty() {
            let event = self.pending_events.remove(0);
            self.events.push(event.clone());
            return Poll::Ready(Some(
                Event::default()
                    .event(get_event_type(&event))
                    .json_data(event),
            ));
        }

        match self.done_state {
            DoneState::SendingDone => {
                self.done_state = DoneState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            DoneState::Done => {
                // Store conversation history if needed
                if self.store {
                    if let Some(history) = self.conversation_history.take() {
                        let cache = get_response_cache();
                        let mut history = history;

                        // Add assistant's response
                        if !self.accumulated_text.is_empty() {
                            history.push(Message {
                                content: Some(MessageContent::from_text(
                                    self.accumulated_text.clone(),
                                )),
                                role: "assistant".to_string(),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }

                        let _ = cache.store_conversation_history(
                            self.streaming_state.response_id.clone(),
                            history,
                        );
                    }
                }

                if let Some(on_done) = &self.on_done {
                    on_done(&self.events);
                }
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }

        // Emit response.created if not sent
        if !self.streaming_state.created_sent {
            self.streaming_state.created_sent = true;
            let seq = self.streaming_state.next_sequence_number();
            let response = self.build_response_resource(ResponseStatus::Queued);
            let event = OpenResponsesStreamEvent::ResponseCreated {
                sequence_number: seq,
                response,
            };
            self.events.push(event.clone());
            return Poll::Ready(Some(
                Event::default().event("response.created").json_data(event),
            ));
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::ModelError(msg, _) => {
                    MistralRs::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.to_string()),
                    );

                    let seq = self.streaming_state.next_sequence_number();
                    let mut response = self.build_current_response(ResponseStatus::Failed);
                    response.error = Some(ResponseError::new("model_error", msg.to_string()));

                    let event = OpenResponsesStreamEvent::ResponseFailed {
                        sequence_number: seq,
                        response,
                    };

                    self.done_state = DoneState::SendingDone;
                    self.events.push(event.clone());
                    Poll::Ready(Some(
                        Event::default().event("response.failed").json_data(event),
                    ))
                }
                Response::ValidationError(e) => {
                    let seq = self.streaming_state.next_sequence_number();
                    let event = OpenResponsesStreamEvent::Error {
                        sequence_number: seq,
                        error: ResponseError::new(
                            "validation_error",
                            sanitize_error_message(e.as_ref()),
                        ),
                    };
                    self.done_state = DoneState::SendingDone;
                    self.events.push(event.clone());
                    Poll::Ready(Some(Event::default().event("error").json_data(event)))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    let seq = self.streaming_state.next_sequence_number();
                    let event = OpenResponsesStreamEvent::Error {
                        sequence_number: seq,
                        error: ResponseError::new(
                            "internal_error",
                            sanitize_error_message(e.as_ref()),
                        ),
                    };
                    self.done_state = DoneState::SendingDone;
                    self.events.push(event.clone());
                    Poll::Ready(Some(Event::default().event("error").json_data(event)))
                }
                Response::Chunk(chat_chunk) => {
                    let mut events_to_emit = Vec::new();

                    // Emit response.in_progress if not sent
                    if !self.streaming_state.in_progress_sent {
                        self.streaming_state.in_progress_sent = true;
                        let seq = self.streaming_state.next_sequence_number();
                        let response = self.build_response_resource(ResponseStatus::InProgress);
                        events_to_emit.push(OpenResponsesStreamEvent::ResponseInProgress {
                            sequence_number: seq,
                            response,
                        });
                    }

                    // Check if all choices are finished
                    let all_finished = chat_chunk.choices.iter().all(|c| c.finish_reason.is_some());

                    for choice in &chat_chunk.choices {
                        // Handle reasoning content
                        if let Some(reasoning) = &choice.delta.reasoning_content {
                            self.accumulated_reasoning.push_str(reasoning);
                        }

                        // Handle text content
                        if let Some(content) = &choice.delta.content {
                            // Emit output_item.added if not done
                            if !self.output_item_added {
                                self.output_item_added = true;
                                let seq = self.streaming_state.next_sequence_number();
                                let item = OutputItem::message(
                                    format!("msg_{}", Uuid::new_v4()),
                                    vec![],
                                    ItemStatus::InProgress,
                                );
                                events_to_emit.push(OpenResponsesStreamEvent::OutputItemAdded {
                                    sequence_number: seq,
                                    output_index: 0,
                                    item,
                                });
                            }

                            // Emit content_part.added if not done
                            if !self.content_part_added {
                                self.content_part_added = true;
                                let seq = self.streaming_state.next_sequence_number();
                                let part = OutputContent::text(String::new());
                                events_to_emit.push(OpenResponsesStreamEvent::ContentPartAdded {
                                    sequence_number: seq,
                                    output_index: 0,
                                    content_index: 0,
                                    part,
                                });
                            }

                            // Accumulate text
                            self.accumulated_text.push_str(content);

                            // Emit text delta
                            let seq = self.streaming_state.next_sequence_number();
                            events_to_emit.push(OpenResponsesStreamEvent::OutputTextDelta {
                                sequence_number: seq,
                                output_index: 0,
                                content_index: 0,
                                delta: content.clone(),
                            });
                        }

                        // Handle tool calls
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tool_call in tool_calls {
                                // Emit function call arguments delta
                                let seq = self.streaming_state.next_sequence_number();
                                events_to_emit.push(
                                    OpenResponsesStreamEvent::FunctionCallArgumentsDelta {
                                        sequence_number: seq,
                                        output_index: 0,
                                        call_id: tool_call.id.clone(),
                                        delta: tool_call.function.arguments.clone(),
                                    },
                                );
                            }
                        }
                    }

                    // If all finished, emit completion events
                    if all_finished {
                        // Emit content_part.done
                        if self.content_part_added {
                            let seq = self.streaming_state.next_sequence_number();
                            let part = OutputContent::text(self.accumulated_text.clone());
                            events_to_emit.push(OpenResponsesStreamEvent::ContentPartDone {
                                sequence_number: seq,
                                output_index: 0,
                                content_index: 0,
                                part,
                            });
                        }

                        // Emit output_item.done
                        if self.output_item_added {
                            let seq = self.streaming_state.next_sequence_number();
                            let content = vec![OutputContent::text(self.accumulated_text.clone())];
                            let item = OutputItem::message(
                                format!("msg_{}", Uuid::new_v4()),
                                content,
                                ItemStatus::Completed,
                            );
                            events_to_emit.push(OpenResponsesStreamEvent::OutputItemDone {
                                sequence_number: seq,
                                output_index: 0,
                                item,
                            });
                        }

                        // Emit response.completed
                        let seq = self.streaming_state.next_sequence_number();
                        let mut response = self.build_current_response(ResponseStatus::Completed);
                        response.completed_at = Some(
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        );

                        // Add usage from chunk if available
                        if let Some(usage) = &chat_chunk.usage {
                            response.usage = Some(ResponseUsage::new(
                                usage.prompt_tokens,
                                usage.completion_tokens,
                            ));
                        }

                        events_to_emit.push(OpenResponsesStreamEvent::ResponseCompleted {
                            sequence_number: seq,
                            response,
                        });

                        self.done_state = DoneState::SendingDone;
                    }

                    MistralRs::maybe_log_response(self.state.clone(), &chat_chunk);

                    // Return first event, queue the rest
                    if !events_to_emit.is_empty() {
                        let first_event = events_to_emit.remove(0);
                        self.pending_events.extend(events_to_emit);
                        self.events.push(first_event.clone());
                        Poll::Ready(Some(
                            Event::default()
                                .event(get_event_type(&first_event))
                                .json_data(first_event),
                        ))
                    } else {
                        Poll::Pending
                    }
                }
                Response::Done(chat_resp) => {
                    // Handle non-streaming completion through chunk path
                    // This shouldn't normally happen in streaming mode
                    let seq = self.streaming_state.next_sequence_number();
                    let response = chat_response_to_response_resource(
                        &chat_resp,
                        self.streaming_state.response_id.clone(),
                        self.metadata.clone(),
                        &self.request_context,
                    );
                    let event = OpenResponsesStreamEvent::ResponseCompleted {
                        sequence_number: seq,
                        response,
                    };
                    self.done_state = DoneState::SendingDone;
                    self.events.push(event.clone());
                    Poll::Ready(Some(
                        Event::default()
                            .event("response.completed")
                            .json_data(event),
                    ))
                }
                _ => Poll::Pending,
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

/// Get the event type string for an event
fn get_event_type(event: &OpenResponsesStreamEvent) -> &'static str {
    match event {
        OpenResponsesStreamEvent::ResponseCreated { .. } => "response.created",
        OpenResponsesStreamEvent::ResponseInProgress { .. } => "response.in_progress",
        OpenResponsesStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
        OpenResponsesStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
        OpenResponsesStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
        OpenResponsesStreamEvent::ContentPartDone { .. } => "response.content_part.done",
        OpenResponsesStreamEvent::OutputItemDone { .. } => "response.output_item.done",
        OpenResponsesStreamEvent::FunctionCallArgumentsDelta { .. } => {
            "response.function_call_arguments.delta"
        }
        OpenResponsesStreamEvent::FunctionCallArgumentsDone { .. } => {
            "response.function_call_arguments.done"
        }
        OpenResponsesStreamEvent::ResponseCompleted { .. } => "response.completed",
        OpenResponsesStreamEvent::ResponseFailed { .. } => "response.failed",
        OpenResponsesStreamEvent::ResponseIncomplete { .. } => "response.incomplete",
        OpenResponsesStreamEvent::Error { .. } => "error",
    }
}

/// Response responder types for OpenResponses API
pub type OpenResponsesResponder =
    BaseCompletionResponder<ResponseResource, KeepAliveStream<OpenResponsesStreamer>>;

type JsonModelError = BaseJsonModelError<ResponseResource>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for OpenResponsesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            OpenResponsesResponder::Sse(s) => s.into_response(),
            OpenResponsesResponder::Json(s) => Json(s).into_response(),
            OpenResponsesResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            OpenResponsesResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            OpenResponsesResponder::ModelError(msg, response) => JsonModelError::new(msg, response)
                .to_response(http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}

/// Convert chat completion response to ResponseResource
fn chat_response_to_response_resource(
    chat_resp: &ChatCompletionResponse,
    request_id: String,
    metadata: Option<Value>,
    request_ctx: &RequestContext,
) -> ResponseResource {
    let created_at = chat_resp.created;
    let mut resource = ResponseResource::new(request_id, chat_resp.model.clone(), created_at);

    let mut output_items = Vec::new();
    let mut output_text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    for choice in &chat_resp.choices {
        let mut content_items = Vec::new();

        // Handle text content
        if let Some(text) = &choice.message.content {
            output_text_parts.push(text.clone());
            content_items.push(OutputContent::text(text.clone()));
        }

        // Handle reasoning content
        if let Some(reasoning) = &choice.message.reasoning_content {
            reasoning_parts.push(reasoning.clone());
        }

        // Handle tool calls - convert to function_call output items
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                let item = OutputItem::function_call(
                    format!("fc_{}", Uuid::new_v4()),
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    tool_call.function.arguments.clone(),
                    ItemStatus::Completed,
                );
                output_items.push(item);
            }
        }

        // Create message output item if there's content
        if !content_items.is_empty() {
            let item = OutputItem::message(
                format!("msg_{}", Uuid::new_v4()),
                content_items,
                ItemStatus::Completed,
            );
            output_items.push(item);
        }
    }

    resource.status = ResponseStatus::Completed;
    resource.output = output_items;
    resource.output_text = if output_text_parts.is_empty() {
        None
    } else {
        Some(output_text_parts.join(""))
    };
    resource.reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join(""))
    };
    resource.usage = Some(ResponseUsage::new(
        chat_resp.usage.prompt_tokens,
        chat_resp.usage.completion_tokens,
    ));
    resource.metadata = metadata;
    resource.completed_at = Some(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );

    // Populate request parameters from context
    resource.tools = request_ctx.tools.clone();
    resource.tool_choice = request_ctx.tool_choice.clone();
    resource.parallel_tool_calls = request_ctx.parallel_tool_calls;
    resource.text = request_ctx.text.clone();
    resource.temperature = request_ctx.temperature;
    resource.top_p = request_ctx.top_p;
    resource.presence_penalty = request_ctx.presence_penalty;
    resource.frequency_penalty = request_ctx.frequency_penalty;
    resource.top_logprobs = request_ctx.top_logprobs;
    resource.max_output_tokens = request_ctx.max_output_tokens;
    resource.max_tool_calls = request_ctx.max_tool_calls;
    resource.store = request_ctx.store;
    resource.background = request_ctx.background;

    resource
}

/// Parse OpenResponses request into internal format
async fn parse_openresponses_request(
    oairequest: OpenResponsesCreateRequest,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<(
    Request,
    bool,
    Option<Vec<Message>>,
    IncludeConfig,
    RequestContext,
)> {
    // Validate unsupported parameters
    // parallel_tool_calls: only `true` (default) or `None` is supported
    if let Some(false) = oairequest.parallel_tool_calls {
        anyhow::bail!(
            "parallel_tool_calls=false is not supported. \
             mistral.rs does not currently support disabling parallel tool calls."
        );
    }

    // max_tool_calls: only `None` (unlimited) is supported
    if oairequest.max_tool_calls.is_some() {
        anyhow::bail!(
            "max_tool_calls is not supported. \
             mistral.rs does not currently support limiting the number of tool calls."
        );
    }

    // Build request context to echo back request parameters
    // Must capture these before consuming oairequest
    let request_context = RequestContext {
        tools: oairequest.tools.clone(),
        tool_choice: oairequest.tool_choice.clone(),
        parallel_tool_calls: oairequest.parallel_tool_calls,
        text: oairequest.text.clone(),
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        top_logprobs: oairequest.top_logprobs,
        max_output_tokens: oairequest.max_output_tokens,
        max_tool_calls: oairequest.max_tool_calls,
        store: oairequest.store,
        background: oairequest.background,
    };

    // Extract include config before consuming oairequest
    let include_config = IncludeConfig::new(oairequest.include.clone());

    // If previous_response_id is provided, get the full conversation history from cache
    let previous_messages = if let Some(prev_id) = &oairequest.previous_response_id {
        let cache = get_response_cache();
        cache.get_conversation_history(prev_id)?
    } else {
        None
    };

    // Get messages from input field
    let messages = oairequest.input.into_either();

    // Build system message from instructions if provided
    let mut final_messages = Vec::new();
    if let Some(instructions) = &oairequest.instructions {
        final_messages.push(Message {
            content: Some(MessageContent::from_text(instructions.clone())),
            role: "system".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Add previous messages if available
    if let Some(prev_msgs) = previous_messages {
        final_messages.extend(prev_msgs);
    }

    // Add current messages
    match messages {
        Either::Left(msgs) => final_messages.extend(msgs),
        Either::Right(prompt) => {
            final_messages.push(Message {
                content: Some(MessageContent::from_text(prompt)),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // Extract reasoning configuration
    let (enable_thinking, reasoning_effort) = if let Some(ref reasoning) = oairequest.reasoning {
        let effort = reasoning.effort.map(|e| match e {
            crate::responses_types::enums::ReasoningEffort::None => "none".to_string(),
            crate::responses_types::enums::ReasoningEffort::Low => "low".to_string(),
            crate::responses_types::enums::ReasoningEffort::Medium => "medium".to_string(),
            crate::responses_types::enums::ReasoningEffort::High => "high".to_string(),
        });
        // Enable thinking if reasoning is configured with any effort level
        let thinking = reasoning
            .effort
            .map(|e| !matches!(e, crate::responses_types::enums::ReasoningEffort::None));
        (thinking, effort)
    } else {
        (None, None)
    };

    // Convert truncation enum to truncate_sequence bool
    let truncate_sequence = oairequest
        .truncation
        .map(|t| matches!(t, crate::responses_types::enums::TruncationStrategy::Auto));

    // Convert OpenResponses `text` field to `response_format`, falling back to legacy field
    let response_format = if let Some(text_config) = oairequest.text {
        text_config.format.map(|fmt| match fmt {
            TextFormat::Text => crate::openai::ResponseFormat::Text,
            TextFormat::JsonSchema {
                name,
                schema,
                strict: _,
            } => crate::openai::ResponseFormat::JsonSchema {
                json_schema: crate::openai::JsonSchemaResponseFormat {
                    name,
                    schema: schema.unwrap_or(serde_json::Value::Object(Default::default())),
                },
            },
            TextFormat::JsonObject => {
                // JsonObject is treated as a schema with empty object
                crate::openai::ResponseFormat::JsonSchema {
                    json_schema: crate::openai::JsonSchemaResponseFormat {
                        name: "json_object".to_string(),
                        schema: serde_json::json!({"type": "object"}),
                    },
                }
            }
        })
    } else {
        oairequest.response_format
    };

    // Convert to ChatCompletionRequest
    let chat_request = ChatCompletionRequest {
        messages: Either::Left(final_messages.clone()),
        model: oairequest.model,
        logit_bias: oairequest.logit_bias,
        logprobs: oairequest.logprobs,
        top_logprobs: oairequest.top_logprobs,
        max_tokens: oairequest.max_output_tokens,
        n_choices: oairequest.n_choices,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        repetition_penalty: oairequest.repetition_penalty,
        stop_seqs: oairequest.stop_seqs,
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        stream: oairequest.stream,
        tools: oairequest.tools,
        tool_choice: oairequest.tool_choice,
        response_format,
        web_search_options: oairequest.web_search_options,
        top_k: oairequest.top_k,
        grammar: oairequest.grammar,
        min_p: oairequest.min_p,
        dry_multiplier: oairequest.dry_multiplier,
        dry_base: oairequest.dry_base,
        dry_allowed_length: oairequest.dry_allowed_length,
        dry_sequence_breakers: oairequest.dry_sequence_breakers,
        enable_thinking,
        truncate_sequence,
        reasoning_effort,
    };

    let (request, is_streaming) = parse_chat_request(chat_request, state, tx).await?;
    Ok((
        request,
        is_streaming,
        Some(final_messages),
        include_config,
        request_context,
    ))
}

/// Create response endpoint - OpenResponses API
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses",
    request_body = OpenResponsesCreateRequest,
    responses((status = 200, description = "Response created"))
)]
pub async fn create_response(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<OpenResponsesCreateRequest>,
) -> OpenResponsesResponder {
    let (tx, rx) = create_response_channel(None);
    let request_id = format!("resp_{}", Uuid::new_v4());
    let metadata = oairequest.metadata.clone();
    let store = oairequest.store.unwrap_or(true);
    let background = oairequest.background.unwrap_or(false);

    // Extract model_id for routing
    let model_id = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let model_name = oairequest.model.clone();

    // Handle background processing
    if background {
        let task_manager = get_background_task_manager();
        let task_id = task_manager.create_task(model_name.clone());

        // Return immediately with queued response
        let response = ResponseResource::new(
            task_id.clone(),
            model_name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
        .with_status(ResponseStatus::Queued)
        .with_metadata(metadata.clone().unwrap_or(Value::Null));

        // Spawn background task
        let state_clone = state.clone();
        let metadata_clone = metadata.clone();
        tokio::spawn(async move {
            let (bg_tx, mut bg_rx) = create_response_channel(None);

            let (request, _, conversation_history, _include_config, request_context) =
                match parse_openresponses_request(oairequest, state_clone.clone(), bg_tx).await {
                    Ok(x) => x,
                    Err(e) => {
                        task_manager.mark_failed(
                            &task_id,
                            ResponseError::new("parse_error", e.to_string()),
                        );
                        return;
                    }
                };

            task_manager.mark_in_progress(&task_id);

            if let Err(e) =
                send_request_with_model(&state_clone, request, model_id.as_deref()).await
            {
                task_manager.mark_failed(&task_id, ResponseError::new("send_error", e.to_string()));
                return;
            }

            // Wait for response
            match bg_rx.recv().await {
                Some(Response::Done(chat_resp)) => {
                    let response = chat_response_to_response_resource(
                        &chat_resp,
                        task_id.clone(),
                        metadata_clone,
                        &request_context,
                    );

                    // Store if requested
                    if store {
                        let cache = get_response_cache();
                        let _ = cache.store_response(task_id.clone(), response.clone());

                        if let Some(mut history) = conversation_history {
                            for choice in &chat_resp.choices {
                                if let Some(content) = &choice.message.content {
                                    history.push(Message {
                                        content: Some(MessageContent::from_text(content.clone())),
                                        role: choice.message.role.clone(),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                    });
                                }
                            }
                            let _ = cache.store_conversation_history(task_id.clone(), history);
                        }
                    }

                    task_manager.mark_completed(&task_id, response);
                }
                Some(Response::ModelError(msg, _partial_resp)) => {
                    task_manager
                        .mark_failed(&task_id, ResponseError::new("model_error", msg.to_string()));
                }
                Some(Response::ValidationError(e)) => {
                    task_manager.mark_failed(
                        &task_id,
                        ResponseError::new("validation_error", e.to_string()),
                    );
                }
                Some(Response::InternalError(e)) => {
                    task_manager.mark_failed(
                        &task_id,
                        ResponseError::new("internal_error", e.to_string()),
                    );
                }
                _ => {
                    task_manager.mark_failed(
                        &task_id,
                        ResponseError::new("unknown_error", "Unexpected response type"),
                    );
                }
            }
        });

        return OpenResponsesResponder::Json(response);
    }

    let (request, is_streaming, conversation_history, _include_config, request_context) =
        match parse_openresponses_request(oairequest, state.clone(), tx).await {
            Ok(x) => x,
            Err(e) => return handle_error(state, e.into()),
        };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        let streamer = OpenResponsesStreamer::new(
            rx,
            state.clone(),
            request_id.clone(),
            model_name,
            metadata,
            store,
            conversation_history,
            request_context,
        );

        let keep_alive_interval = get_keep_alive_interval();
        let sse = Sse::new(streamer)
            .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)));

        OpenResponsesResponder::Sse(sse)
    } else {
        // Non-streaming response
        let mut rx = rx;
        match rx.recv().await {
            Some(Response::Done(chat_resp)) => {
                let response = chat_response_to_response_resource(
                    &chat_resp,
                    request_id.clone(),
                    metadata,
                    &request_context,
                );

                // Store if requested
                if store {
                    let cache = get_response_cache();
                    let _ = cache.store_response(request_id.clone(), response.clone());

                    if let Some(mut history) = conversation_history {
                        for choice in &chat_resp.choices {
                            if let Some(content) = &choice.message.content {
                                history.push(Message {
                                    content: Some(MessageContent::from_text(content.clone())),
                                    role: choice.message.role.clone(),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                });
                            }
                        }
                        let _ = cache.store_conversation_history(request_id, history);
                    }
                }

                OpenResponsesResponder::Json(response)
            }
            Some(Response::ModelError(msg, partial_resp)) => {
                let mut response = chat_response_to_response_resource(
                    &partial_resp,
                    request_id.clone(),
                    metadata,
                    &request_context,
                );
                response.error = Some(ResponseError::new("model_error", msg.to_string()));
                response.status = ResponseStatus::Failed;

                if store {
                    let cache = get_response_cache();
                    let _ = cache.store_response(request_id.clone(), response.clone());
                }

                OpenResponsesResponder::ModelError(msg.to_string(), response)
            }
            Some(Response::ValidationError(e)) => OpenResponsesResponder::ValidationError(e),
            Some(Response::InternalError(e)) => OpenResponsesResponder::InternalError(e),
            _ => OpenResponsesResponder::InternalError(
                anyhow::anyhow!("Unexpected response type").into(),
            ),
        }
    }
}

/// Get response by ID endpoint
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to retrieve")),
    responses((status = 200, description = "Response object"))
)]
pub async fn get_response(
    State(_state): ExtractedMistralRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    // First check background tasks
    let task_manager = get_background_task_manager();
    if let Some(response) = task_manager.get_response(&response_id) {
        return (StatusCode::OK, Json(response)).into_response();
    }

    // Then check cache
    let cache = get_response_cache();
    match cache.get_response(&response_id) {
        Ok(Some(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(None) => JsonError::new(format!("Response with ID '{response_id}' not found"))
            .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error retrieving response: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Delete response by ID endpoint
#[utoipa::path(
    delete,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to delete")),
    responses((status = 200, description = "Response deleted"))
)]
pub async fn delete_response(
    State(_state): ExtractedMistralRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    // Delete from background tasks
    let task_manager = get_background_task_manager();
    let task_deleted = task_manager.delete_task(&response_id);

    // Delete from cache
    let cache = get_response_cache();
    match cache.delete_response(&response_id) {
        Ok(cache_deleted) => {
            if task_deleted || cache_deleted {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "deleted": true,
                        "id": response_id,
                        "object": "response.deleted"
                    })),
                )
                    .into_response()
            } else {
                JsonError::new(format!("Response with ID '{response_id}' not found"))
                    .to_response(StatusCode::NOT_FOUND)
            }
        }
        Err(e) => JsonError::new(format!(
            "Error deleting response: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Cancel response endpoint
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}/cancel",
    params(("response_id" = String, Path, description = "The ID of the response to cancel")),
    responses((status = 200, description = "Response cancelled"))
)]
pub async fn cancel_response(
    State(_state): ExtractedMistralRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    let task_manager = get_background_task_manager();

    if task_manager.request_cancel(&response_id) {
        task_manager.mark_cancelled(&response_id);

        if let Some(response) = task_manager.get_response(&response_id) {
            return (StatusCode::OK, Json(response)).into_response();
        }
    }

    JsonError::new(format!(
        "Response with ID '{response_id}' not found or cannot be cancelled"
    ))
    .to_response(StatusCode::NOT_FOUND)
}

/// Handle errors
fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> OpenResponsesResponder {
    handle_completion_error(state, e)
}
