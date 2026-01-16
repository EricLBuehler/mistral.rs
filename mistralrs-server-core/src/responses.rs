//! ## Responses API functionality and route handlers.
//!
//! This module implements the OpenResponses API specification.
//! See: https://www.openresponses.org/

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

/// Convert InputItem types to legacy Message format
fn convert_input_items_to_messages(items: Vec<InputItem>) -> Vec<Message> {
    let mut messages = Vec::new();

    for item in items {
        match item {
            InputItem::Message(msg_param) => {
                let content = match msg_param.content {
                    MessageContentParam::Text(text) => Some(MessageContent::from_text(text)),
                    MessageContentParam::Parts(parts) => {
                        // For now, extract text from parts
                        let mut text_parts = Vec::new();
                        for part in parts {
                            if let crate::responses_types::content::InputContent::InputText {
                                text,
                            } = part
                            {
                                text_parts.push(text);
                            }
                        }
                        if text_parts.is_empty() {
                            None
                        } else {
                            Some(MessageContent::from_text(text_parts.join(" ")))
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
            InputItem::ItemReference { id: _ } => {
                // Item references should be resolved before this point
                // Skip for now - they'll be handled in parse_responses_request
            }
            InputItem::FunctionCall {
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
            InputItem::FunctionCallOutput { call_id, output } => {
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

/// OpenResponses API create request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct OpenResponsesCreateRequest {
    /// The model to use
    #[serde(default = "default_model")]
    pub model: String,
    /// The input for the response
    pub input: OpenResponsesInput,
    /// System instructions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// ID of a previous response for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Whether to run in background
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Whether to store the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// User-provided metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Maximum output tokens
    #[serde(alias = "max_completion_tokens", alias = "max_output_tokens")]
    pub max_tokens: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f64>,
    /// Top-p sampling
    pub top_p: Option<f64>,
    /// Stop sequences
    #[serde(rename = "stop")]
    pub stop_seqs: Option<crate::openai::StopTokens>,
    /// Tool definitions
    pub tools: Option<Vec<mistralrs_core::Tool>>,
    /// Tool choice
    pub tool_choice: Option<mistralrs_core::ToolChoice>,
    /// Response format
    pub response_format: Option<crate::openai::ResponseFormat>,
    /// Logit bias
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Whether to return logprobs
    #[serde(default)]
    pub logprobs: bool,
    /// Top logprobs to return
    pub top_logprobs: Option<usize>,
    /// Number of choices
    #[serde(rename = "n", default = "default_1usize")]
    pub n_choices: usize,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty
    pub repetition_penalty: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Grammar for constrained generation
    pub grammar: Option<crate::openai::Grammar>,
    /// Min-p sampling
    pub min_p: Option<f64>,
    /// DRY multiplier
    pub dry_multiplier: Option<f32>,
    /// DRY base
    pub dry_base: Option<f32>,
    /// DRY allowed length
    pub dry_allowed_length: Option<usize>,
    /// DRY sequence breakers
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Enable thinking mode
    pub enable_thinking: Option<bool>,
    /// Truncate sequence
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
    /// Reasoning effort
    pub reasoning_effort: Option<String>,
    /// Web search options
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
}

impl OpenResponsesStreamer {
    /// Create a new OpenResponses streamer
    pub fn new(
        rx: Receiver<Response>,
        state: SharedMistralRsState,
        response_id: String,
        model: String,
        metadata: Option<Value>,
        store: bool,
        conversation_history: Option<Vec<Message>>,
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
            content_part_added: false,
            output_item_added: false,
            store,
            conversation_history,
            on_done: None,
            events: Vec::new(),
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
                Event::default()
                    .event("response.created")
                    .json_data(event),
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
                        Event::default()
                            .event("response.failed")
                            .json_data(event),
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
                    let all_finished =
                        chat_chunk.choices.iter().all(|c| c.finish_reason.is_some());

                    for choice in &chat_chunk.choices {
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
) -> ResponseResource {
    let created_at = chat_resp.created as u64;
    let mut resource = ResponseResource::new(request_id, chat_resp.model.clone(), created_at);

    let mut output_items = Vec::new();
    let mut output_text_parts = Vec::new();

    for choice in &chat_resp.choices {
        let mut content_items = Vec::new();

        // Handle text content
        if let Some(text) = &choice.message.content {
            output_text_parts.push(text.clone());
            content_items.push(OutputContent::text(text.clone()));
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

    resource
}

/// Parse OpenResponses request into internal format
async fn parse_openresponses_request(
    oairequest: OpenResponsesCreateRequest,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<(Request, bool, Option<Vec<Message>>)> {
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

    // Convert to ChatCompletionRequest
    let chat_request = ChatCompletionRequest {
        messages: Either::Left(final_messages.clone()),
        model: oairequest.model,
        logit_bias: oairequest.logit_bias,
        logprobs: oairequest.logprobs,
        top_logprobs: oairequest.top_logprobs,
        max_tokens: oairequest.max_tokens,
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
        response_format: oairequest.response_format,
        web_search_options: oairequest.web_search_options,
        top_k: oairequest.top_k,
        grammar: oairequest.grammar,
        min_p: oairequest.min_p,
        dry_multiplier: oairequest.dry_multiplier,
        dry_base: oairequest.dry_base,
        dry_allowed_length: oairequest.dry_allowed_length,
        dry_sequence_breakers: oairequest.dry_sequence_breakers,
        enable_thinking: oairequest.enable_thinking,
        truncate_sequence: oairequest.truncate_sequence,
        reasoning_effort: oairequest.reasoning_effort,
    };

    let (request, is_streaming) = parse_chat_request(chat_request, state, tx).await?;
    Ok((request, is_streaming, Some(final_messages)))
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

            let (request, _, conversation_history) =
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

            if let Err(e) = send_request_with_model(&state_clone, request, model_id.as_deref()).await
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
                    task_manager.mark_failed(
                        &task_id,
                        ResponseError::new("model_error", msg.to_string()),
                    );
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

    let (request, is_streaming, conversation_history) =
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
                let response =
                    chat_response_to_response_resource(&chat_resp, request_id.clone(), metadata);

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
