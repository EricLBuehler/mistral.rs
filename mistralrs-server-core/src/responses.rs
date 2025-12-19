//! ## Responses API functionality and route handlers.

use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    sync::Arc,
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
use base64::{engine::general_purpose::STANDARD, Engine as _};
use either::Either;
use jsonschema::validator_for;
use mistralrs_core::{
    tools::normalize_tool_calls, ChatCompletionResponse, MistralRs, Request, Response,
};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

use crate::{
    cached_responses::ResponseCache,
    chat_completion::parse_request as parse_chat_request,
    completion_core::{handle_completion_error, BaseCompletionResponder},
    handler_core::{
        create_response_channel, send_request_with_model, BaseJsonModelError, ErrorToResponse,
        JsonError, ModelErrorMessage,
    },
    mistral3_chat::{
        canonicalize_messages_for_mistral3_template_if_needed, looks_like_codex_preamble,
    },
    openai::{
        ChatCompletionRequest, Message, MessageContent, MessageInnerContent, ResponsesContent,
        ResponsesCreateRequest, ResponsesError, ResponsesIncompleteDetails,
        ResponsesInputContentPart, ResponsesInputItem, ResponsesInputMessageContent,
        ResponsesObject, ResponsesOutput, ResponsesTextConfig, ResponsesTextFormat,
        ResponsesTextFormatParam, ResponsesUsage, ToolCall,
    },
    streaming::{get_keep_alive_interval, DoneState},
    types::{
        ExtractedMistralRsState, ExtractedResponseCache, ExtractedSamplingDefaults,
        SamplingDefaults, SharedMistralRsState,
    },
    util::{
        maybe_dump_model_output_json, maybe_dump_prompt_json, maybe_dump_request_json,
        sanitize_error_message,
    },
};

#[derive(Clone)]
struct StrictJsonSchema {
    name: String,
    schema: Value,
}

// Guarded helper: only try to parse Codex-style tool calls when the payload appears
// structurally complete (contains both `{` and `}`). This avoids noisy parse errors on
// bare markers such as `[TOOL_CALLS]shell[ARGS]` that lack JSON.
fn parse_tool_calls_if_complete(
    text: &str,
) -> Option<Vec<mistralrs_core::tools::CalledFunctionParameters>> {
    let looks_complete = text.contains('{') && text.contains('}');
    if !looks_complete {
        return None;
    }
    normalize_tool_calls(text)
}

fn validate_strict_json_schema_output(
    schema: &Value,
    output_text: &str,
) -> std::result::Result<(), String> {
    let trimmed = output_text.trim();
    if trimmed.is_empty() {
        return Err("output was empty".to_string());
    }

    let instance: Value =
        serde_json::from_str(trimmed).map_err(|e| format!("output is not valid JSON: {e}"))?;

    let validator = validator_for(schema).map_err(|e| format!("invalid json_schema: {e}"))?;
    if validator.is_valid(&instance) {
        return Ok(());
    }

    let mut errors = validator
        .iter_errors(&instance)
        .map(|e| e.to_string())
        .take(10)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        errors.push("output does not match schema".to_string());
    }
    Err(errors.join("; "))
}

#[derive(serde::Serialize)]
struct ResponseCreatedEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    response: ResponsesObject,
}

#[derive(serde::Serialize)]
struct OutputItemAddedEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    output_index: usize,
    item: ResponsesOutput,
}

#[derive(serde::Serialize)]
struct ContentPartAddedEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    content_index: usize,
    part: ResponsesContent,
}

#[derive(serde::Serialize)]
struct OutputTextDeltaEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    content_index: usize,
    delta: String,
}

#[derive(serde::Serialize)]
struct OutputTextDoneEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    content_index: usize,
    text: String,
}

#[derive(serde::Serialize)]
struct FunctionCallArgumentsDeltaEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    delta: String,
}

#[derive(serde::Serialize)]
struct FunctionCallArgumentsDoneEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    arguments: String,
}

#[derive(serde::Serialize)]
struct ContentPartDoneEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    item_id: String,
    output_index: usize,
    content_index: usize,
    part: ResponsesContent,
}

#[derive(serde::Serialize)]
struct OutputItemDoneEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    output_index: usize,
    item: ResponsesOutput,
}

#[derive(serde::Serialize)]
struct ResponseCompletedEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    response: ResponsesObject,
}

#[derive(serde::Serialize)]
struct ResponseFailedEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    response: ResponsesObject,
}

#[derive(serde::Serialize)]
struct ResponseInProgressEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    response: ResponsesObject,
}

#[derive(serde::Serialize)]
struct ResponseIncompleteEvent {
    #[serde(rename = "type")]
    tp: &'static str,
    sequence_number: u64,
    response: ResponsesObject,
}

struct FunctionCallState {
    output_index: usize,
    name: String,
    arguments: String,
}

#[derive(serde::Serialize)]
struct DebugResponsesModelOutputDump {
    response_id: String,
    created_at: i64,
    model: String,
    status: String,
    message_text: String,
    output_items: Vec<ResponsesOutput>,
    usage: Option<ResponsesUsage>,
}

pub struct ResponsesStreamer {
    rx: tokio::sync::mpsc::Receiver<Response>,
    state: SharedMistralRsState,
    cache: Arc<dyn ResponseCache>,
    response_id: String,
    created_at: i64,
    model: String,
    input_items: Vec<ResponsesInputItem>,
    instructions: Option<String>,
    metadata: Option<Value>,
    store: bool,
    conversation_history: Option<Vec<Message>>,
    tool_choice: Option<Value>,
    tools: Option<Vec<mistralrs_core::Tool>>,
    parallel_tool_calls: Option<bool>,
    truncation: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    previous_response_id: Option<String>,
    max_output_tokens: Option<usize>,
    max_tool_calls: Option<usize>,
    done_state: DoneState,
    sequence_number: u64,
    queued: VecDeque<Event>,
    sent_created: bool,
    sent_in_progress: bool,
    output_items: Vec<ResponsesOutput>,
    message_item_id: Option<String>,
    message_output_index: Option<usize>,
    message_text: String,
    // Buffer for partial Codex-style tool-call text seen in streaming deltas. We
    // accumulate until it parses cleanly, then emit a structured function_call.
    pending_tool_call_text: String,
    function_calls: HashMap<String, FunctionCallState>,
    web_search_calls: HashMap<String, usize>,
    usage: Option<ResponsesUsage>,
    strict_json_schema: Option<StrictJsonSchema>,
}

impl Drop for ResponsesStreamer {
    fn drop(&mut self) {
        let response_id = self.response_id.clone();

        // If the client disconnects mid-stream, we must cancel the in-flight engine request,
        // otherwise it will keep running and can leak PA blocks / GPU memory.
        if matches!(self.done_state, DoneState::Running) {
            let state = self.state.clone();
            let cache = self.cache.clone();
            if tokio::runtime::Handle::try_current().is_ok() {
                tokio::spawn(async move {
                    if let Ok(Some((engine_id, model_id))) =
                        cache.get_active_request_id(&response_id)
                    {
                        if let Ok(sender) = state.get_sender(model_id.as_deref()) {
                            let _ = sender
                                .send(Request::Cancel {
                                    id: engine_id,
                                    model_id: model_id.clone(),
                                })
                                .await;
                        }
                    }
                    let _ = cache.remove_active_request(&response_id);
                });
                return;
            }
        }

        let _ = self.cache.remove_active_request(&response_id);
    }
}

impl ResponsesStreamer {
    fn maybe_dump_model_output(&self, status: &str) {
        let enabled = std::env::var("MISTRALRS_DEBUG_DUMP_MODEL_OUTPUT")
            .ok()
            .as_deref()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        if !enabled {
            return;
        }

        let dump = DebugResponsesModelOutputDump {
            response_id: self.response_id.clone(),
            created_at: self.created_at,
            model: self.model.clone(),
            status: status.to_string(),
            message_text: self.message_text.clone(),
            output_items: self.output_items.clone(),
            usage: self.usage.clone(),
        };

        let response_id = self.response_id.clone();
        tokio::spawn(async move {
            let _ =
                maybe_dump_model_output_json("responses_model_output", &response_id, &dump).await;
        });
    }

    fn push_json_event<T: serde::Serialize>(
        &mut self,
        event_name: &'static str,
        payload: T,
    ) -> Result<(), axum::Error> {
        let ev = Event::default().event(event_name).json_data(payload)?;
        self.queued.push_back(ev);
        Ok(())
    }

    fn next_seq(&mut self) -> u64 {
        let n = self.sequence_number;
        self.sequence_number = self.sequence_number.saturating_add(1);
        n
    }

    fn in_progress_response(&self) -> ResponsesObject {
        ResponsesObject {
            id: self.response_id.clone(),
            object: "response",
            created_at: self.created_at,
            model: self.model.clone(),
            status: "in_progress".to_string(),
            output: vec![],
            output_text: None,
            usage: None,
            error: None,
            metadata: self.metadata.clone(),
            instructions: self.instructions.clone(),
            incomplete_details: None,
            previous_response_id: self.previous_response_id.clone(),
            store: Some(self.store),
            temperature: self.temperature,
            top_p: self.top_p,
            truncation: self.truncation.clone(),
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            text: Some(ResponsesTextConfig {
                format: ResponsesTextFormat {
                    format_type: "text".to_string(),
                },
            }),
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
        }
    }

    fn ensure_message_started(&mut self) -> Result<(), axum::Error> {
        if self.message_item_id.is_some() {
            return Ok(());
        }

        let item_id = format!("msg_{}", Uuid::new_v4());
        let output_index = self.output_items.len();

        let item = ResponsesOutput {
            id: item_id.clone(),
            output_type: "message".to_string(),
            role: "assistant".to_string(),
            status: Some("in_progress".to_string()),
            content: vec![],
            action: None,
            name: None,
            call_id: None,
            arguments: None,
        };

        self.output_items.push(item.clone());
        self.message_item_id = Some(item_id.clone());
        self.message_output_index = Some(output_index);

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_item.added",
            OutputItemAddedEvent {
                tp: "response.output_item.added",
                sequence_number: seq,
                output_index,
                item,
            },
        )?;

        let seq = self.next_seq();
        self.push_json_event(
            "response.content_part.added",
            ContentPartAddedEvent {
                tp: "response.content_part.added",
                sequence_number: seq,
                item_id,
                output_index,
                content_index: 0,
                part: ResponsesContent {
                    content_type: "output_text".to_string(),
                    text: Some(String::new()),
                    annotations: Some(vec![]),
                },
            },
        )?;

        Ok(())
    }

    fn finalize_message(&mut self) -> Result<(), axum::Error> {
        let Some(item_id) = self.message_item_id.clone() else {
            return Ok(());
        };
        let Some(output_index) = self.message_output_index else {
            return Ok(());
        };

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_text.done",
            OutputTextDoneEvent {
                tp: "response.output_text.done",
                sequence_number: seq,
                item_id: item_id.clone(),
                output_index,
                content_index: 0,
                text: self.message_text.clone(),
            },
        )?;

        let part = ResponsesContent {
            content_type: "output_text".to_string(),
            text: Some(self.message_text.clone()),
            annotations: Some(vec![]),
        };
        let seq = self.next_seq();
        self.push_json_event(
            "response.content_part.done",
            ContentPartDoneEvent {
                tp: "response.content_part.done",
                sequence_number: seq,
                item_id: item_id.clone(),
                output_index,
                content_index: 0,
                part: part.clone(),
            },
        )?;

        let done_item = ResponsesOutput {
            id: item_id.clone(),
            output_type: "message".to_string(),
            role: "assistant".to_string(),
            status: Some("completed".to_string()),
            content: vec![part],
            action: None,
            name: None,
            call_id: None,
            arguments: None,
        };
        if output_index < self.output_items.len() {
            self.output_items[output_index] = done_item.clone();
        }

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_item.done",
            OutputItemDoneEvent {
                tp: "response.output_item.done",
                sequence_number: seq,
                output_index,
                item: done_item,
            },
        )?;
        Ok(())
    }

    fn finalize_function_call(&mut self, item_id: &str) -> Result<(), axum::Error> {
        let Some((output_index, name, arguments)) = self
            .function_calls
            .get(item_id)
            .map(|fc| (fc.output_index, fc.name.clone(), fc.arguments.clone()))
        else {
            return Ok(());
        };

        let seq = self.next_seq();
        self.push_json_event(
            "response.function_call_arguments.done",
            FunctionCallArgumentsDoneEvent {
                tp: "response.function_call_arguments.done",
                sequence_number: seq,
                item_id: item_id.to_string(),
                output_index,
                arguments: arguments.clone(),
            },
        )?;

        let done_item = ResponsesOutput {
            id: item_id.to_string(),
            output_type: "function_call".to_string(),
            role: String::new(),
            status: Some("completed".to_string()),
            content: vec![],
            action: None,
            name: Some(name),
            call_id: Some(item_id.to_string()),
            arguments: Some(arguments),
        };
        if output_index < self.output_items.len() {
            self.output_items[output_index] = done_item.clone();
        }

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_item.done",
            OutputItemDoneEvent {
                tp: "response.output_item.done",
                sequence_number: seq,
                output_index,
                item: done_item,
            },
        )?;

        Ok(())
    }

    fn finalize_web_search_call(&mut self, item_id: &str) -> Result<(), axum::Error> {
        let Some(output_index) = self.web_search_calls.get(item_id).copied() else {
            return Ok(());
        };
        let Some(existing) = self.output_items.get(output_index).cloned() else {
            return Ok(());
        };

        let mut done_item = existing;
        done_item.status = Some("completed".to_string());

        if output_index < self.output_items.len() {
            self.output_items[output_index] = done_item.clone();
        }

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_item.done",
            OutputItemDoneEvent {
                tp: "response.output_item.done",
                sequence_number: seq,
                output_index,
                item: done_item,
            },
        )?;

        self.web_search_calls.remove(item_id);
        Ok(())
    }

    fn emit_buffered_message_done(&mut self) -> Result<(), axum::Error> {
        if self.message_text.trim().is_empty() {
            return Ok(());
        }

        let item_id = format!("msg_{}", Uuid::new_v4());
        let output_index = self.output_items.len();

        let part = ResponsesContent {
            content_type: "output_text".to_string(),
            text: Some(self.message_text.clone()),
            annotations: Some(vec![]),
        };
        let item = ResponsesOutput {
            id: item_id,
            output_type: "message".to_string(),
            role: "assistant".to_string(),
            status: Some("completed".to_string()),
            content: vec![part],
            action: None,
            name: None,
            call_id: None,
            arguments: None,
        };
        self.output_items.push(item.clone());

        let seq = self.next_seq();
        self.push_json_event(
            "response.output_item.done",
            OutputItemDoneEvent {
                tp: "response.output_item.done",
                sequence_number: seq,
                output_index,
                item,
            },
        )?;
        Ok(())
    }

    fn complete_response(&mut self) -> Result<(), axum::Error> {
        // If we buffered a tool call that never parsed, keep it available for salvage.
        if !self.pending_tool_call_text.is_empty() && !self.message_text.contains("[TOOL_CALLS]") {
            self.message_text.push_str(&self.pending_tool_call_text);
        }

        // If no structured function calls were captured but the raw message text contains a
        // Codex-style tool call, try to salvage it here so clients still receive a proper
        // `function_call` output item instead of leaked tool-call text.
        if self.function_calls.is_empty()
            && (self.message_text.contains("[TOOL_CALLS]")
                || !self.pending_tool_call_text.is_empty())
        {
            let raw = if !self.pending_tool_call_text.is_empty() {
                self.pending_tool_call_text.clone()
            } else {
                self.message_text.clone()
            };

            let looks_complete = raw.contains('{') && raw.contains('}');
            if looks_complete {
                if let Some(calls) = parse_tool_calls_if_complete(&raw) {
                    self.message_text.clear();
                    self.pending_tool_call_text.clear();
                    for call in calls {
                        let id = format!("call-{}", Uuid::new_v4());
                        let item = ResponsesOutput {
                            id: id.clone(),
                            output_type: "function_call".to_string(),
                            role: String::new(),
                            status: Some("completed".to_string()),
                            content: vec![],
                            action: None,
                            name: Some(call.name.clone()),
                            call_id: Some(id.clone()),
                            arguments: Some(call.parameters.to_string()),
                        };
                        let output_index = self.output_items.len();
                        self.output_items.push(item.clone());

                        // Emit synthetic output_item events so streaming clients (Codex) notice the
                        // tool call even if the model failed to stream it structurally.
                        let added = OutputItemAddedEvent {
                            tp: "response.output_item.added",
                            sequence_number: self.next_seq(),
                            output_index,
                            item,
                        };
                        self.push_json_event("response.output_item.added", added)?;

                        let done = OutputItemDoneEvent {
                            tp: "response.output_item.done",
                            sequence_number: self.next_seq(),
                            output_index,
                            item: self.output_items[output_index].clone(),
                        };
                        self.push_json_event("response.output_item.done", done)?;
                    }
                } else {
                    // Could not parse even though it looked complete; fall through to plain text.
                    let cleaned = raw.replace("[TOOL_CALLS]", "").replace("[ARGS]", "");
                    self.message_text = cleaned;
                    self.pending_tool_call_text.clear();
                }
            } else {
                // Payload never contained JSON; treat buffered text as plain assistant content so
                // the turn is well-formed and avoids template errors.
                let cleaned = raw.replace("[TOOL_CALLS]", "").replace("[ARGS]", "");
                self.message_text = cleaned;
                self.pending_tool_call_text.clear();
            }
        }

        if let Some(strict) = &self.strict_json_schema {
            let candidate = self.message_text.trim();
            // Strict JSON schema output only applies when the model emits a textual message.
            // Tool-only turns should be allowed (Codex may request follow-ups).
            if !candidate.is_empty() {
                if let Err(err) = validate_strict_json_schema_output(&strict.schema, candidate) {
                    let msg = format!(
                        "Model output did not match text.format json_schema `{}`: {err}",
                        strict.name
                    );
                    self.fail_response_with_param(
                        "invalid_json_schema_output",
                        msg,
                        Some("text.format".to_string()),
                    )?;
                    return Ok(());
                }
            }
        }

        // Flush any in-progress items.
        for item_id in self.function_calls.keys().cloned().collect::<Vec<_>>() {
            self.finalize_function_call(&item_id)?;
        }
        for item_id in self.web_search_calls.keys().cloned().collect::<Vec<_>>() {
            let _ = self.finalize_web_search_call(&item_id);
        }
        if self.strict_json_schema.is_some() && self.message_item_id.is_none() {
            self.emit_buffered_message_done()?;
        } else {
            self.finalize_message()?;
        }

        let response = ResponsesObject {
            id: self.response_id.clone(),
            object: "response",
            created_at: self.created_at,
            model: self.model.clone(),
            status: "completed".to_string(),
            output: self.output_items.clone(),
            output_text: if self.message_text.is_empty() {
                None
            } else {
                Some(self.message_text.clone())
            },
            usage: self.usage.clone(),
            error: None,
            metadata: self.metadata.clone(),
            instructions: self.instructions.clone(),
            incomplete_details: None,
            previous_response_id: self.previous_response_id.clone(),
            store: Some(self.store),
            temperature: self.temperature,
            top_p: self.top_p,
            truncation: self.truncation.clone(),
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            text: Some(ResponsesTextConfig {
                format: ResponsesTextFormat {
                    format_type: "text".to_string(),
                },
            }),
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
        };

        if self.store {
            let _ = self
                .cache
                .store_response(self.response_id.clone(), response.clone());
            let _ = self
                .cache
                .store_input_items(self.response_id.clone(), self.input_items.clone());
            if let Some(mut history) = self.conversation_history.clone() {
                if !self.message_text.is_empty() {
                    history.push(Message {
                        content: Some(MessageContent::from_text(self.message_text.clone())),
                        role: "assistant".to_string(),
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                let _ = self
                    .cache
                    .store_conversation_history(self.response_id.clone(), history);
            }
        }

        self.maybe_dump_model_output("completed");

        let seq = self.next_seq();
        self.push_json_event(
            "response.completed",
            ResponseCompletedEvent {
                tp: "response.completed",
                sequence_number: seq,
                response,
            },
        )?;

        self.done_state = DoneState::Done;
        Ok(())
    }

    fn fail_response(
        &mut self,
        error_type: &'static str,
        message: String,
    ) -> Result<(), axum::Error> {
        self.fail_response_with_param(error_type, message, None)
    }

    fn fail_response_with_param(
        &mut self,
        error_type: &'static str,
        message: String,
        param: Option<String>,
    ) -> Result<(), axum::Error> {
        // Flush any in-progress items so clients don't wait for "done" events that never arrive.
        for item_id in self.function_calls.keys().cloned().collect::<Vec<_>>() {
            let _ = self.finalize_function_call(&item_id);
        }
        for item_id in self.web_search_calls.keys().cloned().collect::<Vec<_>>() {
            let _ = self.finalize_web_search_call(&item_id);
        }
        let _ = self.finalize_message();

        let response = ResponsesObject {
            id: self.response_id.clone(),
            object: "response",
            created_at: self.created_at,
            model: self.model.clone(),
            status: "failed".to_string(),
            output: self.output_items.clone(),
            output_text: if self.message_text.is_empty() {
                None
            } else {
                Some(self.message_text.clone())
            },
            usage: self.usage.clone(),
            error: Some(ResponsesError {
                error_type: error_type.to_string(),
                message,
                param,
                code: Some(error_type.to_string()),
            }),
            metadata: self.metadata.clone(),
            instructions: self.instructions.clone(),
            incomplete_details: None,
            previous_response_id: self.previous_response_id.clone(),
            store: Some(self.store),
            temperature: self.temperature,
            top_p: self.top_p,
            truncation: self.truncation.clone(),
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            text: Some(ResponsesTextConfig {
                format: ResponsesTextFormat {
                    format_type: "text".to_string(),
                },
            }),
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
        };

        if self.store {
            let _ = self
                .cache
                .store_response(self.response_id.clone(), response.clone());
            let _ = self
                .cache
                .store_input_items(self.response_id.clone(), self.input_items.clone());
            if let Some(mut history) = self.conversation_history.clone() {
                if !self.message_text.is_empty() {
                    history.push(Message {
                        content: Some(MessageContent::from_text(self.message_text.clone())),
                        role: "assistant".to_string(),
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                let _ = self
                    .cache
                    .store_conversation_history(self.response_id.clone(), history);
            }
        }

        let seq = self.next_seq();
        self.push_json_event(
            "response.failed",
            ResponseFailedEvent {
                tp: "response.failed",
                sequence_number: seq,
                response,
            },
        )?;

        self.maybe_dump_model_output("failed");

        self.done_state = DoneState::Done;
        Ok(())
    }

    fn incomplete_response(&mut self, reason: &'static str) -> Result<(), axum::Error> {
        // Flush any in-progress items so clients don't wait forever.
        for item_id in self.function_calls.keys().cloned().collect::<Vec<_>>() {
            let _ = self.finalize_function_call(&item_id);
        }
        for item_id in self.web_search_calls.keys().cloned().collect::<Vec<_>>() {
            let _ = self.finalize_web_search_call(&item_id);
        }
        let _ = self.finalize_message();

        let response = ResponsesObject {
            id: self.response_id.clone(),
            object: "response",
            created_at: self.created_at,
            model: self.model.clone(),
            status: "incomplete".to_string(),
            output: self.output_items.clone(),
            output_text: if self.message_text.is_empty() {
                None
            } else {
                Some(self.message_text.clone())
            },
            usage: self.usage.clone(),
            error: None,
            metadata: self.metadata.clone(),
            instructions: self.instructions.clone(),
            incomplete_details: Some(ResponsesIncompleteDetails {
                reason: reason.to_string(),
            }),
            previous_response_id: self.previous_response_id.clone(),
            store: Some(self.store),
            temperature: self.temperature,
            top_p: self.top_p,
            truncation: self.truncation.clone(),
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            text: Some(ResponsesTextConfig {
                format: ResponsesTextFormat {
                    format_type: "text".to_string(),
                },
            }),
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
        };

        if self.store {
            let _ = self
                .cache
                .store_response(self.response_id.clone(), response.clone());
            let _ = self
                .cache
                .store_input_items(self.response_id.clone(), self.input_items.clone());
            if let Some(mut history) = self.conversation_history.clone() {
                if !self.message_text.is_empty() {
                    history.push(Message {
                        content: Some(MessageContent::from_text(self.message_text.clone())),
                        role: "assistant".to_string(),
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                let _ = self
                    .cache
                    .store_conversation_history(self.response_id.clone(), history);
            }
        }

        let seq = self.next_seq();
        self.push_json_event(
            "response.incomplete",
            ResponseIncompleteEvent {
                tp: "response.incomplete",
                sequence_number: seq,
                response,
            },
        )?;

        self.maybe_dump_model_output("incomplete");

        self.done_state = DoneState::Done;
        Ok(())
    }
}

impl futures::Stream for ResponsesStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if let Some(ev) = self.queued.pop_front() {
            return Poll::Ready(Some(Ok(ev)));
        }

        if !self.sent_created {
            self.sent_created = true;
            let payload = ResponseCreatedEvent {
                tp: "response.created",
                sequence_number: self.next_seq(),
                response: self.in_progress_response(),
            };
            if let Err(e) = self.push_json_event("response.created", payload) {
                return Poll::Ready(Some(Err(e)));
            }
            if let Some(ev) = self.queued.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
        }

        if !self.sent_in_progress {
            self.sent_in_progress = true;
            let payload = ResponseInProgressEvent {
                tp: "response.in_progress",
                sequence_number: self.next_seq(),
                response: self.in_progress_response(),
            };
            if let Err(e) = self.push_json_event("response.in_progress", payload) {
                return Poll::Ready(Some(Err(e)));
            }
            if let Some(ev) = self.queued.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
        }

        match self.done_state {
            DoneState::Done => return Poll::Ready(None),
            DoneState::SendingDone => {
                self.done_state = DoneState::Done;
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::ModelError(msg, _) => {
                    MistralRs::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.to_string()),
                    );
                    if let Err(e) = self.fail_response("model_error", msg.to_string()) {
                        return Poll::Ready(Some(Err(e)));
                    }
                    if let Some(ev) = self.queued.pop_front() {
                        Poll::Ready(Some(Ok(ev)))
                    } else {
                        Poll::Ready(None)
                    }
                }
                Response::ValidationError(e) => {
                    let msg = sanitize_error_message(e.as_ref());
                    if let Err(e) = self.fail_response("validation_error", msg) {
                        return Poll::Ready(Some(Err(e)));
                    }
                    if let Some(ev) = self.queued.pop_front() {
                        Poll::Ready(Some(Ok(ev)))
                    } else {
                        Poll::Ready(None)
                    }
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    let msg = sanitize_error_message(e.as_ref());
                    if let Err(e) = self.fail_response("internal_error", msg) {
                        return Poll::Ready(Some(Err(e)));
                    }
                    if let Some(ev) = self.queued.pop_front() {
                        Poll::Ready(Some(Ok(ev)))
                    } else {
                        Poll::Ready(None)
                    }
                }
                Response::WebSearchCall { id, status, action } => {
                    let is_completed = status == "completed";

                    let output_index = if let Some(existing) = self.web_search_calls.get(&id) {
                        *existing
                    } else {
                        let output_index = self.output_items.len();
                        let item = ResponsesOutput {
                            id: id.clone(),
                            output_type: "web_search_call".to_string(),
                            role: String::new(),
                            status: Some(status.clone()),
                            content: vec![],
                            action: Some(action.clone()),
                            name: None,
                            call_id: None,
                            arguments: None,
                        };
                        self.output_items.push(item.clone());
                        self.web_search_calls.insert(id.clone(), output_index);

                        let payload = OutputItemAddedEvent {
                            tp: "response.output_item.added",
                            sequence_number: self.next_seq(),
                            output_index,
                            item,
                        };
                        if let Err(e) = self.push_json_event("response.output_item.added", payload)
                        {
                            return Poll::Ready(Some(Err(e)));
                        }
                        output_index
                    };

                    // Keep the latest action/status on the item.
                    if let Some(existing_item) = self.output_items.get_mut(output_index) {
                        existing_item.action = Some(action);
                        existing_item.status = Some(status);
                    }

                    if is_completed {
                        if let Err(e) = self.finalize_web_search_call(&id) {
                            return Poll::Ready(Some(Err(e)));
                        }
                    }

                    if let Some(ev) = self.queued.pop_front() {
                        Poll::Ready(Some(Ok(ev)))
                    } else {
                        Poll::Pending
                    }
                }
                Response::Chunk(chat_chunk) => {
                    // Streaming responses may include usage on the final chunk; preserve it so
                    // `response.completed` can report accurate token counts (Codex uses this for
                    // its context-left indicator).
                    if let Some(u) = &chat_chunk.usage {
                        self.usage = Some(ResponsesUsage {
                            input_tokens: u.prompt_tokens,
                            output_tokens: u.completion_tokens,
                            total_tokens: u.total_tokens,
                            input_tokens_details: None,
                            output_tokens_details: None,
                        });
                    }

                    // Text deltas
                    for choice in &chat_chunk.choices {
                        if let Some(delta) = &choice.delta.content {
                            // Guardrail: if the model is emitting a Codex-style tool call as
                            // plain text, buffer the text until we have a full JSON payload, then
                            // emit a structured function_call output item. This prevents leaking
                            // tool-call text to the client and avoids template errors when the
                            // assistant message would otherwise be empty.
                            if !self.pending_tool_call_text.is_empty()
                                || delta.contains("[TOOL_CALLS]")
                            {
                                self.pending_tool_call_text.push_str(delta);

                                // Avoid spamming `tool_call_parse_failed` logs on clearly
                                // incomplete payloads (e.g., "[TOOL_CALLS]shell[ARGS]" without any
                                // JSON). Only attempt to parse once we have both braces, which
                                // strongly suggests the model started emitting arguments.

                                if let Some(calls) =
                                    parse_tool_calls_if_complete(&self.pending_tool_call_text)
                                {
                                    for call in calls {
                                        let item_id = format!("call-{}", Uuid::new_v4());
                                        let output_index = self.output_items.len();
                                        let item = ResponsesOutput {
                                            id: item_id.clone(),
                                            output_type: "function_call".to_string(),
                                            role: String::new(),
                                            status: Some("in_progress".to_string()),
                                            content: vec![],
                                            action: None,
                                            name: Some(call.name.clone()),
                                            call_id: Some(item_id.clone()),
                                            arguments: Some(String::new()),
                                        };
                                        self.output_items.push(item.clone());
                                        self.function_calls.insert(
                                            item_id.clone(),
                                            FunctionCallState {
                                                output_index,
                                                name: call.name.clone(),
                                                arguments: call.parameters.to_string(),
                                            },
                                        );
                                        let payload = OutputItemAddedEvent {
                                            tp: "response.output_item.added",
                                            sequence_number: self.next_seq(),
                                            output_index,
                                            item,
                                        };
                                        if let Err(e) = self
                                            .push_json_event("response.output_item.added", payload)
                                        {
                                            return Poll::Ready(Some(Err(e)));
                                        }
                                        // Immediately mark done since we already have full args.
                                        let done = OutputItemDoneEvent {
                                            tp: "response.output_item.done",
                                            sequence_number: self.next_seq(),
                                            output_index,
                                            item: ResponsesOutput {
                                                id: item_id.clone(),
                                                output_type: "function_call".to_string(),
                                                role: String::new(),
                                                status: Some("completed".to_string()),
                                                content: vec![],
                                                action: None,
                                                name: Some(call.name),
                                                call_id: Some(item_id.clone()),
                                                arguments: Some(call.parameters.to_string()),
                                            },
                                        };
                                        if let Err(e) =
                                            self.push_json_event("response.output_item.done", done)
                                        {
                                            return Poll::Ready(Some(Err(e)));
                                        }
                                        // Avoid duplicate finalization later.
                                        self.function_calls.remove(&item_id);
                                    }
                                    self.pending_tool_call_text.clear();
                                    // Parsed successfully; no need to treat as normal text.
                                    continue;
                                }

                                continue;
                            }
                            self.message_text.push_str(delta);

                            // If strict JSON schema output is requested, buffer the whole message
                            // and only emit it once it validates. This avoids streaming partial /
                            // invalid JSON to clients like Codex.
                            if self.strict_json_schema.is_some() {
                                // Still emit an in-progress message item so clients can show
                                // "assistant is responding" without consuming invalid deltas.
                                if let Err(e) = self.ensure_message_started() {
                                    return Poll::Ready(Some(Err(e)));
                                }
                                continue;
                            }

                            if let Err(e) = self.ensure_message_started() {
                                return Poll::Ready(Some(Err(e)));
                            }
                            let output_index = self.message_output_index.unwrap_or(0);
                            let item_id = self.message_item_id.clone().unwrap_or_default();

                            let payload = OutputTextDeltaEvent {
                                tp: "response.output_text.delta",
                                sequence_number: self.next_seq(),
                                item_id,
                                output_index,
                                content_index: 0,
                                delta: delta.clone(),
                            };
                            if let Err(e) =
                                self.push_json_event("response.output_text.delta", payload)
                            {
                                return Poll::Ready(Some(Err(e)));
                            }
                        }

                        // Tool call deltas
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                let item_id = tc.id.clone();
                                let (output_index, name) =
                                    if let Some(existing) = self.function_calls.get(&item_id) {
                                        (existing.output_index, existing.name.clone())
                                    } else {
                                        let output_index = self.output_items.len();
                                        let name = tc.function.name.clone();

                                        let item = ResponsesOutput {
                                            id: item_id.clone(),
                                            output_type: "function_call".to_string(),
                                            role: String::new(),
                                            status: Some("in_progress".to_string()),
                                            content: vec![],
                                            action: None,
                                            name: Some(name.clone()),
                                            call_id: Some(item_id.clone()),
                                            arguments: Some(String::new()),
                                        };
                                        self.output_items.push(item.clone());
                                        self.function_calls.insert(
                                            item_id.clone(),
                                            FunctionCallState {
                                                output_index,
                                                name: name.clone(),
                                                arguments: String::new(),
                                            },
                                        );
                                        let payload = OutputItemAddedEvent {
                                            tp: "response.output_item.added",
                                            sequence_number: self.next_seq(),
                                            output_index,
                                            item,
                                        };
                                        if let Err(e) = self
                                            .push_json_event("response.output_item.added", payload)
                                        {
                                            return Poll::Ready(Some(Err(e)));
                                        }
                                        (output_index, name)
                                    };

                                // Determine delta vs full arguments.
                                let new_args = tc.function.arguments.clone();
                                let current_args = self
                                    .function_calls
                                    .get(&item_id)
                                    .map(|s| s.arguments.clone())
                                    .unwrap_or_default();
                                let delta = if new_args.starts_with(&current_args) {
                                    new_args[current_args.len()..].to_string()
                                } else {
                                    new_args.clone()
                                };
                                if !delta.is_empty() {
                                    if let Some(fc) = self.function_calls.get_mut(&item_id) {
                                        fc.name = name;
                                        fc.arguments = if new_args.starts_with(&fc.arguments) {
                                            new_args.clone()
                                        } else if fc.arguments.is_empty() {
                                            new_args.clone()
                                        } else {
                                            // Fallback: append
                                            format!("{}{}", fc.arguments, delta)
                                        };
                                    }

                                    let payload = FunctionCallArgumentsDeltaEvent {
                                        tp: "response.function_call_arguments.delta",
                                        sequence_number: self.next_seq(),
                                        item_id: item_id.clone(),
                                        output_index,
                                        delta,
                                    };
                                    if let Err(e) = self.push_json_event(
                                        "response.function_call_arguments.delta",
                                        payload,
                                    ) {
                                        return Poll::Ready(Some(Err(e)));
                                    }
                                }
                            }
                        }
                    }

                    if let Some(ev) = self.queued.pop_front() {
                        return Poll::Ready(Some(Ok(ev)));
                    }

                    // Completion detected via finish reasons.
                    let all_finished = chat_chunk.choices.iter().all(|c| c.finish_reason.is_some());
                    if all_finished {
                        let any_length_finish = chat_chunk
                            .choices
                            .iter()
                            .any(|c| c.finish_reason.as_deref() == Some("length"));
                        if any_length_finish {
                            if let Err(e) = self.incomplete_response("max_tokens") {
                                return Poll::Ready(Some(Err(e)));
                            }
                        } else if let Err(e) = self.complete_response() {
                            return Poll::Ready(Some(Err(e)));
                        }
                        if let Some(ev) = self.queued.pop_front() {
                            return Poll::Ready(Some(Ok(ev)));
                        }
                        return Poll::Ready(None);
                    }

                    Poll::Pending
                }
                Response::Done(chat_resp) => {
                    self.usage = Some(ResponsesUsage {
                        input_tokens: chat_resp.usage.prompt_tokens,
                        output_tokens: chat_resp.usage.completion_tokens,
                        total_tokens: chat_resp.usage.total_tokens,
                        input_tokens_details: None,
                        output_tokens_details: None,
                    });
                    if let Err(e) = self.complete_response() {
                        return Poll::Ready(Some(Err(e)));
                    }
                    if let Some(ev) = self.queued.pop_front() {
                        Poll::Ready(Some(Ok(ev)))
                    } else {
                        Poll::Ready(None)
                    }
                }
                _ => Poll::Pending,
            },
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // The engine response channel may close without emitting a terminal `Response::Done`
                // or finish_reason-bearing chunk. For the Responses API, clients (notably Codex)
                // require a terminal `response.completed` SSE event to treat the turn as successful.
                //
                // Fall back to emitting a best-effort `response.completed` using whatever output we
                // have accumulated so far (usage may be missing).
                let has_any_output = !self.message_text.is_empty() || !self.output_items.is_empty();
                if has_any_output {
                    tracing::info!(
                        response_id = %self.response_id,
                        output_chars = self.message_text.len(),
                        output_items = self.output_items.len(),
                        "Engine stream closed without a terminal response; emitting best-effort response.completed"
                    );
                } else {
                    tracing::warn!(
                        response_id = %self.response_id,
                        "Engine stream closed without a terminal response and no output; emitting response.completed"
                    );
                }
                if let Err(e) = self.complete_response() {
                    return Poll::Ready(Some(Err(e)));
                }
                if let Some(ev) = self.queued.pop_front() {
                    Poll::Ready(Some(Ok(ev)))
                } else {
                    Poll::Ready(None)
                }
            }
        }
    }
}

/// Response responder types
pub type ResponsesResponder =
    BaseCompletionResponder<ResponsesObject, KeepAliveStream<ResponsesStreamer>>;

type JsonModelError = BaseJsonModelError<ResponsesObject>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for ResponsesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ResponsesResponder::Sse(s) => s.into_response(),
            ResponsesResponder::Json(s) => Json(s).into_response(),
            ResponsesResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ResponsesResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ResponsesResponder::ModelError(msg, response) => JsonModelError::new(msg, response)
                .to_response(http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}

/// Convert chat completion response to responses object
fn chat_response_to_responses_object(
    chat_resp: &ChatCompletionResponse,
    response_id: String,
    created_at: i64,
    metadata: Option<Value>,
    instructions: Option<String>,
    previous_response_id: Option<String>,
    store: bool,
    temperature: Option<f64>,
    top_p: Option<f64>,
    truncation: Option<String>,
    tool_choice: Option<Value>,
    tools: Option<Vec<mistralrs_core::Tool>>,
    parallel_tool_calls: Option<bool>,
    max_output_tokens: Option<usize>,
    max_tool_calls: Option<usize>,
) -> ResponsesObject {
    let mut outputs = Vec::new();
    let mut output_text_parts = Vec::new();

    for choice in &chat_resp.choices {
        if let Some(text) = &choice.message.content {
            output_text_parts.push(text.clone());
            outputs.push(ResponsesOutput {
                id: format!("msg_{}", Uuid::new_v4()),
                output_type: "message".to_string(),
                role: choice.message.role.clone(),
                status: Some("completed".to_string()),
                content: vec![ResponsesContent {
                    content_type: "output_text".to_string(),
                    text: Some(text.clone()),
                    annotations: Some(vec![]),
                }],
                action: None,
                name: None,
                call_id: None,
                arguments: None,
            });
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                outputs.push(ResponsesOutput {
                    id: tc.id.clone(),
                    output_type: "function_call".to_string(),
                    role: String::new(),
                    status: Some("completed".to_string()),
                    content: vec![],
                    action: None,
                    name: Some(tc.function.name.clone()),
                    call_id: Some(tc.id.clone()),
                    arguments: Some(tc.function.arguments.clone()),
                });
            }
        }
    }

    ResponsesObject {
        id: response_id,
        object: "response",
        created_at,
        model: chat_resp.model.clone(),
        status: "completed".to_string(),
        output: outputs,
        output_text: if output_text_parts.is_empty() {
            None
        } else {
            Some(output_text_parts.join(" "))
        },
        usage: Some(ResponsesUsage {
            input_tokens: chat_resp.usage.prompt_tokens,
            output_tokens: chat_resp.usage.completion_tokens,
            total_tokens: chat_resp.usage.total_tokens,
            input_tokens_details: None,
            output_tokens_details: None,
        }),
        error: None,
        metadata,
        instructions,
        incomplete_details: None,
        previous_response_id,
        store: Some(store),
        temperature,
        top_p,
        truncation,
        tool_choice,
        tools,
        parallel_tool_calls,
        text: Some(ResponsesTextConfig {
            format: ResponsesTextFormat {
                format_type: "text".to_string(),
            },
        }),
        max_output_tokens,
        max_tool_calls,
    }
}

/// Parse responses request into internal format
async fn parse_responses_request(
    mut oairequest: ResponsesCreateRequest,
    state: SharedMistralRsState,
    tx: Sender<Response>,
    cache: Arc<dyn ResponseCache>,
    sampling_defaults: SamplingDefaults,
) -> Result<(
    Request,
    bool,
    Option<Vec<Message>>,
    Option<String>,
    Vec<ResponsesInputItem>,
    Option<StrictJsonSchema>,
)> {
    if oairequest.temperature.is_none() {
        oairequest.temperature = sampling_defaults.temperature;
    }
    if oairequest.top_p.is_none() {
        oairequest.top_p = sampling_defaults.top_p;
    }
    if oairequest.min_p.is_none() {
        oairequest.min_p = sampling_defaults.min_p;
    }
    if oairequest.top_k.is_none() {
        oairequest.top_k = sampling_defaults.top_k;
    }

    // Validate modalities
    if let Some(modalities) = &oairequest.modalities {
        for modality in modalities {
            if modality != "text" {
                anyhow::bail!(
                    "Modality '{modality}' is not currently supported. Only 'text' is supported."
                );
            }
        }
    }

    // Responses API supports a top-level `instructions` string. Some clients omit it but still
    // benefit from a server-side default (e.g. enforce language/format). If set, we prepend the
    // default instructions so client-provided instructions still take precedence.
    let mut instructions = oairequest.instructions.clone();
    if let Ok(default_instructions) = std::env::var("MISTRALRS_DEFAULT_INSTRUCTIONS") {
        let default_instructions = default_instructions.trim().to_string();
        if !default_instructions.is_empty() {
            instructions = Some(match instructions {
                Some(client) if !client.trim().is_empty() => {
                    format!("{default_instructions}\n\n{client}")
                }
                _ => default_instructions,
            });
        }
    }
    // If previous_response_id is provided, get the full conversation history from cache
    let previous_messages = if let Some(prev_id) = &oairequest.previous_response_id {
        let mut history = cache.get_conversation_history(prev_id)?;

        // Handle compacting
        if oairequest.compact_history_with_summary.unwrap_or(false) {
            if let Some(msgs) = &history {
                // Construct prompt
                let mut conversation_text = String::new();
                for msg in msgs {
                    let role = &msg.role;
                    let content = msg
                        .content
                        .as_ref()
                        .and_then(|c| c.to_text())
                        .unwrap_or_default();
                    conversation_text.push_str(&format!("{role}: {content}\n"));
                }

                let summary_prompt = format!("Summarize the following conversation history into a concise paragraph to serve as context for future interactions:\n\n{conversation_text}");

                let (sum_tx, mut sum_rx) = create_response_channel(None);

                let summary_messages = vec![Message {
                    content: Some(MessageContent::from_text(summary_prompt)),
                    role: "user".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                }];

                let summary_request = ChatCompletionRequest {
                    messages: Either::Left(summary_messages),
                    model: oairequest.model.clone(),
                    logit_bias: None,
                    logprobs: false,
                    top_logprobs: None,
                    max_tokens: Some(512), // Limit summary length
                    n_choices: 1,
                    presence_penalty: None,
                    frequency_penalty: None,
                    repetition_penalty: None,
                    stop_seqs: None,
                    temperature: None,
                    top_p: None,
                    stream: Some(false),
                    tools: None,
                    tool_choice: None,
                    response_format: None,
                    web_search_options: None,
                    top_k: None,
                    grammar: None,
                    min_p: None,
                    dry_multiplier: None,
                    dry_base: None,
                    dry_allowed_length: None,
                    dry_sequence_breakers: None,
                    enable_thinking: None,
                    truncate_sequence: None,
                };

                // Extract model_id for routing
                let model_id = if oairequest.model == "default" {
                    None
                } else {
                    Some(oairequest.model.clone())
                };

                let (req, _) = parse_chat_request(
                    summary_request,
                    state.clone(),
                    sum_tx,
                    sampling_defaults.clone(),
                )
                .await?;
                send_request_with_model(&state, req, model_id.as_deref()).await?;

                // Wait for response (avoid hanging forever if the model stalls)
                let summary_resp =
                    match tokio::time::timeout(Duration::from_secs(60), sum_rx.recv()).await {
                        Ok(x) => x,
                        Err(_) => anyhow::bail!("Timed out waiting for summary response"),
                    };
                let summary = match summary_resp {
                    Some(Response::Done(resp)) => {
                        resp.choices[0].message.content.clone().unwrap_or_default()
                    }
                    Some(Response::ModelError(msg, _)) => {
                        anyhow::bail!("Model error during summarization: {}", msg);
                    }
                    _ => anyhow::bail!("Failed to get summary response"),
                };

                // Replace history with summary
                history = Some(vec![Message {
                    content: Some(MessageContent::from_text(format!(
                        "Here is a summary of the previous conversation: {summary}"
                    ))),
                    role: "system".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                }]);
            }
        }
        history
    } else {
        None
    };

    let input_items = oairequest.input.into_items();

    let mut req_messages = Vec::new();
    for item in &input_items {
        match item {
            ResponsesInputItem::Message { role, content, .. } => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut structured_parts: Vec<HashMap<String, MessageInnerContent>> = Vec::new();
                let mut has_media = false;

                let push_text =
                    |text_parts: &mut Vec<String>,
                     structured_parts: &mut Vec<HashMap<String, MessageInnerContent>>,
                     text: &str| {
                        text_parts.push(text.to_string());
                        let mut map = HashMap::new();
                        map.insert(
                            "type".to_string(),
                            MessageInnerContent(Either::Left("text".to_string())),
                        );
                        map.insert(
                            "text".to_string(),
                            MessageInnerContent(Either::Left(text.to_string())),
                        );
                        structured_parts.push(map);
                    };

                let push_image_url =
                    |structured_parts: &mut Vec<HashMap<String, MessageInnerContent>>,
                     url: &str| {
                        let mut map = HashMap::new();
                        map.insert(
                            "type".to_string(),
                            MessageInnerContent(Either::Left("image_url".to_string())),
                        );
                        let mut image_url_obj = HashMap::new();
                        image_url_obj.insert("url".to_string(), Value::String(url.to_string()));
                        map.insert(
                            "image_url".to_string(),
                            MessageInnerContent(Either::Right(image_url_obj)),
                        );
                        structured_parts.push(map);
                    };

                match content {
                    crate::openai::ResponsesInputMessageContent::Text(text) => {
                        push_text(&mut text_parts, &mut structured_parts, text);
                    }
                    crate::openai::ResponsesInputMessageContent::Parts(parts) => {
                        for part in parts {
                            match part {
                                ResponsesInputContentPart::InputText { text }
                                | ResponsesInputContentPart::OutputText { text } => {
                                    push_text(&mut text_parts, &mut structured_parts, text);
                                }
                                ResponsesInputContentPart::InputImage { image_url, file_id } => {
                                    if let Some(url) = image_url.as_ref().map(|u| u.url()) {
                                        has_media = true;
                                        push_image_url(&mut structured_parts, url);
                                    } else if let Some(file_id) = file_id {
                                        anyhow::bail!(
                                            "Responses input_image `file_id` is not supported; use `image_url` instead (URL, file path, or data URL). Got file_id={file_id}"
                                        );
                                    } else {
                                        anyhow::bail!(
                                            "Responses input_image must include either `image_url` or `file_id`."
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // If this looks like the Codex harness preamble, split it out as a system message,
                // leaving the last text part as the real user prompt.
                if role == "user"
                    && !has_media
                    && text_parts.len() >= 2
                    && text_parts
                        .first()
                        .is_some_and(|p| looks_like_codex_preamble(p))
                {
                    let preamble = text_parts[..text_parts.len() - 1].join("\n");
                    let user_text = text_parts[text_parts.len() - 1].clone();
                    if !preamble.trim().is_empty() {
                        req_messages.push(Message {
                            content: Some(MessageContent::from_text(preamble)),
                            role: "system".to_string(),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                    if !user_text.trim().is_empty() {
                        req_messages.push(Message {
                            content: Some(MessageContent::from_text(user_text)),
                            role: "user".to_string(),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                    continue;
                }

                req_messages.push(Message {
                    content: Some(if has_media {
                        MessageContent::from_parts(structured_parts)
                    } else {
                        MessageContent::from_text(text_parts.join("\n"))
                    }),
                    role: role.clone(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                });
            }
            ResponsesInputItem::FunctionCallOutput {
                call_id, output, ..
            }
            | ResponsesInputItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut structured_parts: Vec<HashMap<String, MessageInnerContent>> = Vec::new();
                let mut has_media = false;

                let push_text =
                    |text_parts: &mut Vec<String>,
                     structured_parts: &mut Vec<HashMap<String, MessageInnerContent>>,
                     text: &str| {
                        text_parts.push(text.to_string());
                        let mut map = HashMap::new();
                        map.insert(
                            "type".to_string(),
                            MessageInnerContent(Either::Left("text".to_string())),
                        );
                        map.insert(
                            "text".to_string(),
                            MessageInnerContent(Either::Left(text.to_string())),
                        );
                        structured_parts.push(map);
                    };

                let push_image_url =
                    |structured_parts: &mut Vec<HashMap<String, MessageInnerContent>>,
                     url: &str| {
                        let mut map = HashMap::new();
                        map.insert(
                            "type".to_string(),
                            MessageInnerContent(Either::Left("image_url".to_string())),
                        );
                        let mut image_url_obj = HashMap::new();
                        image_url_obj.insert("url".to_string(), Value::String(url.to_string()));
                        map.insert(
                            "image_url".to_string(),
                            MessageInnerContent(Either::Right(image_url_obj)),
                        );
                        structured_parts.push(map);
                    };

                match output {
                    crate::openai::ResponsesInputMessageContent::Text(text) => {
                        push_text(&mut text_parts, &mut structured_parts, text);
                    }
                    crate::openai::ResponsesInputMessageContent::Parts(parts) => {
                        for part in parts {
                            match part {
                                ResponsesInputContentPart::InputText { text }
                                | ResponsesInputContentPart::OutputText { text } => {
                                    push_text(&mut text_parts, &mut structured_parts, text);
                                }
                                ResponsesInputContentPart::InputImage { image_url, file_id } => {
                                    if let Some(url) = image_url.as_ref().map(|u| u.url()) {
                                        has_media = true;
                                        push_image_url(&mut structured_parts, url);
                                    } else if let Some(file_id) = file_id {
                                        anyhow::bail!(
                                            "Responses input_image `file_id` is not supported; use `image_url` instead (URL, file path, or data URL). Got file_id={file_id}"
                                        );
                                    } else {
                                        anyhow::bail!(
                                            "Responses input_image must include either `image_url` or `file_id`."
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                req_messages.push(Message {
                    content: Some(if has_media {
                        MessageContent::from_parts(structured_parts)
                    } else {
                        MessageContent::from_text(text_parts.join("\n"))
                    }),
                    role: "tool".to_string(),
                    name: None,
                    tool_call_id: Some(call_id.clone()),
                    tool_calls: None,
                });
            }
            ResponsesInputItem::FunctionCall {
                name, arguments, ..
            }
            | ResponsesInputItem::CustomToolCall {
                name,
                input: arguments,
                ..
            } => {
                req_messages.push(Message {
                    content: None,
                    role: "assistant".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Some(vec![ToolCall {
                        tp: mistralrs_core::ToolType::Function,
                        function: crate::openai::FunctionCalled {
                            name: name.clone(),
                            parameters: arguments.clone(),
                        },
                    }]),
                });
            }
            ResponsesInputItem::LocalShellCall {
                call_id, action, ..
            } => {
                let args = action
                    .as_ref()
                    .and_then(|a| serde_json::to_string(a).ok())
                    .unwrap_or_else(|| "{}".to_string());
                req_messages.push(Message {
                    content: None,
                    role: "assistant".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Some(vec![ToolCall {
                        tp: mistralrs_core::ToolType::Function,
                        function: crate::openai::FunctionCalled {
                            name: "local_shell".to_string(),
                            parameters: args,
                        },
                    }]),
                });

                // If the call_id is present, add a synthetic tool output marker so the model can
                // associate subsequent outputs, without requiring strict ID preservation.
                if let Some(cid) = call_id.clone() {
                    req_messages.push(Message {
                        content: Some(MessageContent::from_text(String::new())),
                        role: "tool".to_string(),
                        name: None,
                        tool_call_id: Some(cid),
                        tool_calls: None,
                    });
                }
            }
            ResponsesInputItem::Reasoning { .. }
            | ResponsesInputItem::WebSearchCall { .. }
            | ResponsesInputItem::GhostSnapshot { .. }
            | ResponsesInputItem::Compaction { .. }
            | ResponsesInputItem::Other { .. } => {
                // Accepted for history replay but ignored for prompt-building.
            }
        }
    }

    let response_format_from_text = oairequest.text.as_ref().and_then(|t| {
        let fmt = t.format.as_ref()?;
        match fmt {
            crate::openai::ResponsesTextFormatParam::Text => {
                Some(crate::openai::ResponseFormat::Text)
            }
            crate::openai::ResponsesTextFormatParam::JsonSchema { name, schema, .. } => {
                Some(crate::openai::ResponseFormat::JsonSchema {
                    json_schema: crate::openai::JsonSchemaResponseFormat {
                        name: name.clone(),
                        schema: schema.clone(),
                    },
                })
            }
        }
    });

    let strict_json_schema = oairequest.text.as_ref().and_then(|t| {
        let fmt = t.format.as_ref()?;
        match fmt {
            ResponsesTextFormatParam::JsonSchema {
                name,
                schema,
                strict,
            } if *strict => Some(StrictJsonSchema {
                name: name.clone(),
                schema: schema.clone(),
            }),
            _ => None,
        }
    });

    // Convert to ChatCompletionRequest for reuse
    let mut chat_request = ChatCompletionRequest {
        messages: Either::Left(req_messages),
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
        response_format: oairequest.response_format.or(response_format_from_text),
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
    };

    // Prepend previous messages if available
    if let Some(prev_msgs) = previous_messages {
        match &mut chat_request.messages {
            Either::Left(msgs) => {
                let mut combined = prev_msgs;
                combined.extend(msgs.clone());
                chat_request.messages = Either::Left(combined);
            }
            Either::Right(_) => {
                // If it's a prompt string, convert to messages and prepend
                let prompt = chat_request.messages.as_ref().right().unwrap().clone();
                let mut combined = prev_msgs;
                combined.push(Message {
                    content: Some(MessageContent::from_text(prompt)),
                    role: "user".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                });
                chat_request.messages = Either::Left(combined);
            }
        }
    }

    // Add instructions as system message if provided
    if let Some(instructions) = instructions.clone() {
        let system_msg = Message {
            content: Some(MessageContent::from_text(instructions)),
            role: "system".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        };

        match &mut chat_request.messages {
            Either::Left(msgs) => {
                msgs.insert(0, system_msg);
            }
            Either::Right(prompt) => {
                let mut msgs = vec![system_msg];
                msgs.push(Message {
                    content: Some(MessageContent::from_text(prompt.clone())),
                    role: "user".to_string(),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                });
                chat_request.messages = Either::Left(msgs);
            }
        }
    }

    // Devstral/Mistral3 chat templates require strict alternation of counted user/assistant messages
    // (tool calls/results excluded). Some clients (including Codex) may send multiple consecutive user
    // messages for policy/instructions + the actual prompt, which would otherwise error at template render time.
    if let Either::Left(msgs) = &mut chat_request.messages {
        *msgs = canonicalize_messages_for_mistral3_template_if_needed(std::mem::take(msgs));
    }

    // Get all messages for prompt_details
    let all_messages = match &chat_request.messages {
        Either::Left(msgs) => msgs.clone(),
        Either::Right(prompt) => vec![Message {
            content: Some(MessageContent::from_text(prompt.clone())),
            role: "user".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }],
    };

    let (request, is_streaming) =
        parse_chat_request(chat_request, state, tx, sampling_defaults).await?;
    Ok((
        request,
        is_streaming,
        Some(all_messages),
        instructions,
        input_items,
        strict_json_schema,
    ))
}

/// Create response endpoint
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses",
    request_body = ResponsesCreateRequest,
    responses((status = 200, description = "Response created"))
)]
pub async fn create_response(
    State(state): ExtractedMistralRsState,
    State(cache): ExtractedResponseCache,
    State(sampling_defaults): ExtractedSamplingDefaults,
    Json(mut oairequest): Json<ResponsesCreateRequest>,
) -> ResponsesResponder {
    let (tx, mut rx) = create_response_channel(None);
    let response_id = format!("resp_{}", Uuid::new_v4());
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    if oairequest.temperature.is_none() {
        oairequest.temperature = sampling_defaults.temperature;
    }
    if oairequest.top_p.is_none() {
        oairequest.top_p = sampling_defaults.top_p;
    }
    if oairequest.min_p.is_none() {
        oairequest.min_p = sampling_defaults.min_p;
    }
    if oairequest.top_k.is_none() {
        oairequest.top_k = sampling_defaults.top_k;
    }

    let store = oairequest.store.unwrap_or(true);
    let metadata = oairequest.metadata.clone();
    let previous_response_id = oairequest.previous_response_id.clone();
    let temperature = oairequest.temperature;
    let top_p = oairequest.top_p;
    let max_output_tokens = oairequest.max_tokens;
    let max_tool_calls = oairequest.max_tool_calls;
    let parallel_tool_calls = oairequest.parallel_tool_calls;
    let tools = oairequest.tools.clone();
    let tool_choice = oairequest
        .tool_choice
        .clone()
        .and_then(|tc| serde_json::to_value(tc).ok());
    let truncation = oairequest.truncation.clone();

    maybe_dump_request_json("responses", &response_id, &oairequest).await;
    tracing::info!(
        "responses {} params: temperature={:?} top_p={:?} top_k={:?} min_p={:?} max_output_tokens={:?} max_tool_calls={:?} parallel_tool_calls={:?} truncation={:?}",
        response_id,
        oairequest.temperature,
        oairequest.top_p,
        oairequest.top_k,
        oairequest.min_p,
        oairequest.max_tokens,
        oairequest.max_tool_calls,
        oairequest.parallel_tool_calls,
        truncation
    );

    // Extract model_id for routing
    let model_id = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let (
        request,
        is_streaming,
        conversation_history,
        instructions,
        input_items,
        strict_json_schema,
    ) = match parse_responses_request(
        oairequest,
        state.clone(),
        tx,
        cache.clone(),
        sampling_defaults,
    )
    .await
    {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    maybe_dump_prompt_json("responses_parsed", &response_id, &request).await;

    if store {
        let _ = cache.store_input_items(response_id.clone(), input_items.clone());
    }

    // Store active request mapping if it's a normal request
    if let Request::Normal(ref nr) = request {
        let _ = cache.add_active_request(response_id.clone(), nr.id, model_id.clone());
    }

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        let streamer = ResponsesStreamer {
            rx,
            state: state.clone(),
            cache: cache.clone(),
            response_id: response_id.clone(),
            created_at,
            model: model_id.clone().unwrap_or_else(|| "default".to_string()),
            input_items,
            instructions: instructions.clone(),
            metadata,
            store,
            conversation_history: conversation_history.clone(),
            tool_choice,
            tools,
            parallel_tool_calls,
            truncation,
            temperature,
            top_p,
            previous_response_id,
            max_output_tokens,
            max_tool_calls,
            done_state: DoneState::Running,
            sequence_number: 1,
            queued: VecDeque::new(),
            sent_created: false,
            sent_in_progress: false,
            output_items: Vec::new(),
            message_item_id: None,
            message_output_index: None,
            message_text: String::new(),
            pending_tool_call_text: String::new(),
            function_calls: HashMap::new(),
            web_search_calls: HashMap::new(),
            usage: None,
            strict_json_schema: strict_json_schema.clone(),
        };
        ResponsesResponder::Sse(create_streamer(streamer))
    } else {
        let result = match rx.recv().await {
            Some(Response::Done(chat_resp)) => {
                let mut response_obj = chat_response_to_responses_object(
                    &chat_resp,
                    response_id.clone(),
                    created_at,
                    metadata,
                    instructions.clone(),
                    previous_response_id,
                    store,
                    temperature,
                    top_p,
                    truncation,
                    tool_choice,
                    tools,
                    parallel_tool_calls,
                    max_output_tokens,
                    max_tool_calls,
                );

                if let Some(strict) = &strict_json_schema {
                    // Only validate when the model returns an assistant message. Tool-only turns are valid.
                    if let Some(text) = chat_resp
                        .choices
                        .first()
                        .and_then(|c| c.message.content.as_deref())
                    {
                        if !text.trim().is_empty() {
                            if let Err(err) =
                                validate_strict_json_schema_output(&strict.schema, text)
                            {
                                response_obj.status = "failed".to_string();
                                response_obj.error = Some(ResponsesError {
                                    error_type: "invalid_json_schema_output".to_string(),
                                    message: format!(
                                        "Model output did not match text.format json_schema `{}`: {err}",
                                        strict.name
                                    ),
                                    param: Some("text.format".to_string()),
                                    code: Some("invalid_json_schema_output".to_string()),
                                });
                            }
                        }
                    }
                }

                if store {
                    let _ = cache.store_response(response_id.clone(), response_obj.clone());
                    if let Some(mut history) = conversation_history.clone() {
                        for choice in &chat_resp.choices {
                            history.push(Message {
                                content: choice
                                    .message
                                    .content
                                    .clone()
                                    .map(MessageContent::from_text),
                                role: choice.message.role.clone(),
                                name: None,
                                tool_call_id: None,
                                tool_calls: None,
                            });
                        }
                        let _ = cache.store_conversation_history(response_id.clone(), history);
                    }
                }

                ResponsesResponder::Json(response_obj)
            }
            Some(Response::ModelError(msg, partial_resp)) => {
                let mut response_obj = chat_response_to_responses_object(
                    &partial_resp,
                    response_id.clone(),
                    created_at,
                    metadata,
                    instructions.clone(),
                    previous_response_id,
                    store,
                    temperature,
                    top_p,
                    truncation,
                    tool_choice,
                    tools,
                    parallel_tool_calls,
                    max_output_tokens,
                    max_tool_calls,
                );
                response_obj.error = Some(ResponsesError {
                    error_type: "model_error".to_string(),
                    message: msg.to_string(),
                    param: None,
                    code: Some("model_error".to_string()),
                });
                response_obj.status = "failed".to_string();

                if store {
                    let _ = cache.store_response(response_id.clone(), response_obj.clone());
                }
                ResponsesResponder::ModelError(msg.to_string(), response_obj)
            }
            Some(Response::ValidationError(e)) => ResponsesResponder::ValidationError(e),
            Some(Response::InternalError(e)) => ResponsesResponder::InternalError(e),
            _ => ResponsesResponder::InternalError(
                anyhow::anyhow!("Unexpected response type").into(),
            ),
        };
        let _ = cache.remove_active_request(&response_id);
        result
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct ResponsesCompactRequest {
    #[serde(default = "default_compact_model")]
    pub model: String,
    #[serde(default)]
    pub input: Vec<ResponsesInputItem>,
    #[serde(default)]
    pub instructions: String,
}

#[derive(Debug, serde::Serialize)]
struct ResponsesCompactResponse {
    output: Vec<ResponsesInputItem>,
}

fn responses_input_message_content_to_text(content: &ResponsesInputMessageContent) -> String {
    let mut pieces = Vec::new();
    for part in content.clone().into_parts() {
        match part {
            ResponsesInputContentPart::InputText { text }
            | ResponsesInputContentPart::OutputText { text } => {
                if !text.trim().is_empty() {
                    pieces.push(text);
                }
            }
            ResponsesInputContentPart::InputImage { image_url, file_id } => {
                if let Some(url) = image_url.as_ref().map(|u| u.url().to_string()) {
                    pieces.push(format!("[image: {url}]"));
                } else if let Some(file_id) = file_id {
                    pieces.push(format!("[image file_id: {file_id}]"));
                } else {
                    pieces.push("[image]".to_string());
                }
            }
        }
    }
    pieces.join("\n")
}

fn responses_input_item_to_compaction_line(item: &ResponsesInputItem) -> Option<String> {
    match item {
        ResponsesInputItem::Message { role, content, .. } => {
            let text = responses_input_message_content_to_text(content);
            if text.trim().is_empty() {
                None
            } else {
                Some(format!("{role}: {text}"))
            }
        }
        ResponsesInputItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        } => Some(format!(
            "assistant: [function_call name={name} call_id={call_id} arguments={arguments}]"
        )),
        ResponsesInputItem::FunctionCallOutput {
            call_id, output, ..
        } => {
            let text = responses_input_message_content_to_text(output);
            if text.trim().is_empty() {
                Some(format!("tool: [function_call_output call_id={call_id}]"))
            } else {
                Some(format!(
                    "tool: [function_call_output call_id={call_id}] {text}"
                ))
            }
        }
        ResponsesInputItem::CustomToolCall {
            name,
            call_id,
            input,
            ..
        } => Some(format!(
            "assistant: [custom_tool_call name={name} call_id={call_id} input={input}]"
        )),
        ResponsesInputItem::CustomToolCallOutput {
            call_id, output, ..
        } => {
            let text = responses_input_message_content_to_text(output);
            if text.trim().is_empty() {
                Some(format!("tool: [custom_tool_call_output call_id={call_id}]"))
            } else {
                Some(format!(
                    "tool: [custom_tool_call_output call_id={call_id}] {text}"
                ))
            }
        }
        ResponsesInputItem::LocalShellCall {
            call_id,
            status,
            action,
            ..
        } => Some(format!(
            "tool: [local_shell_call call_id={} status={}] action={}",
            call_id.clone().unwrap_or_else(|| "<none>".to_string()),
            status.clone().unwrap_or_else(|| "<none>".to_string()),
            action
                .as_ref()
                .and_then(|a| serde_json::to_string(a).ok())
                .unwrap_or_else(|| "{}".to_string())
        )),
        ResponsesInputItem::WebSearchCall { status, action, .. } => Some(format!(
            "tool: [web_search_call status={}] action={}",
            status.clone().unwrap_or_else(|| "<none>".to_string()),
            serde_json::to_string(action).unwrap_or_else(|_| "\"<unserializable>\"".to_string())
        )),
        ResponsesInputItem::Reasoning { .. }
        | ResponsesInputItem::GhostSnapshot { .. }
        | ResponsesInputItem::Compaction { .. }
        | ResponsesInputItem::Other { .. } => None,
    }
}

fn build_compaction_conversation_text(
    items: &[ResponsesInputItem],
    max_chars: usize,
) -> (String, usize) {
    let mut lines_rev = Vec::new();
    let mut used = 0usize;
    let mut truncated_items = 0usize;

    for item in items.iter().rev() {
        let Some(line) = responses_input_item_to_compaction_line(item) else {
            continue;
        };
        let add_len = line.len().saturating_add(1);
        if used.saturating_add(add_len) > max_chars {
            truncated_items = truncated_items.saturating_add(1);
            continue;
        }
        used = used.saturating_add(add_len);
        lines_rev.push(line);
    }

    lines_rev.reverse();
    let mut text = lines_rev.join("\n");
    if truncated_items > 0 {
        text = format!("[... truncated {truncated_items} older item(s) ...]\n{text}");
    }
    (text, truncated_items)
}

const DEFAULT_COMPACT_MAX_CHARS: usize = 200_000;

fn default_compact_model() -> String {
    "default".to_string()
}

const COMPACT_SYSTEM_PROMPT: &str = r#"You compact conversation history for a coding assistant.

Task:
- Read the transcript and produce a concise summary that preserves: current goal, key decisions, important constraints, file paths or commands mentioned, and any unresolved questions.
- Be factual; do not invent details.
- Do not mention that you are summarizing.
- Output plain text only (no markdown), unless the transcript itself requires code blocks/paths."#;

/// Compact response history endpoint (Codex/OpenAI-style).
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses/compact",
    request_body = serde_json::Value,
    responses((status = 200, description = "Compacted history"))
)]
pub async fn compact_response(
    State(state): ExtractedMistralRsState,
    State(sampling_defaults): ExtractedSamplingDefaults,
    Json(compact_req): Json<ResponsesCompactRequest>,
) -> impl IntoResponse {
    let instructions_len = compact_req.instructions.trim().len();
    if instructions_len > 0 {
        tracing::debug!("responses/compact received instructions (chars={instructions_len})");
    }
    let max_chars = std::env::var("MISTRALRS_COMPACT_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_COMPACT_MAX_CHARS);

    let (conversation_text, truncated_items) =
        build_compaction_conversation_text(&compact_req.input, max_chars);

    // Best-effort: if the transcript is empty after filtering, just return a no-op compaction item.
    if conversation_text.trim().is_empty() {
        let encrypted_content = STANDARD.encode(b"");
        let output = vec![
            ResponsesInputItem::Message {
                id: format!("msg_{}", Uuid::new_v4()),
                role: "user".to_string(),
                content: ResponsesInputMessageContent::Parts(vec![
                    ResponsesInputContentPart::InputText {
                        text: String::new(),
                    },
                ]),
            },
            ResponsesInputItem::Compaction {
                id: format!("cmp_{}", Uuid::new_v4()),
                encrypted_content,
            },
        ];
        return (StatusCode::OK, Json(ResponsesCompactResponse { output })).into_response();
    }

    let (sum_tx, mut sum_rx) = create_response_channel(None);

    let messages = vec![
        Message {
            content: Some(MessageContent::from_text(COMPACT_SYSTEM_PROMPT.to_string())),
            role: "system".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            content: Some(MessageContent::from_text(conversation_text.clone())),
            role: "user".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let summary_request = ChatCompletionRequest {
        messages: Either::Left(messages),
        model: compact_req.model.clone(),
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        max_tokens: Some(1024),
        n_choices: 1,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        stop_seqs: None,
        temperature: Some(0.2),
        top_p: None,
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: None,
        web_search_options: None,
        top_k: None,
        grammar: None,
        min_p: None,
        dry_multiplier: None,
        dry_base: None,
        dry_allowed_length: None,
        dry_sequence_breakers: None,
        enable_thinking: None,
        truncate_sequence: None,
    };

    let model_id = if compact_req.model == "default" {
        None
    } else {
        Some(compact_req.model.clone())
    };

    let summary = async {
        let (req, _) =
            parse_chat_request(summary_request, state.clone(), sum_tx, sampling_defaults).await?;
        send_request_with_model(&state, req, model_id.as_deref()).await?;
        let resp = tokio::time::timeout(Duration::from_secs(120), sum_rx.recv()).await?;
        let Some(resp) = resp else {
            return Err(anyhow::anyhow!("No compaction response received"));
        };
        match resp {
            Response::Done(resp) => Ok(resp.choices[0].message.content.clone().unwrap_or_default()),
            Response::ModelError(msg, _) => Err(anyhow::anyhow!("Model error: {msg}")),
            Response::ValidationError(e) => Err(anyhow::anyhow!(sanitize_error_message(&*e))),
            Response::InternalError(e) => Err(anyhow::anyhow!(sanitize_error_message(&*e))),
            _ => Err(anyhow::anyhow!("Unexpected response type")),
        }
    }
    .await
    .unwrap_or_else(|err| {
        tracing::warn!(
            "responses/compact failed, returning fallback summary (truncated_items={}): {}",
            truncated_items,
            err
        );
        // Keep the newest content so Codex has *something* to replace history with.
        let tail = if conversation_text.len() > 10_000 {
            conversation_text[conversation_text.len().saturating_sub(10_000)..].to_string()
        } else {
            conversation_text.clone()
        };
        format!("Unable to compact reliably. Recent context:\n{tail}")
    });

    // Include truncation info so follow-up prompts understand the summary may be incomplete.
    let summary = if truncated_items > 0 {
        format!("[Note: truncated {truncated_items} older item(s) before compacting]\n{summary}")
    } else {
        summary
    };

    let encrypted_content = STANDARD.encode(summary.as_bytes());
    let output = vec![
        ResponsesInputItem::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: "user".to_string(),
            content: ResponsesInputMessageContent::Parts(vec![
                ResponsesInputContentPart::InputText { text: summary },
            ]),
        },
        ResponsesInputItem::Compaction {
            id: format!("cmp_{}", Uuid::new_v4()),
            encrypted_content,
        },
    ];

    (StatusCode::OK, Json(ResponsesCompactResponse { output })).into_response()
}

/// List all responses endpoint
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/responses",
    responses((status = 200, description = "List of responses"))
)]
pub async fn list_responses(State(cache): ExtractedResponseCache) -> impl IntoResponse {
    match cache.get_all_responses() {
        Ok(responses) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "object": "list",
                "data": responses
            })),
        )
            .into_response(),
        Err(e) => JsonError::new(format!(
            "Error retrieving responses: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
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
    State(cache): ExtractedResponseCache,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mistral3_chat::canonicalize_messages_for_mistral3_template;
    use crate::openai::ResponsesInput;

    #[test]
    fn sampling_defaults_fill_only_missing_fields() {
        let defaults = SamplingDefaults {
            temperature: Some(0.2),
            top_p: Some(0.9),
            min_p: Some(0.05),
            top_k: Some(50),
        };

        let mut req = ResponsesCreateRequest {
            model: "default".to_string(),
            input: ResponsesInput::Items(vec![ResponsesInputItem::Message {
                id: "m1".to_string(),
                role: "user".to_string(),
                content: ResponsesInputMessageContent::Parts(vec![
                    ResponsesInputContentPart::InputText {
                        text: "hello".to_string(),
                    },
                ]),
            }]),
            instructions: None,
            modalities: None,
            previous_response_id: None,
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            max_tokens: None,
            n_choices: 1,
            presence_penalty: None,
            frequency_penalty: None,
            stop_seqs: None,
            temperature: None,
            top_p: None,
            stream: None,
            tools: None,
            tool_choice: None,
            text: None,
            response_format: None,
            web_search_options: None,
            metadata: None,
            output_token_details: None,
            parallel_tool_calls: None,
            store: None,
            max_tool_calls: None,
            reasoning_enabled: None,
            reasoning_max_tokens: None,
            reasoning_top_logprobs: None,
            truncation: None,
            top_k: None,
            grammar: None,
            min_p: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            repetition_penalty: None,
            enable_thinking: None,
            truncate_sequence: None,
            compact_history_with_summary: None,
        };

        if req.temperature.is_none() {
            req.temperature = defaults.temperature;
        }
        if req.top_p.is_none() {
            req.top_p = defaults.top_p;
        }
        if req.min_p.is_none() {
            req.min_p = defaults.min_p;
        }
        if req.top_k.is_none() {
            req.top_k = defaults.top_k;
        }

        assert_eq!(req.temperature, Some(0.2));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.top_k, Some(50));

        // Ensure explicit request values win.
        let defaults = SamplingDefaults {
            temperature: Some(0.2),
            top_p: Some(0.9),
            min_p: Some(0.05),
            top_k: Some(50),
        };
        let mut explicit = req;
        explicit.temperature = Some(1.0);
        explicit.top_p = Some(0.1);
        explicit.min_p = Some(0.2);
        explicit.top_k = Some(5);

        if explicit.temperature.is_none() {
            explicit.temperature = defaults.temperature;
        }
        if explicit.top_p.is_none() {
            explicit.top_p = defaults.top_p;
        }
        if explicit.min_p.is_none() {
            explicit.min_p = defaults.min_p;
        }
        if explicit.top_k.is_none() {
            explicit.top_k = defaults.top_k;
        }

        assert_eq!(explicit.temperature, Some(1.0));
        assert_eq!(explicit.top_p, Some(0.1));
        assert_eq!(explicit.min_p, Some(0.2));
        assert_eq!(explicit.top_k, Some(5));
    }

    #[test]
    fn parse_tool_calls_if_complete_ignores_incomplete_markers() {
        let incomplete = "[TOOL_CALLS]shell[ARGS]";
        assert!(parse_tool_calls_if_complete(incomplete).is_none());
    }

    #[test]
    fn parse_tool_calls_if_complete_parses_complete_payload() {
        let complete = r#"[TOOL_CALLS]shell[ARGS]{"command":["bash","-lc","ls"],"workdir":"/tmp"}"#;
        let parsed = parse_tool_calls_if_complete(complete).expect("should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "shell");
        assert_eq!(parsed[0].parameters["workdir"], "/tmp");
    }

    #[test]
    fn canonicalize_merges_consecutive_users() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("a".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("b".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
        assert_eq!(
            out[0].content.as_ref().unwrap().to_text().unwrap(),
            "a\n\nb"
        );
    }

    #[test]
    fn canonicalize_folds_developer_into_system() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("dev".to_string())),
                role: "developer".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[0].content.as_ref().unwrap().to_text().unwrap(), "dev");
        assert_eq!(out[1].role, "user");
        assert_eq!(out[1].content.as_ref().unwrap().to_text().unwrap(), "hi");
    }

    #[test]
    fn canonicalize_folds_codex_context_user_messages_into_system() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text(
                    "# AGENTS.md instructions for /repo\n...".to_string(),
                )),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>".to_string(),
                )),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("hello".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];
        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        let system_text = out[0].content.as_ref().unwrap().to_text().unwrap();
        assert!(system_text.contains("# AGENTS.md instructions"));
        assert!(system_text.contains("<environment_context>"));
        assert_eq!(out[1].role, "user");
        assert_eq!(out[1].content.as_ref().unwrap().to_text().unwrap(), "hello");
    }

    #[test]
    fn canonicalize_inserts_leading_user_if_assistant_first() {
        let msgs = vec![Message {
            content: Some(MessageContent::from_text("x".to_string())),
            role: "assistant".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }];
        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "user");
        assert_eq!(out[1].role, "assistant");
    }

    #[test]
    fn compaction_text_includes_multiple_input_text_parts() {
        let items = vec![ResponsesInputItem::Message {
            id: "m1".to_string(),
            role: "user".to_string(),
            content: ResponsesInputMessageContent::Parts(vec![
                ResponsesInputContentPart::InputText {
                    text: "hello".to_string(),
                },
                ResponsesInputContentPart::InputText {
                    text: "world".to_string(),
                },
            ]),
        }];

        let (text, truncated) = build_compaction_conversation_text(&items, 10_000);
        assert_eq!(truncated, 0);
        assert!(text.contains("user:"));
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn compaction_text_includes_web_search_call_action() {
        let items = vec![ResponsesInputItem::WebSearchCall {
            id: "ws1".to_string(),
            status: Some("completed".to_string()),
            action: crate::openai::ResponsesWebSearchAction::Search {
                query: Some("weather seattle".to_string()),
            },
        }];

        let (text, truncated) = build_compaction_conversation_text(&items, 10_000);
        assert_eq!(truncated, 0);
        assert!(text.contains("web_search_call"));
        assert!(text.contains("weather seattle"));
    }

    #[test]
    fn compaction_text_prefers_recent_items_when_truncating() {
        let items = vec![
            ResponsesInputItem::Message {
                id: "m_old".to_string(),
                role: "user".to_string(),
                content: ResponsesInputMessageContent::Text(
                    "OLD_MESSAGE_THAT_WILL_BE_DROPPED".to_string(),
                ),
            },
            ResponsesInputItem::Message {
                id: "m_new".to_string(),
                role: "user".to_string(),
                content: ResponsesInputMessageContent::Text("NEW_MESSAGE_KEPT".to_string()),
            },
        ];

        let (text, truncated) = build_compaction_conversation_text(&items, 32);
        assert!(truncated > 0);
        assert!(text.contains("NEW_MESSAGE_KEPT"));
        assert!(!text.contains("OLD_MESSAGE_THAT_WILL_BE_DROPPED"));
        assert!(text.starts_with("[... truncated "));
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
    State(cache): ExtractedResponseCache,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match cache.delete_response(&response_id) {
        Ok(true) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "deleted": true,
                "id": response_id,
                "object": "response.deleted"
            })),
        )
            .into_response(),
        Ok(false) => JsonError::new(format!("Response with ID '{response_id}' not found"))
            .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error deleting response: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Cancel response by ID endpoint
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}/cancel",
    params(("response_id" = String, Path, description = "The ID of the response to cancel")),
    responses((status = 200, description = "Response canceled"))
)]
pub async fn cancel_response(
    State(state): ExtractedMistralRsState,
    State(cache): ExtractedResponseCache,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match cache.get_active_request_id(&response_id) {
        Ok(Some((engine_id, model_id))) => {
            let sender = match state.get_sender(model_id.as_deref()) {
                Ok(s) => s,
                Err(e) => {
                    return JsonError::new(format!("Failed to get engine sender: {e:?}"))
                        .to_response(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            if let Err(e) = sender
                .send(Request::Cancel {
                    id: engine_id,
                    model_id: model_id.clone(),
                })
                .await
            {
                return JsonError::new(format!("Failed to send cancel request: {e}"))
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR);
            }

            // Also remove from active requests immediately?
            // The engine might take a moment to abort.
            // But from API perspective, we submitted the cancel.
            // We can leave it to the completion callback to cleanup, or do it here.
            // If we do it here, we might get a "completion" later that tries to remove it again (harmless).
            // But the response object might be updated to "cancelled" status by the engine response?
            // If we abort, the engine sends a chunk with StopReason::Canceled.
            // That chunk/response will trigger the cleanup logic.
            // So we just send the signal.

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "canceled": true,
                    "id": response_id,
                    "object": "response.canceled"
                })),
            )
                .into_response()
        }
        Ok(None) => JsonError::new(format!(
            "Active response with ID '{response_id}' not found or already completed"
        ))
        .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error retrieving active request: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Handle errors
fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ResponsesResponder {
    handle_completion_error(state, e)
}

/// Create SSE streamer
fn create_streamer(streamer: ResponsesStreamer) -> Sse<KeepAliveStream<ResponsesStreamer>> {
    let keep_alive_interval = get_keep_alive_interval();

    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// Get input items for a response
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}/input_items",
    params(("response_id" = String, Path, description = "The ID of the response to retrieve input items for")),
    responses((status = 200, description = "List of input items"))
)]
pub async fn get_input_items(
    State(cache): ExtractedResponseCache,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match cache.get_input_items(&response_id) {
        Ok(Some(items)) => {
            let first_id = items.first().map(|i| match i {
                ResponsesInputItem::Message { id, .. } => id.clone(),
                ResponsesInputItem::FunctionCallOutput { id, .. } => id.clone(),
                ResponsesInputItem::FunctionCall { id, .. } => id.clone(),
                ResponsesInputItem::CustomToolCall { id, .. } => id.clone(),
                ResponsesInputItem::CustomToolCallOutput { id, .. } => id.clone(),
                ResponsesInputItem::LocalShellCall { id, .. } => id.clone(),
                ResponsesInputItem::Reasoning { id, .. } => id.clone(),
                ResponsesInputItem::WebSearchCall { id, .. } => id.clone(),
                ResponsesInputItem::GhostSnapshot { id, .. } => id.clone(),
                ResponsesInputItem::Compaction { id, .. } => id.clone(),
                ResponsesInputItem::Other { id, .. } => id.clone(),
            });
            let last_id = items.last().map(|i| match i {
                ResponsesInputItem::Message { id, .. } => id.clone(),
                ResponsesInputItem::FunctionCallOutput { id, .. } => id.clone(),
                ResponsesInputItem::FunctionCall { id, .. } => id.clone(),
                ResponsesInputItem::CustomToolCall { id, .. } => id.clone(),
                ResponsesInputItem::CustomToolCallOutput { id, .. } => id.clone(),
                ResponsesInputItem::LocalShellCall { id, .. } => id.clone(),
                ResponsesInputItem::Reasoning { id, .. } => id.clone(),
                ResponsesInputItem::WebSearchCall { id, .. } => id.clone(),
                ResponsesInputItem::GhostSnapshot { id, .. } => id.clone(),
                ResponsesInputItem::Compaction { id, .. } => id.clone(),
                ResponsesInputItem::Other { id, .. } => id.clone(),
            });
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "object": "list",
                    "data": items,
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": false
                })),
            )
                .into_response()
        }
        Ok(None) => JsonError::new(format!(
            "Input items for response ID '{response_id}' not found"
        ))
        .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error retrieving input items: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}
