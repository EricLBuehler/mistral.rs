use std::sync::Arc;

use either::Either;
use indexmap::IndexMap;

use serde_json::Value;

use crate::{
    get_mut_arcmutex, search, MessageContent, NormalRequest, RequestMessage, Response,
    ToolCallResponse, ToolChoice, WebSearchOptions,
};

use super::Engine;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Get a mutable reference to the messages vec inside a request.
fn get_messages_mut(
    request: &mut NormalRequest,
) -> &mut Vec<IndexMap<String, MessageContent>> {
    match &mut request.messages {
        RequestMessage::Chat { messages, .. }
        | RequestMessage::MultimodalChat { messages, .. } => messages,
        _ => unreachable!(),
    }
}

/// Build a `tool_calls` structured field for an assistant message.
/// Enables Gemma 4 (and other templates that use `message.tool_calls`)
/// to render the tool call correctly.
fn build_tool_calls_field(tc: &ToolCallResponse) -> MessageContent {
    let mut tc_map = IndexMap::new();
    tc_map.insert("id".to_string(), Value::String(tc.id.clone()));
    tc_map.insert("type".to_string(), Value::String("function".to_string()));
    let mut function_map = serde_json::Map::new();
    function_map.insert("name".to_string(), Value::String(tc.function.name.clone()));
    let args_value = serde_json::from_str(&tc.function.arguments)
        .unwrap_or(Value::String(tc.function.arguments.clone()));
    function_map.insert("arguments".to_string(), args_value);
    tc_map.insert("function".to_string(), Value::Object(function_map));
    Either::Right(vec![tc_map])
}

/// Append an assistant message recording the tool call invocation.
fn append_assistant_tool_call(
    messages: &mut Vec<IndexMap<String, MessageContent>>,
    tc: &ToolCallResponse,
) {
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("assistant".to_string()));
    message.insert("content".to_string(), Either::Left(String::new()));
    message.insert("tool_calls".to_string(), build_tool_calls_field(tc));
    messages.push(message);
}

/// Append a tool response message with the execution result.
fn append_tool_response(
    messages: &mut Vec<IndexMap<String, MessageContent>>,
    tool_name: &str,
    content: String,
) {
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("tool".to_string()));
    message.insert("name".to_string(), Either::Left(tool_name.to_string()));
    message.insert("content".to_string(), Either::Left(content));
    messages.push(message);
}

/// Ensure a system message exists at the start of the conversation.
/// Models need a system message alongside tool declarations to reliably
/// trigger tool calls.
fn ensure_system_message(messages: &mut Vec<IndexMap<String, MessageContent>>) {
    let has_system = messages
        .first()
        .and_then(|m| m.get("role"))
        .and_then(|r| match r {
            Either::Left(s) => Some(s.as_str()),
            _ => None,
        })
        .is_some_and(|r| r == "system" || r == "developer");
    if !has_system {
        let mut sys_msg: IndexMap<String, MessageContent> = IndexMap::new();
        sys_msg.insert("role".to_string(), Either::Left("system".to_string()));
        sys_msg.insert("content".to_string(), Either::Left(String::new()));
        messages.insert(0, sys_msg);
    }
}

/// Forward a non-chat-completion response to the user sender.
/// Returns `true` if the response was forwarded (caller should return),
/// `false` if it's a `Done` or `Chunk` that the caller should handle.
async fn forward_passthrough(
    resp: Response,
    user_sender: &tokio::sync::mpsc::Sender<Response>,
) -> Option<Response> {
    match resp {
        // These are the ones the tool loop actually handles:
        Response::Done(_) | Response::Chunk(_) => Some(resp),
        // Everything else gets forwarded directly:
        other => {
            let _ = user_sender.send(other).await;
            None
        }
    }
}

// ── Tool executors ─────────────────────────────────────────────────────────
//
// Each executor: append assistant tool-call message, run the tool via
// `tool_dispatch`, append tool response, and configure the request for the
// next turn.

use super::tool_dispatch;

async fn do_search(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> NormalRequest {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_search(&engine, tc, opts).await;
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    request
}

async fn do_extraction(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> NormalRequest {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_extraction(&engine, tc, opts).await;
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    request
}

async fn do_custom_tool(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
) -> NormalRequest {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_custom_tool(&engine, tc);
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    request
}

/// Drive one or more web-search / extraction rounds without recursion.
///
/// Strategy:
/// 1. Send a "probe" request that may call the search/extract tools.
/// 2. If such a tool is called, run it (`do_search` / `do_extraction`) to
///    mutate the conversational context and build the next request.
/// 3. Repeat until no further tool call is made.
/// 4. Forward every user-visible reply **except** the first, which is just the
///    probe that discovers whether a tool call is needed.
pub(super) async fn search_request(this: Arc<Engine>, request: NormalRequest) {
    let web_search_options = request.web_search_options.clone();

    // The sender that ultimately delivers data back to the caller.
    let user_sender = request.response.clone();
    let is_streaming = request.is_streaming;

    // ---------------------------------------------------------------------
    // Build the *first* request (the “probe”).
    // ---------------------------------------------------------------------
    let mut probe = request.clone();
    if let Some(ref opts) = web_search_options {
        probe
            .tools
            .get_or_insert_with(Vec::new)
            .extend(search::get_search_tools(opts).unwrap());
    }

    // Add Tool definitions from registered tool callbacks
    if !this.tool_callbacks.is_empty() {
        let tools = probe.tools.get_or_insert_with(Vec::new);
        let existing_tool_names: Vec<String> =
            tools.iter().map(|t| t.function.name.clone()).collect();

        for (name, callback_with_tool) in &this.tool_callbacks {
            if !existing_tool_names.contains(name) {
                tools.push(callback_with_tool.tool.clone());
            }
        }
    }

    // Models need a system message alongside tool declarations to reliably
    ensure_system_message(get_messages_mut(&mut probe));

    probe.tool_choice = Some(ToolChoice::Auto);
    // Prevent accidental infinite recursion on the probe itself.
    probe.web_search_options = None;

    // The conversation context that the user *will* see.
    let mut visible_req = probe.clone();
    visible_req.response = user_sender.clone();

    // We'll drive everything inside a single spawned task.
    let this_clone = this.clone();
    let handle = tokio::spawn(async move {
        // `current` is what we actually dispatch each loop.
        // The very first time that is the hidden probe.
        let mut current = probe;

        loop {
            // Each dispatch gets its own one-shot channel so we can peek at
            // the response before (optionally) forwarding it.
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            current.response = sender;

            // Kick the request into the engine via the channel.
            // Clear web_search_options so the engine doesn't re-enter the
            // search flow — this loop already manages tool orchestration.
            current.web_search_options = None;
            let _ = this_clone
                .tx
                .send(crate::request::Request::Normal(Box::new(current)))
                .await;

            // ----------------------- NON-STREAMING ------------------------
            if !is_streaming {
                let resp = receiver.recv().await.unwrap();
                let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                    return;
                };
                let done = match resp {
                    Response::Done(done) => done,
                    _ => {
                        let _ = user_sender.send(resp).await;
                        return;
                    }
                };

                // Did the assistant ask to run a tool?
                let tc_opt = match &done.choices[0].message.tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        if calls.len() > 1 {
                            tracing::warn!(
                                "Model returned {} tool calls; executing only the first.",
                                calls.len()
                            );
                        }
                        Some(&calls[0])
                    }
                    _ => None,
                };

                // No tool call? We are finished.
                if tc_opt.is_none() {
                    user_sender
                        .send(Response::Done(done.clone()))
                        .await
                        .unwrap();
                    return;
                }

                // Tool requested -> build the next turn.
                let tc = tc_opt.unwrap();
                let next_visible = if search::search_tool_called(&tc.function.name) {
                    let web_search_options = web_search_options.as_ref().unwrap();
                    if tc.function.name == search::SEARCH_TOOL_NAME {
                        do_search(this_clone.clone(), visible_req, tc, web_search_options).await
                    } else {
                        do_extraction(this_clone.clone(), visible_req, tc, web_search_options).await
                    }
                } else {
                    do_custom_tool(this_clone.clone(), visible_req, tc).await
                };

                // The fresh request becomes both the user-visible context and
                // the next `current` we will dispatch.
                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
            // ------------------------- STREAMING -------------------------
            else {
                // We need the *last* chunk to see whether a tool was called.
                let mut last_choice = None;

                while let Some(resp) = receiver.recv().await {
                    let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                        return;
                    };
                    match resp {
                        Response::Chunk(chunk) => {
                            // Forward content-bearing chunks, suppress tool-call chunks.
                            let first_choice = &chunk.choices[0];
                            if first_choice.delta.tool_calls.is_none() {
                                let _ = user_sender.send(Response::Chunk(chunk.clone())).await;
                            }
                            last_choice = Some(first_choice.clone());

                            if last_choice
                                .as_ref()
                                .and_then(|c| c.finish_reason.as_ref())
                                .is_some()
                            {
                                break;
                            }
                        }
                        other => {
                            // Done or unexpected in streaming — forward and stop.
                            let _ = user_sender.send(other).await;
                            return;
                        }
                    }
                }

                let Some(choice) = last_choice else { break };

                let tc_opt = match &choice.delta.tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        if calls.len() > 1 {
                            tracing::warn!(
                                "Model returned {} tool calls; executing only the first.",
                                calls.len()
                            );
                        }
                        Some(&calls[0])
                    }
                    _ => None,
                };

                if tc_opt.is_none() {
                    break; // No more tool calls -> done.
                }

                let tc = tc_opt.unwrap();
                let next_visible = if search::search_tool_called(&tc.function.name) {
                    let web_search_options = web_search_options.as_ref().unwrap();
                    if tc.function.name == search::SEARCH_TOOL_NAME {
                        do_search(this_clone.clone(), visible_req, tc, web_search_options).await
                    } else {
                        do_extraction(this_clone.clone(), visible_req, tc, web_search_options).await
                    }
                } else {
                    do_custom_tool(this_clone.clone(), visible_req, tc).await
                };

                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
        }
    });

    get_mut_arcmutex!(this.handles).push(handle);
}
