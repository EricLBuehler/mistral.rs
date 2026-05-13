use std::sync::Arc;

use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;

use serde_json::Value;

use crate::{
    files::{
        compact_tool_message_content, compose_tool_response_with_files,
        merge_required_outputs_into_args, prepend_system_message,
        system_message_for_required_files, tool_file_to_file, File, RequestedFile,
    },
    get_mut_arcmutex,
    pipeline::SupportedModality,
    response::{AgenticToolCallData, AgenticToolCallPhase},
    search, MessageContent, NormalRequest, RequestMessage, Response, ToolCallResponse, ToolChoice,
    WebSearchOptions,
};

use super::file_tools::{do_list_files, do_read_file};
use super::Engine;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Count user-role messages in the request's conversation. Used to
/// derive the conversation turn at agentic-loop entry.
fn count_user_messages(request: &NormalRequest) -> usize {
    get_messages(request)
        .iter()
        .filter(|m| {
            matches!(
                m.get("role"),
                Some(Either::Left(s)) if s == "user"
            )
        })
        .count()
        .saturating_sub(1) // Zero-based: the first user message is turn 0.
}

/// Get a reference to the messages vec inside a request.
fn get_messages(request: &NormalRequest) -> &Vec<IndexMap<String, MessageContent>> {
    match &request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::MultimodalChat { messages, .. } => {
            messages
        }
        _ => unreachable!(),
    }
}

/// Get a mutable reference to the messages vec inside a request.
pub(super) fn get_messages_mut(
    request: &mut NormalRequest,
) -> &mut Vec<IndexMap<String, MessageContent>> {
    match &mut request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::MultimodalChat { messages, .. } => {
            messages
        }
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
pub(super) fn append_assistant_tool_call(
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
pub(super) fn append_tool_response(
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

/// Upgrade a `Chat` request to `MultimodalChat` in-place. No-op if already multimodal.
pub(super) fn upgrade_to_multimodal(request: &mut NormalRequest) {
    let dummy = RequestMessage::Chat {
        messages: vec![],
        enable_thinking: None,
        reasoning_effort: None,
    };
    let old = std::mem::replace(&mut request.messages, dummy);
    request.messages = match old {
        RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort,
        } => RequestMessage::MultimodalChat {
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
            messages,
            enable_thinking,
            reasoning_effort,
        },
        other @ RequestMessage::MultimodalChat { .. } => other,
        _ => unreachable!(),
    };
}

pub(super) fn get_images_mut(request: &mut NormalRequest) -> &mut Vec<DynamicImage> {
    match &mut request.messages {
        RequestMessage::MultimodalChat { images, .. } => images,
        _ => unreachable!("must call upgrade_to_multimodal first"),
    }
}

pub(super) fn get_videos_mut(request: &mut NormalRequest) -> &mut Vec<crate::VideoInput> {
    match &mut request.messages {
        RequestMessage::MultimodalChat { videos, .. } => videos,
        _ => unreachable!("must call upgrade_to_multimodal first"),
    }
}

/// Append a tool response that may include images and/or video frames.
///
/// When images/videos are present and the model supports the modality,
/// the content is built as an array of typed parts and media is added to
/// the request-level vecs. Otherwise a textual note is appended.
fn append_multimodal_tool_response(
    request: &mut NormalRequest,
    tool_name: &str,
    mut content: String,
    images: Vec<DynamicImage>,
    video_frames: Vec<DynamicImage>,
    supports_vision: bool,
    supports_video: bool,
) {
    let inject_images = !images.is_empty() && supports_vision;
    let inject_video = !video_frames.is_empty() && supports_video;

    if !images.is_empty() && !supports_vision {
        content.push_str(&format!(
            "\n[ERROR: {} image(s) were generated but this model does not support vision input. Do not attempt to generate images.]",
            images.len()
        ));
    }
    if !video_frames.is_empty() && !supports_video {
        content.push_str(&format!(
            "\n[ERROR: {} video frame(s) were generated but this model does not support video input. Do not attempt to generate video.]",
            video_frames.len()
        ));
    }

    if !inject_images && !inject_video {
        let messages = get_messages_mut(request);
        append_tool_response(messages, tool_name, content);
        return;
    }

    upgrade_to_multimodal(request);

    let mut parts: Vec<IndexMap<String, Value>> = Vec::new();

    // Images.
    if inject_images {
        let req_images = get_images_mut(request);
        for img in &images {
            req_images.push(img.clone());
            let mut part = IndexMap::new();
            part.insert("type".to_string(), Value::String("image".to_string()));
            parts.push(part);
        }
    }

    // Video frames → VideoInput.
    if inject_video {
        let video = crate::VideoInput::from_frames(video_frames, 1.0, None);
        get_videos_mut(request).push(video);
        let mut part = IndexMap::new();
        part.insert("type".to_string(), Value::String("video".to_string()));
        parts.push(part);
    }

    // Text part.
    let mut text_part = IndexMap::new();
    text_part.insert("type".to_string(), Value::String("text".to_string()));
    text_part.insert("text".to_string(), Value::String(content));
    parts.push(text_part);

    let messages = get_messages_mut(request);
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("tool".to_string()));
    message.insert("name".to_string(), Either::Left(tool_name.to_string()));
    message.insert("content".to_string(), Either::Right(parts));
    messages.push(message);
}

/// Ensure a system message exists at the start of the conversation.
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

/// Save the current conversation state to the session store. Tool
/// messages have their inline file content stripped; bodies remain
/// reachable by id in the [`crate::files::FileStore`].
fn save_session(engine: &Arc<Engine>, session_id: &str, visible_req: &NormalRequest) {
    let mut messages = get_messages(visible_req).clone();
    for msg in &mut messages {
        let role = msg
            .get("role")
            .and_then(|r| match r {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("");
        if role != "tool" {
            continue;
        }
        if let Some(Either::Left(content)) = msg.get_mut("content") {
            *content = compact_tool_message_content(content);
        }
    }
    let (images, videos) = match &visible_req.messages {
        RequestMessage::MultimodalChat { images, videos, .. } => (images.clone(), videos.clone()),
        _ => (Vec::new(), Vec::new()),
    };
    let entry = super::agentic_session::AgenticSessionEntry::new(messages, images, videos);
    engine
        .session_store
        .lock()
        .unwrap()
        .save(session_id.to_string(), entry);
    // Refresh file TTLs alongside the session save so files don't
    // expire while their session is still active.
    engine.file_store.touch_session(session_id);
}

use super::tool_dispatch;

#[cfg(feature = "code-execution")]
fn is_code_exec_tool(name: &str) -> bool {
    mistralrs_code_exec::code_exec_tool_called(name)
}
#[cfg(not(feature = "code-execution"))]
fn is_code_exec_tool(_name: &str) -> bool {
    false
}

#[cfg(feature = "code-execution")]
fn is_read_file_tool(name: &str) -> bool {
    name == mistralrs_code_exec::READ_FILE_TOOL_NAME
}
#[cfg(not(feature = "code-execution"))]
fn is_read_file_tool(_name: &str) -> bool {
    false
}

#[cfg(feature = "code-execution")]
fn is_list_files_tool(name: &str) -> bool {
    name == mistralrs_code_exec::LIST_FILES_TOOL_NAME
}
#[cfg(not(feature = "code-execution"))]
fn is_list_files_tool(_name: &str) -> bool {
    false
}

/// Build `AgenticToolCallData` for the Calling phase from a tool call.
fn calling_data_for_tool(tc: &ToolCallResponse) -> AgenticToolCallData {
    if search::search_tool_called(&tc.function.name) {
        let query = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| {
                v.get("query")
                    .and_then(|q| q.as_str())
                    .map(|s| s.to_string())
            });
        AgenticToolCallData::WebSearch {
            query,
            results_count: None,
        }
    } else if is_code_exec_tool(&tc.function.name) {
        let code = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| {
                v.get("code")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            });
        AgenticToolCallData::CodeExecution {
            code,
            stdout: None,
            stderr: None,
            exception: None,
            images: vec![],
            video_frame_count: None,
            video_frames: vec![],
            working_directory: None,
            execution_time_ms: None,
        }
    } else {
        AgenticToolCallData::Custom {
            arguments: tc.function.arguments.clone(),
            content: String::new(),
        }
    }
}

async fn do_search(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_search(&engine, tc, opts).await;

    // Count results for the progress data.
    let results_count = serde_json::from_str::<serde_json::Value>(&result.content)
        .ok()
        .and_then(|v| v.get("output")?.as_array().map(|a| a.len()));

    let data = AgenticToolCallData::WebSearch {
        query: None, // already sent in Calling phase
        results_count,
    };
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    (request, data, Vec::new())
}

async fn do_extraction(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_extraction(&engine, tc, opts).await;
    let data = AgenticToolCallData::WebSearch {
        query: None,
        results_count: Some(1),
    };
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    (request, data, Vec::new())
}

#[allow(clippy::too_many_arguments)]
async fn do_custom_tool(
    engine: Arc<Engine>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    supports_vision: bool,
    supports_video: bool,
    ctx: &mistralrs_mcp::ToolCallContext,
    run_id: &str,
    round: usize,
    turn: usize,
    required_files: &[RequestedFile],
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    // For code-exec calls, merge the request-level required files into
    // the tool's `outputs` parameter so the executor reads them
    // regardless of whether the model declared them.
    let dispatched_tc;
    let dispatched_ref: &ToolCallResponse =
        if is_code_exec_tool(&tc.function.name) && !required_files.is_empty() {
            dispatched_tc = merge_required_outputs_into_args(tc, required_files);
            &dispatched_tc
        } else {
            tc
        };

    let result = tool_dispatch::execute_custom_tool(&engine, dispatched_ref, ctx);

    // Convert any tool-produced files into typed File artifacts.
    let files: Vec<File> = result
        .files
        .iter()
        .enumerate()
        .map(|(idx, tf)| tool_file_to_file(tf, run_id, round, turn, idx, &tc.function.name))
        .collect();

    // Build tool-specific result data.
    let is_code_exec = is_code_exec_tool(&tc.function.name);
    let data = if is_code_exec {
        // Parse the code-exec result JSON for structured fields.
        let val = serde_json::from_str::<serde_json::Value>(&result.content).ok();
        AgenticToolCallData::CodeExecution {
            code: None, // already sent in Calling phase
            stdout: val
                .as_ref()
                .and_then(|v| v.get("stdout"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
            stderr: val
                .as_ref()
                .and_then(|v| v.get("stderr"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
            exception: val
                .as_ref()
                .and_then(|v| v.get("exception"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            images: result.images.clone(),
            video_frame_count: if result.video_frames.is_empty() {
                None
            } else {
                Some(result.video_frames.len())
            },
            video_frames: result.video_frames.clone(),
            working_directory: val
                .as_ref()
                .and_then(|v| v.get("working_directory"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            execution_time_ms: val
                .as_ref()
                .and_then(|v| v.get("execution_time_ms"))
                .and_then(|v| v.as_u64()),
        }
    } else {
        AgenticToolCallData::Custom {
            arguments: String::new(), // already sent in Calling phase
            content: result.content.clone(),
        }
    };

    let composed_content = compose_tool_response_with_files(&result.content, &files);

    let has_multimodal = !result.images.is_empty() || !result.video_frames.is_empty();
    if !has_multimodal {
        let messages = get_messages_mut(&mut request);
        append_tool_response(messages, &tc.function.name, composed_content);
    } else {
        append_multimodal_tool_response(
            &mut request,
            &tc.function.name,
            composed_content,
            result.images,
            result.video_frames,
            supports_vision,
            supports_video,
        );
    }

    request.tool_choice = Some(ToolChoice::Auto);
    (request, data, files)
}

fn do_http_tool(
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    url: &str,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let result = tool_dispatch::execute_http_tool(tc, url);
    let data = AgenticToolCallData::Custom {
        arguments: String::new(),
        content: result.content.clone(),
    };
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    (request, data, Vec::new())
}

/// Tool-name → executor cascade. Returns `None` when no dispatcher is
/// configured for the requested tool (caller short-circuits the loop).
#[allow(clippy::too_many_arguments)]
async fn dispatch_tool(
    engine: &Arc<Engine>,
    visible_req: NormalRequest,
    tc: &ToolCallResponse,
    web_search_options: Option<&WebSearchOptions>,
    dispatch_url: Option<&str>,
    supports_vision: bool,
    supports_video: bool,
    tool_call_ctx: &mistralrs_mcp::ToolCallContext,
    run_id: &str,
    round: usize,
    turn: usize,
    session_id: &str,
    required_files: &[RequestedFile],
) -> Option<(NormalRequest, AgenticToolCallData, Vec<File>)> {
    if is_read_file_tool(&tc.function.name) {
        return Some(do_read_file(visible_req, tc, &engine.file_store));
    }
    if is_list_files_tool(&tc.function.name) {
        return Some(do_list_files(visible_req, tc, &engine.file_store, session_id));
    }
    if search::search_tool_called(&tc.function.name) {
        let opts = web_search_options?;
        return Some(if tc.function.name == search::SEARCH_TOOL_NAME {
            do_search(engine.clone(), visible_req, tc, opts).await
        } else {
            do_extraction(engine.clone(), visible_req, tc, opts).await
        });
    }
    if engine.tool_callbacks.contains_key(&tc.function.name) {
        return Some(
            do_custom_tool(
                engine.clone(),
                visible_req,
                tc,
                supports_vision,
                supports_video,
                tool_call_ctx,
                run_id,
                round,
                turn,
                required_files,
            )
            .await,
        );
    }
    if let Some(url) = dispatch_url {
        return Some(do_http_tool(visible_req, tc, url));
    }
    None
}

/// Insert each produced file into the engine's [`FileStore`] and emit
/// it on the user-facing channel. Consumes `files` so each body is
/// cloned exactly once (for the store); the original is moved into the
/// response.
async fn emit_files(
    engine: &Engine,
    session_id: &str,
    files: Vec<File>,
    sender: &tokio::sync::mpsc::Sender<Response>,
) {
    for f in files {
        engine
            .file_store
            .insert(f.clone(), Some(session_id.to_string()));
        let _ = sender.send(Response::File(f)).await;
    }
}

/// Drive one or more tool-use rounds (web-search, code execution, custom
/// tools, etc.) without recursion.
///
/// Strategy:
/// 1. Send a "probe" request that may call available tools.
/// 2. If a tool is called, run it to mutate the conversational context and
///    build the next request.
/// 3. Repeat until no further tool call is made.
/// 4. Forward every user-visible reply **except** the first, which is just the
///    probe that discovers whether a tool call is needed.
pub(super) async fn agentic_loop(this: Arc<Engine>, mut request: NormalRequest) {
    let web_search_options = request.web_search_options.clone();
    let dispatch_url = request.tool_dispatch_url.clone();
    let required_files: Vec<RequestedFile> = request.files.clone().unwrap_or_default();

    // Short tag for File ids; stable within a run, unique across runs.
    let run_id: String = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();

    // Resolve session ID: use explicit, generate new, or match by content.
    let session_id = request
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Look up existing session and splice stored tool history into request.
    {
        let mut store = this.session_store.lock().unwrap();
        let existing = if request.session_id.is_some() {
            store.get(&session_id)
        } else {
            let msgs = get_messages(&request);
            store.find_by_messages(msgs).map(|(_, e)| e)
        };
        if let Some(entry) = existing {
            super::agentic_session::splice_session_into_request(&mut request, &entry);
        }
    }

    // Refresh TTL on every file already associated with this session so
    // an active multi-turn session doesn't lose earlier files to expiry.
    this.file_store.touch_session(&session_id);

    // Turn = number of user messages in the conversation at agentic-
    // loop entry. First turn is 0. Constant for the duration of the
    // loop (round increments within a turn; turn does not).
    let turn = count_user_messages(&request);

    // If the request declared required files, prepend a system message
    // telling the model what to produce. Done after splicing so the
    // contract sits at message-vec start (or merges into an existing
    // system message).
    if let Some(sys) = system_message_for_required_files(&required_files) {
        let messages = get_messages_mut(&mut request);
        prepend_system_message(messages, &sys);
    }

    // Cache model modality support for multimodal tool results.
    let modalities = {
        let pipeline = get_mut_arcmutex!(this.pipeline);
        pipeline.get_metadata().modalities.clone()
    };
    let supports_vision = modalities.input.contains(&SupportedModality::Vision);
    let supports_video = modalities.input.contains(&SupportedModality::Video);

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

    // Validate: reject user tools that conflict with registered internal tools.
    if let Some(user_tools) = &probe.tools {
        for t in user_tools {
            if this.tool_callbacks.contains_key(&t.function.name) {
                let _ = user_sender
                    .send(Response::ValidationError(
                        format!(
                            "Tool '{}' conflicts with a registered internal tool. \
                             Internal tool names cannot be overridden.",
                            t.function.name
                        )
                        .into(),
                    ))
                    .await;
                return;
            }
        }
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

    probe.tool_choice = Some(ToolChoice::Auto);
    // Prevent accidental infinite recursion on the probe itself.
    probe.web_search_options = None;

    // The conversation context that the user *will* see.
    let mut visible_req = probe.clone();
    visible_req.response = user_sender.clone();

    // We'll drive everything inside a single spawned task.
    let this_clone = this.clone();
    let handle = tokio::spawn(async move {
        // Build tool call context with session ID for tool callbacks.
        let tool_call_ctx = mistralrs_mcp::ToolCallContext {
            session_id: Some(session_id.clone()),
        };

        // `current` is what we actually dispatch each loop.
        // The very first time that is the hidden probe.
        let mut current = probe;
        let max_rounds = current.max_tool_rounds.unwrap_or(16);
        let mut round = 0;

        loop {
            // Each dispatch gets its own one-shot channel so we can peek at
            // the response before (optionally) forwarding it.
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            current.response = sender;

            // Kick the request into the engine via the channel.
            // Clear fields that would cause the engine to re-enter the
            // agentic loop — this loop already manages tool orchestration.
            // Clear fields that would cause re-entry into the agentic loop.
            // max_tool_rounds = Some(0) is a sentinel recognized by
            // add_request to skip the agentic loop check.
            current.web_search_options = None;
            current.enable_code_execution = false;
            current.max_tool_rounds = Some(0);
            current.tool_dispatch_url = None;
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

                // No tool call, or max rounds reached? We are finished.
                if tc_opt.is_none() || round >= max_rounds {
                    // Save conversation state for future requests.
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    final_resp.session_id = Some(session_id.clone());
                    user_sender.send(Response::Done(final_resp)).await.unwrap();
                    return;
                }

                // Tool requested -> build the next turn.
                let tc = tc_opt.unwrap();

                // Notify client that a tool call is starting.
                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Calling(calling_data_for_tool(tc)),
                    })
                    .await;

                let outcome = dispatch_tool(
                    &this_clone,
                    visible_req.clone(),
                    tc,
                    web_search_options.as_ref(),
                    dispatch_url.as_deref(),
                    supports_vision,
                    supports_video,
                    &tool_call_ctx,
                    &run_id,
                    round,
                    turn,
                    &session_id,
                    &required_files,
                )
                .await;
                let Some((next_visible, complete_data, files)) = outcome else {
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    final_resp.session_id = Some(session_id.clone());
                    user_sender.send(Response::Done(final_resp)).await.unwrap();
                    return;
                };

                emit_files(&this_clone, &session_id, files, &user_sender).await;

                // Notify client that the tool call completed.
                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Complete(complete_data),
                    })
                    .await;

                round += 1;

                // The fresh request becomes both the user-visible context and
                // the next `current` we will dispatch.
                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
            // ------------------------- STREAMING -------------------------
            else {
                // We need the *last* chunk to see whether a tool was called.
                // The finish-reason chunk is held back so we can stamp the
                // session ID on it if this turns out to be the final round.
                let mut last_choice = None;
                let mut held_final_chunk: Option<crate::ChatCompletionChunkResponse> = None;

                while let Some(resp) = receiver.recv().await {
                    let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                        return;
                    };
                    match resp {
                        Response::Chunk(chunk) => {
                            // Forward content-bearing chunks, suppress tool-call chunks.
                            // Forwarding tool call chunks would cause streaming clients
                            // to see a premature finish_reason before the tool loop
                            // has a chance to execute the tool and continue.
                            let first_choice = &chunk.choices[0];
                            let is_final = first_choice.finish_reason.is_some();
                            if first_choice.delta.tool_calls.is_none() {
                                if is_final {
                                    // Hold back — we may need to stamp the session ID.
                                    held_final_chunk = Some(chunk.clone());
                                } else {
                                    let _ = user_sender.send(Response::Chunk(chunk.clone())).await;
                                }
                            }
                            last_choice = Some(first_choice.clone());

                            if is_final {
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

                // No tool call or max rounds reached -> done.
                if tc_opt.is_none() || round >= max_rounds {
                    // Save conversation state for future requests.
                    save_session(&this_clone, &session_id, &visible_req);
                    // Stamp the session ID on the held final chunk and send it.
                    if let Some(mut final_chunk) = held_final_chunk {
                        final_chunk.session_id = Some(session_id.clone());
                        let _ = user_sender.send(Response::Chunk(final_chunk)).await;
                    }
                    break;
                }

                let tc = tc_opt.unwrap();

                // Notify client that a tool call is starting.
                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Calling(calling_data_for_tool(tc)),
                    })
                    .await;

                let outcome = dispatch_tool(
                    &this_clone,
                    visible_req.clone(),
                    tc,
                    web_search_options.as_ref(),
                    dispatch_url.as_deref(),
                    supports_vision,
                    supports_video,
                    &tool_call_ctx,
                    &run_id,
                    round,
                    turn,
                    &session_id,
                    &required_files,
                )
                .await;
                let Some((next_visible, complete_data, files)) = outcome else {
                    save_session(&this_clone, &session_id, &visible_req);
                    break;
                };

                emit_files(&this_clone, &session_id, files, &user_sender).await;

                // Notify client that the tool call completed.
                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Complete(complete_data),
                    })
                    .await;

                round += 1;

                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
        }
    });

    get_mut_arcmutex!(this.handles).push(handle);
}
