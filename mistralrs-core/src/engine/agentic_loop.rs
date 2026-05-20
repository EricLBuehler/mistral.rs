use std::sync::Arc;

use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;

use serde_json::Value;

use crate::{
    files::{
        compose_tool_response_with_files, merge_required_outputs_into_args,
        required_files_tool_addendum, tool_file_to_file, File, RequestedFile,
    },
    get_mut_arcmutex,
    pipeline::SupportedModality,
    response::{AgenticToolCallData, AgenticToolCallPhase},
    search, AgentPermission, AgentToolApproval, AgentToolApprovalCallback,
    AgentToolApprovalDecision, AgentToolApprovalHandler, AgentToolKind, AgentToolMetadata,
    AgentToolSource, MessageContent, NormalRequest, RequestMessage, Response, ToolCallResponse,
    ToolChoice, WebSearchOptions,
};

use super::file_tools::{do_list_files, do_read_file};
use super::Engine;

/// Default cap on tool-use rounds when the request doesn't set one.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 256;

/// Set on inner probe requests so `handle_request` doesn't re-enter the loop. Distinct from `None` (unset).
pub const AGENTIC_LOOP_REENTRY_SENTINEL: Option<usize> = Some(0);

/// Turn = number of completed user messages.
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
        .saturating_sub(1)
}

fn get_messages(request: &NormalRequest) -> &Vec<IndexMap<String, MessageContent>> {
    match &request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::MultimodalChat { messages, .. } => {
            messages
        }
        _ => unreachable!(),
    }
}

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

/// Structured `tool_calls` field for the assistant message. Required by templates (Gemma 4 etc.) that render from `message.tool_calls`.
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

/// Upgrade `Chat` to `MultimodalChat` in place. No-op if already multimodal.
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

/// Append a tool response, routing images/video to the request's multimodal vecs when supported. Otherwise text-only with an error note.
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

    if inject_images {
        let req_images = get_images_mut(request);
        for img in &images {
            req_images.push(img.clone());
            let mut part = IndexMap::new();
            part.insert("type".to_string(), Value::String("image".to_string()));
            parts.push(part);
        }
    }

    if inject_video {
        let video = crate::VideoInput::from_frames(video_frames, 1.0, None);
        get_videos_mut(request).push(video);
        let mut part = IndexMap::new();
        part.insert("type".to_string(), Value::String("video".to_string()));
        parts.push(part);
    }

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

/// `Some(resp)` for `Done`/`Chunk` (caller handles); forwards everything else and returns `None`.
async fn forward_passthrough(
    resp: Response,
    user_sender: &tokio::sync::mpsc::Sender<Response>,
) -> Option<Response> {
    match resp {
        Response::Done(_) | Response::Chunk(_) => Some(resp),
        other => {
            let _ = user_sender.send(other).await;
            None
        }
    }
}

/// Persist the conversation as-is. Refreshes file TTLs.
fn save_session(engine: &Arc<Engine>, session_id: &str, visible_req: &NormalRequest) {
    let messages = get_messages(visible_req).clone();
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
            sources: Vec::new(),
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

fn tool_arguments(tc: &ToolCallResponse) -> Value {
    serde_json::from_str(&tc.function.arguments)
        .unwrap_or_else(|_| Value::String(tc.function.arguments.clone()))
}

fn tool_metadata_for(ctx: &DispatchCtx<'_>, tc: &ToolCallResponse) -> AgentToolMetadata {
    let name = &tc.function.name;
    if is_code_exec_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::CodeExecution,
            label: "Python code".to_string(),
        }
    } else if search::search_tool_called(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::WebSearch,
            label: if name == search::SEARCH_TOOL_NAME {
                "Web search".to_string()
            } else {
                "Web page extraction".to_string()
            },
        }
    } else if is_read_file_tool(name) || is_list_files_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::File,
            label: "File access".to_string(),
        }
    } else if ctx.engine.tool_callbacks.contains_key(name) {
        AgentToolMetadata {
            source: AgentToolSource::User,
            kind: AgentToolKind::Custom,
            label: name.clone(),
        }
    } else if ctx.dispatch_url.is_some() {
        AgentToolMetadata {
            source: AgentToolSource::External,
            kind: AgentToolKind::External,
            label: name.clone(),
        }
    } else {
        AgentToolMetadata {
            source: AgentToolSource::User,
            kind: AgentToolKind::Custom,
            label: name.clone(),
        }
    }
}

async fn call_agent_approval_callback(
    callback: AgentToolApprovalCallback,
    approval: AgentToolApproval,
) -> AgentToolApprovalDecision {
    match tokio::task::spawn_blocking(move || callback(&approval)).await {
        Ok(decision) => decision,
        Err(_) => AgentToolApprovalDecision::deny_with_message(
            "Agent action requires approval, but the approval handler failed.",
        ),
    }
}

async fn call_agent_approval_handler(
    handler: AgentToolApprovalHandler,
    approval: AgentToolApproval,
) -> AgentToolApprovalDecision {
    match handler {
        AgentToolApprovalHandler::Sync(callback) => {
            call_agent_approval_callback(callback, approval).await
        }
        AgentToolApprovalHandler::Async(callback) => {
            match tokio::spawn(async move { callback(approval).await }).await {
                Ok(decision) => decision,
                Err(_) => AgentToolApprovalDecision::deny_with_message(
                    "Agent action requires approval, but the approval handler failed.",
                ),
            }
        }
    }
}

async fn approve_agent_tool(
    ctx: &DispatchCtx<'_>,
    tc: &ToolCallResponse,
    round: usize,
) -> AgentToolApprovalDecision {
    let tool = tool_metadata_for(ctx, tc);
    let message = match ctx.agent_permission {
        AgentPermission::Auto => return AgentToolApprovalDecision::approve(),
        AgentPermission::Deny => format!("{} was denied by policy.", tool.label),
        AgentPermission::Ask => {
            if ctx
                .engine
                .session_store
                .lock()
                .unwrap()
                .agent_actions_approved(ctx.session_id)
            {
                return AgentToolApprovalDecision::approve();
            }
            let Some(handler) = &ctx.agent_approval_handler else {
                return AgentToolApprovalDecision::deny_with_message(
                    "Agent action requires approval, but no approval handler is configured.",
                );
            };
            let approval = AgentToolApproval {
                approval_id: format!("appr_{}", uuid::Uuid::new_v4().simple()),
                session_id: ctx.session_id.to_string(),
                round,
                tool,
                arguments: tool_arguments(tc),
            };
            if let Some(notifier) = &ctx.tool_call_ctx.agent_approval_notifier {
                notifier(mistralrs_mcp::AgentToolApprovalRequest {
                    approval_id: approval.approval_id.clone(),
                    session_id: approval.session_id.clone(),
                    round: approval.round,
                    tool: approval.tool.clone(),
                    arguments: approval.arguments.clone(),
                });
            }
            let decision = call_agent_approval_handler(handler.clone(), approval).await;
            if decision.approve && decision.remember_for_session {
                ctx.engine
                    .session_store
                    .lock()
                    .unwrap()
                    .approve_agent_actions(ctx.session_id.to_string());
            }
            return decision;
        }
    };
    AgentToolApprovalDecision::deny_with_message(message)
}

fn denied_tool_result(
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    message: String,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);
    let content = serde_json::json!({
        "status": "denied",
        "exception": message,
    })
    .to_string();
    append_tool_response(messages, &tc.function.name, content.clone());
    request.tool_choice = Some(ToolChoice::Auto);

    let data = if is_code_exec_tool(&tc.function.name) {
        AgenticToolCallData::CodeExecution {
            code: None,
            stdout: None,
            stderr: None,
            exception: Some(message),
            images: vec![],
            video_frame_count: None,
            video_frames: vec![],
            working_directory: None,
            execution_time_ms: None,
        }
    } else {
        AgenticToolCallData::Custom {
            arguments: String::new(),
            content,
        }
    };

    (request, data, Vec::new())
}

/// Per-loop dispatch context. Borrows data owned by the loop's task; round/tc/visible_req are passed alongside.
struct DispatchCtx<'a> {
    engine: &'a Arc<Engine>,
    web_search_options: Option<&'a WebSearchOptions>,
    dispatch_url: Option<&'a str>,
    supports_vision: bool,
    supports_video: bool,
    tool_call_ctx: &'a mistralrs_mcp::ToolCallContext,
    run_id: &'a str,
    turn: usize,
    session_id: &'a str,
    required_files: &'a [RequestedFile],
    agent_permission: AgentPermission,
    agent_approval_handler: Option<AgentToolApprovalHandler>,
}

fn web_search_metadata(content: &str) -> (Option<usize>, Vec<String>) {
    let Ok(value) = serde_json::from_str::<Value>(content) else {
        return (None, Vec::new());
    };

    let results_count = value.get("output").and_then(|output| {
        if let Some(results) = output.as_array() {
            Some(results.len())
        } else if output.is_string() {
            Some(1)
        } else {
            None
        }
    });

    let sources = value
        .get("sources")
        .and_then(|sources| sources.as_array())
        .map(|sources| {
            sources
                .iter()
                .filter_map(|source| source.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| {
            value
                .get("output")
                .and_then(|output| output.as_array())
                .map(|results| {
                    search::source_domains(
                        results
                            .iter()
                            .filter_map(|result| result.get("url").and_then(|url| url.as_str())),
                    )
                })
                .unwrap_or_default()
        });

    (results_count, sources)
}

fn extraction_sources(tc: &ToolCallResponse) -> Vec<String> {
    serde_json::from_str::<Value>(&tc.function.arguments)
        .ok()
        .and_then(|value| {
            value
                .get("url")
                .and_then(|url| url.as_str())
                .map(|url| search::source_domains([url]))
        })
        .unwrap_or_default()
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

    let (results_count, sources) = web_search_metadata(&result.content);

    let data = AgenticToolCallData::WebSearch {
        query: None, // already sent in Calling phase
        results_count,
        sources,
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
        sources: extraction_sources(tc),
    };
    append_tool_response(messages, &tc.function.name, result.content);

    request.tool_choice = Some(ToolChoice::Auto);
    (request, data, Vec::new())
}

async fn do_custom_tool(
    ctx: &DispatchCtx<'_>,
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    round: usize,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    // For code-exec, merge required files into `outputs` so the executor reads them even if the model omitted them.
    let dispatched_tc;
    let dispatched_ref: &ToolCallResponse =
        if is_code_exec_tool(&tc.function.name) && !ctx.required_files.is_empty() {
            dispatched_tc = merge_required_outputs_into_args(tc, ctx.required_files);
            &dispatched_tc
        } else {
            tc
        };

    let mut tool_call_ctx;
    let dispatch_tool_ctx = if is_code_exec_tool(&tc.function.name) {
        tool_call_ctx = ctx.tool_call_ctx.clone();
        tool_call_ctx.round = Some(round);
        tool_call_ctx.tool_name = Some(tc.function.name.clone());
        &tool_call_ctx
    } else {
        ctx.tool_call_ctx
    };

    let result = tool_dispatch::execute_custom_tool(ctx.engine, dispatched_ref, dispatch_tool_ctx);

    let files: Vec<File> = result
        .files
        .iter()
        .enumerate()
        .map(|(idx, tf)| tool_file_to_file(tf, ctx.run_id, round, ctx.turn, idx, &tc.function.name))
        .collect();

    let is_code_exec = is_code_exec_tool(&tc.function.name);
    let data = if is_code_exec {
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
            ctx.supports_vision,
            ctx.supports_video,
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

/// `None` when no dispatcher is configured for `tc.function.name`. Caller short-circuits the loop.
async fn dispatch_tool(
    ctx: &DispatchCtx<'_>,
    visible_req: NormalRequest,
    tc: &ToolCallResponse,
    round: usize,
) -> Option<(NormalRequest, AgenticToolCallData, Vec<File>)> {
    let name = &tc.function.name;
    if is_read_file_tool(name) {
        return Some(do_read_file(visible_req, tc, &ctx.engine.file_store));
    }
    if is_list_files_tool(name) {
        return Some(do_list_files(
            visible_req,
            tc,
            &ctx.engine.file_store,
            ctx.session_id,
        ));
    }
    if search::search_tool_called(name) {
        let opts = ctx.web_search_options?;
        return Some(if name == search::SEARCH_TOOL_NAME {
            do_search(ctx.engine.clone(), visible_req, tc, opts).await
        } else {
            do_extraction(ctx.engine.clone(), visible_req, tc, opts).await
        });
    }
    if ctx.engine.tool_callbacks.contains_key(name) {
        return Some(do_custom_tool(ctx, visible_req, tc, round).await);
    }
    if let Some(url) = ctx.dispatch_url {
        return Some(do_http_tool(visible_req, tc, url));
    }
    None
}

/// Store full file bodies and emit wire-elided clones on the user channel. Truncated bodies stay fetchable via the store.
async fn emit_files(
    engine: &Engine,
    session_id: &str,
    files: Vec<File>,
    sender: &tokio::sync::mpsc::Sender<Response>,
) {
    for f in files {
        let wire = f.elide_for_wire();
        engine.file_store.insert(f, Some(session_id.to_string()));
        let _ = sender.send(Response::File(wire)).await;
    }
}

/// Drive tool-use rounds (search, code exec, custom tools) without recursion. Forwards every reply except the first probe.
pub(super) async fn agentic_loop(this: Arc<Engine>, mut request: NormalRequest) {
    let web_search_options = request.web_search_options.clone();
    let dispatch_url = request.tool_dispatch_url.clone();
    let code_execution_permission = request.code_execution_permission;
    let code_execution_approval_notifier = request.code_execution_approval_notifier.clone();
    let agent_permission = request.agent_permission.unwrap_or_default();
    let agent_approval_handler = request.agent_approval_handler.clone();
    let agent_approval_notifier = request.agent_approval_notifier.clone();
    let required_files: Vec<RequestedFile> = request.files.clone().unwrap_or_default();

    let run_id: String = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();

    let mut session_id = request
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    {
        let mut store = this.session_store.lock().unwrap();
        let existing = if request.session_id.is_some() {
            store.get(&session_id).map(|e| (session_id.clone(), e))
        } else {
            let msgs = get_messages(&request);
            store.find_by_messages(msgs)
        };
        if let Some((matched_id, entry)) = existing {
            session_id = matched_id;
            super::agentic_session::splice_session_into_request(&mut request, &entry);
        }
    }

    this.file_store.touch_session(&session_id);

    let turn = count_user_messages(&request);

    let modalities = {
        let pipeline = get_mut_arcmutex!(this.pipeline);
        pipeline.get_metadata().modalities.clone()
    };
    let supports_vision = modalities.input.contains(&SupportedModality::Vision);
    let supports_video = modalities.input.contains(&SupportedModality::Video);

    let user_sender = request.response.clone();
    let is_streaming = request.is_streaming;

    let mut probe = request.clone();
    if let Some(ref opts) = web_search_options {
        probe
            .tools
            .get_or_insert_with(Vec::new)
            .extend(search::get_search_tools(opts).unwrap());
    }

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

    if let Some(addendum) = required_files_tool_addendum(&required_files) {
        if let Some(tools) = probe.tools.as_mut() {
            for t in tools.iter_mut() {
                if is_code_exec_tool(&t.function.name) {
                    let desc = t.function.description.get_or_insert_with(String::new);
                    desc.push_str(&addendum);
                }
            }
        }
    }

    probe.tool_choice = Some(ToolChoice::Auto);
    probe.web_search_options = None;

    let mut visible_req = probe.clone();
    visible_req.response = user_sender.clone();

    let this_clone = this.clone();
    let handle = tokio::spawn(async move {
        let tool_call_ctx = mistralrs_mcp::ToolCallContext {
            session_id: Some(session_id.clone()),
            round: None,
            tool_name: None,
            agent_permission: Some(agent_permission),
            agent_approval_notifier,
            code_execution_permission,
            code_execution_approval_notifier,
        };
        let dispatch_ctx = DispatchCtx {
            engine: &this_clone,
            web_search_options: web_search_options.as_ref(),
            dispatch_url: dispatch_url.as_deref(),
            supports_vision,
            supports_video,
            tool_call_ctx: &tool_call_ctx,
            run_id: &run_id,
            turn,
            session_id: &session_id,
            required_files: &required_files,
            agent_permission,
            agent_approval_handler,
        };

        let mut current = probe;
        let max_rounds = current.max_tool_rounds.unwrap_or(DEFAULT_MAX_TOOL_ROUNDS);
        let mut round = 0;

        loop {
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            current.response = sender;

            // Prevent the inner probe from re-entering the agentic loop or being rejected
            // by the files-without-agentic-surface guard in `add_request`.
            current.web_search_options = None;
            current.enable_code_execution = false;
            current.max_tool_rounds = AGENTIC_LOOP_REENTRY_SENTINEL;
            current.tool_dispatch_url = None;
            current.files = None;
            let _ = this_clone
                .tx
                .send(crate::request::Request::Normal(Box::new(current)))
                .await;

            if !is_streaming {
                let Some(resp) = receiver.recv().await else {
                    tracing::warn!("Engine closed without sending a response.");
                    return;
                };
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

                if tc_opt.is_none() || round >= max_rounds {
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    final_resp.session_id = Some(session_id.clone());
                    let _ = user_sender.send(Response::Done(final_resp)).await;
                    return;
                }

                let tc = tc_opt.unwrap();

                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Calling(calling_data_for_tool(tc)),
                    })
                    .await;
                tokio::task::yield_now().await;

                let approval = approve_agent_tool(&dispatch_ctx, tc, round).await;
                let outcome = if approval.approve {
                    dispatch_tool(&dispatch_ctx, visible_req.clone(), tc, round).await
                } else {
                    Some(denied_tool_result(
                        visible_req.clone(),
                        tc,
                        approval
                            .message
                            .unwrap_or_else(|| "Agent action was denied.".to_string()),
                    ))
                };
                let Some((next_visible, complete_data, files)) = outcome else {
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    final_resp.session_id = Some(session_id.clone());
                    let _ = user_sender.send(Response::Done(final_resp)).await;
                    return;
                };

                emit_files(&this_clone, &session_id, files, &user_sender).await;

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
            } else {
                // Hold the finish-reason chunk so we can stamp the session ID on it if this is the final round.
                let mut last_choice = None;
                let mut held_final_chunk: Option<crate::ChatCompletionChunkResponse> = None;

                while let Some(resp) = receiver.recv().await {
                    let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                        return;
                    };
                    match resp {
                        Response::Chunk(chunk) => {
                            // Suppress tool-call chunks. Forwarding them would surface a premature finish_reason before the tool loop continues.
                            let first_choice = &chunk.choices[0];
                            let is_final = first_choice.finish_reason.is_some();
                            if first_choice.delta.tool_calls.is_none() {
                                if is_final {
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
                            let _ = user_sender.send(other).await;
                            return;
                        }
                    }
                }

                let Some(choice) = last_choice else {
                    tracing::warn!("Engine closed without sending any chunks.");
                    save_session(&this_clone, &session_id, &visible_req);
                    break;
                };

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

                if tc_opt.is_none() || round >= max_rounds {
                    save_session(&this_clone, &session_id, &visible_req);
                    if let Some(mut final_chunk) = held_final_chunk {
                        final_chunk.session_id = Some(session_id.clone());
                        let _ = user_sender.send(Response::Chunk(final_chunk)).await;
                    }
                    break;
                }

                let tc = tc_opt.unwrap();

                let _ = user_sender
                    .send(Response::AgenticToolCallProgress {
                        round,
                        tool_name: tc.function.name.clone(),
                        phase: AgenticToolCallPhase::Calling(calling_data_for_tool(tc)),
                    })
                    .await;
                tokio::task::yield_now().await;

                let approval = approve_agent_tool(&dispatch_ctx, tc, round).await;
                let outcome = if approval.approve {
                    dispatch_tool(&dispatch_ctx, visible_req.clone(), tc, round).await
                } else {
                    Some(denied_tool_result(
                        visible_req.clone(),
                        tc,
                        approval
                            .message
                            .unwrap_or_else(|| "Agent action was denied.".to_string()),
                    ))
                };
                let Some((next_visible, complete_data, files)) = outcome else {
                    save_session(&this_clone, &session_id, &visible_req);
                    break;
                };

                emit_files(&this_clone, &session_id, files, &user_sender).await;

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
