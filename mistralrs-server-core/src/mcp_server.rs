//! MCP server support: exposes the loaded model as an MCP tool over HTTP JSON-RPC.

use std::sync::Arc;

use axum::{extract::State, response::Json, routing::post, Router};
use mistralrs_core::{Response, SupportedModality};
use serde_json::{json, Value};

use crate::{
    chat_completion::{parse_request, ChatCompletionParseContext},
    handler_core::{create_response_channel, send_request},
    openai::{ChatCompletionRequest, OpenAiToolSurface},
    types::SharedMistralRsState,
};

pub const MCP_ROUTE: &str = "/mcp";
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

const JSONRPC_VERSION: &str = "2.0";
const CHAT_TOOL_NAME: &str = "chat";

const INVALID_REQUEST: i32 = -32600;
const METHOD_NOT_FOUND: i32 = -32601;
const INTERNAL_ERROR: i32 = -32603;

const MCP_INSTRUCTIONS: &str = r#"
This server provides LLM text and multimodal model inference. You can use the following tools:
- `chat` for sending a chat completion request with a model message history
"#;

#[derive(serde::Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(serde::Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(serde::Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn ok_response(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: JSONRPC_VERSION,
        id,
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Option<Value>, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: JSONRPC_VERSION,
        id,
        result: None,
        error: Some(JsonRpcError { code, message }),
    }
}

fn initialize_result() -> Value {
    json!({
        "capabilities": { "tools": {} },
        "instructions": MCP_INSTRUCTIONS,
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": { "name": "mistralrs", "version": env!("CARGO_PKG_VERSION") },
    })
}

fn chat_tool() -> Value {
    json!({
        "name": CHAT_TOOL_NAME,
        "description": "Send a chat completion request with messages and other hyperparameters.",
        "inputSchema": {
            "type": "object",
            "required": ["messages"],
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages so far",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": { "type": "string", "enum": ["user", "assistant", "system"] },
                            "content": { "type": "string" }
                        }
                    }
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate"
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature between 0 and 2",
                    "minimum": 0.0,
                    "maximum": 2.0
                }
            }
        }
    })
}

fn list_tools_result(chat_enabled: bool) -> Value {
    let tools = if chat_enabled {
        vec![chat_tool()]
    } else {
        Vec::new()
    };
    json!({ "tools": tools })
}

// Handles every method that does not need model access; tools/call returns None for the caller.
fn dispatch_stateless(
    method: &str,
    id: Option<Value>,
    chat_enabled: bool,
) -> Option<JsonRpcResponse> {
    match method {
        "initialize" => Some(ok_response(id, initialize_result())),
        "ping" => Some(ok_response(id, json!({}))),
        "tools/list" => Some(ok_response(id, list_tools_result(chat_enabled))),
        "tools/call" => None,
        other => Some(error_response(
            id,
            METHOD_NOT_FOUND,
            format!("Method not found: {other}"),
        )),
    }
}

struct McpState {
    mistralrs: SharedMistralRsState,
    chat_enabled: bool,
}

/// Build the MCP router (`POST /mcp`, JSON-RPC 2.0). Mount on its own port or into an existing app.
pub fn create_mcp_router(mistralrs: SharedMistralRsState) -> Router {
    let chat_enabled = mistralrs
        .config(None)
        .map(|c| {
            c.modalities.input.contains(&SupportedModality::Text)
                && c.modalities.output.contains(&SupportedModality::Text)
        })
        .unwrap_or(false);
    let state = Arc::new(McpState {
        mistralrs,
        chat_enabled,
    });
    Router::new()
        .route(MCP_ROUTE, post(handle_jsonrpc))
        .with_state(state)
}

async fn handle_jsonrpc(
    State(state): State<Arc<McpState>>,
    Json(request): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    Json(handle_request(&state, request).await)
}

async fn handle_request(state: &McpState, request: JsonRpcRequest) -> JsonRpcResponse {
    if request.jsonrpc != JSONRPC_VERSION {
        return error_response(
            request.id,
            INVALID_REQUEST,
            "Expected jsonrpc to be 2.0".to_string(),
        );
    }

    if let Some(response) =
        dispatch_stateless(&request.method, request.id.clone(), state.chat_enabled)
    {
        return response;
    }

    let params = request.params.unwrap_or(json!({}));
    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if tool_name != CHAT_TOOL_NAME || !state.chat_enabled {
        return error_response(
            request.id,
            METHOD_NOT_FOUND,
            format!("Unknown tool: {tool_name}"),
        );
    }

    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    match call_chat_tool(&state.mistralrs, args).await {
        Ok(result) => ok_response(request.id, result),
        Err(msg) => error_response(
            request.id,
            INTERNAL_ERROR,
            format!("Tool execution error: {msg}"),
        ),
    }
}

async fn call_chat_tool(state: &SharedMistralRsState, args: Value) -> Result<Value, String> {
    let chat_req: ChatCompletionRequest =
        serde_json::from_value(args).map_err(|e| e.to_string())?;

    let (tx, mut rx) = create_response_channel(None);
    let (request, _is_streaming) = parse_request(
        chat_req,
        ChatCompletionParseContext {
            state: state.clone(),
            tx,
            tool_dispatch_url: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            tool_surface: OpenAiToolSurface::ChatCompletions,
            skill_store: None,
        },
    )
    .await
    .map_err(|e| e.to_string())?;
    send_request(state, request)
        .await
        .map_err(|e| e.to_string())?;

    loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress { .. })
            | Some(Response::BlockDenoisingProgress(_))
            | Some(Response::File(_)) => continue,
            Some(Response::Done(resp)) => {
                let content = resp
                    .choices
                    .iter()
                    .filter_map(|c| c.message.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(json!({ "content": [{ "type": "text", "text": content }] }));
            }
            Some(Response::ModelError(msg, _)) => return Err(msg),
            Some(_) | None => return Err("no response".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(method: &str, chat_enabled: bool) -> JsonRpcResponse {
        dispatch_stateless(method, Some(json!(1)), chat_enabled).unwrap()
    }

    #[test]
    fn initialize_reports_protocol_and_tools_capability() {
        let resp = dispatch("initialize", true);
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], "mistralrs");
        assert!(resp.error.is_none());
    }

    #[test]
    fn tools_list_exposes_chat_for_text_models() {
        let result = dispatch("tools/list", true).result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], CHAT_TOOL_NAME);
        assert_eq!(tools[0]["inputSchema"]["required"], json!(["messages"]));
    }

    #[test]
    fn tools_list_is_empty_without_text_modality() {
        let result = dispatch("tools/list", false).result.unwrap();
        assert!(result["tools"].as_array().unwrap().is_empty());
    }

    #[test]
    fn ping_returns_empty_object() {
        assert_eq!(dispatch("ping", true).result.unwrap(), json!({}));
    }

    #[test]
    fn unknown_method_is_a_jsonrpc_error() {
        let resp = dispatch("bogus", true);
        assert!(resp.result.is_none());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn tools_call_is_deferred_to_the_stateful_path() {
        assert!(dispatch_stateless("tools/call", None, true).is_none());
    }

    #[test]
    fn error_envelope_omits_result() {
        let resp = error_response(Some(json!(7)), INVALID_REQUEST, "bad".to_string());
        let wire = serde_json::to_value(&resp).unwrap();
        assert_eq!(wire["jsonrpc"], "2.0");
        assert_eq!(wire["error"]["code"], INVALID_REQUEST);
        assert!(wire.get("result").is_none());
    }
}
