use async_trait::async_trait;
use axum::{extract::State, http::StatusCode, response::Json, routing::post, Router};
use mistralrs_core::SupportedModality;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use tokio::net::TcpListener;

use mistralrs_server_core::{
    chat_completion::parse_request, handler_core::create_response_channel,
    types::SharedMistralRsState,
};

// Import your existing types
use rust_mcp_sdk::schema::{
    schema_utils::CallToolError, CallToolResult, CallToolResultContentItem, Implementation,
    InitializeResult, ListToolsResult, ServerCapabilities, ServerCapabilitiesTools, TextContent,
    Tool, ToolInputSchema, LATEST_PROTOCOL_VERSION,
};

mod errors {
    #![allow(dead_code)]

    /// JSON-RPC error codes based on MCPEx.Protocol.Errors
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

// JSON-RPC types
#[derive(serde::Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    params: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(serde::Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

// Keep your existing McpTool trait and ChatTool implementation
#[async_trait]
pub trait McpTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    fn input_schema(&self) -> &ToolInputSchema;

    fn as_tool_record(&self) -> Tool {
        Tool {
            name: self.name().to_string(),
            description: self.description().map(|s| s.to_string()),
            input_schema: self.input_schema().clone(),
            annotations: None,
        }
    }

    async fn call(
        &self,
        args: serde_json::Value,
        state: &SharedMistralRsState,
    ) -> std::result::Result<CallToolResult, CallToolError>;
}

pub struct ChatTool {
    input_schema: ToolInputSchema,
}

impl ChatTool {
    pub fn new() -> Self {
        let required = vec!["messages".to_string()];

        let mut properties: HashMap<String, Map<String, Value>> = HashMap::new();
        properties.insert(
            "messages".to_string(),
            json!({
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
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        properties.insert(
            "maxTokens".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum number of tokens to generate"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        properties.insert(
            "temperature".to_string(),
            json!({
                "type": "number",
                "description": "Sampling temperature between 0 and 2",
                "minimum": 0.0,
                "maximum": 2.0
            })
            .as_object()
            .unwrap()
            .clone(),
        );

        let input_schema = ToolInputSchema::new(required, Some(properties));
        Self { input_schema }
    }
}

#[async_trait]
impl McpTool for ChatTool {
    fn name(&self) -> &str {
        "chat"
    }

    fn description(&self) -> Option<&str> {
        Some("Send a chat completion request with messages and other hyperparameters.")
    }

    fn input_schema(&self) -> &ToolInputSchema {
        &self.input_schema
    }

    async fn call(
        &self,
        args: serde_json::Value,
        state: &SharedMistralRsState,
    ) -> std::result::Result<CallToolResult, CallToolError> {
        // Translate to the internal ChatCompletionRequest.
        let chat_req: mistralrs_server_core::openai::ChatCompletionRequest =
            serde_json::from_value(args).map_err(CallToolError::new)?;

        // Execute the request using existing helper utilities.
        let (tx, mut rx) = create_response_channel(None);
        let (request, _is_streaming) = parse_request(chat_req, state.clone(), tx)
            .await
            .map_err(|e| CallToolError::new(io::Error::other(e.to_string())))?;

        mistralrs_server_core::handler_core::send_request(state, request)
            .await
            .map_err(|e| CallToolError::new(io::Error::other(e.to_string())))?;

        match rx.recv().await {
            Some(mistralrs_core::Response::Done(resp)) => {
                let content = resp
                    .choices
                    .iter()
                    .filter_map(|c| c.message.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(CallToolResult {
                    content: vec![CallToolResultContentItem::TextContent(TextContent::new(
                        content, None,
                    ))],
                    is_error: None,
                    meta: None,
                })
            }
            Some(mistralrs_core::Response::ModelError(msg, _)) => {
                Err(CallToolError::new(io::Error::other(msg)))
            }
            Some(_) | None => Err(CallToolError::new(io::Error::other("no response"))),
        }
    }
}

const MCP_INSTRUCTIONS: &str = r#"
This server provides LLM text and multimodal model inference. You can use the following tools:
- `chat` for sending a chat completion request with a model message history
"#;

// HTTP MCP Handler
pub struct HttpMcpHandler {
    pub state: SharedMistralRsState,
    tools: HashMap<String, Arc<dyn McpTool>>,
    server_info: InitializeResult,
}

impl HttpMcpHandler {
    pub fn new(state: SharedMistralRsState) -> Self {
        let modalities = &state.config(None).unwrap().modalities;

        let mut tools: HashMap<String, Arc<dyn McpTool>> = HashMap::new();
        if modalities.input.contains(&SupportedModality::Text)
            && modalities.output.contains(&SupportedModality::Text)
        {
            tools.insert("chat".to_string(), Arc::new(ChatTool::new()));
        }

        let server_info = InitializeResult {
            server_info: Implementation {
                name: "mistralrs".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ServerCapabilitiesTools { list_changed: None }),
                ..Default::default()
            },
            meta: None,
            instructions: Some(MCP_INSTRUCTIONS.to_string()),
            protocol_version: LATEST_PROTOCOL_VERSION.to_string(),
        };

        Self {
            state,
            tools,
            server_info,
        }
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        if request.jsonrpc != "2.0" {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: errors::INVALID_REQUEST,
                    message: "Expected jsonrpc to be 2.0".to_string(),
                    data: None,
                }),
            };
        }

        match request.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(serde_json::to_value(&self.server_info).unwrap()),
                error: None,
            },
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(json!({})),
                error: None,
            },
            "tools/list" => {
                let tools: Vec<Tool> = self.tools.values().map(|t| t.as_tool_record()).collect();
                let result = ListToolsResult {
                    tools,
                    meta: None,
                    next_cursor: None,
                };
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(serde_json::to_value(result).unwrap()),
                    error: None,
                }
            }
            "tools/call" => {
                let params = request.params.unwrap_or(json!({}));

                // Extract tool name and arguments from params
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));

                match self.tools.get(tool_name) {
                    Some(tool) => match tool.call(args, &self.state).await {
                        Ok(result) => JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request.id,
                            result: Some(serde_json::to_value(result).unwrap()),
                            error: None,
                        },
                        Err(e) => JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: errors::INTERNAL_ERROR,
                                message: format!("Tool execution error: {e}"),
                                data: None,
                            }),
                        },
                    },
                    None => JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: errors::METHOD_NOT_FOUND,
                            message: format!("Unknown tool: {tool_name}"),
                            data: None,
                        }),
                    },
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: errors::METHOD_NOT_FOUND,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        }
    }
}

// Axum handler
async fn handle_jsonrpc(
    State(handler): State<Arc<HttpMcpHandler>>,
    Json(request): Json<JsonRpcRequest>,
) -> Result<Json<JsonRpcResponse>, StatusCode> {
    let response = handler.handle_request(request).await;
    Ok(Json(response))
}

// Create HTTP MCP server - this replaces your old create_mcp_server function
pub async fn create_http_mcp_server(
    state: SharedMistralRsState,
    host: String,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let handler = Arc::new(HttpMcpHandler::new(state));

    let app = Router::new()
        .route("/mcp", post(handle_jsonrpc))
        .with_state(handler);

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr).await?;

    axum::serve(listener, app).await?;

    Ok(())
}
