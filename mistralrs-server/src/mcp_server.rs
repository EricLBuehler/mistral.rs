use async_trait::async_trait;
use rust_mcp_sdk::{
    mcp_server::{hyper_server, HyperServerOptions, ServerHandler},
    schema::{
        schema_utils::CallToolError, CallToolRequest, CallToolResult, CallToolResultContentItem,
        Implementation, InitializeResult, ListToolsRequest, ListToolsResult, RpcError,
        ServerCapabilities, ServerCapabilitiesTools, TextContent, Tool, ToolInputSchema,
        LATEST_PROTOCOL_VERSION,
    },
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io;

use mistralrs_server_core::{
    chat_completion::{create_response_channel, parse_request},
    types::SharedMistralRsState,
};

const MCP_INSTRUCTIONS: &str = r#"
This server provides LLM text and multimodal model inference. You can use the following tools:
- `chat` for sending a chat completion request with a model message history
"#;

pub struct MistralMcpHandler {
    pub state: SharedMistralRsState,
}

#[async_trait]
impl ServerHandler for MistralMcpHandler {
    async fn handle_call_tool_request(
        &self,
        request: CallToolRequest,
        _runtime: &dyn rust_mcp_sdk::McpServer,
    ) -> std::result::Result<CallToolResult, CallToolError> {
        if request.params.name != "chat" {
            return Err(CallToolError::unknown_tool(request.params.name));
        }
        let args = request.params.arguments.into();
        let req: rust_mcp_sdk::schema::CreateMessageRequest =
            serde_json::from_value(args).map_err(|e| CallToolError::new(io::Error::other(e)))?;
        // Translate to ChatCompletionRequest
        let chat_req: mistralrs_server_core::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::to_value(req).unwrap())
                .map_err(CallToolError::new)?;

        let (tx, mut rx) = create_response_channel(None);
        let (request, _is_streaming) = parse_request(chat_req, self.state.clone(), tx)
            .await
            .map_err(|e| CallToolError::new(io::Error::other(e.to_string())))?;
        mistralrs_server_core::chat_completion::send_request(&self.state, request)
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

    async fn handle_list_tools_request(
        &self,
        _request: ListToolsRequest,
        _runtime: &dyn rust_mcp_sdk::McpServer,
    ) -> std::result::Result<ListToolsResult, RpcError> {
        // Currently we expose only one tool: `chat`
        // JSON‑Schema description for the `chat` tool arguments
        // Required parameters follow the `CreateMessageRequest` schema: `messages`.
        let required = vec!["messages".to_string()];

        // Build the `properties` map
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
                "description": "Sampling temperature between 0 and 1",
                "minimum": 0.0,
                "maximum": 1.0
            })
            .as_object()
            .unwrap()
            .clone(),
        );

        properties.insert(
            "systemPrompt".to_string(),
            json!({
                "type": "string",
                "description": "Optional system prompt to prepend to the conversation"
            })
            .as_object()
            .unwrap()
            .clone(),
        );

        let input_schema = ToolInputSchema::new(required, Some(properties));

        let tool = Tool {
            name: "chat".to_string(),
            description: Some(
                "Send a chat completion request with messages and other hyperparameters."
                    .to_string(),
            ),
            input_schema,
            annotations: None,
        };

        Ok(ListToolsResult {
            tools: vec![tool],
            meta: None,
            next_cursor: None,
        })
    }
}

pub fn create_mcp_server(
    state: SharedMistralRsState,
    host: String,
    port: u16,
) -> rust_mcp_sdk::mcp_server::HyperServer {
    let server_details = InitializeResult {
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
    let handler = MistralMcpHandler { state };
    let opts = HyperServerOptions {
        host,
        port,
        ..Default::default()
    };
    hyper_server::create_server(server_details, handler, opts)
}
