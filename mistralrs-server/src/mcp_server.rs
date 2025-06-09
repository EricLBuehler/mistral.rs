use async_trait::async_trait;
use rust_mcp_sdk::{
    mcp_server::{hyper_server, HyperServerOptions, ServerHandler},
    schema::{
        schema_utils::CallToolError, CallToolRequest, CallToolResult, CallToolResultContentItem,
        Implementation, InitializeResult, ServerCapabilities, ServerCapabilitiesTools, TextContent,
        LATEST_PROTOCOL_VERSION,
    },
};
use std::io;

use mistralrs_server_core::{
    chat_completion::{create_response_channel, parse_request},
    types::SharedMistralRsState,
};

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
        instructions: Some("use tool 'chat'".to_string()),
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
