use crate::client::McpServerConnection;

use rmcp::transport::common::client_side_sse::ExponentialBackoff;
use rmcp::{
    model::{
        CallToolRequestParam, ClientCapabilities, ClientInfo, ClientRequest, Implementation,
        PingRequest, ReadResourceRequestParam, Tool as McpSdkTool,
    },
    ServiceExt,
};

use crate::McpToolInfo;
use anyhow::Result;
use rust_mcp_schema::Resource;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// SSE MCP server connection using rmcp SDK
pub struct SseMcpConnection {
    client:
        Arc<rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>>,
    server_id: String,
    server_name: String,
}

impl SseMcpConnection {
    /// Create a new SSE MCP connection.
    ///
    /// The configuration options max_times, base_duration, and use_message_endpoint are parsed and
    /// logged for future enhanced SSE support.
    pub async fn new(
        server_id: String,
        server_name: String,
        uri: String,
        max_times: Option<usize>,
        base_duration: Option<std::time::Duration>,
        use_message_endpoint: Option<String>,
    ) -> Result<Self> {
        tracing::info!(
            ?max_times,
            ?base_duration,
            ?use_message_endpoint,
            "SSE connection configuration"
        );
        let retry_policy = ExponentialBackoff {
            max_times: Some(max_times.unwrap_or(3)),
            base_duration: base_duration.unwrap_or_else(|| Duration::from_millis(100)),
        };
        let config = rmcp::transport::sse_client::SseClientConfig {
            sse_endpoint: Arc::<str>::from(uri.as_str()),
            retry_policy: Arc::new(retry_policy),
            use_message_endpoint,
        };
        // TODO: Integrate max_times, base_duration, and use_message_endpoint into SSE transport when supported.
        let transport = rmcp::transport::SseClientTransport::start_with_client(
            reqwest::Client::default(),
            config,
        )
        .await?;
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: server_name.clone(),
                version: "0.0.1".to_string(),
            },
        };
        let client: Arc<
            rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>,
        > = Arc::new(client_info.serve(transport).await?);

        Ok(Self {
            client,
            server_id,
            server_name,
        })
    }
}

#[async_trait::async_trait]
impl McpServerConnection for SseMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }
    fn server_name(&self) -> &str {
        &self.server_name
    }
    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self.client.list_tools(Default::default()).await?;
        let tools: Vec<McpSdkTool> = result.tools;
        Ok(tools
            .into_iter()
            .map(|t| McpToolInfo {
                name: t.name.to_string(),
                description: t.description.as_ref().map(|c| c.to_string()),
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
                annotations: t
                    .annotations
                    .as_ref()
                    .map(|a| serde_json::to_value(a).unwrap_or(Value::Null)),
            })
            .collect())
    }
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .client
            .call_tool(CallToolRequestParam {
                name: name.to_string().into(),
                arguments: arguments.as_object().cloned(),
            })
            .await?;
        Ok(format!("{result:?}"))
    }
    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self.client.list_resources(None).await?;
        let resources = result
            .resources
            .into_iter()
            .map(|res| {
                let value = serde_json::to_value(res)?;
                let resource = serde_json::from_value(value)?;
                Ok(resource)
            })
            .collect::<Result<Vec<Resource>, anyhow::Error>>()?;
        Ok(resources)
    }
    // TODO: Revise this method to handle multiple resource contents.
    // Currently, only the first content (text or blob) is returned as a String.
    // Consider supporting returning all contents or handling multi-part resources.
    async fn read_resource(&self, uri: &str) -> Result<String> {
        let result = self
            .client
            .read_resource(ReadResourceRequestParam {
                uri: uri.to_string(),
            })
            .await?;
        let contents = result.contents;
        if let Some(content) = contents.into_iter().next() {
            match content {
                rmcp::model::ResourceContents::TextResourceContents { text, .. } => Ok(text),
                rmcp::model::ResourceContents::BlobResourceContents { blob, .. } => Ok(blob),
            }
        } else {
            Ok(String::new())
        }
    }

    async fn ping(&self) -> Result<()> {
        self.client
            .send_request(ClientRequest::PingRequest(PingRequest::default()))
            .await?;
        Ok(())
    }
}
