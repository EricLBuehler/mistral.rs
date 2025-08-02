pub mod http;
pub mod process;
pub mod sse;
pub mod streamable_http;
pub mod websocket;

use crate::tools::{Function, Tool as LocalTool, ToolCallback, ToolCallbackWithTool, ToolType};
use crate::{McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo};
use anyhow::Result;
use rust_mcp_schema::Resource;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Trait for MCP server connections
#[async_trait::async_trait]
pub trait McpServerConnection: Send + Sync {
    /// Get the server ID
    fn server_id(&self) -> &str;

    /// Get the server name
    fn server_name(&self) -> &str;

    /// List available tools from this server
    async fn list_tools(&self) -> Result<Vec<McpToolInfo>>;

    /// Call a tool on this server
    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String>;

    /// List available resources from this server
    async fn list_resources(&self) -> Result<Vec<Resource>>;

    /// Read a resource from this server
    async fn read_resource(&self, uri: &str) -> Result<String>;

    /// Check if the connection is healthy
    async fn ping(&self) -> Result<()>;
}

pub struct McpClient {
    /// Configuration for the client including server list and policies
    config: McpClientConfig,
    /// Active connections to MCP servers, indexed by server ID
    servers: HashMap<String, Arc<dyn McpServerConnection>>,
    /// Registry of discovered tools from all connected servers
    tools: HashMap<String, McpToolInfo>,
    /// Legacy tool callbacks for backward compatibility
    tool_callbacks: HashMap<String, Arc<ToolCallback>>,
    /// Tool callbacks with associated Tool definitions for automatic tool calling
    tool_callbacks_with_tools: HashMap<String, ToolCallbackWithTool>,
    /// Semaphore to control maximum concurrent tool calls
    concurrency_semaphore: Arc<Semaphore>,
}

impl McpClient {
    /// Create a new MCP client with the given configuration
    pub fn new(config: McpClientConfig) -> Self {
        let max_concurrent = config.max_concurrent_calls.unwrap_or(10);
        Self {
            config,
            servers: HashMap::new(),
            tools: HashMap::new(),
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            concurrency_semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Initialize connections to all configured servers
    pub async fn initialize(&mut self) -> Result<()> {
        for server_config in &self.config.servers {
            if server_config.enabled {
                let connection = self.create_connection(server_config).await?;
                self.servers.insert(server_config.id.clone(), connection);
            }
        }

        if self.config.auto_register_tools {
            self.discover_and_register_tools().await?;
        }

        Ok(())
    }

    /// Get tool callbacks that can be used with the existing tool calling system
    pub fn get_tool_callbacks(&self) -> &HashMap<String, Arc<ToolCallback>> {
        &self.tool_callbacks
    }

    /// Get tool callbacks with their associated Tool definitions
    pub fn get_tool_callbacks_with_tools(&self) -> &HashMap<String, ToolCallbackWithTool> {
        &self.tool_callbacks_with_tools
    }

    /// Get discovered tools information
    pub fn get_tools(&self) -> &HashMap<String, McpToolInfo> {
        &self.tools
    }

    /// Create connection based on server source type
    async fn create_connection(
        &self,
        config: &McpServerConfig,
    ) -> Result<Arc<dyn McpServerConnection>> {
        match &config.source {
            McpServerSource::StreamableHttp {
                uri,
                max_times,
                base_duration,
                channel_buffer_capacity,
                allow_stateless,
            } => {
                let connection = streamable_http::StreamableHttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    uri.clone(),
                    *max_times,
                    *base_duration,
                    *channel_buffer_capacity,
                    *allow_stateless,
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Sse {
                uri,
                max_times,
                base_duration,
                use_message_endpoint,
            } => {
                let connection = sse::SseMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    uri.clone(),
                    *max_times,
                    *base_duration,
                    use_message_endpoint.clone(),
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Http {
                url,
                timeout_secs,
                headers,
            } => {
                // Merge Bearer token with existing headers if provided
                let mut merged_headers = headers.clone().unwrap_or_default();
                if let Some(token) = &config.bearer_token {
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {token}"));
                }

                let connection = http::HttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    Some(merged_headers),
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Process {
                command,
                args,
                work_dir,
                env,
            } => {
                let connection = process::ProcessMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    command.clone(),
                    args.clone(),
                    work_dir.clone(),
                    env.clone(),
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::WebSocket {
                url,
                timeout_secs,
                headers,
            } => {
                // Merge Bearer token with existing headers if provided
                let mut merged_headers = headers.clone().unwrap_or_default();
                if let Some(token) = &config.bearer_token {
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {token}"));
                }

                let connection = websocket::WebSocketMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    Some(merged_headers),
                )
                .await?;
                Ok(Arc::new(connection))
            }
        }
    }

    /// Discover tools from all connected servers and register them
    async fn discover_and_register_tools(&mut self) -> Result<()> {
        for (server_id, connection) in &self.servers {
            let tools = connection.list_tools().await?;
            let server_config = self
                .config
                .servers
                .iter()
                .find(|s| &s.id == server_id)
                .ok_or_else(|| anyhow::anyhow!("Server config not found for {}", server_id))?;

            for tool in tools {
                let tool_name = if let Some(prefix) = &server_config.tool_prefix {
                    format!("{}_{}", prefix, tool.name)
                } else {
                    tool.name.clone()
                };

                // Create tool callback that calls the MCP server with timeout and concurrency controls
                let connection_clone = Arc::clone(connection);
                let original_tool_name = tool.name.clone();
                let semaphore_clone = Arc::clone(&self.concurrency_semaphore);
                let timeout_duration =
                    Duration::from_secs(self.config.tool_timeout_secs.unwrap_or(30));

                let callback: Arc<ToolCallback> = Arc::new(move |called_function| {
                    let connection = Arc::clone(&connection_clone);
                    let tool_name = original_tool_name.clone();
                    let semaphore = Arc::clone(&semaphore_clone);
                    let arguments: serde_json::Value =
                        serde_json::from_str(&called_function.arguments)?;

                    // Use tokio::task::spawn_blocking to handle the async-to-sync bridge
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || {
                        rt.block_on(async move {
                            // Acquire semaphore permit for concurrency control
                            let _permit = semaphore.acquire().await.map_err(|_| {
                                anyhow::anyhow!("Failed to acquire concurrency permit")
                            })?;

                            // Execute tool call with timeout
                            match tokio::time::timeout(
                                timeout_duration,
                                connection.call_tool(&tool_name, arguments),
                            )
                            .await
                            {
                                Ok(result) => result,
                                Err(_) => Err(anyhow::anyhow!(
                                    "Tool call timed out after {} seconds",
                                    timeout_duration.as_secs()
                                )),
                            }
                        })
                    })
                    .join()
                    .map_err(|_| anyhow::anyhow!("Tool call thread panicked"))?
                });

                let mut tool_info = tool.clone();
                tool_info.name = tool_name.clone(); // Update name with prefix if present
                let tool_def = tool_from_mcp(&tool_info);

                // Store in both collections for backward compatibility
                self.tool_callbacks
                    .insert(tool_name.clone(), callback.clone());
                self.tool_callbacks_with_tools.insert(
                    tool_name.clone(),
                    ToolCallbackWithTool {
                        callback,
                        tool: tool_def,
                    },
                );
                self.tools.insert(tool_name, tool);
            }
        }

        Ok(())
    }
}

/// Converts an McpToolInfo to an internal Tool definition.
fn tool_from_mcp(info: &McpToolInfo) -> LocalTool {
    LocalTool {
        tp: ToolType::Function,
        function: Function {
            name: info.name.clone(),
            description: info.description.clone(),
            parameters: Some(info.input_schema.clone()),
        },
    }
}
