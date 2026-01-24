use crate::tools::{Function, Tool, ToolCallback, ToolCallbackWithTool, ToolType};
use crate::transport::{HttpTransport, McpTransport, ProcessTransport, WebSocketTransport};
use crate::types::McpToolResult;
use crate::{McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo};
use anyhow::Result;
use rust_mcp_schema::Resource;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::warn;

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

    /// Close the server connection
    async fn close(&self) -> Result<()>;
}

/// MCP client that manages connections to multiple MCP servers
///
/// The main interface for interacting with Model Context Protocol servers.
/// Handles connection lifecycle, tool discovery, and provides integration
/// with tool calling systems.
///
/// # Features
///
/// - **Multi-server Management**: Connects to and manages multiple MCP servers simultaneously
/// - **Automatic Tool Discovery**: Discovers available tools from connected servers
/// - **Tool Registration**: Converts MCP tools to internal Tool format for seamless integration
/// - **Connection Pooling**: Maintains persistent connections for efficient tool execution
/// - **Error Handling**: Robust error handling with proper cleanup and reconnection logic
///
/// # Example
///
/// ```rust,no_run
/// use mistralrs_mcp::{McpClient, McpClientConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = McpClientConfig::default();
///     let mut client = McpClient::new(config);
///     
///     // Initialize all configured server connections
///     client.initialize().await?;
///     
///     // Get tool callbacks for model integration
///     let callbacks = client.get_tool_callbacks_with_tools();
///     
///     Ok(())
/// }
/// ```
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

    /// Get tool callbacks for use with legacy tool calling systems.
    ///
    /// Returns a map of tool names to their callback functions. These callbacks
    /// handle argument parsing, concurrency control, and timeout enforcement
    /// automatically.
    ///
    /// For new integrations, prefer [`Self::get_tool_callbacks_with_tools`] which
    /// includes tool definitions alongside callbacks.
    pub fn get_tool_callbacks(&self) -> &HashMap<String, Arc<ToolCallback>> {
        &self.tool_callbacks
    }

    /// Get tool callbacks paired with their tool definitions.
    ///
    /// This is the primary method for integrating MCP tools with the model's
    /// automatic tool calling system. Each entry contains:
    /// - A callback function that executes the tool with timeout and concurrency controls
    /// - A [`Tool`] definition with name, description, and parameter schema
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// let tools = client.get_tool_callbacks_with_tools();
    /// for (name, tool_with_callback) in tools {
    ///     println!("Tool: {} - {:?}", name, tool_with_callback.tool.function.description);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_tool_callbacks_with_tools(&self) -> &HashMap<String, ToolCallbackWithTool> {
        &self.tool_callbacks_with_tools
    }

    /// Get information about all discovered tools.
    ///
    /// Returns metadata about tools discovered from connected MCP servers,
    /// including their names, descriptions, input schemas, and which server
    /// they came from.
    pub fn get_tools(&self) -> &HashMap<String, McpToolInfo> {
        &self.tools
    }

    /// Get a reference to all connected MCP server connections.
    ///
    /// This provides direct access to server connections, allowing you to:
    /// - List available resources with [`McpServerConnection::list_resources`]
    /// - Read resources with [`McpServerConnection::read_resource`]
    /// - Check server health with [`McpServerConnection::ping`]
    /// - Call tools directly with [`McpServerConnection::call_tool`]
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// for (server_id, connection) in client.servers() {
    ///     println!("Server: {} ({})", connection.server_name(), server_id);
    ///     let resources = connection.list_resources().await?;
    ///     println!("  Resources: {:?}", resources);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn servers(&self) -> &HashMap<String, Arc<dyn McpServerConnection>> {
        &self.servers
    }

    /// Get a specific server connection by its ID.
    ///
    /// Returns `None` if no server with the given ID is connected.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// if let Some(server) = client.server("my_server_id") {
    ///     server.ping().await?;
    ///     let resources = server.list_resources().await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn server(&self, id: &str) -> Option<&Arc<dyn McpServerConnection>> {
        self.servers.get(id)
    }

    /// Get the client configuration.
    pub fn config(&self) -> &McpClientConfig {
        &self.config
    }

    /// Create connection based on server source type
    async fn create_connection(
        &self,
        config: &McpServerConfig,
    ) -> Result<Arc<dyn McpServerConnection>> {
        match &config.source {
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

                let connection = HttpMcpConnection::new(
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
                let connection = ProcessMcpConnection::new(
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

                let connection = WebSocketMcpConnection::new(
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

                // Convert MCP tool schema to Tool definition
                let function_def = Function {
                    name: tool_name.clone(),
                    description: tool.description.clone(),
                    parameters: Self::convert_mcp_schema_to_parameters(&tool.input_schema),
                };

                let tool_def = Tool {
                    tp: ToolType::Function,
                    function: function_def,
                };

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

    /// Convert MCP tool input schema to Tool parameters format
    fn convert_mcp_schema_to_parameters(
        schema: &serde_json::Value,
    ) -> Option<HashMap<String, serde_json::Value>> {
        // MCP tools can have various schema formats, we'll try to convert common ones
        match schema {
            serde_json::Value::Object(obj) => {
                let mut params = HashMap::new();

                // If it's a JSON schema object, extract properties
                if let Some(properties) = obj.get("properties") {
                    if let serde_json::Value::Object(props) = properties {
                        for (key, value) in props {
                            params.insert(key.clone(), value.clone());
                        }
                    }
                } else {
                    // If it's just a direct object, use it as-is
                    for (key, value) in obj {
                        params.insert(key.clone(), value.clone());
                    }
                }

                if params.is_empty() {
                    None
                } else {
                    Some(params)
                }
            }
            _ => {
                // For non-object schemas, we can't easily convert to parameters
                None
            }
        }
    }

    /// Remove tools associated with a specific server
    fn remove_tools_for_server(&mut self, server_id: &str) {
        let tools_to_remove: Vec<String> = self
            .tools
            .iter()
            .filter(|(_, info)| info.server_id == server_id)
            .map(|(name, _)| name.clone())
            .collect();

        for name in tools_to_remove {
            self.tools.remove(&name);
            self.tool_callbacks.remove(&name);
            self.tool_callbacks_with_tools.remove(&name);
        }
    }

    /// Register tools for a single server
    async fn register_tools_for_server(&mut self, server_id: &str) -> Result<()> {
        let connection = self
            .servers
            .get(server_id)
            .ok_or_else(|| anyhow::anyhow!("Server not connected: {}", server_id))?
            .clone();

        let server_config = self
            .config
            .servers
            .iter()
            .find(|s| s.id == server_id)
            .ok_or_else(|| anyhow::anyhow!("Server config not found for {}", server_id))?
            .clone();

        let tools = connection.list_tools().await?;

        for tool in tools {
            let tool_name = if let Some(prefix) = &server_config.tool_prefix {
                format!("{}_{}", prefix, tool.name)
            } else {
                tool.name.clone()
            };

            // Create tool callback that calls the MCP server with timeout and concurrency controls
            let connection_clone = Arc::clone(&connection);
            let original_tool_name = tool.name.clone();
            let semaphore_clone = Arc::clone(&self.concurrency_semaphore);
            let timeout_duration = Duration::from_secs(self.config.tool_timeout_secs.unwrap_or(30));

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
                        let _permit = semaphore
                            .acquire()
                            .await
                            .map_err(|_| anyhow::anyhow!("Failed to acquire concurrency permit"))?;

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

            // Convert MCP tool schema to Tool definition
            let function_def = Function {
                name: tool_name.clone(),
                description: tool.description.clone(),
                parameters: Self::convert_mcp_schema_to_parameters(&tool.input_schema),
            };

            let tool_def = Tool {
                tp: ToolType::Function,
                function: function_def,
            };

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

        Ok(())
    }

    // ==================== Connection Management Methods ====================

    /// Gracefully shutdown all server connections.
    ///
    /// Closes all active connections and clears the tools and callbacks.
    /// The client cannot be used after calling this method without re-initializing.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // ... use the client ...
    ///
    /// // Gracefully shutdown when done
    /// client.shutdown().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn shutdown(&mut self) -> Result<()> {
        // Close all connections
        for connection in self.servers.values() {
            let _ = connection.close().await;
        }

        // Clear all state
        self.servers.clear();
        self.tools.clear();
        self.tool_callbacks.clear();
        self.tool_callbacks_with_tools.clear();

        Ok(())
    }

    /// Disconnect a specific server by its ID.
    ///
    /// Removes the server from active connections and clears its associated tools.
    /// Returns an error if the server ID is not found.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // Disconnect a specific server
    /// client.disconnect("my_server_id").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn disconnect(&mut self, id: &str) -> Result<()> {
        let connection = self
            .servers
            .remove(id)
            .ok_or_else(|| anyhow::anyhow!("Server not connected: {}", id))?;

        connection.close().await?;
        self.remove_tools_for_server(id);

        Ok(())
    }

    /// Reconnect to a specific server by its ID.
    ///
    /// Re-establishes the connection using the stored configuration.
    /// Returns an error if the server ID is not in the configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // Reconnect to a server after it was disconnected or lost connection
    /// client.reconnect("my_server_id").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn reconnect(&mut self, id: &str) -> Result<()> {
        // Find the server config
        let server_config = self
            .config
            .servers
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| anyhow::anyhow!("Server config not found: {}", id))?
            .clone();

        // Close existing connection if any
        if let Some(connection) = self.servers.remove(id) {
            let _ = connection.close().await;
        }

        // Remove old tools for this server
        self.remove_tools_for_server(id);

        // Create new connection
        let connection = self.create_connection(&server_config).await?;
        self.servers.insert(id.to_string(), connection);

        // Re-register tools if auto_register_tools is enabled
        if self.config.auto_register_tools {
            self.register_tools_for_server(id).await?;
        }

        Ok(())
    }

    /// Check if a specific server is currently connected.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// if client.is_connected("my_server_id") {
    ///     println!("Server is connected");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_connected(&self, id: &str) -> bool {
        self.servers.contains_key(id)
    }

    /// Dynamically add and connect a new server at runtime.
    ///
    /// Adds the server configuration and establishes the connection.
    /// If auto_register_tools is enabled, discovers and registers the server's tools.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig, McpServerConfig, McpServerSource};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // Add a new server dynamically
    /// let new_server = McpServerConfig {
    ///     id: "new_server".to_string(),
    ///     name: "New MCP Server".to_string(),
    ///     source: McpServerSource::Http {
    ///         url: "https://api.example.com/mcp".to_string(),
    ///         timeout_secs: Some(30),
    ///         headers: None,
    ///     },
    ///     ..Default::default()
    /// };
    /// client.add_server(new_server).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_server(&mut self, config: McpServerConfig) -> Result<()> {
        let id = config.id.clone();

        // Check if server already exists
        if self.servers.contains_key(&id) {
            return Err(anyhow::anyhow!("Server already exists: {}", id));
        }

        // Create connection
        let connection = self.create_connection(&config).await?;
        self.servers.insert(id.clone(), connection);

        // Store config
        self.config.servers.push(config);

        // Register tools if enabled
        if self.config.auto_register_tools {
            self.register_tools_for_server(&id).await?;
        }

        Ok(())
    }

    /// Disconnect and remove a server from the client.
    ///
    /// Closes the connection and removes the server from the configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // Remove a server completely
    /// client.remove_server("my_server_id").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn remove_server(&mut self, id: &str) -> Result<()> {
        // Disconnect first
        if let Some(connection) = self.servers.remove(id) {
            let _ = connection.close().await;
        }

        // Remove tools
        self.remove_tools_for_server(id);

        // Remove from config
        self.config.servers.retain(|s| s.id != id);

        Ok(())
    }

    // ==================== Tool Management Methods ====================

    /// Re-discover tools from all connected servers.
    ///
    /// Clears existing tool registrations and re-queries all servers.
    /// Useful for long-running clients when servers update their tools.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// // Refresh tools after servers have been updated
    /// client.refresh_tools().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn refresh_tools(&mut self) -> Result<()> {
        // Clear all existing tools
        self.tools.clear();
        self.tool_callbacks.clear();
        self.tool_callbacks_with_tools.clear();

        // Re-discover tools from all servers
        self.discover_and_register_tools().await
    }

    /// Get a specific tool by name.
    ///
    /// Returns `None` if no tool with the given name is registered.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// if let Some(tool) = client.get_tool("web_search") {
    ///     println!("Found tool: {:?}", tool.description);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_tool(&self, name: &str) -> Option<&McpToolInfo> {
        self.tools.get(name)
    }

    /// Check if a tool with the given name exists.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// if client.has_tool("web_search") {
    ///     println!("Tool is available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Directly call a tool by name with the given arguments.
    ///
    /// This bypasses the callback system and calls the tool directly
    /// on the appropriate server with timeout and concurrency controls.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # use serde_json::json;
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// let result = client.call_tool("web_search", json!({"query": "rust programming"})).await?;
    /// println!("Result: {}", result);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let tool_info = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", name))?;

        let connection = self
            .servers
            .get(&tool_info.server_id)
            .ok_or_else(|| anyhow::anyhow!("Server not connected: {}", tool_info.server_id))?;

        // Acquire semaphore permit for concurrency control
        let _permit = self
            .concurrency_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("Failed to acquire concurrency permit"))?;

        let timeout_duration = Duration::from_secs(self.config.tool_timeout_secs.unwrap_or(30));

        // Execute tool call with timeout
        match tokio::time::timeout(
            timeout_duration,
            connection.call_tool(&tool_info.name, arguments),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!(
                "Tool call timed out after {} seconds",
                timeout_duration.as_secs()
            )),
        }
    }

    /// Get the total number of registered tools.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// println!("Total tools: {}", client.tool_count());
    /// # Ok(())
    /// # }
    /// ```
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    // ==================== Status / Convenience Methods ====================

    /// Get the number of connected servers.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// println!("Connected servers: {}", client.server_count());
    /// # Ok(())
    /// # }
    /// ```
    pub fn server_count(&self) -> usize {
        self.servers.len()
    }

    /// Get a list of all connected server IDs.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// for id in client.server_ids() {
    ///     println!("Server: {}", id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn server_ids(&self) -> Vec<&str> {
        self.servers.keys().map(|s| s.as_str()).collect()
    }

    /// Ping all connected servers and return results per server.
    ///
    /// Returns a map of server ID to ping result. Useful for health monitoring.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// let results = client.ping_all().await;
    /// for (server_id, result) in results {
    ///     match result {
    ///         Ok(()) => println!("{}: healthy", server_id),
    ///         Err(e) => println!("{}: unhealthy - {}", server_id, e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn ping_all(&self) -> HashMap<String, Result<()>> {
        let mut results = HashMap::new();

        for (server_id, connection) in &self.servers {
            let result = connection.ping().await;
            results.insert(server_id.clone(), result);
        }

        results
    }

    // ==================== Resource Access Methods ====================

    /// List resources from all connected servers.
    ///
    /// Returns a vector of (server_id, resource) tuples.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mistralrs_mcp::{McpClient, McpClientConfig};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = McpClientConfig::default();
    /// let mut client = McpClient::new(config);
    /// client.initialize().await?;
    ///
    /// let resources = client.list_all_resources().await?;
    /// for (server_id, resource) in resources {
    ///     println!("Server {}: {:?}", server_id, resource);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_all_resources(&self) -> Result<Vec<(String, Resource)>> {
        let mut all_resources = Vec::new();

        for (server_id, connection) in &self.servers {
            match connection.list_resources().await {
                Ok(resources) => {
                    for resource in resources {
                        all_resources.push((server_id.clone(), resource));
                    }
                }
                Err(e) => {
                    // Log error but continue with other servers
                    warn!("Failed to list resources from server {}: {}", server_id, e);
                }
            }
        }

        Ok(all_resources)
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Try to get the tokio runtime handle
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let servers = std::mem::take(&mut self.servers);
            handle.spawn(async move {
                for (_, connection) in servers {
                    let _ = connection.close().await;
                }
            });
        }
    }
}

/// HTTP-based MCP server connection
pub struct HttpMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl HttpMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = HttpTransport::new(url, timeout_secs, headers)?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": rust_mcp_schema::ProtocolVersion::latest().to_string(),
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        self.transport.send_initialization_notification().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for HttpMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!("Tool execution failed: {tool_result}"));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

/// Process-based MCP server connection
pub struct ProcessMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl ProcessMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        command: String,
        args: Vec<String>,
        work_dir: Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = ProcessTransport::new(command, args, work_dir, env).await?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": rust_mcp_schema::ProtocolVersion::latest().to_string(),
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        self.transport.send_initialization_notification().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for ProcessMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!("Tool execution failed: {tool_result}"));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

/// WebSocket-based MCP server connection
pub struct WebSocketMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl WebSocketMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = WebSocketTransport::new(url, timeout_secs, headers).await?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": rust_mcp_schema::ProtocolVersion::latest().to_string(),
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        self.transport.send_initialization_notification().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for WebSocketMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!("Tool execution failed: {tool_result}"));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}
