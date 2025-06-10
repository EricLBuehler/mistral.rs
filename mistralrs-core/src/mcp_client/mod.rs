//! Model Context Protocol (MCP) Client Implementation
//!
//! This module provides a comprehensive client implementation for the Model Context Protocol (MCP),
//! enabling AI assistants to connect to and interact with external tools and resources through
//! standardized server interfaces.
//!
//! # Overview
//!
//! The MCP client supports multiple transport protocols and provides automatic tool discovery,
//! registration, and execution. It seamlessly integrates with the existing tool calling system
//! in mistral.rs, allowing MCP tools to be used alongside built-in tools.
//!
//! # Features
//!
//! - **Multiple Transport Protocols**: HTTP, WebSocket, and Process-based connections
//! - **Automatic Tool Discovery**: Discovers and registers tools from connected MCP servers
//! - **Bearer Token Authentication**: Supports authentication for secured MCP servers
//! - **Concurrent Tool Execution**: Handles multiple tool calls efficiently
//! - **Resource Access**: Access to MCP server resources like files and data
//! - **Tool Naming Prefix**: Avoid conflicts with customizable tool name prefixes
//!
//! # Transport Protocols
//!
//! ## HTTP Transport
//!
//! For MCP servers accessible via HTTP endpoints with JSON-RPC over HTTP.
//! Supports both regular JSON responses and Server-Sent Events (SSE).
//!
//! ## WebSocket Transport
//!
//! For real-time bidirectional communication with MCP servers over WebSocket.
//! Ideal for interactive applications requiring low-latency tool calls.
//!
//! ## Process Transport
//!
//! For local MCP servers running as separate processes, communicating via stdin/stdout
//! using JSON-RPC messages.
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use mistralrs_core::mcp_client::{McpClientConfig, McpServerConfig, McpServerSource, McpClient};
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Configure MCP client with multiple servers
//!     let config = McpClientConfig {
//!         servers: vec![
//!             // HTTP server with Bearer token
//!             McpServerConfig {
//!                 id: "web_search".to_string(),
//!                 name: "Web Search MCP".to_string(),
//!                 source: McpServerSource::Http {
//!                     url: "https://api.example.com/mcp".to_string(),
//!                     timeout_secs: Some(30),
//!                     headers: None,
//!                 },
//!                 enabled: true,
//!                 tool_prefix: Some("web".to_string()),
//!                 resources: None,
//!                 bearer_token: Some("your-api-token".to_string()),
//!             },
//!             // WebSocket server
//!             McpServerConfig {
//!                 id: "realtime_data".to_string(),
//!                 name: "Real-time Data MCP".to_string(),
//!                 source: McpServerSource::WebSocket {
//!                     url: "wss://realtime.example.com/mcp".to_string(),
//!                     timeout_secs: Some(60),
//!                     headers: None,
//!                 },
//!                 enabled: true,
//!                 tool_prefix: Some("rt".to_string()),
//!                 resources: None,
//!                 bearer_token: Some("ws-token".to_string()),
//!             },
//!             // Process-based server
//!             McpServerConfig {
//!                 id: "filesystem".to_string(),
//!                 name: "Filesystem MCP".to_string(),
//!                 source: McpServerSource::Process {
//!                     command: "mcp-server-filesystem".to_string(),
//!                     args: vec!["--root".to_string(), "/tmp".to_string()],
//!                     work_dir: None,
//!                     env: None,
//!                 },
//!                 enabled: true,
//!                 tool_prefix: Some("fs".to_string()),
//!                 resources: Some(vec!["file://**".to_string()]),
//!                 bearer_token: None,
//!             },
//!         ],
//!         auto_register_tools: true,
//!         tool_timeout_secs: Some(30),
//!         max_concurrent_calls: Some(10),
//!     };
//!     
//!     // Initialize MCP client
//!     let mut client = McpClient::new(config);
//!     client.initialize().await?;
//!     
//!     // Get tool callbacks for integration with model builder
//!     let tool_callbacks = client.get_tool_callbacks_with_tools();
//!     println!("Registered {} MCP tools", tool_callbacks.len());
//!     
//!     Ok(())
//! }
//! ```
//!

use anyhow::Result;
use rust_mcp_schema::Resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

pub mod client;
pub mod transport;
pub mod types;

use crate::tools::{Function, Tool, ToolCallback, ToolCallbackWithTool, ToolType};

/// Supported MCP server transport sources
///
/// Defines the different ways to connect to MCP servers, each optimized for
/// specific use cases and deployment scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpServerSource {
    /// HTTP-based MCP server using JSON-RPC over HTTP
    ///
    /// Best for: Public APIs, RESTful services, servers behind load balancers
    /// Features: SSE support, standard HTTP semantics, easy debugging
    Http {
        /// Base URL of the MCP server (http:// or https://)
        url: String,
        /// Optional timeout in seconds for HTTP requests
        timeout_secs: Option<u64>,
        /// Optional headers to include in requests (e.g., API keys, custom headers)
        headers: Option<HashMap<String, String>>,
    },
    /// Local process-based MCP server using stdin/stdout communication
    ///
    /// Best for: Local tools, development servers, sandboxed environments
    /// Features: Process isolation, no network overhead, easy deployment
    Process {
        /// Command to execute (e.g., "mcp-server-filesystem")
        command: String,
        /// Arguments to pass to the command
        args: Vec<String>,
        /// Optional working directory for the process
        work_dir: Option<String>,
        /// Optional environment variables for the process
        env: Option<HashMap<String, String>>,
    },
    /// WebSocket-based MCP server for real-time bidirectional communication
    ///
    /// Best for: Interactive applications, real-time data, low-latency requirements
    /// Features: Persistent connections, server-initiated notifications, minimal overhead
    WebSocket {
        /// WebSocket URL (ws:// or wss://)
        url: String,
        /// Optional timeout in seconds for connection establishment
        timeout_secs: Option<u64>,
        /// Optional headers for the WebSocket handshake
        headers: Option<HashMap<String, String>>,
    },
}

/// Configuration for MCP client integration
///
/// This structure defines how the MCP client should connect to and manage
/// multiple MCP servers, including authentication, tool registration, and
/// execution policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientConfig {
    /// List of MCP servers to connect to
    pub servers: Vec<McpServerConfig>,
    /// Whether to automatically register discovered tools with the model
    ///
    /// When enabled, tools from MCP servers are automatically converted to
    /// the internal Tool format and registered for automatic tool calling.
    pub auto_register_tools: bool,
    /// Timeout for individual tool execution in seconds
    ///
    /// Controls how long to wait for a tool call to complete before timing out.
    /// Defaults to 30 seconds if not specified.
    pub tool_timeout_secs: Option<u64>,
    /// Maximum number of concurrent tool calls across all MCP servers
    ///
    /// Limits resource usage and prevents overwhelming servers with too many
    /// simultaneous requests. Defaults to 10 if not specified.
    pub max_concurrent_calls: Option<usize>,
}

/// Configuration for an individual MCP server
///
/// Defines connection parameters, authentication, and tool management
/// settings for a single MCP server instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique identifier for this server within the client
    ///
    /// Used internally to track connections and must be unique across all servers.
    pub id: String,
    /// Human-readable name for this server
    ///
    /// Used for logging, debugging, and user-facing displays.
    pub name: String,
    /// Transport configuration for connecting to the server
    pub source: McpServerSource,
    /// Whether this server should be activated
    ///
    /// Disabled servers are ignored during initialization, allowing for
    /// easy toggling of server connections.
    pub enabled: bool,
    /// Optional prefix to add to tool names from this server
    ///
    /// Helps avoid naming conflicts when multiple servers provide tools
    /// with similar names. For example, "web_search" vs "local_search".
    pub tool_prefix: Option<String>,
    /// Resource URI patterns this client is interested in (optional)
    ///
    /// Allows subscribing to specific resource types or patterns from
    /// the server. Uses glob-like patterns (e.g., "file://**", "db://users/*").
    pub resources: Option<Vec<String>>,
    /// Bearer token for authentication (optional)
    ///
    /// Automatically added as "Authorization: Bearer <token>" header for
    /// HTTP and WebSocket connections. Process connections typically don't
    /// need authentication.
    pub bearer_token: Option<String>,
}

/// Information about a discovered MCP tool
///
/// Contains metadata about tools discovered from MCP servers, including
/// their schemas and server association for proper routing and execution.
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    /// Name of the tool as reported by the MCP server
    pub name: String,
    /// Human-readable description of the tool's purpose
    pub description: Option<String>,
    /// JSON schema defining the tool's input parameters
    ///
    /// This schema is used to validate tool calls and can be converted
    /// to the internal Tool parameter format for automatic tool calling.
    pub input_schema: serde_json::Value,
    /// ID of the server this tool comes from
    ///
    /// Used to route tool calls to the correct MCP server connection.
    pub server_id: String,
    /// Display name of the server for logging and debugging
    pub server_name: String,
}

/// MCP client that manages connections to multiple MCP servers
///
/// The main interface for interacting with Model Context Protocol servers.
/// Handles connection lifecycle, tool discovery, and provides integration
/// with the mistral.rs tool calling system.
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
/// use mistralrs_core::mcp_client::{McpClient, McpClientConfig};
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

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            servers: Vec::new(),
            auto_register_tools: true,
            tool_timeout_secs: Some(30),
            max_concurrent_calls: Some(10),
        }
    }
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
            McpServerSource::Http {
                url,
                timeout_secs,
                headers,
            } => {
                // Merge Bearer token with existing headers if provided
                let mut merged_headers = headers.clone().unwrap_or_default();
                if let Some(token) = &config.bearer_token {
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {}", token));
                }

                let connection = client::HttpMcpConnection::new(
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
                let connection = client::ProcessMcpConnection::new(
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
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {}", token));
                }

                let connection = client::WebSocketMcpConnection::new(
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
}
