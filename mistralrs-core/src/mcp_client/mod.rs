use anyhow::Result;
use rust_mcp_schema::Resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub mod client;
pub mod transport;
pub mod types;

use crate::tools::ToolCallback;

/// Supported MCP server sources
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpServerSource {
    /// HTTP-based MCP server
    Http {
        /// Base URL of the MCP server
        url: String,
        /// Optional timeout in seconds
        timeout_secs: Option<u64>,
        /// Optional headers to include in requests
        headers: Option<HashMap<String, String>>,
    },
    /// Local process-based MCP server
    Process {
        /// Command to execute
        command: String,
        /// Arguments to pass to the command
        args: Vec<String>,
        /// Optional working directory
        work_dir: Option<String>,
        /// Optional environment variables
        env: Option<HashMap<String, String>>,
    },
    /// WebSocket-based MCP server
    WebSocket {
        /// WebSocket URL
        url: String,
        /// Optional timeout in seconds
        timeout_secs: Option<u64>,
        /// Optional headers for the WebSocket handshake
        headers: Option<HashMap<String, String>>,
    },
}

/// Configuration for MCP client integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientConfig {
    /// List of MCP servers to connect to
    pub servers: Vec<McpServerConfig>,
    /// Whether to automatically register discovered tools
    pub auto_register_tools: bool,
    /// Timeout for tool execution in seconds
    pub tool_timeout_secs: Option<u64>,
    /// Maximum number of concurrent tool calls
    pub max_concurrent_calls: Option<usize>,
}

/// Configuration for individual MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique identifier for this server
    pub id: String,
    /// Human-readable name for this server
    pub name: String,
    /// Source specification for connecting to the server
    pub source: McpServerSource,
    /// Whether this server is enabled
    pub enabled: bool,
    /// Optional tool name prefix to avoid conflicts
    pub tool_prefix: Option<String>,
    /// Resource patterns to subscribe to (optional)
    pub resources: Option<Vec<String>>,
}

/// Information about a discovered MCP tool
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    /// Name of the tool
    pub name: String,
    /// Description of the tool
    pub description: Option<String>,
    /// JSON schema for the tool's input parameters
    pub input_schema: serde_json::Value,
    /// ID of the server this tool comes from
    pub server_id: String,
    /// Server name for display purposes
    pub server_name: String,
}

/// MCP client that manages connections to multiple MCP servers
pub struct McpClient {
    /// Configuration for the client
    config: McpClientConfig,
    /// Connected MCP servers
    servers: HashMap<String, Arc<dyn McpServerConnection>>,
    /// Discovered tools from all servers
    tools: HashMap<String, McpToolInfo>,
    /// Tool callbacks that can be used with the existing tool calling system
    tool_callbacks: HashMap<String, Arc<ToolCallback>>,
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
        Self {
            config,
            servers: HashMap::new(),
            tools: HashMap::new(),
            tool_callbacks: HashMap::new(),
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
    
    /// Get discovered tools information
    pub fn get_tools(&self) -> &HashMap<String, McpToolInfo> {
        &self.tools
    }
    
    /// Create connection based on server source type
    async fn create_connection(&self, config: &McpServerConfig) -> Result<Arc<dyn McpServerConnection>> {
        match &config.source {
            McpServerSource::Http { url, timeout_secs, headers } => {
                let connection = client::HttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    headers.clone(),
                ).await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Process { command, args, work_dir, env } => {
                let connection = client::ProcessMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    command.clone(),
                    args.clone(),
                    work_dir.clone(),
                    env.clone(),
                ).await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::WebSocket { url, timeout_secs, headers } => {
                let connection = client::WebSocketMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    headers.clone(),
                ).await?;
                Ok(Arc::new(connection))
            }
        }
    }
    
    /// Discover tools from all connected servers and register them
    async fn discover_and_register_tools(&mut self) -> Result<()> {
        for (server_id, connection) in &self.servers {
            let tools = connection.list_tools().await?;
            let server_config = self.config.servers.iter()
                .find(|s| &s.id == server_id)
                .ok_or_else(|| anyhow::anyhow!("Server config not found for {}", server_id))?;
            
            for tool in tools {
                let tool_name = if let Some(prefix) = &server_config.tool_prefix {
                    format!("{}_{}", prefix, tool.name)
                } else {
                    tool.name.clone()
                };
                
                // Create tool callback that calls the MCP server
                let connection_clone = Arc::clone(connection);
                let original_tool_name = tool.name.clone();
                let callback: Arc<ToolCallback> = Arc::new(move |called_function| {
                    let connection = Arc::clone(&connection_clone);
                    let tool_name = original_tool_name.clone();
                    let arguments: serde_json::Value = serde_json::from_str(&called_function.arguments)?;
                    
                    // Use tokio runtime to call async function
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async move {
                        connection.call_tool(&tool_name, arguments).await
                    })
                });
                
                self.tool_callbacks.insert(tool_name.clone(), callback);
                self.tools.insert(tool_name, tool);
            }
        }
        
        Ok(())
    }
}