//! Model Context Protocol (MCP) Client Implementation
//!
//! This crate provides a comprehensive client implementation for the Model Context Protocol (MCP),
//! enabling AI assistants to connect to and interact with external tools and resources through
//! standardized server interfaces.
//!
//! # Overview
//!
//! The MCP client supports multiple transport protocols and provides automatic tool discovery,
//! registration, and execution. It seamlessly integrates with tool calling systems,
//! allowing MCP tools to be used alongside built-in tools.
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
//! ## Simple Configuration
//!
//! ```rust,no_run
//! use mistralrs_mcp::{McpClientConfig, McpServerConfig, McpServerSource, McpClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Simple configuration with minimal settings
//!     // Most fields use sensible defaults (enabled=true, UUID for id/prefix, no timeouts)
//!     let config = McpClientConfig {
//!         servers: vec![
//!             McpServerConfig {
//!                 name: "Hugging Face MCP Server".to_string(),
//!                 source: McpServerSource::Http {
//!                     url: "https://hf.co/mcp".to_string(),
//!                     timeout_secs: None,
//!                     headers: None,
//!                 },
//!                 bearer_token: Some("hf_xxx".to_string()),
//!                 ..Default::default()
//!             },
//!         ],
//!         ..Default::default()
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
//! ## Advanced Configuration
//!
//! ```rust,no_run
//! use mistralrs_mcp::{McpClientConfig, McpServerConfig, McpServerSource, McpClient};
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Configure MCP client with multiple servers and custom settings
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
//!         max_concurrent_calls: Some(5),
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

pub mod client;
pub mod tools;
pub mod transport;
pub mod types;

pub use client::{McpClient, McpServerConnection};
pub use tools::{CalledFunction, Function, Tool, ToolCallback, ToolCallbackWithTool, ToolType};
pub use types::McpToolResult;

pub use rust_mcp_schema;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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
        /// Defaults to no timeout if not specified.
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
        /// Defaults to no timeout if not specified.
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
    /// Defaults to no timeout if not specified.
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
#[serde(default)]
pub struct McpServerConfig {
    /// Unique identifier for this server
    ///
    /// Used internally to track connections and route tool calls.
    /// Must be unique across all servers in a single MCP client configuration.
    /// Defaults to a UUID if not specified.
    #[serde(default = "generate_uuid")]
    pub id: String,
    /// Human-readable name for this server
    ///
    /// Used for logging, debugging, and user-facing displays.
    pub name: String,
    /// Transport-specific connection configuration
    pub source: McpServerSource,
    /// Whether this server should be activated
    ///
    /// Disabled servers are ignored during client initialization.
    /// Defaults to true if not specified.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional prefix to add to all tool names from this server
    ///
    /// Helps prevent naming conflicts when multiple servers provide
    /// tools with similar names. For example, with prefix "web",
    /// a tool named "search" becomes "web_search".
    /// Defaults to a UUID-based prefix if not specified.
    #[serde(default = "generate_uuid_prefix")]
    pub tool_prefix: Option<String>,
    /// Optional resource URI patterns this server provides
    ///
    /// Used for resource discovery and subscription.
    /// Supports glob patterns like "file://**" for filesystem access.
    pub resources: Option<Vec<String>>,
    /// Optional Bearer token for authentication
    ///
    /// Automatically included as `Authorization: Bearer <token>` header
    /// for HTTP and WebSocket connections. Process connections typically
    /// don't require authentication tokens.
    pub bearer_token: Option<String>,
}

/// Information about a tool discovered from an MCP server
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    /// Name of the tool as reported by the MCP server
    pub name: String,
    /// Optional human-readable description of what the tool does
    pub description: Option<String>,
    /// JSON schema describing the tool's input parameters
    pub input_schema: serde_json::Value,
    /// ID of the server this tool comes from
    ///
    /// Used to route tool calls to the correct MCP server connection.
    pub server_id: String,
    /// Display name of the server for logging and debugging
    pub server_name: String,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            servers: Vec::new(),
            auto_register_tools: true,
            tool_timeout_secs: None,
            max_concurrent_calls: Some(1),
        }
    }
}

fn generate_uuid() -> String {
    Uuid::new_v4().to_string()
}

fn default_true() -> bool {
    true
}

fn generate_uuid_prefix() -> Option<String> {
    Some(format!("mcp_{}", Uuid::new_v4().simple()))
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            id: generate_uuid(),
            name: String::new(),
            source: McpServerSource::Http {
                url: String::new(),
                timeout_secs: None,
                headers: None,
            },
            enabled: true,
            tool_prefix: generate_uuid_prefix(),
            resources: None,
            bearer_token: None,
        }
    }
}
