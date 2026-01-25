//! Server configuration options

use clap::Args;
use std::path::PathBuf;

/// HTTP server configuration
#[derive(Args, Clone)]
pub struct ServerOptions {
    /// HTTP server port
    #[arg(short = 'p', long, default_value_t = 8080)]
    pub port: u16,

    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// MCP protocol server port (enables MCP if set)
    #[arg(long)]
    pub mcp_port: Option<u16>,

    /// MCP client configuration file path
    #[arg(long)]
    pub mcp_config: Option<PathBuf>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            mcp_port: None,
            mcp_config: None,
        }
    }
}
