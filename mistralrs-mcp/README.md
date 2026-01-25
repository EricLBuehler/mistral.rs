# mistralrs-mcp

Model Context Protocol (MCP) client implementation for mistral.rs.

This crate provides a client library for connecting to MCP servers and integrating external tools with language models. It supports multiple transport protocols (HTTP, WebSocket, Process) and provides automatic tool discovery and registration.

## Features

- **Multi-transport Support**: HTTP, WebSocket, and Process-based connections
- **Automatic Tool Discovery**: Discovers and registers tools from connected MCP servers
- **Bearer Token Authentication**: Supports authentication for secured MCP servers
- **Concurrent Tool Execution**: Handles multiple tool calls efficiently with configurable limits
- **Timeout Control**: Configurable timeouts for individual tool calls
- **Resource Access**: Access to MCP server resources like files and data

## Usage

```rust
use mistralrs_mcp::{McpClient, McpClientConfig, McpServerConfig, McpServerSource};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = McpClientConfig {
        servers: vec![
            McpServerConfig {
                id: "web_search".to_string(),
                name: "Web Search MCP".to_string(),
                source: McpServerSource::Http {
                    url: "https://api.example.com/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: None,
                },
                enabled: true,
                tool_prefix: Some("web".to_string()),
                resources: None,
                bearer_token: Some("your-api-token".to_string()),
            },
        ],
        auto_register_tools: true,
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(10),
    };
    
    let mut client = McpClient::new(config);
    client.initialize().await?;
    
    let tools = client.get_tools();
    println!("Discovered {} tools", tools.len());
    
    Ok(())
}
```

See the [MCP Client documentation](../docs/MCP/README.md) for more details.
