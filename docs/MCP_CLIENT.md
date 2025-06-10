# MCP Client - Connect to External Tools

Connect your mistral.rs models to external tools and services using the Model Context Protocol (MCP). Works with Rust, Python, and HTTP APIs.

## Quick Start Examples

### HTTP Server with MCP Tools

```bash
# 1. Create mcp-config.json
cat > mcp-config.json << 'EOF'
{
  "servers": [
    {
      "id": "web_search",
      "name": "Web Search API",
      "source": {
        "type": "Http", 
        "url": "https://api.example.com/mcp",
        "timeout_secs": 30
      },
      "enabled": true,
      "tool_prefix": "web"
    }
  ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 10
}
EOF

# 2. Start server with MCP tools
mistralrs-server --mcp-config mcp-config.json --port 1234 run -m microsoft/Phi-3.5-mini-instruct

# 3. Use tools automatically in chat
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"phi-3.5","messages":[{"role":"user","content":"Search for AI news"}]}'
```

### Python API

```python
from mistralrs import Runner, Which, McpClientConfig, McpServerConfig, McpServerSource

# Configure MCP servers
mcp_config = McpClientConfig(
    servers=[
        McpServerConfig(
            id="web_search",
            name="Web Search",
            source=McpServerSource.Http(url="https://api.example.com/mcp", timeout_secs=30),
            enabled=True,
            tool_prefix="web"
        )
    ]
)

# Create runner with MCP tools
runner = Runner(
    Which.Plain(model_id="microsoft/Phi-3.5-mini-instruct"),
    mcp_client_config=mcp_config
)

# Tools are automatically available!
response = runner.send_chat_completion_request({
    "model": "phi-3.5",
    "messages": [{"role": "user", "content": "Search for the latest AI developments"}]
})
```

### Rust API

```rust
use mistralrs::{TextModelBuilder, McpClientConfig, McpServerConfig, McpServerSource};

let mcp_config = McpClientConfig {
    servers: vec![
        McpServerConfig {
            id: "web_search".to_string(),
            name: "Web Search".to_string(),
            source: McpServerSource::Http {
                url: "https://api.example.com/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            enabled: true,
            tool_prefix: Some("web".to_string()),
            resources: None,
            bearer_token: None,
        },
    ],
    auto_register_tools: true,      // Automatically register discovered tools
    tool_timeout_secs: Some(30),    // 30 second timeout per tool call
    max_concurrent_calls: Some(5),  // Max 5 concurrent tool calls
};

let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
    .with_mcp_client(mcp_config)  // Tools automatically integrated!
    .build()
    .await?;
```

## What is MCP?

The **Model Context Protocol** allows AI models to connect to external tools and data sources. With mistral.rs MCP client support, your models can:

- **Search the web** for real-time information
- **Access databases** for data retrieval
- **Control local tools** like file systems or scripts
- **Connect to APIs** for specialized services
- **Stream real-time data** via WebSocket connections

**All tools are automatically discovered and made available to your models.**

## Transport Protocols

### HTTP/HTTPS Transport

**Best for**: Public APIs, RESTful services, servers behind load balancers

**Features**:
- HTTPS/TLS encryption for secure communication
- Server-Sent Events (SSE) support for streaming responses
- Bearer token authentication
- Custom headers for API keys and versioning
- Configurable request timeouts

```rust
McpServerSource::Http {
    url: "https://api.example.com/mcp".to_string(),
    timeout_secs: Some(30),
    headers: None, // Custom headers can be added here
}
```

### WebSocket Transport

**Best for**: Interactive applications, real-time data, low-latency requirements

**Features**:
- Persistent bidirectional connections
- Real-time communication with minimal overhead
- Bearer token authentication in handshake
- Automatic request/response correlation
- Concurrent operations with split streams

```rust
McpServerSource::WebSocket {
    url: "wss://realtime.example.com/mcp".to_string(),
    timeout_secs: Some(60),
    headers: None, // Headers for WebSocket handshake
}
```

### Process Transport

**Best for**: Local tools, development servers, sandboxed environments

**Features**:
- Process isolation for security
- No network overhead (direct pipes)
- Full environment control
- Automatic process lifecycle management
- JSON-RPC over stdin/stdout

```rust
McpServerSource::Process {
    command: "mcp-server-filesystem".to_string(),
    args: vec!["--root".to_string(), "/tmp".to_string()],
    work_dir: Some("/path/to/workdir".to_string()),
    env: Some({
        let mut env = HashMap::new();
        env.insert("MCP_LOG_LEVEL".to_string(), "debug".to_string());
        env
    }),
}
```

## Configuration

Configure multiple MCP servers with different transport protocols and authentication:

```rust
use mistralrs::{McpClientConfig, McpServerConfig, McpServerSource};
use std::collections::HashMap;

let mcp_config = McpClientConfig {
    servers: vec![
        // HTTP server with Bearer token authentication
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
        // WebSocket server for real-time communication
        McpServerConfig {
            id: "realtime_data".to_string(),
            name: "Real-time Data MCP".to_string(),
            source: McpServerSource::WebSocket {
                url: "wss://realtime.example.com/mcp".to_string(),
                timeout_secs: Some(60),
                headers: None,
            },
            enabled: true,
            tool_prefix: Some("rt".to_string()),
            resources: None,
            bearer_token: Some("ws-auth-token".to_string()),
        },
        // Process-based local server
        McpServerConfig {
            id: "filesystem_server".to_string(),
            name: "Filesystem MCP Server".to_string(),
            source: McpServerSource::Process {
                command: "mcp-server-filesystem".to_string(),
                args: vec!["--root".to_string(), "/tmp".to_string()],
                work_dir: None,
                env: None,
            },
            enabled: true,
            tool_prefix: Some("fs".to_string()),
            resources: Some(vec!["file://**".to_string()]),
            bearer_token: None, // Process servers don't typically need authentication
        },
    ],
    tool_timeout_secs: Some(30),
    max_concurrent_calls: Some(10),
};
```

## Authentication

### Bearer Token Authentication

Automatically handled for HTTP and WebSocket connections:

```rust
McpServerConfig {
    // ... other fields
    bearer_token: Some("your-secret-token".to_string()),
    // Automatically adds: Authorization: Bearer your-secret-token
}
```

### Custom Headers

Add additional headers for API keys, versioning, etc.:

```rust
McpServerSource::Http {
    url: "https://api.example.com/mcp".to_string(),
    timeout_secs: Some(30),
    headers: Some({
        let mut headers = HashMap::new();
        headers.insert("X-API-Version".to_string(), "v1".to_string());
        headers.insert("X-Client-ID".to_string(), "mistral-rs".to_string());
        headers
    }),
}
```

## Usage with Model Builders

Configure MCP client support when building your model:

```rust
use mistralrs::{TextModelBuilder, IsqType};

let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
    .with_isq(IsqType::Q8_0)
    .with_mcp_client(mcp_config) // MCP tools automatically available
    .build()
    .await?;
```

## Tool Discovery and Registration

When the MCP client initializes:

1. **Connection**: Establishes connections to all enabled MCP servers
2. **Discovery**: Lists available tools from each connected server
3. **Schema Conversion**: Converts MCP tool schemas to internal Tool format
4. **Registration**: Automatically registers tools with the tool calling system
5. **Prefixing**: Applies configurable prefixes to avoid naming conflicts
6. **Integration**: Tools become immediately available for automatic tool calling

### Tool Naming and Conflicts

Tools from MCP servers are registered with optional prefixes:

- Without prefix: `search_web` (original tool name)
- With prefix: `web_search_web` (with "web" prefix)

This prevents conflicts when multiple servers provide similar tools.

## Automatic Tool Calling Integration

MCP tools integrate seamlessly with mistral.rs's automatic tool calling:

```rust
let messages = TextMessages::new()
    .add_message(
        TextMessageRole::System,
        "You have access to web search, filesystem, and real-time data tools via MCP servers."
    )
    .add_message(
        TextMessageRole::User,
        "Search for recent AI news and save a summary to /tmp/ai_news.txt"
    );

let response = model.send_chat_request(messages).await?;

// Model automatically calls MCP tools as needed
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    println!("MCP tools called:");
    for tool_call in tool_calls {
        println!("- {}: {}", tool_call.function.name, tool_call.function.arguments);
    }
}
```

## Error Handling and Resilience

The MCP client provides robust error handling:

- **Connection Failures**: Logs warnings and continues with available servers
- **Authentication Errors**: Clear error messages for token/credential issues
- **Tool Call Timeouts**: Configurable timeouts with graceful fallback
- **Server Disconnections**: Automatic reconnection attempts for persistent connections
- **Invalid Responses**: Graceful handling with informative error messages
- **Process Crashes**: Automatic cleanup and optional restart for process servers

## Configuration Options

### McpClientConfig

- `servers`: List of MCP server configurations to connect to
- `auto_register_tools`: Whether to automatically register discovered tools with the model (default: true)
- `tool_timeout_secs`: Timeout for individual tool calls in seconds (default: 30)
- `max_concurrent_calls`: Maximum concurrent tool calls across all servers (default: 10)

### McpServerConfig

- `id`: Unique identifier for the server (used for internal tracking)
- `name`: Human-readable name for logging and debugging
- `source`: Transport configuration (HTTP, WebSocket, or Process)
- `enabled`: Whether this server should be activated
- `tool_prefix`: Optional prefix to add to tool names (prevents conflicts)
- `resources`: Optional resource URI patterns to subscribe to
- `bearer_token`: Optional Bearer token for authentication

## Performance Considerations

### Transport Selection

- **HTTP**: Best for stateless interactions, caching, load balancing
- **WebSocket**: Best for interactive applications, real-time updates
- **Process**: Best for local tools, no network overhead, maximum security

### Concurrent Operations

- Configure `max_concurrent_calls` based on server capacity
- WebSocket transport supports true concurrent operations
- HTTP transport benefits from connection pooling
- Process transport is limited by stdin/stdout serialization

## Examples

### Complete Example

```rust
use mistralrs::{
    TextModelBuilder, TextMessages, TextMessageRole, 
    McpClientConfig, McpServerConfig, McpServerSource,
    IsqType
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mcp_config = McpClientConfig {
        servers: vec![
            McpServerConfig {
                id: "web_search".to_string(),
                name: "Web Search API".to_string(),
                source: McpServerSource::Http {
                    url: "https://api.example.com/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: None,
                },
                enabled: true,
                tool_prefix: Some("web".to_string()),
                resources: None,
                bearer_token: Some("your-api-key".to_string()),
            },
        ],
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(5),
    };

    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
        .with_mcp_client(mcp_config)
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::User,
            "Search for information about Rust programming language"
        );

    let response = model.send_chat_request(messages).await?;
    println!("Response: {}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```

See also:
- **Rust**: `mistralrs/examples/mcp_client/main.rs`
- **Python**: `examples/python/mcp_client.py`

## Compatibility

The MCP client is compatible with:

- **MCP Protocol**: Version 2025-03-26 and compatible versions
- **MCP Servers**: Any server implementing the standard MCP protocol
- **Model Types**: All mistral.rs model types that support tool calling
- **Existing Tools**: Works alongside built-in and custom tool calling functions

## Security Considerations

### Process Servers
- Child processes inherit the environment and permissions of mistral.rs
- Use dedicated user accounts with minimal privileges when possible
- Validate all inputs to prevent command injection

### HTTP/WebSocket Servers
- Always use HTTPS/WSS in production environments
- Implement proper Bearer token rotation and management
- Validate server certificates and use pinning when appropriate
- Monitor for authentication failures and suspicious activity

### Tool Execution
- MCP tools execute with the same privileges as the mistral.rs process
- Implement input validation and sanitization in MCP servers
- Consider sandboxing for untrusted tools
- Log all tool executions for audit purposes

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check network connectivity and firewall settings
   - Verify server URLs and authentication credentials
   - Review server logs for detailed error information

2. **Authentication Errors**
   - Ensure Bearer tokens are valid and not expired
   - Check custom headers for correct format
   - Verify server authentication requirements

3. **Tool Discovery Issues**
   - Confirm MCP servers implement the tools/list method
   - Check server logs for tool registration errors
   - Verify tool schemas are valid JSON

4. **Performance Issues**
   - Adjust `max_concurrent_calls` based on server capacity
   - Monitor tool execution times and adjust timeouts
   - Consider using WebSocket for high-frequency operations

### Debug Configuration

Enable detailed logging for troubleshooting:

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_env_filter("mistralrs_core::mcp_client=debug")
    .init();
```

## Future Enhancements

- **Advanced Authentication**: OAuth 2.0, JWT, mutual TLS support
- **Resource Management**: Full resource subscription and notification handling
- **Connection Pooling**: Advanced connection management and load balancing
- **Metrics Integration**: Prometheus metrics and health monitoring
- **Configuration Hot-reload**: Dynamic server configuration updates
- **Caching Layer**: Intelligent caching for tool results and schemas