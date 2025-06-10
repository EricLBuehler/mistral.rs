# MCP Client Support

mistral.rs supports acting as a Model Context Protocol (MCP) client, allowing it to connect to external MCP servers and automatically register their tools for use in automatic tool calling.

## Overview

The MCP client feature enables mistral.rs to:

- Connect to multiple MCP servers simultaneously
- Automatically discover tools from connected servers
- Register discovered tools with the existing automatic tool calling system
- Support HTTP, process-based, and WebSocket MCP server connections
- Handle tool name conflicts with configurable prefixes
- Manage server connections and handle failures gracefully

## Configuration

MCP client functionality is configured using the `McpClientConfig` struct:

```rust
use mistralrs::{McpClientConfig, McpServerConfig, McpServerSource};
use std::collections::HashMap;

let mcp_config = McpClientConfig {
    servers: vec![
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
        },
    ],
    auto_register_tools: true,
    tool_timeout_secs: Some(30),
    max_concurrent_calls: Some(10),
};
```

## Server Source Types

### HTTP Servers

Connect to MCP servers over HTTP:

```rust
McpServerSource::Http {
    url: "http://localhost:8080/mcp".to_string(),
    timeout_secs: Some(30),
    headers: Some({
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token".to_string());
        headers
    }),
}
```

### Process Servers

Launch and communicate with MCP servers as child processes:

```rust
McpServerSource::Process {
    command: "mcp-server-filesystem".to_string(),
    args: vec!["--root".to_string(), "/tmp".to_string()],
    work_dir: Some("/path/to/workdir".to_string()),
    env: Some({
        let mut env = HashMap::new();
        env.insert("API_KEY".to_string(), "your-api-key".to_string());
        env
    }),
}
```

### WebSocket Servers

Connect to MCP servers via WebSocket (planned):

```rust
McpServerSource::WebSocket {
    url: "ws://localhost:9090/mcp".to_string(),
    timeout_secs: Some(30),
    headers: None,
}
```

## Usage with TextModelBuilder

Configure MCP client support when building your model:

```rust
use mistralrs::{TextModelBuilder, McpClientConfig, McpServerConfig, McpServerSource};

let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
    .with_mcp_client(mcp_config)
    .build()
    .await?;
```

## Tool Discovery and Registration

When the MCP client is initialized:

1. **Connection**: Connects to all enabled MCP servers
2. **Discovery**: Lists available tools from each server
3. **Registration**: Automatically registers tools with the tool calling system
4. **Prefixing**: Applies tool prefixes to avoid naming conflicts
5. **Integration**: Tools become available for automatic tool calling

### Tool Naming

Tools from MCP servers are registered with optional prefixes:

- Without prefix: `list_files` (from filesystem server)
- With prefix: `fs_list_files` (with "fs" prefix)

This prevents naming conflicts when multiple servers provide similar tools.

## Automatic Tool Calling Integration

MCP tools integrate seamlessly with mistral.rs's automatic tool calling:

```rust
let messages = TextMessages::new()
    .add_message(
        TextMessageRole::User,
        "List the files in /tmp and search for Rust information"
    );

let response = model.send_chat_request(messages).await?;

// Model will automatically call MCP tools as needed
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    for tool_call in tool_calls {
        println!("Called tool: {}", tool_call.function.name);
    }
}
```

## Error Handling

The MCP client handles various error scenarios:

- **Connection failures**: Logs warnings and continues with other servers
- **Tool call timeouts**: Configurable timeouts with fallback behavior
- **Server disconnections**: Automatic reconnection attempts
- **Invalid responses**: Graceful error handling with informative messages

## Configuration Options

### McpClientConfig

- `servers`: List of MCP server configurations
- `auto_register_tools`: Whether to automatically register discovered tools (default: true)
- `tool_timeout_secs`: Timeout for individual tool calls (default: 30)
- `max_concurrent_calls`: Maximum concurrent tool calls (default: 10)

### McpServerConfig

- `id`: Unique identifier for the server
- `name`: Human-readable name
- `source`: Connection configuration (HTTP, Process, or WebSocket)
- `enabled`: Whether this server should be used
- `tool_prefix`: Optional prefix for tool names
- `resources`: Optional resource patterns to subscribe to

## Examples

See the example implementations:

- **Rust**: `mistralrs/examples/mcp_client/main.rs`
- **Python**: `examples/python/mcp_client.py`

## Current Limitations

1. **WebSocket Transport**: Not yet implemented (HTTP and Process work)
2. **Resource Subscriptions**: Basic implementation, full resource management pending
3. **Reconnection Logic**: Basic reconnection, advanced scenarios pending
4. **Authentication**: Basic header-based auth, advanced auth methods pending

## Future Enhancements

- Full WebSocket transport implementation
- Advanced authentication mechanisms (OAuth, JWT, etc.)
- Resource subscription and notification handling
- Connection pooling and load balancing
- Metrics and monitoring integration
- Configuration hot-reloading

## Compatibility

The MCP client is compatible with:

- MCP protocol version 2025-03-26
- Any MCP server implementing the standard protocol
- Existing mistral.rs tool calling functionality
- All model types that support tool calling

## Security Considerations

- **Process Servers**: Child processes inherit environment and permissions
- **HTTP Servers**: Use HTTPS and proper authentication for production
- **Tool Execution**: MCP tools run with the same privileges as mistral.rs
- **Input Validation**: Ensure MCP servers properly validate tool arguments