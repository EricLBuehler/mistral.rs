# MCP (Model Context Protocol) Client

mistral.rs includes a built-in MCP client that allows models to connect to external tools and services through the Model Context Protocol. This enables automatic tool discovery and usage from any MCP-compatible server.

## Quick Start

Examples below show HTTP (Hugging Face), Process (filesystem), and WebSocket transports. Replace `hf_xxx` with your actual Hugging Face token for HTTP examples.

### Rust SDK

```rust
use mistralrs::{
    TextModelBuilder, McpClientConfig, McpServerConfig, McpServerSource,
    TextMessages, TextMessageRole,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Process example (filesystem server - recommended for getting started)
    let mcp_config = McpClientConfig {
        servers: vec![McpServerConfig {
            name: "Filesystem Tools".to_string(),
            source: McpServerSource::Process {
                command: "npx".to_string(),
                args: vec!["@modelcontextprotocol/server-filesystem".to_string(), ".".to_string()],
                work_dir: None,
                env: None,
            },
            ..Default::default()
        }],
        auto_register_tools: true,
        ..Default::default()
    };

    // Alternative HTTP example (Hugging Face MCP server)
    let _mcp_config_http = McpClientConfig {
        servers: vec![McpServerConfig {
            id: "hf_server".to_string(),
            name: "Hugging Face MCP".to_string(),
            source: McpServerSource::Http {
                url: "https://hf.co/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            enabled: false, // Disabled by default
            tool_prefix: Some("hf".to_string()),
            resources: None,
            bearer_token: Some("hf_xxx".to_string()), // Your HF token
        }],
        auto_register_tools: true,
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(5),
    };

    // Alternative WebSocket example
    let _mcp_config_websocket = McpClientConfig {
        servers: vec![McpServerConfig {
            name: "WebSocket Example".to_string(),
            source: McpServerSource::WebSocket {
                url: "wss://api.example.com/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            enabled: false, // Disabled by default
            ..Default::default()
        }],
        auto_register_tools: true,
        ..Default::default()
    };

    // Build model with MCP support
    let model = TextModelBuilder::new("Qwen/Qwen3-4B")
        .with_mcp_client(mcp_config)
        .build()
        .await?;

    // Use the model - tools are automatically available
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::User,
            "List the files in the current directory and create a test.txt file"
        );

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    
    Ok(())
}
```

### Python SDK

```python
import mistralrs

# Process example (filesystem server - recommended for getting started)
filesystem_server = mistralrs.McpServerConfigPy(
    name="Filesystem Tools",
    source=mistralrs.McpServerSourcePy.Process(
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", "."],
        work_dir=None,
        env=None
    )
)

# Alternative HTTP example (Hugging Face MCP server)
hf_server = mistralrs.McpServerConfigPy(
    id="hf_server",
    name="Hugging Face MCP",
    source=mistralrs.McpServerSourcePy.Http(
        url="https://hf.co/mcp",
        timeout_secs=30,
        headers=None
    ),
    enabled=False,  # Disabled by default
    tool_prefix="hf",
    resources=None,
    bearer_token="hf_xxx"  # Your HF token
)

# Alternative WebSocket example
websocket_server = mistralrs.McpServerConfigPy(
    name="WebSocket Example",
    source=mistralrs.McpServerSourcePy.WebSocket(
        url="wss://api.example.com/mcp",
        timeout_secs=30,
        headers=None
    ),
    enabled=False  # Disabled by default
)

# Create MCP client config using filesystem server (others are disabled)
mcp_config = mistralrs.McpClientConfigPy(
    servers=[filesystem_server], # hf_server, websocket_server can be added when enabled
    auto_register_tools=True,
    tool_timeout_secs=30,
    max_concurrent_calls=5
)

# Build model with MCP support
runner = mistralrs.Runner(
    which=mistralrs.Which.Plain(
        model_id="Qwen/Qwen3-4B",
        arch=mistralrs.Architecture.Qwen3,
    ),
    mcp_client_config=mcp_config
)

# Use the model - tools are automatically available
res = runner.send_chat_completion_request(
    mistralrs.ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "List the files in the current directory and create a test.txt file"}
        ],
        max_tokens=500,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
```

### HTTP API

1. Create `mcp-config.json`:

**Process Example (Recommended for getting started):**
```json
{
  "servers": [{
    "name": "Filesystem Tools",
    "source": {
      "type": "Process",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "."]
    }
  }],
  "auto_register_tools": true
}
```

> **Note:** To install the filesystem server, run: `npx @modelcontextprotocol/server-filesystem . -y`

**HTTP Example (Hugging Face MCP Server):**
```json
{
  "servers": [
    {
      "name": "Hugging Face MCP",
      "source": {
        "type": "Http",
        "url": "https://hf.co/mcp",
        "timeout_secs": 30
      },
      "bearer_token": "hf_xxx",
      "tool_prefix": "hf",
      "enabled": false
    },
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 5
}
```

**WebSocket Example:**
```json
{
  "servers": [
    {
      "name": "WebSocket Example",
      "source": {
        "type": "WebSocket",
        "url": "wss://api.example.com/mcp",
        "timeout_secs": 30
      },
      "enabled": false
    },
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```

2. Start server with MCP:
```bash
mistralrs serve \
  -p 1234 \
  --mcp-config mcp-config.json \
  -m Qwen/Qwen3-4B
```

3. Use the API:
```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "List the files in the current directory and create a test.txt file"}
    ],
    "max_tokens": 500,
    "temperature": 0.1
  }'
```

## Key Features

- **Automatic Tool Discovery**: Tools are discovered from MCP servers at startup
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Transport Flexibility**: HTTP, WebSocket, and Process transports supported
- **Authentication**: Bearer token support for secure connections
- **Tool Prefixing**: Avoid naming conflicts between servers
- **Concurrency Control**: Limit parallel tool executions
- **Timeout Management**: Control individual tool execution timeouts

## Next Steps

- [Configuration Reference](./configuration.md) - Detailed configuration options
- [Transport Types](./transports.md) - HTTP, WebSocket, and Process transports
- [Advanced Usage](./advanced.md) - Multi-server setups, custom headers, and more
- [MCP Server Development](./server.md) - Building your own MCP server

## Common MCP Servers

- **Filesystem**: `@modelcontextprotocol/server-filesystem` - Local file operations (Process)
- **Hugging Face**: `https://hf.co/mcp` - Access HF models, datasets, and spaces (HTTP)
- **Postgres**: `@modelcontextprotocol/server-postgres` - Database operations (Process)

**Additional servers (install separately):**
- [Brave Search](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search) - Web search capabilities
- [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/github) - GitHub API access

Replace placeholder tokens and URLs with actual values for your use case.

## Troubleshooting

### Common Issues

**"MCP server failed to start" or "npx command not found"**
- Install Node.js and npm: `curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs`
- Install the filesystem server: `npx @modelcontextprotocol/server-filesystem . -y`

**"No tools available" or "tools_available: false"**
- Check server logs for MCP connection errors
- Verify the MCP config file path is correct
- Ensure the MCP server process is running: `ps aux | grep mcp`

**"Tool call failed" or timeout errors**
- Increase `tool_timeout_secs` in your config (default: 30)
- Check `max_concurrent_calls` setting (start with 1-5)
- Verify file permissions for filesystem operations

**Authentication errors with HTTP servers**
- Double-check `bearer_token` values (e.g., HF tokens start with `hf_`)
- Verify API endpoints are accessible: `curl -H "Authorization: Bearer YOUR_TOKEN" https://hf.co/mcp`

**Need help?**
- [MCP Server Registry](https://github.com/modelcontextprotocol/servers) - Find more servers
- [Discord Community](https://discord.gg/SZrecqK8qw) - Get support