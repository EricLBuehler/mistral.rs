# MCP (Model Context Protocol) Client

mistral.rs includes a built-in MCP client that allows models to connect to external tools and services through the Model Context Protocol. This enables automatic tool discovery and usage from any MCP-compatible server.

## Quick Start

All examples below use the Hugging Face MCP server. Replace `hf_xxx` with your actual Hugging Face token.

### Rust API

```rust
use mistralrs::{
    TextModelBuilder, McpClientConfig, McpServerConfig, McpServerSource,
    TextMessages, TextMessageRole,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure HF MCP server
    let mcp_config = McpClientConfig {
        servers: vec![McpServerConfig {
            id: "hf_server".to_string(),
            name: "Hugging Face MCP".to_string(),
            source: McpServerSource::Http {
                url: "https://hf.co/mcp".to_string(),
                timeout_secs: Some(30),
                headers: None,
            },
            enabled: true,
            tool_prefix: Some("hf".to_string()),
            resources: None,
            bearer_token: Some("hf_xxx".to_string()), // Your HF token
        }],
        auto_register_tools: true,
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(5),
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
            "What are the top trending models on Hugging Face?"
        );

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    
    Ok(())
}
```

### Python API

```python
import mistralrs

# Configure HF MCP server
hf_server = mistralrs.McpServerConfigPy(
    id="hf_server",
    name="Hugging Face MCP",
    source=mistralrs.McpServerSourcePy.Http(
        url="https://hf.co/mcp",
        timeout_secs=30,
        headers=None
    ),
    enabled=True,
    tool_prefix="hf",
    resources=None,
    bearer_token="hf_xxx"  # Your HF token
)

# Create MCP client config
mcp_config = mistralrs.McpClientConfigPy(
    servers=[hf_server],
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
        model="mistral",
        messages=[
            {"role": "user", "content": "What are the top trending models on Hugging Face?"}
        ],
        max_tokens=500,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
```

### HTTP API

1. Create `mcp-config.json`:
```json
{
  "servers": [{
    "id": "hf_server",
    "name": "Hugging Face MCP Server",
    "source": {
      "type": "Http",
      "url": "https://hf.co/mcp",
      "timeout_secs": 30
    },
    "enabled": true,
    "tool_prefix": "hf",
    "bearer_token": "hf_xxx"
  }],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 5
}
```

2. Start server with MCP:
```bash
mistralrs-server \
  --port 1234 \
  --mcp-config mcp-config.json \
  plain \
  -m Qwen/Qwen3-4B \
  -a qwen3
```

3. Use the API:
```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "What are the top trending models on Hugging Face?"}
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

- **Hugging Face**: `https://hf.co/mcp` - Access HF models, datasets, and spaces
- **Filesystem**: `mcp-server-filesystem` - Local file operations
- **GitHub**: `mcp-server-github` - GitHub API access
- **Web Search**: Various providers - Web search capabilities

Replace placeholder tokens and URLs with actual values for your use case.