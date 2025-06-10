# MCP (Model Context Protocol) Examples

This directory contains examples demonstrating how to use mistral.rs as an MCP client to connect to external MCP servers and automatically use their tools.

## Overview

The Model Context Protocol (MCP) allows AI assistants to connect to external tools and resources. mistral.rs supports acting as an MCP client that can:

- Connect to multiple MCP servers simultaneously
- Support HTTP, WebSocket, and Process-based transports
- Automatically discover and register tools from MCP servers
- Handle Bearer token authentication
- Control tool execution with timeouts and concurrency limits

## Featured Example: Hugging Face MCP Server

All examples are configured to use the Hugging Face MCP server as the primary demonstration:

```json
{
  "id": "hf_server",
  "name": "Hugging Face MCP Server",
  "source": {
    "type": "Http",
    "url": "https://hf.co/mcp",
    "timeout_secs": 30
  },
  "enabled": true,
  "tool_prefix": "hf",
  "bearer_token": "hf_xxx"  // Replace with your HF token
}
```

## Examples by API Type

### 1. Rust API Examples

#### Basic Usage
**File:** `mistralrs/examples/mcp_client/main.rs`

```rust
use mistralrs::{McpClientConfig, McpServerConfig, McpServerSource, TextModelBuilder};

let mcp_config = McpClientConfig {
    servers: vec![
        McpServerConfig {
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
            bearer_token: Some("hf_xxx".to_string()),
        },
    ],
    auto_register_tools: true,
    tool_timeout_secs: Some(30),
    max_concurrent_calls: Some(5),
};

let model = TextModelBuilder::new("Qwen/Qwen3-4B".to_string())
    .with_mcp_client(mcp_config)
    .build()
    .await?;
```

### 2. Python API Examples

#### Main Example
**File:** `examples/python/mcp_client_example.py`

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
    bearer_token="hf_xxx"
)

mcp_config = mistralrs.McpClientConfigPy(
    servers=[hf_server],
    auto_register_tools=True,
    tool_timeout_secs=30,
    max_concurrent_calls=5
)

runner = mistralrs.Runner(
    which=mistralrs.Which.Plain(
        model_id="Qwen/Qwen3-4B",
        arch=mistralrs.Architecture.Qwen3,
    ),
    mcp_client_config=mcp_config
)
```

#### Simplified Example
**File:** `examples/python/mcp_client.py`

A simpler version focusing on the basic MCP setup with clear comments.

### 3. HTTP API Examples

#### JSON Configuration Files

**Simple Config:** `examples/mcp-test-config.json`
```json
{
  "servers": [
    {
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
    }
  ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 5
}
```

**Advanced Config:** `examples/mcp-server-config.json`
Contains multiple transport examples (HTTP, Process, WebSocket) with the HF server enabled and others commented out.

#### Server Usage

**Start server with MCP:**
```bash
cargo run --release --bin mistralrs-server -- \
  --port 1234 \
  plain \
  -m Qwen/Qwen3-4B \
  -a qwen3 \
  --mcp-config examples/mcp-test-config.json
```

**Python client:** `examples/server/mcp_chat.py`
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="placeholder"
)

response = client.chat.completions.create(
    model="mistral",
    messages=[
        {"role": "system", "content": "You have access to HF tools via MCP..."},
        {"role": "user", "content": "Get the top 10 HF models right now?"}
    ],
    tool_choice="auto"
)
```

**Curl/Shell example:** `examples/mcp_http_example.sh`
Complete shell script demonstrating HTTP API usage with health checks and JSON parsing.

## Transport Types

### HTTP Transport (Featured in Examples)
- **Use case:** Public APIs, RESTful services, Hugging Face
- **Features:** Bearer token auth, custom headers, SSE support
- **Example:** Hugging Face MCP server at `https://hf.co/mcp`

### Process Transport (Commented in Examples)
- **Use case:** Local tools, development servers, filesystem access
- **Features:** Process isolation, stdin/stdout communication
- **Example:** `mcp-server-filesystem` for file operations

### WebSocket Transport (Commented in Examples)  
- **Use case:** Real-time data, interactive applications
- **Features:** Persistent connections, bidirectional communication
- **Example:** Real-time data feeds, interactive tools

## Configuration Reference

### MCP Client Config
```json
{
  "servers": [...],                    // Array of server configurations
  "auto_register_tools": true,         // Auto-discover and register tools
  "tool_timeout_secs": 30,            // Individual tool call timeout
  "max_concurrent_calls": 5           // Max concurrent tool executions
}
```

### Server Config
```json
{
  "id": "unique_server_id",           // Unique identifier
  "name": "Display Name",             // Human-readable name
  "source": {...},                    // Transport configuration
  "enabled": true,                    // Enable/disable server
  "tool_prefix": "prefix",            // Tool name prefix (optional)
  "resources": [...],                 // Resource patterns (optional)
  "bearer_token": "token"             // Authentication token (optional)
}
```

## Getting Started

1. **Choose your API:** Rust, Python, or HTTP
2. **Configure your MCP server:** Start with the HF example
3. **Replace placeholder token:** Use your actual HF token (`hf_xxx`)
4. **Run the example:** Follow the usage instructions for your chosen API
5. **Expand:** Uncomment additional transport examples as needed

## Troubleshooting

- **No tool calls made:** Check that `auto_register_tools` is `true` and servers are `enabled`
- **Authentication errors:** Verify your Bearer token is correct
- **Connection timeouts:** Increase `timeout_secs` for slow servers
- **Tool conflicts:** Use different `tool_prefix` values for each server

## Additional Resources

- [MCP Documentation](docs/MCP.md) - Detailed MCP client documentation
- [Tool Calling Guide](docs/TOOL_CALLING.md) - General tool calling documentation
- [HTTP API Reference](docs/HTTP.md) - Complete HTTP API documentation