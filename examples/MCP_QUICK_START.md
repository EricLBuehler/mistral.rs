# MCP Client Usage Guide

Connect your models to external tools and services using the Model Context Protocol (MCP).

You can find [more details here](../docs/MCP/README.md).

## Quick Start

### 1. Create Configuration File

Create `mcp-config.json`:
```json
{
  "servers": [
    {
      "name": "Web Search Tool",
      "source": {
        "type": "Http",
        "url": "https://api.example.com/mcp"
      },
    }
  ]
}
```

### 2. Start Server

```bash
# With config file
mistralrs-server --mcp-config mcp-config.json --port 1234 run -m Qwen/Qwen3-4B

# Or with environment variable
export MCP_CONFIG_PATH=mcp-config.json
mistralrs-server --port 1234 run -m Qwen/Qwen3-4B
```

### 3. Use Tools Automatically

```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [
      {"role": "user", "content": "Search for the latest AI news"}
    ]
  }'
```

**Tools are automatically available!** The model can now use your MCP servers.

## Check Integration Status

```bash
curl http://localhost:1234/v1/models
```

Response includes MCP information:
```json
{
  "object": "list",
  "data": [{
    "id": "Qwen/Qwen3-4B",
    "object": "model",
    "created": 1699027200,
    "owned_by": "local",
    "tools_available": true,
    "mcp_tools_count": 3,
    "mcp_servers_connected": 1
  }]
}
```

## Transport Types

| Transport | Use Case | Example |
|-----------|----------|---------|
| **HTTP** | REST APIs, cloud services | Web search, databases |
| **Process** | Local tools, CLI tools | File operations, local scripts |
| **WebSocket** | Real-time data, streaming | Live data feeds, interactive tools |

## Configuration Examples

### HTTP Server
```json
{
  "name": "web_api",
  "source": {
    "type": "Http",
    "url": "https://api.example.com/mcp",
    "headers": {
      "User-Agent": "mistral-rs/0.6.0"
    }
  },
  "bearer_token": "your-api-key"
}
```

### Local Process
```json
{
  "name": "filesystem",
  "source": {
    "type": "Process",
    "command": "mcp-server-filesystem",
    "args": ["--root", "/workspace", "--readonly"],
    "env": {
      "LOG_LEVEL": "info"
    }
  }
}
```

### WebSocket Server
```json
{
  "name": "realtime",
  "source": {
    "type": "WebSocket",
    "url": "wss://data.example.com/mcp"
  }
}
```

## Error Handling

The system gracefully handles failures:
- **Startup**: Invalid configurations are caught and reported with helpful messages
- **Runtime**: Failed MCP connections are logged as warnings, server continues without MCP
- **Tools**: Individual tool failures don't crash the server

## See Also

- [Configuration Reference](mcp-config-reference.json) - Complete configuration options
- [Configuration Examples](mcp-server-config.json) - Real-world examples
- [Test Configuration](mcp-test-config.json) - Simple test setup