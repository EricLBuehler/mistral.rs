# MCP Configuration Reference

This page provides a complete reference for configuring the MCP client in mistral.rs.

## Quick Start - Minimal Configuration

For simple use cases, you can now use a minimal configuration that leverages smart defaults:

```json
{
  "servers": [{
    "name": "Hugging Face MCP Server",
    "source": {
      "type": "Http",
      "url": "https://hf.co/mcp"
    },
    "bearer_token": "hf_xxx"
  }]
}
```

This automatically provides:
- **UUID-based server ID**: Unique identifier generated automatically
- **Enabled by default**: Server is active without explicit `enabled: true`
- **UUID-based tool prefix**: Prevents naming conflicts automatically
- **No timeouts**: Tools and connections don't timeout by default
- **Sequential execution**: Only 1 concurrent tool call to prevent overwhelming servers
- **Auto-registration**: Tools are automatically discovered and registered

## Configuration Structure

### McpClientConfig

The top-level configuration for the MCP client:

```json
{
  "servers": [...],                    // Array of MCP server configurations
  "auto_register_tools": true,         // Automatically register discovered tools (default: true)
  "tool_timeout_secs": null,           // Timeout for individual tool calls, null = no timeout (default: null)
  "max_concurrent_calls": 1            // Maximum concurrent tool executions (default: 1)
}
```

### McpServerConfig

Configuration for each MCP server:

```json
{
  "id": "unique_id",                  // Unique identifier (default: UUID if not specified)
  "name": "Display Name",             // Human-readable name
  "source": {...},                    // Transport configuration (see below)
  "enabled": true,                    // Enable/disable this server (default: true)
  "tool_prefix": "mcp_abc123",         // Prefix for tool names (default: UUID-based if not specified)
  "resources": ["pattern"],           // Optional resource patterns
  "bearer_token": "token"             // Optional authentication token
}
```

## Transport Source Configuration

### HTTP Transport

```json
{
  "type": "Http",
  "url": "https://api.example.com/mcp",
  "timeout_secs": null,               // Optional, null = no timeout (default)
  "headers": {                        // Optional custom headers
    "X-API-Version": "v1",
    "User-Agent": "mistral-rs/0.6.0"
  }
}
```

### WebSocket Transport

```json
{
  "type": "WebSocket", 
  "url": "wss://realtime.example.com/mcp",
  "timeout_secs": null,               // Optional, null = no timeout (default)
  "headers": {                        // Optional WebSocket headers
    "Origin": "https://mistral.rs",
    "Sec-WebSocket-Protocol": "mcp"
  }
}
```

### Process Transport

```json
{
  "type": "Process",
  "command": "mcp-server-filesystem",
  "args": ["--root", "/tmp"],         // Command arguments
  "work_dir": "/home/user",           // Optional working directory
  "env": {                            // Optional environment variables
    "MCP_LOG_LEVEL": "info"
  }
}
```

## Field Reference

### McpClientConfig Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `servers` | Array | Yes | - | List of MCP server configurations |
| `auto_register_tools` | Boolean | No | `true` | Automatically discover and register tools at startup |
| `tool_timeout_secs` | Integer | No | `null` | Timeout in seconds for individual tool calls (null = no timeout) |
| `max_concurrent_calls` | Integer | No | `1` | Maximum number of concurrent tool executions |

### McpServerConfig Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | String | No | UUID | Unique identifier for the server (UUID generated if not provided) |
| `name` | String | Yes | - | Human-readable server name |
| `source` | Object | Yes | - | Transport configuration |
| `enabled` | Boolean | No | `true` | Whether to connect to this server |
| `tool_prefix` | String | No | UUID-based | Prefix to add to all tool names (UUID-based if not provided) |
| `resources` | Array | No | None | Resource URI patterns to subscribe to |
| `bearer_token` | String | No | None | Bearer token for authentication |

### Transport Source Fields

#### HTTP Source
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | String | Yes | - | Must be "Http" |
| `url` | String | Yes | - | HTTP/HTTPS URL of the MCP server |
| `timeout_secs` | Integer | No | `null` | Request timeout in seconds (null = no timeout) |
| `headers` | Object | No | None | Additional HTTP headers |

#### WebSocket Source
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | String | Yes | - | Must be "WebSocket" |
| `url` | String | Yes | - | WS/WSS URL of the MCP server |
| `timeout_secs` | Integer | No | `null` | Connection timeout in seconds (null = no timeout) |
| `headers` | Object | No | None | WebSocket handshake headers |

#### Process Source
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | String | Yes | - | Must be "Process" |
| `command` | String | Yes | - | Executable command to run |
| `args` | Array | No | `[]` | Command line arguments |
| `work_dir` | String | No | Current dir | Working directory |
| `env` | Object | No | None | Environment variables |

## Authentication

### Bearer Token
The `bearer_token` field is automatically added as an `Authorization: Bearer <token>` header for HTTP and WebSocket connections.

```json
{
  "bearer_token": "hf_AbCdEfGhIjKlMnOpQrStUvWxYz"
}
```

### Custom Headers
For other authentication schemes, use the `headers` field:

```json
{
  "source": {
    "type": "Http",
    "url": "https://api.example.com/mcp",
    "headers": {
      "X-API-Key": "your-api-key",
      "X-Client-ID": "your-client-id"
    }
  }
}
```

## Tool Naming

### Without Prefix
Tools are registered with their original names:
- MCP tool: `search` -> Registered as: `search`

### With Prefix
When `tool_prefix` is set, all tools from that server get prefixed:
- MCP tool: `search` with prefix `web` -> Registered as: `web_search`

This prevents conflicts when multiple servers provide tools with the same name.

## Resource Patterns

The `resources` field accepts glob-like patterns:

```json
{
  "resources": [
    "file://**/*.txt",      // All .txt files
    "file://data/**",       // Everything under data/
    "db://users/*",         // All user records
    "api://v1/metrics"      // Specific endpoint
  ]
}
```

## Environment Variables

### Using Environment Variables in Configuration

While JSON doesn't support environment variables directly, you can use them when building configurations programmatically:

```rust
McpServerConfig {
    bearer_token: std::env::var("HF_TOKEN").ok(),
    source: McpServerSource::Http {
        url: std::env::var("MCP_SERVER_URL")
            .unwrap_or_else(|_| "https://hf.co/mcp".to_string()),
        // ...
    },
    // ...
}
```

```python
import os

McpServerConfigPy(
    bearer_token=os.getenv("HF_TOKEN"),
    source=McpServerSourcePy.Http(
        url=os.getenv("MCP_SERVER_URL", "https://hf.co/mcp")
    )
)
```

### MCP-Related Environment Variables

| Variable | Description |
|----------|-------------|
| `MCP_CONFIG_PATH` | Path to MCP configuration file |
| `MCP_LOG_LEVEL` | Logging level for MCP operations |
| `MCP_POOL_SIZE` | Connection pool size for HTTP/WebSocket |

## Validation Rules

1. **Unique Server IDs**: All server `id` values must be unique
2. **Valid URLs**: HTTP URLs must start with `http://` or `https://`
3. **Valid WebSocket URLs**: Must start with `ws://` or `wss://`
4. **Executable Commands**: Process commands must be executable
5. **Tool Name Conflicts**: Use `tool_prefix` to avoid conflicts

## Example Configurations

### Single Server (Hugging Face) - Minimal
```json
{
  "servers": [{
    "name": "Hugging Face MCP Server",
    "source": {
      "type": "Http",
      "url": "https://hf.co/mcp"
    },
    "bearer_token": "hf_xxx"
  }]
}
```

### Single Server (Hugging Face) - Full Configuration
```json
{
  "servers": [{
    "id": "hf",
    "name": "Hugging Face MCP",
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

### Multi-Server Setup
```json
{
  "servers": [
    {
      "id": "hf",
      "name": "Hugging Face",
      "source": {"type": "Http", "url": "https://hf.co/mcp"},
      "tool_prefix": "hf",
      "bearer_token": "hf_xxx"
    },
    {
      "id": "github",
      "name": "GitHub API",
      "source": {"type": "Http", "url": "https://api.github.com/mcp"},
      "tool_prefix": "gh",
      "bearer_token": "ghp_xxx"
    },
    {
      "id": "local_fs",
      "name": "Filesystem",
      "source": {
        "type": "Process",
        "command": "mcp-server-filesystem",
        "args": ["--root", "/data", "--readonly"]
      },
      "tool_prefix": "fs"
    }
  ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 10
}
```