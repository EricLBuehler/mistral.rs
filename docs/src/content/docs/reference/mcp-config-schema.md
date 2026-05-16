---
title: MCP configuration schema
description: JSON schema for mistralrs MCP client configuration files.
sidebar:
  order: 8
---

When mistral.rs acts as an MCP client (see [connect to MCP server](/mistral.rs/guides/agents/connect-mcp-server/)), it reads a JSON config describing servers to connect to.

## Top-level fields

```json
{
  "servers": [ ... ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 10
}
```

| Field | Type | Default | Purpose |
|---|---|---|---|
| `servers` | array | required | List of MCP servers. |
| `auto_register_tools` | bool | `true` | Expose every tool from every connected server to the model. |
| `tool_timeout_secs` | int | 30 | Per-tool-call timeout. |
| `max_concurrent_calls` | int | 10 | Cap on concurrent MCP calls. |

## Server entry

```json
{
  "name": "filesystem",
  "source": { ... },
  "enabled": true,
  "tool_prefix": "fs",
  "bearer_token": "..."
}
```

| Field | Type | Required | Purpose |
|---|---|---|---|
| `id` | string | no (auto UUID) | Stable identifier for the server. |
| `name` | string | yes | Server name. |
| `source` | object | yes | Transport configuration. |
| `enabled` | bool | no (default `true`) | Disable a server without removing the entry. |
| `tool_prefix` | string | no (auto-generated `mcp_<uuid>`) | Prefix applied to tool names. |
| `resources` | array | no | Optional resource list. |
| `bearer_token` | string | no | Optional bearer token. |

## Transports, `source` object

### Process (stdio)

```json
{
  "type": "Process",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  "env": { "KEY": "value" },
  "work_dir": "/path"
}
```

| Field | Purpose |
|---|---|
| `type` | Literal `"Process"`. |
| `command` | Executable to run. |
| `args` | Arguments. |
| `env` | Optional environment variables. |
| `work_dir` | Optional working directory. |

### HTTP

```json
{
  "type": "Http",
  "url": "https://mcp.example.com",
  "headers": { "Authorization": "Bearer ..." },
  "timeout_secs": 60
}
```

| Field | Purpose |
|---|---|
| `type` | Literal `"Http"`. |
| `url` | Endpoint URL. |
| `headers` | Optional request headers. |
| `timeout_secs` | Optional per-source timeout. |

### WebSocket

```json
{
  "type": "WebSocket",
  "url": "wss://mcp.example.com/ws",
  "headers": { ... },
  "timeout_secs": 60
}
```

Same fields as HTTP with a WebSocket URL.

## Tool name prefix

Tools from a server with `tool_prefix = "fs"` are exposed as `fs_read_file`, `fs_list_directory`, etc. The separator is an underscore. Without `tool_prefix`, an auto-generated `mcp_<uuid>` prefix is used.

## Example

```json
{
  "servers": [
    {
      "name": "filesystem",
      "tool_prefix": "fs",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/agent-workspace"]
      }
    },
    {
      "name": "github",
      "source": {
        "type": "Http",
        "url": "https://mcp.github.example.com",
        "headers": {
          "Authorization": "Bearer xxx"
        }
      }
    }
  ],
  "auto_register_tools": true,
  "max_concurrent_calls": 8,
  "tool_timeout_secs": 45
}
```

Pass it on the CLI:

```bash
mistralrs serve --mcp-config mcp.json -m Qwen/Qwen3-4B
```

The same path can be supplied via the `MCP_CONFIG_PATH` environment variable.
