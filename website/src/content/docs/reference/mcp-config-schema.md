---
title: MCP configuration schema
description: JSON schema for mistralrs MCP client configuration files.
sidebar:
  order: 8
---

When mistral.rs acts as an MCP client (see [connect to MCP server](/mistral.rs/guides/agents/connect-mcp-server/)), it reads a JSON config describing servers to connect to. This page is the full schema.

## Top-level fields

```json
{
  "servers": [ ... ],
  "auto_register_tools": true,
  "max_concurrent_calls": 4,
  "default_timeout_secs": 30
}
```

| Field | Type | Default | Purpose |
|---|---|---|---|
| `servers` | array | required | List of MCP servers to connect to. |
| `auto_register_tools` | bool | `true` | If true, every tool from every connected server is exposed to the model. |
| `max_concurrent_calls` | int | 4 | Cap on concurrent MCP calls. |
| `default_timeout_secs` | int | 30 | Global timeout for tool calls. |

## Server entry

Each entry in `servers`:

```json
{
  "name": "filesystem",
  "source": { ... },
  "tool_filter": ["read_file"],
  "timeout_secs": 60
}
```

| Field | Type | Required | Purpose |
|---|---|---|---|
| `name` | string | yes | Unique name for this server. Used as a prefix on tool names. |
| `source` | object | yes | Transport configuration. See below. |
| `tool_filter` | array | no | When set, only the named tools are exposed. Overrides `auto_register_tools`. |
| `timeout_secs` | int | no | Per-server timeout override. |
| `description` | string | no | Human-readable description. |

## Transports: `source` object

Three transport kinds.

### Process (stdio)

Launch the server as a subprocess and talk over stdin/stdout.

```json
{
  "type": "Process",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  "env": { "KEY": "value" },
  "working_dir": "/path"
}
```

| Field | Purpose |
|---|---|
| `type` | Literal `"Process"`. |
| `command` | Executable to run. |
| `args` | Arguments to pass. |
| `env` | Environment variables for the subprocess (optional). |
| `working_dir` | Working directory (optional, defaults to mistralrs's CWD). |

### HTTP

```json
{
  "type": "Http",
  "url": "https://mcp.example.com",
  "headers": { "Authorization": "Bearer ..." }
}
```

| Field | Purpose |
|---|---|
| `type` | Literal `"Http"`. |
| `url` | Full URL of the MCP endpoint. |
| `headers` | Optional request headers. |

### WebSocket

```json
{
  "type": "WebSocket",
  "url": "wss://mcp.example.com/ws",
  "headers": { ... }
}
```

Same fields as HTTP, with a WebSocket URL.

## Tool name prefix

Tools from an MCP server named `filesystem` are exposed as `filesystem.read_file`, `filesystem.list_directory`, etc. The model sees the prefix in the tool schema.

If `name` conflicts with an existing tool namespace (built-in search, code execution), MCP tools take precedence on overlapping names.

## Validation

At startup, mistral.rs validates:

- Required fields per transport are present.
- `name` is unique across servers.
- `Process` `command` entries resolve in `PATH` or are absolute paths.
- `Http` and `WebSocket` URLs parse.

Validation failures abort startup with the offending entry identified. Invalid configs are never silently ignored.

## Connection lifecycle

mistral.rs connects to every server at startup. Connection failures mark a server as unavailable; the rest proceed. Failed servers retry periodically and tools become available on reconnect without restart.

During normal operation, drops trigger automatic reconnect. In-flight tool calls against a dropped server fail with a timeout; the model can retry or proceed without that tool.

## Example: full config

```json
{
  "servers": [
    {
      "name": "filesystem",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/agent-workspace"]
      },
      "tool_filter": ["read_file", "list_directory", "write_file"]
    },
    {
      "name": "github",
      "source": {
        "type": "Http",
        "url": "https://mcp.github.example.com",
        "headers": {
          "Authorization": "Bearer ${GITHUB_MCP_TOKEN}"
        }
      }
    },
    {
      "name": "slack",
      "source": {
        "type": "Process",
        "command": "/usr/local/bin/slack-mcp-server",
        "env": {
          "SLACK_TOKEN": "xoxb-...",
          "SLACK_WORKSPACE": "example"
        }
      }
    }
  ],
  "auto_register_tools": true,
  "max_concurrent_calls": 8,
  "default_timeout_secs": 45
}
```

Pass it to the CLI:

```bash
mistralrs serve --mcp-config mcp.json -m Qwen/Qwen3-4B
```

Environment variable interpolation (`${VAR}`) is not supported in the config file. Expand variables in the caller or use the `env` field on Process sources.
