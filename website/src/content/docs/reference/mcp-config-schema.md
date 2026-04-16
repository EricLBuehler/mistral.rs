---
title: MCP configuration schema
description: JSON schema for mistralrs MCP client configuration files.
sidebar:
  order: 8
---

When mistralrs acts as an MCP client (see [connect to MCP server](/mistral.rs/guides/agents/connect-mcp-server/)), it reads a JSON config file that describes the servers to connect to. This page is the full schema.

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

Tools from an MCP server named `filesystem` are exposed to the model as `filesystem.read_file`, `filesystem.list_directory`, etc. The model sees the prefix in the tool schema.

If the `name` field conflicts with an existing tool namespace (built-in search or code execution), the MCP tools take precedence for any names that overlap.

## Validation

At startup, mistralrs validates:

- Every transport's required fields are present.
- The `name` field is unique across all servers.
- `command` entries for `Process` are resolvable in `PATH` (or are absolute paths).
- URLs for `Http` and `WebSocket` are parseable.

Validation failures abort startup with a message pointing at the offending entry. Invalid configs are not silently ignored.

## Connection lifecycle

mistralrs connects to every server in the config at startup. If a connection fails, the server is marked as unavailable and the rest proceed. The failed server is retried periodically; if it comes back, its tools become available without a mistralrs restart.

During normal operation, connection drops trigger an automatic reconnect. In-flight tool calls against a dropped server fail with a timeout and the model can retry or proceed without that tool.

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

Environment variable interpolation (`${VAR}`) is not supported in the config file itself; expand them in the caller or use the `env` field on Process sources.
