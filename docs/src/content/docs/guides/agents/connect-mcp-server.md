---
title: Connect to an MCP server
description: Use mistralrs as an MCP client.
---

mistral.rs can act as an MCP client, connecting to one or more MCP servers at startup and merging their tools into the model's available set.

MCP tools automatically use [strict tool calling](/mistral.rs/guides/agents/tool-calling-basics/#strict-tool-calling) when the MCP server provides an input schema.

## Starting with a config file

```bash
mistralrs serve --mcp-config mcp.json -m <model>
```

The `MCP_CONFIG_PATH` environment variable is an alternative to the flag.

Minimal config:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      }
    }
  ],
  "auto_register_tools": true,
  "tool_timeout_secs": 30,
  "max_concurrent_calls": 4
}
```

Full schema: [MCP config schema reference](/mistral.rs/reference/mcp-config-schema/).

## Transports

Three `source.type` values are supported:

- `Process`: launch the server as a subprocess and communicate over stdio.
- `Http`: connect over HTTP.
- `WebSocket`: connect over WebSockets.

Remote servers can carry per-server auth and naming. A Hugging Face MCP entry, for example:

```json
{
  "servers": [
    {
      "name": "Hugging Face MCP",
      "source": {"type": "Http", "url": "https://hf.co/mcp", "timeout_secs": 30},
      "bearer_token": "hf_xxx",
      "tool_prefix": "hf"
    }
  ],
  "auto_register_tools": true
}
```

`bearer_token` is sent as an `Authorization` header; `enabled: false` keeps an entry in the file without connecting it.

## Tool name prefix

Each server's tools are exposed to the model as `<prefix>_<tool>`. The prefix is `tool_prefix` if set on the server entry, otherwise an auto-generated `mcp_<uuid>`. Separator is an underscore.

## Concurrency and timeouts

`max_concurrent_calls` caps in-flight MCP calls. `tool_timeout_secs` is the per-call timeout. `Http` and `WebSocket` sources can override the timeout per-source.

## Verifying the connection

`GET /v1/models` reports MCP status per model:

```json
{
  "object": "list",
  "data": [{
    "id": "Qwen/Qwen3-4B",
    "object": "model",
    "tools_available": true,
    "mcp_tools_count": 3,
    "mcp_servers_connected": 1
  }]
}
```

Failures degrade gracefully: invalid configurations are reported at startup, failed MCP connections are logged as warnings while the server continues without them, and individual tool failures do not crash the server.

## Observability

MCP tool calls appear in `agentic_tool_calls` records and `agentic_tool_call_progress` streaming events with the prefixed name. They share the [tool-round cap](/mistral.rs/guides/agents/tool-calling-basics/#configuring-the-tool-loop).

Full examples: [Python MCP client](/mistral.rs/examples/python/mcp-client/), [Rust MCP client](/mistral.rs/examples/rust/advanced/mcp-client/), [HTTP chat with MCP tools](/mistral.rs/examples/server/mcp-chat/).
