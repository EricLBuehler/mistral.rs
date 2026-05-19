---
title: Connect to an MCP server
description: Use mistralrs as an MCP client.
sidebar:
  order: 7
---

mistral.rs can act as an MCP client, connecting to one or more MCP servers at startup and merging their tools into the model's available set.

MCP tools automatically use [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) when the MCP server provides an input schema.

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

## Tool name prefix

Each server's tools are exposed to the model as `<prefix>_<tool>`. The prefix is `tool_prefix` if set on the server entry, otherwise an auto-generated `mcp_<uuid>`. Separator is an underscore.

## Concurrency and timeouts

`max_concurrent_calls` caps in-flight MCP calls. `tool_timeout_secs` is the per-call timeout. `Http` and `WebSocket` sources can override the timeout per-source.

## Observability

MCP tool calls appear in `agentic_tool_calls` records and `agentic_tool_call_progress` streaming events with the prefixed name.
