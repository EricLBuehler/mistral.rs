---
title: Connect to an MCP server
description: Use mistralrs as an MCP client. Wire up external tool providers (Slack, GitHub, databases, custom ones) so the model can call them.
sidebar:
  order: 4
---

The Model Context Protocol is a standard for tool providers exposing tools to LLM hosts. Many tools ship as MCP servers: Slack, GitHub, filesystem access, databases, and project-specific tools. mistral.rs acts as an MCP client, connecting to multiple servers at startup and merging their tools into the model's available set.

This is typically the cleanest way to add custom tools — write the tool once, expose via MCP, and any MCP-aware host (mistralrs, Claude Desktop, Cursor) can use it.

## Starting with a config file

MCP clients are configured via JSON. Pass the file with `--mcp-config`:

```bash
mistralrs serve --mcp-config mcp.json -m <model>
```

The config lists MCP servers and per-server transport details:

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
    },
    {
      "name": "slack",
      "source": {
        "type": "Http",
        "url": "https://mcp.slack.example.com",
        "headers": { "Authorization": "Bearer ..." }
      }
    }
  ],
  "auto_register_tools": true,
  "max_concurrent_calls": 4
}
```

Three transports are supported. Details: [MCP transports reference](/mistral.rs/reference/mcp-config-schema/).

- **Process** — launch the server as a subprocess and communicate over stdio. Most common for npm or Python MCP packages.
- **HTTP** — connect over plain HTTP. Used for remote servers.
- **WebSocket** — connect over WebSockets. Used for long-lived connections.

## Auto-registration versus explicit

`auto_register_tools: true` exposes every tool from every connected server to the model.

For more control (e.g., excluding a `delete` tool), set `auto_register_tools: false` and list tools per server:

```json
{
  "name": "filesystem",
  "source": { "type": "Process", ... },
  "tool_filter": ["read_file", "list_directory"]
}
```

Only listed tools pass through.

## Concurrency and timeouts

`max_concurrent_calls` caps in-flight MCP calls. Higher values let the model use multiple tools at once. Lower values are safer for tools with side effects.

`default_timeout_secs` controls per-call timeout. Default 30 seconds. Raise globally or per-server for known-slow tools (e.g., embedding-based retrieval).

## Observability

MCP tool calls appear in the same `agentic_tool_calls` response field as built-in tools, prefixed with the server name:

```json
{
  "name": "slack.send_message",
  "arguments": "{\"channel\": \"general\", \"text\": \"hi\"}",
  "result_content": "{\"ok\": true, \"ts\": \"1234567890.1234\"}"
}
```

The web UI renders MCP tool calls identically to built-in ones.

## Writing your own MCP server

For tools without an existing MCP server, write one. The [MCP reference implementation](https://github.com/modelcontextprotocol) provides SDKs in several languages with templates. Internal tools can be stood up in a couple dozen lines of Python or TypeScript.

After the server speaks MCP, add it to the mistralrs config. No mistralrs changes are required.

## Troubleshooting

If MCP tools are not reaching the model:

- Run `mistralrs doctor`. It reports MCP connection attempts and outcomes.
- Check server logs for "Connected to MCP server: <name>" and "Registered N tools from <name>".
- If `auto_register_tools` is on but tools do not appear, the MCP server may not be advertising them correctly. Verify with another MCP client (e.g., Claude Desktop).
- If connection succeeds but tool calls fail, `DEBUG`-level mistralrs request logs show the exact JSON exchanged.
