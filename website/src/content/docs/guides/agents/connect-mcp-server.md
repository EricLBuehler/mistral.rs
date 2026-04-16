---
title: Connect to an MCP server
description: Use mistralrs as an MCP client. Wire up external tool providers (Slack, GitHub, databases, custom ones) so the model can call them.
sidebar:
  order: 4
---

The Model Context Protocol is a standard for tool providers to expose their tools to LLM hosts. Lots of tools already ship as MCP servers: Slack, GitHub, filesystem access, databases, and plenty of project-specific ones. mistral.rs can act as an MCP client, connecting to any number of such servers at startup and merging their tools into the set the model can use.

This is usually the cleanest way to add custom tools. You write the tool once, expose it via MCP, and every MCP-aware host (mistralrs, Claude Desktop, Cursor, and others) can use it without further integration work.

## Starting with a config file

MCP clients are configured through a JSON file. Point `mistralrs serve` at it with `--mcp-config`:

```bash
mistralrs serve --mcp-config mcp.json -m <model>
```

The config lists the MCP servers to connect to and how to talk to each one:

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

Three kinds of transport are supported, covered in detail in the [MCP transports reference](/mistral.rs/reference/mcp-config-schema/):

- **Process**: Launch the MCP server as a subprocess and talk to it over stdio. The most common choice for MCP servers that ship as npm or Python packages.
- **HTTP**: Connect over plain HTTP. Used for remote MCP servers.
- **WebSocket**: Connect over WebSockets. Used when the server needs long-lived connections.

## Auto-registration versus explicit

`auto_register_tools: true` tells mistralrs to expose every tool from every connected server to the model. This is the easy mode.

If you want more control (maybe the filesystem server exposes a `delete` tool that you do not want the model touching), set `auto_register_tools: false` and list the tools you want explicitly per server:

```json
{
  "name": "filesystem",
  "source": { "type": "Process", ... },
  "tool_filter": ["read_file", "list_directory"]
}
```

Only the listed tools are passed through.

## Concurrency and timeouts

`max_concurrent_calls` caps the number of MCP calls in flight at any moment. Higher values let the model use several tools at once, which speeds up complex tasks. Lower values are safer if the tools have side effects.

`default_timeout_secs` controls how long to wait for a tool response before giving up. The default is 30 seconds. If you have MCP servers that are known to be slow (retrieval servers doing embedding lookups, for example), raise this globally or per-server.

## Observability

When an MCP tool is called during a request, it shows up in the same `agentic_tool_calls` response field as the built-in tools. The tool name includes the server name prefix, so a Slack tool call looks like:

```json
{
  "name": "slack.send_message",
  "arguments": "{\"channel\": \"general\", \"text\": \"hi\"}",
  "result_content": "{\"ok\": true, \"ts\": \"1234567890.1234\"}"
}
```

In the web UI, MCP tool calls render the same way as built-in ones.

## Writing your own MCP server

If you need a tool that does not already exist as an MCP server, writing one is straightforward. The [MCP reference implementation](https://github.com/modelcontextprotocol) has SDKs in several languages with working templates. For one-off internal tools, the Python and TypeScript SDKs let you stand one up in a couple dozen lines.

Once your server speaks MCP, add it to the mistralrs config like any other. No changes to mistralrs itself are needed.

## Troubleshooting

A few things to check if MCP tools are not reaching the model:

- Run `mistralrs doctor`. It reports the MCP connections attempted at startup and whether they succeeded.
- Check the server logs for a "Connected to MCP server: <name>" line and a subsequent "Registered N tools from <name>" line.
- If `auto_register_tools` is on and tools still do not appear, the MCP server might not be advertising them correctly; try connecting to it with an MCP-aware client like Claude Desktop to verify.
- If the server connects but tool calls fail, the mistralrs request logs at `DEBUG` level show the exact JSON sent and received, which usually makes the problem obvious.
