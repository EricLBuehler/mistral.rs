---
title: Expose mistralrs as an MCP server
description: Let other tools (Claude Desktop, Cursor, custom agents) call your mistralrs instance as an MCP tool provider.
sidebar:
  order: 5
---

The other direction of the MCP pattern: instead of mistral.rs consuming tools from other MCP servers, mistral.rs can expose itself *as* an MCP server. This lets another agent host, like Claude Desktop or a custom orchestrator, call your models as if they were tools.

This is useful when:

- You have a locally hosted model that you want to make available to an external agent.
- You want to offer your models to other MCP-aware applications without them having to know about OpenAI-compatible APIs.
- You are building a multi-agent system where different models play different roles.

## Starting an MCP server

```bash
mistralrs serve --mcp -m Qwen/Qwen3-4B
```

The `--mcp` flag enables the MCP endpoint alongside the normal HTTP API. Both run in the same process; clients can use whichever protocol suits them.

The MCP endpoint speaks the standard MCP transport over stdio or HTTP. For a Claude Desktop configuration, stdio is the usual shape:

```json
{
  "mcpServers": {
    "mistralrs": {
      "command": "mistralrs",
      "args": ["serve", "--mcp", "--mcp-transport", "stdio", "-m", "Qwen/Qwen3-4B"]
    }
  }
}
```

Claude Desktop will spawn mistralrs as a subprocess and talk to it through stdin/stdout. No HTTP port, no CORS, no auth to configure.

For other MCP clients that want an HTTP or WebSocket endpoint, use `--mcp-transport http`:

```bash
mistralrs serve --mcp --mcp-transport http --mcp-port 3030 -m <model>
```

The client then connects to `http://localhost:3030/mcp`.

## What tools get exposed

By default, each loaded model gets exposed as a tool. A single-model server advertises one tool (named after the model); a multi-model server advertises one per model.

The tool schema includes the standard chat completion parameters: messages, sampling options, max tokens. The external agent calls the tool with a list of messages and gets back a completion.

If you want to restrict which loaded models are exposed, pass `--mcp-models` at startup:

```bash
mistralrs serve --mcp --mcp-models "default,qwen-fast" -m <model>
```

Only the listed models are advertised.

## Nested tool calls

If the external agent calls mistralrs-as-MCP, and mistralrs itself has tools enabled (search, code execution, connected MCP servers), those tools run as part of the call. The external agent sees the final answer; the internal tool loop is invisible to it.

This composition is useful but has a consistency cost. The external agent cannot see what mistralrs did along the way. If that visibility matters for your use case, the `agentic_tool_calls` field is available through the regular HTTP API but not over the MCP endpoint.

## Authentication

MCP over stdio runs inside a client-controlled subprocess, so authentication is implicit in the subprocess boundary. MCP over HTTP does not have a standard auth story, so for anything beyond localhost development, put an authenticating proxy in front of it. The same reverse proxies that work for the HTTP API (nginx, Caddy, Traefik) work for MCP-over-HTTP.

## Limits and notes

- Each MCP call from the external agent corresponds to one chat completion on mistralrs's side. There is no persistent session across calls unless the external agent passes the session id through.
- Streaming responses over MCP are supported but not every MCP client handles them. When in doubt, use non-streaming.
- The tool schema is generated from the loaded model's capabilities; if a model does not support multimodal input, the corresponding parameters are omitted from the schema.
