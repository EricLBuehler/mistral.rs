---
title: Expose mistralrs as an MCP server
description: Let other tools (Claude Desktop, Cursor, custom agents) call your mistralrs instance as an MCP tool provider.
sidebar:
  order: 5
---

mistral.rs can also expose itself *as* an MCP server. An external agent host (Claude Desktop, custom orchestrator) can then call your models as tools.

Use cases:

- A locally hosted model made available to an external agent.
- Model offerings to MCP-aware applications without OpenAI API knowledge.
- Multi-agent systems with different models in different roles.

## Starting an MCP server

```bash
mistralrs serve --mcp -m Qwen/Qwen3-4B
```

`--mcp` enables the MCP endpoint alongside the HTTP API. Both run in one process.

The MCP endpoint speaks standard MCP transport over stdio or HTTP. For Claude Desktop, stdio is typical:

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

Claude Desktop spawns mistralrs as a subprocess and communicates over stdin/stdout. No HTTP port, CORS, or auth.

For HTTP/WebSocket clients, use `--mcp-transport http`:

```bash
mistralrs serve --mcp --mcp-transport http --mcp-port 3030 -m <model>
```

The client connects to `http://localhost:3030/mcp`.

## What tools get exposed

Each loaded model is exposed as a tool by default. A single-model server advertises one tool (named after the model); multi-model serves advertise one per model.

The tool schema includes standard chat completion parameters: messages, sampling options, max tokens. The external agent calls the tool with messages and receives a completion.

To restrict exposed models, pass `--mcp-models`:

```bash
mistralrs serve --mcp --mcp-models "default,qwen-fast" -m <model>
```

Only listed models are advertised.

## Nested tool calls

If the external agent calls mistralrs-as-MCP and mistralrs has its own tools enabled (search, code execution, connected MCP servers), those tools run as part of the call. The external agent sees only the final answer; the internal loop is invisible.

This composition is useful but has a visibility cost. The external agent cannot inspect intermediate steps. The `agentic_tool_calls` field is available via the HTTP API but not over MCP.

## Authentication

MCP over stdio runs in a client-controlled subprocess; auth is implicit in the subprocess boundary. MCP over HTTP has no standard auth story — for non-localhost use, place an authenticating proxy in front. The same reverse proxies that work for the HTTP API (nginx, Caddy, Traefik) work for MCP-over-HTTP.

## Limits and notes

- Each MCP call corresponds to one chat completion. There is no persistent session across calls unless the external agent passes the session id through.
- Streaming over MCP is supported but not all clients handle it. When in doubt, use non-streaming.
- The tool schema is generated from the loaded model's capabilities. Multimodal-incapable models omit the corresponding parameters.
