---
title: Expose mistralrs as an MCP server
description: Serve the loaded model as an MCP tool other agents can call.
---

mistral.rs can expose the loaded model as an MCP server: a `chat` tool over JSON-RPC 2.0 that any MCP client can call.

```bash
mistralrs serve -m Qwen/Qwen3-4B --mcp-port 4321
```

`--mcp-port` starts an additional listener on the same `--host` as the main HTTP API. The OpenAI-compatible API on `--port` (default 1234) always runs alongside; `--mcp-port` must differ from `--port`, and the bind is checked at startup so failures surface before serving. In a [TOML config](/mistral.rs/reference/cli-toml-config/), the equivalent is `mcp_port` under `[server]`:

```toml
command = "serve"

[server]
port = 1234
mcp_port = 4321
```

Clients connect to `http://<host>:<mcp_port>/mcp`. Each call is a `POST /mcp` with a JSON-RPC 2.0 body.

## Methods

- `initialize`: returns `{"capabilities":{"tools":{}},"instructions":...,"protocolVersion":"2025-11-25","serverInfo":{"name":"mistralrs","version":...}}`.
- `ping`: returns `{}`.
- `tools/list`: returns the `chat` tool. The list is empty if the loaded model does not have text input and output modalities.
- `tools/call`: runs the `chat` tool.

Anything else returns JSON-RPC error -32601 (method not found). A body with `jsonrpc` other than `"2.0"` returns -32600; tool execution failures return -32603.

## The chat tool

The advertised input schema requires `messages` (an array of `{role, content}` objects with roles `user`, `assistant`, or `system`) and documents `max_tokens` and `temperature`. The `arguments` object is actually deserialized as a full OpenAI [`ChatCompletionRequest`](/mistral.rs/reference/http-api/), so any of its fields work; `model` defaults to `"default"`.

```bash
curl http://localhost:4321/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "chat",
      "arguments": {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
      }
    }
  }'
```

The result is MCP tool-call content:

```json
{"content": [{"type": "text", "text": "Hello! How can I help?"}]}
```

## Authentication

The MCP endpoint has no built-in authentication. For non-localhost use, place an authenticating proxy in front.

## See also

- [Connect to an MCP server](/mistral.rs/guides/agents/connect-mcp-server/): the opposite direction, using external MCP tools from mistral.rs.
- [Serve CLI reference](/mistral.rs/reference/cli/serve/).
