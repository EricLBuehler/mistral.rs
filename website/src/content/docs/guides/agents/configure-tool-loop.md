---
title: Configure the tool loop
description: Round caps, dispatch URLs, default behaviors. What the knobs do and when to reach for each.
sidebar:
  order: 7
---

The server-side tool loop runs whenever the model invokes a tool during a request. Defaults work for most cases. This guide covers the knobs that override them.

## Maximum tool rounds

By default, the loop can run for many rounds within one request. The max-tool-rounds setting caps rounds before the server forces a final response.

```bash
mistralrs serve --max-tool-rounds 10 -m <model>
```

When the cap is reached, the model gets one final chance to produce a user-facing answer with whatever information it has, with no further tool access. If the final response is still a tool call, the server returns an error.

Reasonable values:

- `3`–`5` — interactive use prioritizing fast responses. Insufficient for complex tasks.
- `10` — default. Handles most agentic workflows without artificial limits.
- `20`+ — long-running planning. Often unnecessary; if needed, model quality is usually the bottleneck.

## Dispatch URL for custom tool execution

By default, tools execute inside the mistralrs process. For execution in a separate process (sandboxed container, separate service, another runtime), set a dispatch URL:

```bash
mistralrs serve --tool-dispatch-url http://localhost:7070/tools -m <model>
```

When configured, mistralrs POSTs each tool call to that URL and waits for the response. The external service runs the tool and returns the result.

Request body:

```json
{
  "name": "search",
  "arguments": "{\"query\": \"mistralrs\"}",
  "session_id": "user-123"
}
```

Expected response: a string (the tool result) or a JSON object matching the OpenAI tool-result shape.

This pattern fits centralized tool audit logging or existing tool infrastructure.

## Per-request versus server-level defaults

Some tool behavior is per-request; some is startup-only.

Per-request:

- `tools` — tools the model can see.
- `tool_choice` — force, disable, or model-decides.
- `web_search_options.search_context_size`
- `session_id`

Server-level only:

- `--max-tool-rounds`
- `--tool-dispatch-url`
- `--enable-search`, `--enable-code-execution`
- `--search-embedding-model`
- `--code-working-dir`, `--code-timeout-secs`

Correctness and safety controls (which tools exist, how they run) are server-level. Request-specific behavior is client-controllable.

## Tools from MCP servers

Connected MCP server tools (see [connect-mcp-server](/mistral.rs/guides/agents/connect-mcp-server/)) merge into the tool set the model sees. Auto-registration includes all of them; manual filtering is available.

MCP tools share the `--max-tool-rounds` cap with built-in tools. The round counter is cumulative across all tools.

## Debugging the loop

**Logging.** `RUST_LOG=mistralrs_core::engine::agentic_loop=debug` prints one line per round with tool name, arguments, and result. Noisy but complete.

**Response inspection.** Non-streaming responses include the full `agentic_tool_calls` array. Reading it is usually faster than reading logs.

**Constraining tools.** If the loop misbehaves, temporarily disable specific tools (via `tool_choice` or by unloading the relevant MCP server) to narrow the cause.
