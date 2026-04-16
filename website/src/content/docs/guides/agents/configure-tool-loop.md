---
title: Configure the tool loop
description: Round caps, dispatch URLs, default behaviors. What the knobs do and when to reach for each.
sidebar:
  order: 7
---

The server-side tool loop is what runs when the model decides to call a tool during a request. The defaults are chosen to work for most cases. This guide covers the knobs that let you override them when the defaults are not right for your workload.

## Maximum tool rounds

By default, the loop can run for many rounds within a single request. A badly behaved model or an adversarial prompt could theoretically keep invoking tools forever. The max-tool-rounds setting caps how many rounds the server will run before giving up and producing a response anyway.

```bash
mistralrs serve --max-tool-rounds 10 -m <model>
```

When the cap is reached, the model gets one final chance to produce a user-facing answer with whatever information it has. It does not receive another opportunity to call tools; if the final response is still a tool call, the server produces an error response to the client.

Reasonable values:

- `3` to `5` for interactive use where you want fast responses. Insufficient for complex tasks.
- `10` is the default. Handles most agentic workflows without artificial limits.
- `20` or higher for long-running planning tasks. Often unnecessary; if your workload needs this, the bottleneck is usually model quality, not the round cap.

## Dispatch URL for custom tool execution

By default, tools execute inside the mistralrs process. If you want to run them in a different process (a sandboxed container, a separate service, another language runtime), set a dispatch URL:

```bash
mistralrs serve --tool-dispatch-url http://localhost:7070/tools -m <model>
```

When a model calls a tool and a dispatch URL is configured, mistralrs POSTs the tool call to that URL and waits for the response. The external service is responsible for running the tool and returning the result.

The request body looks like:

```json
{
  "name": "search",
  "arguments": "{\"query\": \"mistralrs\"}",
  "session_id": "user-123"
}
```

The expected response body is a string (the tool result) or a JSON object with the OpenAI tool-result shape.

This is the pattern for organizations that want centralized tool audit logs, or that have existing tool infrastructure they want mistralrs to use rather than replicate.

## Per-request versus server-level defaults

Some tool behavior can be changed per-request via the request body, and some can only be set at server startup.

Per-request:

- `tools` (the list of tools the model can see)
- `tool_choice` (whether to force a specific tool, disable tools, or let the model decide)
- `web_search_options.search_context_size`
- `session_id`

Server-level only:

- `--max-tool-rounds`
- `--tool-dispatch-url`
- `--enable-search`, `--enable-code-execution`
- `--search-embedding-model`
- `--code-working-dir`, `--code-timeout-secs`

The division is meant to keep things that affect correctness and safety (what tools exist, how they run) at the server level, and leave request-specific behavior to clients.

## Tools from MCP servers

When MCP servers are connected (see [connect-mcp-server](/mistral.rs/guides/agents/connect-mcp-server/)), their tools are merged into the tool set the model sees. Auto-registration includes all of them; manual filtering is also available.

MCP tools are subject to the same `--max-tool-rounds` cap as built-in tools. The round counter is cumulative across all tools in a request.

## Debugging the loop

Two levers help when the loop is misbehaving:

**Logging.** `RUST_LOG=mistralrs_core::engine::agentic_loop=debug` prints one line per round with the tool name, arguments, and result. Noisy but complete.

**Response inspection.** Non-streaming responses include the full `agentic_tool_calls` array. If the model is doing something weird, reading that array is usually faster than reading logs.

**Constraining tools.** If the loop is getting into bad patterns, temporarily disabling specific tools (via `tool_choice` or by unloading the relevant MCP server) narrows down the cause.
