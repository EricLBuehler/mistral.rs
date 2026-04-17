---
title: Tool calling basics
description: How tool calling works in mistralrs, what the request and response shapes look like, and when to use the built-in loop vs handling it yourself.
sidebar:
  order: 1
---

Tool calling is how a model requests external work. Tools are described to the model (name, arguments, purpose), and the model can emit a structured invocation instead of a text reply. The caller runs the tool, returns the result, and the model continues.

This guide covers the two tool-calling modes in mistral.rs, the request and response shapes, and when to use each.

## The two modes

**Server-side loop.** mistral.rs runs the entire tool loop inside one HTTP request. Tools are enabled with CLI flags; the client sends one request and receives one final reply. [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) demonstrates this.

**Client-side loop.** Standard OpenAI-style flow: the model emits a `tool_calls` field, the caller runs the tool, sends the result back as a `tool` message, and the model produces another response. The loop runs in client code with one HTTP round trip per round.

The server-side loop is the default — less work, fewer round trips, no model-state re-hydration, and works with built-in tools (web search, code execution, MCP). The client-side loop is for cases requiring tool-call interception (audit, modification, custom routing) or porting from existing OpenAI code.

## Client-side: defining a tool

Tool definitions follow the OpenAI schema. The `tools` field is a list of tool specs:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

When the model calls the tool, the response carries a `tool_calls` array:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Tokyo\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

The caller parses `arguments`, invokes the real API, and sends the result back:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "tool_calls": [...]},
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temperature\": 18, \"conditions\": \"rain\"}"
    }
  ],
  "tools": [...]
}
```

The model produces a final user-facing reply.

## Server-side: registering tools at startup

Tools available on every request are registered at server start. Built-in tools (search, code execution) are flag-gated:

```bash
mistralrs serve --enable-search --enable-code-execution -m <model>
```

For custom tools (database lookup, calendar API, application-specific tools), the cleanest path is to run them as an MCP server and connect mistralrs as a client. See the [MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/).

For lighter use, the Rust SDK accepts a map of tool callbacks via `ModelBuilder::with_tool_callbacks`. See the [Rust API reference](/mistral.rs/reference/rust-api/).

## Forcing or suppressing tool use

Two request fields control tool behavior:

- `tool_choice: "none"` — disable tool calling for the request.
- `tool_choice: "auto"` — default; the model decides.
- `tool_choice: {"type": "function", "function": {"name": "get_weather"}}` — force a specific tool.

## mistral.rs-specific extensions

When the server-side loop runs, the response includes an `agentic_tool_calls` array. Each entry documents one round: tool name, arguments, result, and any images or videos produced. Full schema: [HTTP API reference](/mistral.rs/reference/http-api/). See [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/#what-the-api-returns) for an example.

For streaming responses, tool rounds emit `agentic_tool_call_progress` events with `phase: "calling"` before and `phase: "complete"` after each tool. The web UI renders these inline.
