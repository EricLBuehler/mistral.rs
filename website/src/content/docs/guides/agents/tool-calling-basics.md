---
title: Tool calling basics
description: How tool calling works in mistralrs, what the request and response shapes look like, and when to use the built-in loop vs handling it yourself.
sidebar:
  order: 1
---

Tool calling is the mechanism by which a model asks for external work to be done. In its most basic form: you describe a tool to the model (its name, its arguments, what it does), and the model can emit a structured request to invoke that tool instead of generating a regular text reply. Your code runs the tool, feeds the result back in, and the model continues.

This guide covers the two ways mistral.rs exposes tool calling, the request and response shapes involved, and when to use each.

## The two modes

**Server-side loop.** mistral.rs runs the entire tool-call loop inside a single HTTP request. You turn on the relevant tools with CLI flags, and from the client's point of view you just send one request and get one final reply. This is what [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) walks through.

**Client-side loop.** The classic OpenAI-style interaction: the model emits a `tool_calls` field in its response, your code runs the tool, you send the result back as a `tool` message, and the model produces a new response. The loop runs in your code, one HTTP round trip per round.

The server-side loop is the right default. It is less work, more efficient (fewer round trips, model state does not have to be re-hydrated), and works with tools that are already built into the engine (web search, code execution, MCP). The client-side loop is what you want when you need to intercept tool calls (to audit them, modify them, or route them to custom handlers) or when you are porting code from an existing OpenAI-based setup.

## Client-side: defining a tool

Tool definitions follow the OpenAI schema. A `tools` field on the request is a list of tool specs:

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

When the model decides to call the tool, the response has a `tool_calls` array instead of (or alongside) a normal content:

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

Your code parses `arguments`, calls the real weather API, and sends the result back:

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

The model sees the result and produces a final user-facing reply.

## Server-side: registering tools at startup

For tools that should be available on every request, register them when the server starts. The built-in ones (search, code execution) are opt-in via flags:

```bash
mistralrs serve --enable-search --enable-code-execution -m <model>
```

For custom tools (a database lookup, a calendar API, anything specific to your application), the cleanest path is to run them as an MCP server and connect mistralrs to it as a client. See the [MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/).

If MCP is too much ceremony for a one-off, the Rust SDK accepts a map of tool callbacks directly. See the [Rust API reference](/mistral.rs/reference/rust-api/) for `ModelBuilder::with_tool_callbacks`.

## Forcing or suppressing tool use

Two request-level fields control tool behavior:

- `tool_choice: "none"` disables tool calling entirely for that request. The model must produce a regular text reply.
- `tool_choice: "auto"` is the default: the model decides whether to call a tool.
- `tool_choice: {"type": "function", "function": {"name": "get_weather"}}` forces a specific tool call. The model has to invoke that function; it cannot produce a regular reply.

## How tool calls appear in mistralrs-specific extensions

When the server-side loop runs, the final response includes an `agentic_tool_calls` array. Each entry documents one round of the loop: tool name, arguments, result, any images or videos produced. The [HTTP API reference](/mistral.rs/reference/http-api/) has the full schema, and [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/#what-the-api-returns) shows a full example.

For streaming responses, tool rounds emit `agentic_tool_call_progress` events with `phase: "calling"` before and `phase: "complete"` after each tool runs. The web UI renders these inline; client code can too.
