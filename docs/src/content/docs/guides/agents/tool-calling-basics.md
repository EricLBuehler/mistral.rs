---
title: Tool calling basics
description: Server-side and client-side tool calling.
sidebar:
  order: 1
---

Tool calling lets the model request external work via a structured invocation. mistral.rs supports both server-side and client-side tool loops, plus strict tool schemas for constraining generated arguments before a tool runs.

## The two modes

**Server-side loop.** mistral.rs runs the entire tool loop inside one HTTP request. Tools are enabled with CLI flags; the client sends one request and receives one final reply.

**Client-side loop.** Standard OpenAI flow: the model emits a `tool_calls` field, the caller runs the tool, sends the result back as a `tool` message, and the model produces another response.

## Client-side: defining tools

Tool definitions follow the OpenAI schema:

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
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

Set `function.strict` to `true` when you want mistral.rs to constrain the generated arguments to the tool's JSON Schema. See [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) for HTTP, Python, and Rust examples.

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

The caller invokes the real API and sends the result back:

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

## Server-side: enabling built-in tools

```bash
mistralrs serve --enable-search --enable-code-execution -m <model>
```

For custom tools, the cleanest path is to run them as an MCP server and connect mistralrs as a client. See the [MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/).

The SDKs can also register custom callbacks directly: Python uses `Runner(tool_callbacks=...)`; Rust builders use `with_tool_callback(...)` or `with_tool_callback_and_tool(...)`.

Built-in search, code execution, and file helper tools use strict schemas by default. MCP tools also use strict schemas when the MCP server provides an input schema.

## Forcing or suppressing tool use

- `tool_choice: "none"`: disable tool calling for the request.
- `tool_choice: "auto"` (default): model decides.
- `tool_choice: {"type": "function", "function": {"name": "..."}}`: force a specific tool.

`"required"` is not supported. Use a specific function object to force tool use.

## mistral.rs response extensions

When the server-side loop runs, the response includes an `agentic_tool_calls` array. Each entry: `round`, `name`, `arguments`, `result_content`, `result_images_base64`.

For streaming, tool rounds emit `agentic_tool_call_progress` events with `phase: "calling"` before and `phase: "complete"` after each tool. Event `data` includes tool-type-specific fields.
