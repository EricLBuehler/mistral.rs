---
title: Build an agent
description: Turn on tool calling, web search, and code execution so the model can take actions, not just reply. Watch the agentic loop work end to end. About fifteen minutes.
sidebar:
  order: 5
---

The agentic loop lets the server handle tool calls inside a single request: the model requests a tool, the server runs it, feeds the result back, and continues until the model produces a normal reply. Two built-in tools are covered here: web search and Python code execution. The model is Qwen3-4B.

## Starting the server with agents enabled

Both features are off by default. Each carries a cost: network access for search, a Python subprocess for code execution.

```bash
mistralrs serve --ui \
  --enable-search \
  --enable-code-execution \
  -m Qwen/Qwen3-4B
```

`--enable-search` enables the built-in web search tool. `--enable-code-execution` enables a Python subprocess that persists across calls within a session. Code execution is compiled in by default.

Open `http://localhost:1234/ui` once the server is ready.

## Watching it work in the UI

Paste into the chat box:

```
What is the current population of Tokyo, and what fraction of Japan's total population does that represent? Show your working.
```

The reply takes longer than a normal chat response because the loop runs multiple rounds. The UI shows:

1. A collapsed search block listing the query Qwen issued, with retrieved URLs and snippets.
2. A code execution block with the Python the model ran and its stdout.
3. Sometimes additional rounds for follow-up searches or calculations.
4. A final reply citing the numbers and showing the arithmetic.

Everything between the question and the final reply happened inside a single HTTP request.

The UI renders structured events the server emits as part of the response. The UI is not required, the same events are available to any client.

## API response

Stop the UI and call the endpoint directly:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the square root of 99856? Verify by squaring the result."}
    ]
  }'
```

The response body adds an `agentic_tool_calls` field alongside the standard `choices` array:

```json
{
  "id": "...",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "316. Squaring 316 gives 99856, which matches."
      },
      "finish_reason": "stop"
    }
  ],
  "agentic_tool_calls": [
    {
      "round": 0,
      "name": "execute_python",
      "arguments": "{\"code\":\"import math\\nprint(math.sqrt(99856))\\nprint(316**2)\"}",
      "result_content": "316.0\n99856\n",
      "result_images_base64": []
    }
  ],
  "usage": { "..." : "..." }
}
```

`choices` is OpenAI-compatible. `agentic_tool_calls` is a mistral.rs extension. Each entry records one round: tool name, arguments, and result. Tool-produced images (e.g., matplotlib plots) appear as base64 strings in `result_images_base64`.

For streaming, the loop emits Server-Sent Events with type `agentic_tool_call_progress`. Each event has a `phase` of `"calling"` or `"complete"`. The full schema is in the [HTTP API reference](/mistral.rs/reference/http-api/).

## Notes

Enabling the flags does not force tool use. The model is given the tools and their descriptions and decides when to call them.

Code execution is stateful within a session. Subsequent requests reusing the same session id share the Python subprocess, so prior variables remain available. If no session id is passed, one is created and returned in the response. See the [persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/).

The two flags above enable the built-in tools only. To expose custom tools (calendar API, vector search, shell), implement them as MCP servers and connect mistral.rs as a client, or register tool callbacks through the Rust or Python SDK. See the [agent guides](/mistral.rs/guides/agents/).

## Next steps

- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/): fit larger models on the available GPU.
- [The MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/): connect to a third-party MCP server.
- [The persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/): keep state across separate requests.
