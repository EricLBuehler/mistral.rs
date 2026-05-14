---
title: Configure the tool loop
description: Round caps and dispatch URLs.
sidebar:
  order: 8
---

The server-side tool loop runs whenever the model invokes a tool during a request.

## Maximum tool rounds

```bash
mistralrs serve --max-tool-rounds 10 -m <model>
```

`--max-tool-rounds` caps rounds before the server forces a final response. The flag is unset by default; the loop's internal fallback cap is 16.

## Dispatch URL

`--tool-dispatch-url` POSTs each tool call to an external URL instead of running it in-process.

```bash
mistralrs serve --tool-dispatch-url http://localhost:7070/tools -m <model>
```

Request body sent to the dispatch URL:

```json
{
  "name": "search",
  "arguments": { "query": "mistralrs" }
}
```

Expected response: a string (the tool result) or `{"content": "..."}`.

## Per-request versus server-level

Per-request:

- `tools`: tools the model can see.
- `tool_choice`: force, disable, or model-decides.
- `web_search_options.search_context_size`.
- `max_tool_rounds`: override the server default for one request.
- `session_id`: reuse persistent agentic state.
- `enable_code_execution`: opt into the built-in Python tools when the server has code execution enabled.

Server-level:

- `--max-tool-rounds`
- `--tool-dispatch-url`
- `--enable-search`, `--enable-code-execution`
- `--search-embedding-model`
- `--code-exec-python`, `--code-exec-workdir`, `--code-exec-timeout`

## MCP tools

Connected MCP server tools (see [connect-mcp-server](/mistral.rs/guides/agents/connect-mcp-server/)) merge into the tool set. They share the `--max-tool-rounds` cap.

## Streaming progress events

Tool rounds emit `agentic_tool_call_progress` SSE events with `phase: "calling"` before and `phase: "complete"` after each tool. The event `data` includes tool-type-specific fields (e.g., `code`, `stdout`, `stderr`, `images_base64` for code execution).

For the full app-facing event shape, see [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).
