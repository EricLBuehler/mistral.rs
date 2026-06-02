---
title: The agentic loop
description: How the server-side tool loop runs inside a single HTTP request.
sidebar:
  order: 2
---

With tools enabled, the engine runs the tool and continues the request without returning control to the client.

## Entry conditions

The loop runs only when the model emits a tool call. The server advertises tools to the model in any of these cases:

- `--enable-search` is on (advertises the web search tool).
- `--enable-code-execution` is on and the request sets `enable_code_execution: true` (advertises the `mistralrs_execute_python` tool).
- A registered tool callback exists (Rust/Python SDK `tool_callbacks` or MCP client tools).
- `--tool-dispatch-url` is set.

Otherwise the request is dispatched normally and this page does not apply.

## Round structure

Each iteration:

1. The engine runs inference. The result is a model response that either contains tool calls or does not.
2. If the response contains no tool calls, the loop exits and the response is forwarded to the client. If more than one tool call is returned, only the first is executed and a warning is logged.
3. The loop emits a progress event with phase `calling` and the tool arguments.
4. The tool is executed through one of four paths: built-in web search, built-in code execution, a registered callback, or a POST to `--tool-dispatch-url`.
5. The loop emits a progress event with phase `complete` and the structured result.
6. The message history is extended with the assistant's tool-call message and a `tool`-role response, so the next inference pass sees the outcome.
7. If the round counter reaches the configured cap, the loop exits without another tool opportunity.

The cap is set by `--max-tool-rounds`. When unset, the loop uses an internal fallback of 256 rounds.

## Progress events

Non-streaming responses include an `agentic_tool_calls` array with one entry per executed round. Streaming responses emit `agentic_tool_call_progress` Server-Sent Events around each tool execution.

Event shape:

- Phase `calling`: before the tool runs. Includes the tool name and parsed arguments.
- Phase `complete`: after the tool runs. Data is tool-type-specific:
  - Code execution: `code`, `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `video_frame_count`, `working_directory`, `execution_time_ms`.
  - Web search: `query`, `results_count`.
  - Custom tools: `arguments`, `content`.

## Files

The loop also produces typed `File` outputs alongside the tool-call records. When the request declares `files: [...]` or a tool writes into the working directory and lists the file in its `outputs` parameter, the runtime captures it, attributes it to the producing round, and emits it as a `file_produced` SSE event during streaming or as a top-level `files[]` entry on the non-streaming response. Each `agentic_tool_calls[*].file_ids` lists the ids attributable to that round. See [agentic runtime: files](/mistral.rs/guides/agents/agentic-runtime/#files).

## Session interaction

At termination, the expanded message list (synthesized assistant tool-call messages and `tool`-role responses) is written back to the session. On the next request with the same session id, that history is spliced back in.

## Client-side path

If none of the entry conditions are met, the request is dispatched directly. The model's `tool_calls` field is returned to the client and the client runs the next round. This is the standard OpenAI-compatible flow.

## See also

- Guide: [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/), [tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/), [configure the tool loop](/mistral.rs/guides/agents/configure-tool-loop/).
- Reference: [HTTP API](/mistral.rs/reference/http-api/).
