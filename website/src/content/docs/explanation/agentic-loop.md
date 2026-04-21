---
title: The agentic loop
description: How the server-side tool loop runs inside a single HTTP request.
sidebar:
  order: 2
---

When a model with tools enabled asks to invoke one, the engine runs the tool and continues the request without returning control to the client.

## Entry conditions

The loop is entered when at least one of the following holds on the incoming request:

- `--enable-search` is on.
- `--enable-code-execution` is on (exposes the `execute_python` tool).
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

The cap is set by `--max-tool-rounds`. When unset, the loop uses an internal fallback of 16 rounds.

## Progress events

Non-streaming responses include an `agentic_tool_calls` array with one entry per executed round. Streaming responses emit `agentic_tool_call_progress` Server-Sent Events around each tool execution.

Event shape:

- Phase `calling` — before the tool runs. Includes the tool name and parsed arguments.
- Phase `complete` — after the tool runs. Data is tool-type-specific:
  - Code execution: `code`, `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `video_frame_count`, `working_directory`, `execution_time_ms`.
  - Web search: `query`, `results_count`.
  - Custom tools: `arguments`, `content`.

## Session interaction

The loop is session-aware. At termination the full expanded message list — including the synthesized assistant tool-call messages and the `tool`-role responses the client never sent — is written back to the session. On the next request with the same session id, that history is spliced back in so the model sees a consistent conversation.

## Client-side path

If none of the entry conditions are met, the request is dispatched directly. The model's `tool_calls` field is returned to the client and the client runs the next round. This is the standard OpenAI-compatible flow.

## See also

- Guide: [tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/), [configure the tool loop](/mistral.rs/guides/agents/configure-tool-loop/).
- Reference: [HTTP API](/mistral.rs/reference/http-api/).
