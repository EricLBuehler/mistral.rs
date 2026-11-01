---
title: Agentic runtime for apps
description: Consume tool progress, files, and sessions in an agent application.
---

Use the server-side runtime when your app needs model generation and tool execution in one request. Start with [Build an agent](/guides/agents/build-an-agent/), then use this page to handle streamed progress and outputs.

## How the loop runs

Enable executors at startup and request the tools you need. The runtime runs a tool call, appends its result to history, and resumes generation until the model answers or reaches `max_tool_rounds`. If a round contains several tool calls, only the first is executed.

Without a registered executor or dispatch URL, handle `tool_calls` in your application. See [tool calling](/guides/agents/tool-calling-basics/) for callbacks, dispatch, and round limits.

## HTTP run stream

Start the tools and send a streaming request to `/v1/chat/completions`:

```bash
mistralrs serve --agent -m google/gemma-4-E4B-it
```

```json
{
  "model": "default",
  "stream": true,
  "messages": [{"role": "user", "content": "Plot sin(x) as plot.png and explain it."}],
  "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
  "files": [{"name": "plot.png", "format": "png"}],
  "max_tool_rounds": 4,
  "session_id": "analysis-demo"
}
```

Read ordinary chat chunks for model output and named SSE events for tools and files:

| Event | App behavior |
|---|---|
| `agentic_tool_call_progress`, phase `calling` | Show the tool arguments and running status. |
| `agentic_tool_call_progress`, phase `complete` | Show the result, stdout/stderr, or generated media. |
| `file_produced` | Attach the file to the conversation and offer a download. |

Use `round` and the opaque tool identifier for correlation. Non-streaming replies include `agentic_tool_calls` instead. See the [HTTP reference](/reference/http-api/) for payloads and the [HTTP example](/examples/server/agentic-tool-rounds/) for a client.

## Files

Tools automatically surface new or modified output files from their working directory. Declare required outputs in `files`, as above, to supply metadata and receive error placeholders for missing files. Use `mistralrs_surface_outputs` to retrieve files from earlier shell calls.

Chat Completions and Anthropic Messages return a top-level `files` array. Responses attaches `container_file_citation` annotations to assistant text. Download bytes through `GET /v1/files/{id}/content`; Responses also supports the container-file content route.

Bodies up to 8 MB are included inline. Fetch larger files by ID; SDK `File.is_truncated()` reports an omitted body. Text output previews are limited to 1,024 bytes, so use the file content for complete results.

For uploads, inline input files, and text-preview limits, see [file inputs](/guides/agents/file-inputs/). For output declarations in each SDK, see [code execution](/guides/agents/enable-code-execution/#declaring-output-files) and [shell outputs](/guides/agents/enable-shell/#file-outputs).

## Sessions

Pass a unique `session_id` to retain history, tool records, media, and Python interpreter state. Sessions are in memory; export them for persistence. Interpreter state itself cannot be exported. See [sessions](/guides/agents/persist-sessions/) for reuse, expiry, and lifecycle methods.

## SDK boundaries

| Surface | Streaming support |
|---|---|
| HTTP | Model chunks, tool progress, files, and approval events. |
| Rust | `Model::stream_chat_request` yields raw responses, including `Response::AgenticToolCallProgress`. |
| Python | The native iterator yields model chunks; use HTTP SSE for the full tool timeline. |
| Web UI | Renders tool work, media, files, and approvals. |

Both native SDKs support input files, callbacks, code execution, shell, local Skills, and sessions. See [Rust streaming](/examples/rust/advanced/agent-streaming/) and [Python agentic tools](/examples/python/agentic-tools/) examples.

## Security

CLI agent mode uses the `developer` sandbox profile on supported platforms. Choose `restricted` and tighter network limits for untrusted workloads. Native SDKs do not sandbox by default; attach a `SandboxPolicy` to enable isolation. Use [permissions and approvals](/guides/agents/permissions-and-approvals/) to gate actions; requests cannot loosen server-wide `ask` or `deny` policies. See the [sandbox reference](/reference/sandbox/) for isolation limits.
