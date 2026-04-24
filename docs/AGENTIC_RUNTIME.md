# Agentic Runtime

mistral.rs can be used as a local-first runtime for agent applications. A **run** is one request that may include model generation, server-side tool execution, Python code execution, web search, generated media, and persistent session state.

This page describes the current public surfaces. The most complete app-facing event stream today is the HTTP chat-completions stream, which includes normal OpenAI-compatible model chunks plus mistral.rs tool-progress events.

## Runtime Model

A run has four moving parts:

| Part | What it carries |
|------|-----------------|
| Model output | Normal chat-completion responses or streaming chunks |
| Tool progress | Built-in search, code execution, MCP, callbacks, or HTTP dispatch activity |
| Generated media | Images and video frames produced by tools, encoded as base64 fields |
| Session state | A reusable `session_id` for multi-turn tool and code state |

The same engine loop powers the HTTP API, Rust SDK, Python SDK, CLI, and web UI. The surfaces differ in how much of the execution timeline they expose.

| Surface | Current timeline support |
|---------|--------------------------|
| HTTP `/v1/chat/completions` | Streams `agentic_tool_call_progress` SSE events alongside model chunks |
| Rust SDK | `stream_chat_request` yields raw `Response::AgenticToolCallProgress` events |
| Python SDK | Supports agentic requests, callbacks, code execution, and sessions; streaming iterator currently yields model chunks |
| Web UI | Renders code execution, search, reasoning blocks, and generated media inline |

## HTTP API

Start a server with the tools your application is allowed to use:

```bash
mistralrs serve \
  --enable-code-execution \
  --enable-search \
  --ui \
  -m google/gemma-4-E4B-it
```

Then call `/v1/chat/completions` with the per-request runtime features enabled:

```json
{
  "model": "default",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "Use Python to plot sin(x), then explain the chart."
    }
  ],
  "enable_code_execution": true,
  "web_search_options": {},
  "max_tool_rounds": 4,
  "session_id": "analysis-demo"
}
```

When `stream` is `true`, normal model chunks are streamed as OpenAI-compatible chat-completion chunks. Tool activity is streamed as named SSE events:

```text
event: agentic_tool_call_progress
data: {"type":"agentic_tool_call_progress","round":0,"tool_name":"mistralrs_execute_python","phase":"calling","data":{"tool_type":"code_execution","code":"print('hello')"}}
```

When a tool finishes, the completion event includes the result payload. Code execution events can include stdout, stderr, exceptions, captured images, captured video frames, working directory, and execution time:

```json
{
  "type": "agentic_tool_call_progress",
  "round": 0,
  "tool_name": "mistralrs_execute_python",
  "phase": "complete",
  "data": {
    "tool_type": "code_execution",
    "stdout": "saved plot\n",
    "images_base64": ["..."],
    "video_frames_base64": ["..."],
    "video_frame_count": 12,
    "working_directory": "/tmp/mistralrs-code-demo",
    "execution_time_ms": 118
  }
}
```

For non-streaming HTTP requests, the final `ChatCompletionResponse` includes `agentic_tool_calls` with the collected execution history. Use streaming when a UI needs a live tool timeline.

### Request Fields

| Field | Use |
|-------|-----|
| `enable_code_execution` | Enables the built-in Python tools for this request. The server must also be started with `--enable-code-execution`. |
| `web_search_options` | Enables built-in web search for this request. The server must also be started with `--enable-search`. |
| `max_tool_rounds` | Lets the server continue tool execution and model generation without client round-trips. |
| `session_id` | Reuses persistent agentic state across requests. |

For security, HTTP tool dispatch URLs are configured with the server flag `--tool-dispatch-url`; they are not accepted as arbitrary per-request HTTP fields.

### Sessions

Reusable sessions make agent apps feel continuous without making the client manually replay every intermediate tool message.

| Endpoint | Purpose |
|----------|---------|
| `GET /v1/sessions/{session_id}` | Export a serialized session |
| `PUT /v1/sessions/{session_id}` | Import or replace a session |
| `DELETE /v1/sessions/{session_id}` | Delete a session |

## Rust SDK

The Rust SDK exposes the same engine events through `Model::stream_chat_request`. Match on `Response::AgenticToolCallProgress` to render tool activity:

```rust
use mistralrs::core::{AgenticToolCallData, AgenticToolCallPhase};
use mistralrs::{
    CodeExecutionConfig, IsqBits, ModelBuilder, RequestBuilder, Response, TextMessageRole,
    TextMessages,
};

let model = ModelBuilder::new("google/gemma-4-E4B-it")
    .with_auto_isq(IsqBits::Four)
    .with_code_execution(CodeExecutionConfig::default())
    .build()
    .await?;

let messages = TextMessages::new().add_message(
    TextMessageRole::User,
    "Use Python to plot sin(x), then explain the chart.",
);
let request = RequestBuilder::from(messages)
    .with_code_execution()
    .set_max_tool_rounds(4);

let mut stream = model.stream_chat_request(request).await?;
while let Some(event) = stream.next().await {
    match event {
        Response::Chunk(chunk) => {
            if let Some(text) = chunk
                .choices
                .first()
                .and_then(|choice| choice.delta.content.as_deref())
            {
                print!("{text}");
            }
        }
        Response::AgenticToolCallProgress {
            tool_name, phase, ..
        } => match phase {
            AgenticToolCallPhase::Calling(_) => {
                eprintln!("{tool_name}: calling");
            }
            AgenticToolCallPhase::Complete(AgenticToolCallData::CodeExecution {
                stdout,
                images,
                video_frames,
                ..
            }) => {
                eprintln!(
                    "{tool_name}: stdout={} images={} video_frames={}",
                    stdout.as_deref().unwrap_or(""),
                    images.len(),
                    video_frames.len()
                );
            }
            AgenticToolCallPhase::Complete(_) => {
                eprintln!("{tool_name}: complete");
            }
        },
        Response::Done(response) => {
            eprintln!("session: {:?}", response.session_id);
        }
        _ => {}
    }
}
```

For simpler use, `send_chat_request` keeps the chat-completion shape and skips intermediate progress events.

## Python SDK

The Python SDK can build agentic requests with callbacks, web search, code execution, and persistent sessions:

```python
from mistralrs import (
    ChatCompletionRequest,
    CodeExecutionConfig,
    Runner,
    Which,
)

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
    code_execution_config=CodeExecutionConfig(),
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "Use Python to calculate the first 20 primes and their sum.",
            }
        ],
        enable_code_execution=True,
        max_tool_rounds=4,
        session_id="prime-demo",
    )
)

print(response.choices[0].message.content)
```

When `stream=True`, the Python iterator yields model chunks. If a Python application needs the complete tool-progress timeline today, call the HTTP API directly and consume `agentic_tool_call_progress` SSE events.

The Python SDK also exposes session management helpers:

```python
session_json = runner.export_session("prime-demo")
runner.import_session("restored-demo", session_json)
runner.delete_session("prime-demo")
```

## Event Schema

Tool progress events use this JSON shape in HTTP SSE:

| Field | Meaning |
|-------|---------|
| `type` | `agentic_tool_call_progress` |
| `round` | Agentic loop round |
| `tool_name` | Tool that started or completed |
| `phase` | `calling` or `complete` |
| `data.tool_type` | `code_execution`, `web_search`, or `custom` |

Tool-specific data:

| Tool type | Fields |
|-----------|--------|
| `code_execution` | `code`, `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `video_frame_count`, `working_directory`, `execution_time_ms` |
| `web_search` | `query`, `results_count` |
| `custom` | `arguments`, `content` |

## Current Boundaries

- `/v1/chat/completions` is the best current API for streaming tool progress into an app UI.
- The Rust SDK exposes raw progress events; Python SDK streaming currently keeps OpenAI-style model chunks.
- Generated media is exposed as base64 image and video-frame fields, not as a separate artifact store.
- Code execution runs with the permissions of the configured Python interpreter. Use a sandbox for untrusted workloads.
