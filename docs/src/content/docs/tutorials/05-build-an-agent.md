---
title: Build an agent
description: Turn on tool calling, web search, and code execution so the model can take actions. Use the same agent from the web UI, HTTP, Python, and Rust. About fifteen minutes.
sidebar:
  order: 5
---

The agentic loop lets the server handle tool calls inside a single request: the model requests a tool, the server runs it, feeds the result back, and continues until the model produces a normal reply. Unlike a plain OpenAI-compatible model server, mistral.rs can run the tool loop locally and stream both model text and tool progress from the same request.

This tutorial builds one local agent that can search the web, run Python, stream tool progress, return structured files, and keep state across requests. The model is Qwen3-4B.

## What you'll build

You will ask the agent to find population figures, calculate a percentage, and produce a chart. That one task exercises the main agentic features:

- Web search finds current source material.
- Code execution performs the calculation and creates a plot.
- Strict tool schemas constrain built-in tool arguments before dispatch.
- The web UI renders search and code activity as it happens.
- HTTP responses expose tool traces and produced files.
- Python and Rust clients can call the same runtime.
- Sessions let follow-up requests continue the same work.

## Start the agent runtime

The fastest way to start an agent is the `--agent` flag:

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

`--agent` (alias `--agentic`) desugars to `--enable-search --enable-code-execution`, runs the code-execution tool in a per-session temp working directory, and uses the built-in 256-turn agentic loop. The web UI is mounted at `/ui` by default; pass `--no-ui` to skip it.

If you want to turn the pieces on individually -- for example, search without code execution -- use the underlying flags:

```bash
mistralrs serve \
  --enable-search \
  --enable-code-execution \
  -m Qwen/Qwen3-4B
```

`--enable-search` enables the built-in web search tool. `--enable-code-execution` enables a Python subprocess that persists across calls within a session. On Linux and macOS, code execution is [sandboxed by default](/mistral.rs/reference/sandbox/) with `--sandbox auto`.

Open `http://localhost:1234/ui` once the server is ready.

## From the web UI

Paste into the chat box:

```
Find recent population figures for Tokyo and Japan, calculate Tokyo's share of Japan's population, and create a simple bar chart. Cite sources and show the calculation.
```

The reply takes longer than a normal chat response because the loop runs multiple rounds. The UI renders, in order:

1. A collapsed search block with the query, retrieved URLs, and snippets.
2. A code execution block with the Python the model ran and its stdout.
3. Generated media when the Python tool produces an image.
4. Further rounds for follow-up searches or calculations, when the model requests them.
5. A final reply citing the sources and showing the arithmetic.

Everything between the question and the final reply happens inside a single HTTP request. The UI renders structured events the server emits as part of the response. The same events are available to any client.

## From HTTP

Apps can make the output contract explicit by declaring files up front. This request asks the model to save a PNG chart and tells mistral.rs to surface it as a typed file:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Find recent population figures for Tokyo and Japan, calculate the population share for Tokyo relative to Japan, and save a bar chart as tokyo-population.png. Cite sources and show the calculation."
      }
    ],
    "web_search_options": {},
    "enable_code_execution": true,
    "max_tool_rounds": 6,
    "session_id": "tokyo-demo",
    "files": [
      {
        "name": "tokyo-population.png",
        "format": "png",
        "description": "Bar chart comparing Tokyo and Japan population"
      }
    ]
  }'
```

The response body keeps the normal OpenAI-compatible `choices` array and adds mistral.rs fields for tool work, files, and session state:

```json
{
  "id": "...",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Tokyo is about ... of Japan's population. Sources: ..."
      },
      "finish_reason": "stop"
    }
  ],
  "agentic_tool_calls": [
    {
      "round": 0,
      "name": "web_search",
      "arguments": "{\"query\":\"Tokyo population Japan population\"}",
      "result_content": "..."
    },
    {
      "round": 1,
      "name": "mistralrs_execute_python",
      "arguments": "{\"code\":\"...\"}",
      "result_content": "Tokyo share: ...",
      "file_ids": ["file_tokyo_r1_0"]
    }
  ],
  "files": [
    {
      "id": "file_tokyo_r1_0",
      "name": "tokyo-population.png",
      "format": "png",
      "mime_type": "image/png",
      "bytes": 14823,
      "data_base64": "iVBORw0KGgo..."
    }
  ],
  "session_id": "tokyo-demo",
  "usage": {"prompt_tokens": 148, "completion_tokens": 82, "total_tokens": 230}
}
```

`agentic_tool_calls` records the work the server did on behalf of the model. `files` contains structured outputs produced by tools. Small files are returned inline; larger files can be fetched through the files API.

## Streaming progress

For streaming requests, normal model text arrives as OpenAI-compatible chunks. Tool progress arrives as named Server-Sent Events:

```text
event: agentic_tool_call_progress
data: {"type":"agentic_tool_call_progress","round":0,"tool_name":"web_search","phase":"calling","data":{"tool_type":"web_search","query":"Tokyo population Japan population"}}
```

When code execution produces a declared file, the stream emits it immediately:

```text
event: file_produced
data: {"id":"file_tokyo_r1_0","name":"tokyo-population.png","format":"png","mime_type":"image/png","bytes":14823}
```

The full schema is in the [HTTP API reference](/mistral.rs/reference/http-api/) and the [agentic runtime guide](/mistral.rs/guides/agents/agentic-runtime/).

Minimal Python client that calls the HTTP server and prints the tool trace:

```python
import requests

prompt = (
    "Find recent population figures for Tokyo and Japan, calculate the population "
    "share for Tokyo relative to Japan, and save a bar chart as tokyo-population.png. "
    "Cite sources and show the calculation."
)

r = requests.post("http://localhost:1234/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": prompt}],
    "web_search_options": {},
    "enable_code_execution": True,
    "max_tool_rounds": 6,
    "session_id": "tokyo-demo",
    "files": [{"name": "tokyo-population.png", "format": "png"}],
}).json()

for call in r.get("agentic_tool_calls", []):
    print(f"round {call['round']}: {call['name']}({call['arguments']})")
    print(f"  -> {call['result_content'].strip()}")

print(r["choices"][0]["message"]["content"])
for f in r.get("files", []):
    print(f"file: {f['name']} ({f['bytes']} bytes)")
```

## From the Python SDK

`Runner` enables both built-in tools in-process. Web search is enabled on the runner; code execution also requires a `CodeExecutionConfig`. Per-request, set `web_search_options`, `enable_code_execution`, and any required files on the `ChatCompletionRequest`.

```python
from mistralrs import (
    ChatCompletionRequest,
    CodeExecutionConfig,
    RequestedFile,
    Runner,
    WebSearchOptions,
    Which,
)

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
    enable_search=True,
    code_execution_config=CodeExecutionConfig(),  # defaults: python3, 30 s timeout
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{
            "role": "user",
            "content": "Find recent population figures for Tokyo and Japan, calculate Tokyo's share of Japan's population, and save a bar chart as tokyo-population.png. Cite sources and show the calculation.",
        }],
        max_tokens=512,
        web_search_options=WebSearchOptions(),
        enable_code_execution=True,
        max_tool_rounds=6,
        session_id="tokyo-demo",
        files=[RequestedFile("tokyo-population.png", "png")],
    )
)
print(response.choices[0].message.content)
for file in response.files or []:
    file.save(file.name)
    print(f"saved {file.name} ({file.bytes} bytes)")
```

`CodeExecutionConfig` accepts `python_path`, `timeout_secs`, and `working_directory`. See [`CodeExecutionConfig`](/mistral.rs/reference/python/code-execution/).

For custom tools, pass `tool_callbacks={name: callable}` to `Runner`; each callable receives the tool name and a dict of arguments and returns a string. See [`Runner`](/mistral.rs/reference/python/runner/).

## From the Rust SDK

The Rust SDK supports the same tools in-process. Enable search and code execution at load time, then opt into them on the request:

```rust
use anyhow::Result;
use futures::StreamExt;
use mistralrs::{
    CodeExecutionConfig, IsqBits, RequestBuilder, SearchEmbeddingModel, TextMessageRole,
    TextMessages, TextModelBuilder, WebSearchOptions,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_search(SearchEmbeddingModel::default())
        .with_code_execution(CodeExecutionConfig::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Find recent population figures for Tokyo and Japan, calculate Tokyo's share of Japan's population, and save a bar chart as tokyo-population.png.",
    );

    let request = RequestBuilder::from(messages)
        .with_web_search_options(WebSearchOptions::default())
        .with_code_execution()
        .with_session_id("tokyo-demo")
        .set_max_tool_rounds(6)
        .require_file("tokyo-population.png");

    let mut stream = model.stream_chat_request(request).await?;
    while let Some(event) = stream.next().await {
        match event {
            mistralrs::Response::Chunk(chunk) => {
                if let Some(text) = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.as_deref())
                {
                    print!("{text}");
                }
            }
            mistralrs::Response::AgenticToolCallProgress { tool_name, .. } => {
                eprintln!("tool progress: {tool_name}");
            }
            _ => {}
        }
    }

    Ok(())
}
```

`CodeExecutionConfig::default()` uses `python3` (or `python` on Windows) with a 30 s per-call timeout. Override via `CodeExecutionConfig { python_path, timeout_secs, working_directory }`.

Per-request control is on [`RequestBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.RequestBuilder.html): `.with_code_execution()`, `.set_max_tool_rounds(...)`, `.with_session_id(...)`, `.with_web_search_options(...)`. Use `stream_chat_request` to observe `Response::AgenticToolCallProgress` events.

## Structured outputs

When code execution produces files (plots, CSVs, JSON), the runtime surfaces them as typed `File` objects on the response. Declare required outputs on the request and the runtime tells the model what to write, then collects whatever appears in the working directory.

HTTP:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Plot sin(x) for x in [0, 2pi] and save as plot.png."}
    ],
    "enable_code_execution": true,
    "files": [{"name": "plot.png", "format": "png"}]
  }'
```

The response gains a top-level `files` array; each entry has `id`, `name`, `mime_type`, `bytes`, and either inline `data_base64` / `text` or a `url` to fetch.

Python SDK:

```python
from mistralrs import ChatCompletionRequest, RequestedFile

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Plot sin(x) and save as plot.png."}],
        enable_code_execution=True,
        files=[RequestedFile("plot.png", "png")],
    )
)
for f in response.files or []:
    f.save(f.name)
    print(f"saved {f.name} ({f.bytes} bytes)")
```

Rust SDK:

```rust
let request = mistralrs::RequestBuilder::from(messages)
    .with_code_execution()
    .require_file("plot.png");

let response = model.send_chat_request(request).await?;
for f in response.files.as_deref().unwrap_or_default() {
    f.save(&f.name)?;
    println!("saved {} ({} bytes)", f.name, f.bytes);
}
```

Full schema, size policy, the `read_file` / `list_files` model tools, and the streaming `file_produced` event are documented in [agentic runtime: files](/mistral.rs/guides/agents/agentic-runtime/#files).

## Sessions

Use `session_id` when you want later requests to continue the same agent state. Sessions preserve message history, tool-call records, file references, and the Python subprocess used for code execution.

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "session_id": "tokyo-demo",
    "messages": [
      {"role": "user", "content": "Using the same analysis, explain the chart in one paragraph."}
    ],
    "enable_code_execution": true
  }'
```

If no `session_id` is passed, the server creates one and returns it in the response. See the [persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/) for export, import, deletion, TTLs, and SDK methods.

## Notes

Enabling the flags does not force tool use. The model is given the tools and their descriptions and decides when to call them.

Code execution runs in a subprocess as the same OS user as mistral.rs. It is not a sandbox. For untrusted users, run mistral.rs in a container or VM, use a low-privilege user, and constrain network access.

The two flags above enable the built-in tools only. To expose custom tools (calendar API, vector search, shell), implement them as MCP servers and connect mistral.rs as a client, or register tool callbacks through the Rust or Python SDK. See the [agent guides](/mistral.rs/guides/agents/).

## Next steps

- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/): fit larger models on the available GPU.
- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/): consume the runtime event stream in an application.
- [The MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/): connect to a third-party MCP server.
- [The persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/): keep state across separate requests.
