---
title: Build an agent
description: Turn on tool calling, web search, and code execution so the model can take actions. Use the same agent from the web UI, HTTP, Python, and Rust. About fifteen minutes.
sidebar:
  order: 5
---

The agentic loop lets the server handle tool calls inside a single request: the model requests a tool, the server runs it, feeds the result back, and continues until the model produces a normal reply. Two built-in tools are covered here: web search and Python code execution. The model is Qwen3-4B.

## Enabling the tools

Both features are off by default. Each carries a cost: network access for search, a Python subprocess for code execution.

```bash
mistralrs serve --ui \
  --enable-search \
  --enable-code-execution \
  -m Qwen/Qwen3-4B
```

`--enable-search` enables the built-in web search tool. `--enable-code-execution` enables a Python subprocess that persists across calls within a session.

Open `http://localhost:1234/ui` once the server is ready.

## From the web UI

Paste into the chat box:

```
What is the population of Tokyo, and what fraction of Japan's total population does that represent? Show your working.
```

The reply takes longer than a normal chat response because the loop runs multiple rounds. The UI renders, in order:

1. A collapsed search block with the query, retrieved URLs, and snippets.
2. A code execution block with the Python the model ran and its stdout.
3. Further rounds for follow-up searches or calculations, when the model requests them.
4. A final reply citing the numbers and showing the arithmetic.

Everything between the question and the final reply happens inside a single HTTP request. The UI renders structured events the server emits as part of the response. The same events are available to any client.

## From HTTP

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the square root of 99856? Verify by squaring the result."}
    ],
    "enable_code_execution": true,
    "max_tool_rounds": 4
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
      "name": "mistralrs_execute_python",
      "arguments": "{\"code\":\"import math\\nprint(math.sqrt(99856))\\nprint(316**2)\"}",
      "result_content": "316.0\n99856\n",
      "result_images_base64": []
    }
  ],
  "usage": {"prompt_tokens": 148, "completion_tokens": 82, "total_tokens": 230}
}
```

`choices` is OpenAI-compatible. `agentic_tool_calls` is a mistral.rs extension. Each entry records one round: tool name, arguments, and result. Tool-produced images (e.g., matplotlib plots) appear as base64 strings in `result_images_base64`.

For streaming, the loop emits Server-Sent Events with type `agentic_tool_call_progress`. Each event has a `phase` of `"calling"` or `"complete"`. The full schema is in the [HTTP API reference](/mistral.rs/reference/http-api/).

Minimal Python client that prints the tool trace:

```python
import requests

r = requests.post("http://localhost:1234/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Factorial of 37?"}],
    "enable_code_execution": True,
    "max_tool_rounds": 4,
}).json()

for call in r.get("agentic_tool_calls", []):
    print(f"round {call['round']}: {call['name']}({call['arguments']})")
    print(f"  -> {call['result_content'].strip()}")

print(r["choices"][0]["message"]["content"])
```

## From the Python SDK

`Runner` enables both built-in tools in-process. Web search with `enable_search=True`; code execution with a `CodeExecutionConfig`. Per-request, set `enable_code_execution=True` on the `ChatCompletionRequest`.

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
    enable_search=True,
    code_execution_config=CodeExecutionConfig(),  # defaults: python3, 30 s timeout
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Factorial of 37?"}],
        max_tokens=512,
        enable_code_execution=True,
    )
)
print(response.choices[0].message.content)
```

`CodeExecutionConfig` accepts `python_path`, `timeout_secs`, and `working_directory`. See [`CodeExecutionConfig`](/mistral.rs/reference/python/code-execution/).

For custom tools, pass `tool_callbacks={name: callable}` to `Runner`; each callable receives the tool name and a dict of arguments and returns a string. See [`Runner`](/mistral.rs/reference/python/runner/).

## From the Rust SDK

The Rust SDK supports both tools in-process. Enable them at load time:

```rust
use anyhow::Result;
use mistralrs::{
    CodeExecutionConfig, IsqBits, SearchEmbeddingModel, TextMessageRole, TextMessages,
    TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_search(SearchEmbeddingModel::default())
        .with_code_execution(CodeExecutionConfig::default())
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Factorial of 37?");

    let request = mistralrs::RequestBuilder::from(messages)
        .with_code_execution()
        .set_max_tool_rounds(4);

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

## Collecting structured outputs

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
for f in &response.files {
    f.save(&f.name)?;
    println!("saved {} ({} bytes)", f.name, f.bytes);
}
```

Full schema, size policy, the `read_file` / `list_files` model tools, and the streaming `file_produced` event are documented in [agentic runtime: files](/mistral.rs/guides/agents/agentic-runtime/#files).

## Notes

Enabling the flags does not force tool use. The model is given the tools and their descriptions and decides when to call them.

Code execution is stateful within a session. Subsequent requests reusing the same session id share the Python subprocess, so prior variables remain available. If no session id is passed, one is created and returned in the response. See the [persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/).

The two flags above enable the built-in tools only. To expose custom tools (calendar API, vector search, shell), implement them as MCP servers and connect mistral.rs as a client, or register tool callbacks through the Rust or Python SDK. See the [agent guides](/mistral.rs/guides/agents/).

## Next steps

- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/): fit larger models on the available GPU.
- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/): consume the runtime event stream in an application.
- [The MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/): connect to a third-party MCP server.
- [The persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/): keep state across separate requests.
