---
title: Enable code execution
description: Turn on the Python executor.
sidebar:
  order: 3
---

`--enable-code-execution` registers a code execution tool with the model. The tool runs Python in an isolated subprocess.

The code execution and file helper tools use [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) by default, so generated arguments are constrained to the declared JSON Schema before the tool runs.

## Turning it on

```bash
mistralrs serve --enable-code-execution -m <model>
```

The `code-execution` Cargo feature is in the default feature set. Binaries built with `--no-default-features` need it added explicitly.

HTTP requests must also opt into the tool:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": "Use Python to calculate and plot the first 20 primes."
    }
  ],
  "enable_code_execution": true,
  "max_tool_rounds": 4
}
```

## Declaring outputs

Apps can declare required output files on the request. The runtime tells the model about them and surfaces any matching files written into the working directory as first-class `File` objects on the response (or as `file_produced` SSE events when streaming). Missing files surface as error placeholders so the app always knows what came back.

HTTP:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Plot a sine wave and save as plot.png."}
  ],
  "enable_code_execution": true,
  "files": [{"name": "plot.png"}]
}
```

The response gains a top-level `files` array:

```json
{
  "files": [
    {
      "id": "file_abc_r0_0",
      "name": "plot.png",
      "mime_type": "image/png",
      "bytes": 14823,
      "data_base64": "iVBORw0KGgo..."
    }
  ]
}
```

Python SDK:

```python
from mistralrs import ChatCompletionRequest, CodeExecutionConfig, RequestedFile, Runner, Which

runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B"), code_execution_config=CodeExecutionConfig())
resp = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Plot sin(x) as plot.png."}],
        enable_code_execution=True,
        files=[RequestedFile("plot.png", "png")],
    )
)
for f in resp.files or []:
    f.save(f.name)
```

Rust SDK:

```rust
let req = mistralrs::RequestBuilder::from(messages)
    .with_code_execution()
    .require_file("plot.png");

let resp = model.send_chat_request(req).await?;
for f in &resp.files {
    f.save(&f.name)?;
}
```

The `mistralrs_execute_python` tool also accepts an `outputs` parameter so the model can list files it wrote that were not declared on the request. The runtime always surfaces files declared in `request.files`, regardless of whether the model lists them.

For full schema, size limits, and the `read_file` / `list_files` model tools, see [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/#files).

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `--code-exec-python <path>` | `python` on Windows, `python3` elsewhere | Python interpreter. |
| `--code-exec-timeout <secs>` | 30 | Per-call timeout in seconds. |
| `--code-exec-workdir <path>` | per-session temp dir | Working directory. |

## Sessions and state

Each session gets its own Python subprocess on first call. Subsequent calls reuse it; variables and imports persist within the session. Without a session id, each request gets a fresh interpreter.

Subprocesses idle for more than 1 hour are reaped. The reaper runs every 5 minutes.

## stdin / stdout / stderr

stdout and stderr from user code are captured and returned in the tool result. stdin reads raise `EOFError`.

When streaming chat completions, code execution progress is emitted as `agentic_tool_call_progress` SSE events. Complete events can include `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `working_directory`, and `execution_time_ms`.

## Working directory

Without `--code-exec-workdir`, each session gets a unique `mistralrs-code-<random>` temp directory.

With `--code-exec-workdir /path`, all sessions share the directory.

## Isolation

The subprocess runs as the same user as mistral.rs. It is not a sandbox. For untrusted users, run mistral.rs in a container, with a dedicated low-privilege user, and constrain network egress.

## Implementation

The executor lives in the `mistralrs-code-exec` crate. The Python side is `mistralrs-code-exec/python/executor.py`.

## See also

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).
- [Code execution design](/mistral.rs/explanation/code-execution-design/).
