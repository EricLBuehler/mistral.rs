---
title: Enable code execution
description: Turn on the Python executor.
sidebar:
  order: 3
---

`--enable-code-execution` registers a code execution tool with the model. The tool runs Python in a subprocess; on Linux and macOS it is wrapped in an [OS-level sandbox](/mistral.rs/reference/sandbox/) by default (`--sandbox auto`).

The code execution and file helper tools use [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) by default, so generated arguments are constrained to the declared JSON Schema before the tool runs.

## Turning it on

```bash
mistralrs serve --enable-code-execution -m <model>
```

The `code-execution` Cargo feature is in the default feature set. Binaries built with `--no-default-features` need it added explicitly.

Server startup makes the tool available. HTTP requests opt into it per request with `enable_code_execution: true`; without that field, the request stays a plain chat request even when the server has code execution enabled. The web UI sends the field when its Code Execution toggle is on.

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

## Programmatic use and examples

- Server and HTTP: [build an agent](/mistral.rs/tutorials/05-build-an-agent/#from-http) shows the request shape, streaming events, declared files, and sessions.
- Server runtime details: [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/) documents request fields, SSE progress events, and file output behavior.
- Web UI: [use the built-in web UI](/mistral.rs/guides/serve/with-web-ui/) covers the browser interface and how tool results render.
- Python: see the [Python code execution example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/code_execution.py), the [Python SDK flow](/mistral.rs/tutorials/05-build-an-agent/#from-the-python-sdk), and the [Python API reference](/mistral.rs/reference/python/code-execution/).
- Rust: see the [Rust code execution example](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/code_execution/main.rs), the [Rust file-output example](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/code_execution_files/main.rs), and the [Rust SDK flow](/mistral.rs/tutorials/05-build-an-agent/#from-the-rust-sdk).
- CLI config: [cli-config.toml](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/cli-config.toml) includes the same runtime and sandbox settings in file form.

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
| `--code-exec-workdir <path>` | per-session temp dir | Working directory for Python and produced files. |
| `--agent-permission <mode>` | `auto` | `auto`, `ask`, or `deny`. Controls whether model-requested agent actions run automatically, require approval, or are denied. |
| `--sandbox <mode>` | `auto` | OS-level sandbox: `auto`, `on`, `off`. See [sandbox reference](/mistral.rs/reference/sandbox/) for the full set of sandbox knobs. |

`--agent-permission` is separate from the sandbox. Permission mode decides whether the runtime may execute a model-requested action. The sandbox decides what Python can access after it starts. For the centralized permission model, each API surface, and approval examples, see [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions).

## Sessions and state

Each session gets its own Python subprocess on first call. Subsequent calls reuse it; variables and imports persist within the session. Without a session id, each request gets a fresh interpreter.

Subprocesses idle for more than 1 hour are reaped. The reaper runs every 5 minutes.

## stdin / stdout / stderr

stdout and stderr from user code are captured and returned in the tool result. stdin reads raise `EOFError`.

When streaming chat completions, code execution progress is emitted as `agentic_tool_call_progress` SSE events. Complete events can include `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `working_directory`, and `execution_time_ms`.

## Working directory

Without `--code-exec-workdir`, each session gets a unique `mistralrs-code-<random>` temp directory.

With `--code-exec-workdir /path`, all sessions share the directory.

`--code-exec-workdir` chooses where Python starts and where output files are collected. It does not turn the sandbox on or off. When the sandbox is enabled, this directory is included as the writable working directory.

## Isolation

On Linux and macOS the subprocess is wrapped in an OS-level sandbox by default (`--sandbox auto`). Layers include env scrubbing, namespace isolation, Landlock FS allowlist, `setrlimit`-based caps, a seccomp deny-list, and optional cgroup v2 limits on Linux; macOS uses Seatbelt and env scrubbing, without rlimit caps. The threat model is **model misbehavior**. For higher-assurance deployments, also run mistral.rs in a container or VM with a dedicated low-privilege user and constrained network egress.

Example with explicit sandbox settings:

```bash
mistralrs serve \
  -m mistralrs-community/gemma-4-E4B-it-UQFF \
  --from-uqff 8 \
  --enable-code-execution \
  --sandbox on \
  --sandbox-network none \
  --sb-max-memory-mb 2048 \
  --code-exec-workdir . \
  --enable-search
```

`--sandbox on` makes missing sandbox support a hard error. `--sandbox-network none` blocks network access from model-generated Python; web search still runs through the server-side search tool. `--code-exec-workdir .` keeps produced files in the directory where the server was started.

Resource limit flags such as `--sb-max-memory-mb` are enforced on Linux. On macOS, the same command still applies Seatbelt filesystem and network isolation, but hard memory, CPU, and process-count caps require an outer container or VM.

See the full [sandbox reference](/mistral.rs/reference/sandbox/) for what each layer does, how to tune the limits, and how to disable it (`--sandbox off`).

Programmatic sandbox configuration:

```rust
use mistralrs::{CodeExecutionConfig, NetworkMode, SandboxPolicy};

let cfg = CodeExecutionConfig {
    sandbox_policy: Some(SandboxPolicy {
        max_memory_mb: 1024,
        network: NetworkMode::None,
        ..SandboxPolicy::default()
    }),
    ..CodeExecutionConfig::default()
};
```

```python
from mistralrs import CodeExecutionConfig, NetworkMode, Runner, SandboxPolicy, Which

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    code_execution_config=CodeExecutionConfig(
        sandbox_policy=SandboxPolicy(
            max_memory_mb=1024,
            network=NetworkMode.NoNetwork,
        ),
    ),
)
```

Omit `sandbox_policy` (or pass `None`) to disable the sandbox entirely.

## See also

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).
- [Code execution design](/mistral.rs/explanation/code-execution-design/).
