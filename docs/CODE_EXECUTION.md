# Code Execution

mistral.rs supports Python code execution as a built-in tool. When enabled, the model is given two tools:
- **`mistralrs_execute_python`** -- Execute Python code in a persistent session
- **`mistralrs_reset_python_session`** -- Reset the session, clearing all variables and imports

Internal tools are prefixed with `mistralrs_` to prevent name collisions with user-provided tools.

This feature is similar to code interpreters in ChatGPT and other AI assistants. The model can write and run Python code to answer questions, perform calculations, create visualizations, and more.

## Features

- **Persistent sessions**: Variables, imports, and state are preserved across multiple tool calls within a single request (like Jupyter notebooks)
- **Last-expression capture**: If the final statement in a code block is an expression, its value is automatically returned (Jupyter-style)
- **Rich outputs**:
  - **Matplotlib figures**: Automatically captured as PNG images
  - **PIL Images**: Detected and captured when they are the last expression
  - **Pandas DataFrames/Series**: Formatted repr returned
- **Multimodal feedback**: For vision-capable models, generated images are returned as actual visual content the model can see. For text-only models, a note is appended instead.
- **Timeouts**: Configurable execution timeout with graceful interrupt (SIGINT) before forceful kill (SIGKILL)
- **Isolated working directory**: Each request gets a unique temporary directory

## Configuration

### CLI

```bash
mistralrs serve \
  --enable-code-execution \
  --code-exec-python /path/to/python3 \
  --code-exec-timeout 60 \
  -m <model_id>
```

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-code-execution` | Enable the code execution tools | Off |
| `--code-exec-python <PATH>` | Path to Python interpreter | `python3` |
| `--code-exec-timeout <SECS>` | Execution timeout in seconds | `30` |

### Rust SDK

```rust
use mistralrs::{ModelBuilder, CodeExecutionConfig};

let model = ModelBuilder::new("model-id")
    .with_code_execution(CodeExecutionConfig {
        python_path: "python3".into(),
        timeout_secs: 30,
    })
    .build()
    .await;
```

### HTTP API

Code execution must be enabled in two places:

1. Start the server with `--enable-code-execution` so the Python tools are registered.
2. Set `"enable_code_execution": true` on each chat request that may use the tools.

Once both are enabled, the `mistralrs_execute_python` and `mistralrs_reset_python_session` tools are automatically available to the model. The model decides when to use code execution based on the conversation.

### Web UI

When using the built-in web UI (`--ui`), code execution results are displayed inline with syntax-highlighted Python code blocks, stdout output, stderr/exceptions in red, and matplotlib/PIL images rendered directly in the chat. Enable code execution via the settings panel or the `--enable-code-execution` CLI flag.

## Session Management

### Persistence

Within a single request (which may involve multiple tool calls via the agentic loop), the Python session is persistent:

```
User: "Calculate the first 10 Fibonacci numbers and then sum them."
  -> Model calls mistralrs_execute_python: "fibs = [0, 1]\nfor i in range(8): fibs.append(fibs[-1] + fibs[-2])\nfibs"
  <- Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  -> Model calls mistralrs_execute_python: "sum(fibs)"  # `fibs` is still available!
  <- Result: 88
```

### Reset

The model can call `mistralrs_reset_python_session` to clear all variables and imports while keeping the session alive.

### Timeouts

When code execution exceeds the configured timeout:
1. A `SIGINT` (KeyboardInterrupt) is sent to the Python process
2. The system waits 3 seconds for graceful handling
3. If still unresponsive, a `SIGKILL` is sent (forceful termination)

After a `SIGKILL`, the session is respawned on the next tool call. The model is informed about the full timeout details including whether the interrupt was graceful or forced.

## Installed Packages

When code execution is initialized, `pip list` is run to capture all installed packages. This list is included in the tool description so the model knows what libraries are available.

Package installation (`pip install`) is disabled. To make additional packages available, install them in the Python environment before starting the server.

### Using a Virtual Environment

To use a specific virtual environment, point `--code-exec-python` to its interpreter:

```bash
mistralrs serve \
  --enable-code-execution \
  --code-exec-python /path/to/venv/bin/python \
  -m <model_id>
```

## Cargo Feature

Code execution is gated behind the `code-execution` Cargo feature. It is **enabled by default** for the CLI (`mistralrs-cli`).

```bash
# The CLI includes code-execution by default:
cargo build --release

# Build with CUDA and code execution (already included):
cargo build --release --features cuda
```

For other crates (`mistralrs`, `mistralrs-server-core`, `mistralrs-pyo3`), the feature must be explicitly enabled:

```bash
cargo build --release -p mistralrs --features code-execution
```

## Tool Call Progress Events

When tools are called during the agentic loop, mistral.rs emits structured progress events. These allow clients to display tool activity, captured stdout/stderr, generated images, generated video frames, and session state in real time.

### Streaming (SSE)

For streaming chat completions, progress events are sent as SSE events with `event: agentic_tool_call_progress`:

```json
{"type":"agentic_tool_call_progress","round":0,"tool_name":"mistralrs_execute_python","phase":"calling","data":{"tool_type":"code_execution","code":"print('hello')"}}
{"type":"agentic_tool_call_progress","round":0,"tool_name":"mistralrs_execute_python","phase":"complete","data":{"tool_type":"code_execution","stdout":"hello\n","execution_time_ms":42}}
```

When code produces media, the completion event includes base64 image and video-frame fields:

```json
{
  "type": "agentic_tool_call_progress",
  "round": 0,
  "tool_name": "mistralrs_execute_python",
  "phase": "complete",
  "data": {
    "tool_type": "code_execution",
    "images_base64": ["..."],
    "video_frames_base64": ["..."],
    "video_frame_count": 12
  }
}
```

### Non-Streaming

For non-streaming chat completions, the final `ChatCompletionResponse` includes an `agentic_tool_calls` field with the full history:

```json
{
  "choices": [...],
  "agentic_tool_calls": [
    {
      "round": 0,
      "name": "mistralrs_execute_python",
      "arguments": "{\"code\":\"print('hello')\"}",
      "result_content": "stdout: hello",
      "result_images_base64": []
    }
  ]
}
```

### Tool-Specific Data

Progress events carry typed data depending on the tool:

- **`code_execution`**: `code`, `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `video_frame_count`, `working_directory`, `execution_time_ms`
- **`web_search`**: `query`, `results_count`
- **`custom`**: `arguments`, `content` (opaque strings)

### Tool Name Conflicts

Internal tools are prefixed with `mistralrs_` (e.g., `mistralrs_execute_python`). If a user-provided tool has the same name as a registered internal tool, the request will be rejected with a validation error.

## Security Considerations

**Code execution allows the model to run arbitrary Python code on your machine.** This includes:

- **Full filesystem access**: The model can read and write any files accessible to the Python process
- **Full network access**: The model can make HTTP requests, open sockets, etc.
- **System command execution**: Via `os.system()`, `subprocess`, etc.

### Recommendations

1. **Do not enable in production** without proper sandboxing (containers, VMs, etc.)
2. **Use a dedicated Python environment** with minimal packages installed
3. **Run with least privilege** -- use a restricted user account
4. **Monitor execution** -- the security warning in the logs is there for a reason
5. **Consider network isolation** if the machine has access to internal services

### Security Warning

When code execution is enabled, a prominent warning is displayed in the logs:

```
WARN  CODE EXECUTION IS ENABLED
WARN  The model can execute ARBITRARY Python code on this machine.
WARN  Network access and filesystem access are NOT restricted.
WARN  Only enable this in trusted environments.
```
