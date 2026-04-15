# Code Execution

mistral.rs supports Python code execution as a built-in tool. When enabled, the model is given two tools:
- **`execute_python`** -- Execute Python code in a persistent session
- **`reset_python_session`** -- Reset the session, clearing all variables and imports

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

Code execution is enabled server-side. Once enabled, the `execute_python` and `reset_python_session` tools are automatically available to the model. No additional API parameters are needed -- the model decides when to use code execution based on the conversation.

## Session Management

### Persistence

Within a single request (which may involve multiple tool calls via the agentic loop), the Python session is persistent:

```
User: "Calculate the first 10 Fibonacci numbers and then sum them."
  -> Model calls execute_python: "fibs = [0, 1]\nfor i in range(8): fibs.append(fibs[-1] + fibs[-2])\nfibs"
  <- Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  -> Model calls execute_python: "sum(fibs)"  # `fibs` is still available!
  <- Result: 88
```

### Reset

The model can call `reset_python_session` to clear all variables and imports while keeping the session alive.

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
