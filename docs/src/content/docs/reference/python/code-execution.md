---
title: Code execution
description: "Configuration for the built-in Python code executor."
sidebar:
  order: 9
---
## `CodeExecutionConfig`

Configuration for the built-in Python code execution tool.

Pass to `Runner(code_execution_config=...)` to enable the `execute_python`
tool. Per-request, set `ChatCompletionRequest.enable_code_execution=True`.

All fields are optional; defaults match the CLI:

- `python_path`: interpreter to run. Defaults to `python` on Windows,
  `python3` elsewhere.
- `timeout_secs`: per-call timeout. Defaults to 30.
- `working_directory`: shared working directory. Defaults to a per-session
  temp directory.
- `sandbox_policy`: an OS-level sandbox to apply to the spawned interpreter
  on Linux and macOS. `None` (the default) disables the sandbox - the
  model has full filesystem, network, and subprocess access. Pass a
  `SandboxPolicy` to enable isolation. See the
  [sandbox reference](/mistral.rs/reference/sandbox/).

### `CodeExecutionConfig.__init__`

```text
__init__(
    python_path: str | None = None,
    timeout_secs: int | None = None,
    working_directory: str | None = None,
    sandbox_policy: SandboxPolicy | None = None,
) -> None
```

## `SandboxPolicy`

OS-level sandbox applied to the spawned Python interpreter. Available on
Linux (Landlock + seccomp + namespaces + rlimits) and macOS (Seatbelt +
rlimits). The threat model is model misbehavior, not kernel escape.

- `max_memory_mb`: per-session memory cap. Default 2048.
- `max_cpu_secs`: per-session CPU time cap. Default 300.
- `max_procs`: per-session process/thread cap. Default 64.
- `max_open_fds`: per-session open-fd cap. Default 1024.
- `max_file_sz_mb`: per-session max written-file size. Default 256.
- `network`: `"none"`, `"loopback"`, or `"full"`. Default `"loopback"`.
- `extra_fs_read`: additional paths the sandboxed process may read,
  appended to the built-in system allowlist.
- `extra_fs_write`: additional paths the sandboxed process may read and
  write, appended to the per-session workdir.
- `extra_env`: additional environment variable names allowed through the
  env scrub. The default allowlist excludes secrets like `HF_TOKEN` to
  prevent model-generated code from exfiltrating them.
- `strict`: when true, missing layers (no Landlock, no user namespaces)
  cause `Runner` construction to fail rather than silently degrade.

### `SandboxPolicy.__init__`

```text
__init__(
    max_memory_mb: int = 2048,
    max_cpu_secs: int = 300,
    max_procs: int = 64,
    max_open_fds: int = 1024,
    max_file_sz_mb: int = 256,
    network: str = "loopback",
    extra_fs_read: list[str] = [],
    extra_fs_write: list[str] = [],
    extra_env: list[str] = [],
    strict: bool = False,
) -> None
```

### Example

```python
from mistralrs import CodeExecutionConfig, Runner, SandboxPolicy, Which

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    code_execution_config=CodeExecutionConfig(
        sandbox_policy=SandboxPolicy(
            max_memory_mb=1024,
            network="none",
            extra_fs_read=["/opt/datasets"],
            extra_env=["AWS_REGION"],
        ),
    ),
)
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
