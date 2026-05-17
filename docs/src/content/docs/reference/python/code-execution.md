---
title: Code execution
description: "Configuration for the built-in Python code executor."
sidebar:
  order: 9
---
## `SandboxPolicy`

OS-level sandbox applied to the code-execution subprocess on Linux/macOS.

Pass to `CodeExecutionConfig(sandbox_policy=...)` to enable the sandbox;
omit (or pass `None`) to disable it. See the sandbox reference for the
layered defenses: env scrub, namespaces, Landlock FS allowlist, rlimits,
seccomp deny-list, and optional cgroup v2 on Linux.

- `max_memory_mb`: per-session memory cap (default 2048).
- `max_cpu_secs`: per-session CPU time cap (default 300).
- `max_procs`: per-session process/thread cap (default 64).
- `max_open_fds`: per-session open-fd cap (default 1024).
- `max_file_sz_mb`: per-session max written-file size (default 256).
- `network`: `"none"`, `"loopback"`, or `"full"`. Default `"loopback"`.
- `extra_fs_read`: additional paths the sandboxed process may read.
- `extra_fs_write`: additional paths the sandboxed process may read/write.
- `extra_env`: additional environment variable names allowed through.
- `strict`: fail initialization if requested filesystem or network
  isolation is unavailable.

### `SandboxPolicy.__init__`

```text
__init__(
    max_memory_mb: int = 2048,
    max_cpu_secs: int = 300,
    max_procs: int = 64,
    max_open_fds: int = 1024,
    max_file_sz_mb: int = 256,
    network: str = 'loopback',
    extra_fs_read: list[str] = [],
    extra_fs_write: list[str] = [],
    extra_env: list[str] = [],
    strict: bool = False,
) -> None
```


## `CodeExecutionConfig`

Configuration for the built-in Python code execution tool.

Pass to `Runner(code_execution_config=...)` to enable the `execute_python`
tool. Per-request, set `ChatCompletionRequest.enable_code_execution=True`.

All fields are optional:

- `python_path`: interpreter to run. Defaults to `python` on Windows,
  `python3` elsewhere.
- `timeout_secs`: per-call timeout. Defaults to 30.
- `working_directory`: shared working directory. Defaults to a per-session
  temp directory.
- `sandbox_policy`: an OS-level sandbox to apply to the spawned interpreter
  on Linux/macOS. `None` (default) disables the sandbox; passing a
  `SandboxPolicy` enables it with the configured limits.

### `CodeExecutionConfig.__init__`

```text
__init__(
    python_path: str | None = None,
    timeout_secs: int | None = None,
    working_directory: str | None = None,
    sandbox_policy: SandboxPolicy | None = None,
) -> None
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
