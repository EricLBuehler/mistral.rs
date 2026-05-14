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

**Security warning:** code execution allows the model to run arbitrary
Python on the host with full network and filesystem access. Only enable
in trusted contexts or inside a sandbox.

### `CodeExecutionConfig.__init__`

```text
__init__(
    python_path: str | None = None,
    timeout_secs: int | None = None,
    working_directory: str | None = None,
) -> None
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
