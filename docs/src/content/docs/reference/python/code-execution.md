---
title: Code and shell execution
description: "Configuration for the built-in Python and shell executors."
sidebar:
  order: 9
---
## `SandboxPolicy`

Limits on the processes that the Python and shell tools start, on Linux and macOS.

Pass one as `CodeExecutionConfig(sandbox_policy=...)` or `ShellConfig(sandbox_policy=...)`. Without
one, those processes run with no sandbox. The sandbox controls:

- which files they can read and write
- which network they can reach
- which environment variables they see
- how much memory, CPU time, and how many processes they can use

macOS applies the file and network rules but not the resource limits. The
[sandbox reference](/reference/sandbox/) explains how each rule is enforced.

The defaults match the CLI's `restricted` profile.

### `SandboxPolicy.__init__`

```text
__init__(
    max_memory_mb: int = 2048,
    max_cpu_secs: int = 600,
    max_procs: int = 64,
    max_open_fds: int = 1024,
    max_file_sz_mb: int = 256,
    network: NetworkMode = NetworkMode.Loopback,
    extra_fs_read: list[str] = [],
    extra_fs_write: list[str] = [],
    extra_env: list[str] = [],
    strict: bool = False,
) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `max_memory_mb` | `int` | `2048` | Memory (address space) limit, in MB. |
| `max_cpu_secs` | `int` | `600` | CPU time limit, in seconds. On Linux, a limit shorter than the tool's timeout is raised to match it. |
| `max_procs` | `int` | `64` | Limit on processes and threads. |
| `max_open_fds` | `int` | `1024` | Limit on open file descriptors. |
| `max_file_sz_mb` | `int` | `256` | Largest file a process can write, in MB. |
| `network` | `NetworkMode` | `NetworkMode.Loopback` | Network access. See `NetworkMode`. |
| `extra_fs_read` | `list[str]` | `[]` | More paths the process can read. |
| `extra_fs_write` | `list[str]` | `[]` | More paths the process can read and write. |
| `extra_env` | `list[str]` | `[]` | More environment variable names to pass through. Their values come from the current environment. |
| `strict` | `bool` | `False` | Fail at startup if a requested file or network restriction isn't available, instead of skipping it. |


## `CodeExecutionConfig`

Turns on the built-in Python tool, `execute_python`.

Pass it as `Runner(code_execution_config=...)`, then set
`ChatCompletionRequest.enable_code_execution=True` on each request that may run code.

### `CodeExecutionConfig.__init__`

```text
__init__(
    python_path: str | None = None,
    timeout_secs: int | None = None,
    working_directory: str | None = None,
    sandbox_policy: SandboxPolicy | None = None,
    permission: CodeExecutionPermission | None = None,
    approval_callback: Callable[[dict[str, object]], bool] | None = None,
) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `python_path` | `str \| None` | `None` | Interpreter to run. Defaults to `python` on Windows and `python3` elsewhere. |
| `timeout_secs` | `int \| None` | `None` | Time limit per call, in seconds. Defaults to 60. |
| `working_directory` | `str \| None` | `None` | One directory that every session runs in. Defaults to a new temporary directory per session. |
| `sandbox_policy` | `SandboxPolicy \| None` | `None` | Sandbox for the interpreter. `None` runs it without one. |
| `permission` | `CodeExecutionPermission \| None` | `None` | Older form of `ChatCompletionRequest.agent_permission`, which you should prefer. Defaults to `Auto`. |
| `approval_callback` | `Callable[[dict[str, object]], bool] \| None` | `None` | Older, code-only approval callback. Prefer `ChatCompletionRequest.agent_approval_callback`. It receives a dict with `approval_id`, `session_id`, `code`, `outputs`, and `working_directory`, and returns `True` to run the code. An exception counts as a denial. |


## `ShellConfig`

Turns on the built-in shell tool.

Pass it as `Runner(shell_config=...)`, then set `ChatCompletionRequest.enable_shell=True` or pass
`shell_skills` on each request that may run commands.

### `ShellConfig.__init__`

```text
__init__(
    shell_path: str | None = None,
    timeout_secs: int | None = None,
    working_directory: str | None = None,
    sandbox_policy: SandboxPolicy | None = None,
    permission: AgentPermission | None = None,
) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `shell_path` | `str \| None` | `None` | Shell to run. Defaults to `cmd` on Windows and `/bin/sh` elsewhere. |
| `timeout_secs` | `int \| None` | `None` | Time limit per call, in seconds. Defaults to 600. |
| `working_directory` | `str \| None` | `None` | Parent directory for sessions. Each session runs in its own `<working_directory>/<session_id>`. Defaults to a new temporary directory per session. |
| `sandbox_policy` | `SandboxPolicy \| None` | `None` | Sandbox for the shell. `None` runs it without one. |
| `permission` | `AgentPermission \| None` | `None` | Default permission for shell commands. Defaults to `Auto`. For per-request control, use `ChatCompletionRequest.agent_permission`. |


## `ShellSkillMount`

A local skill directory to mount for the shell tool, in the OpenAI Skill layout (a folder with a `SKILL.md`).

Pass instances in `ChatCompletionRequest.shell_skills`. With the HTTP server, upload skills through
`/v1/skills` instead. Each skill appears at `skills/<name>` in the shell's working directory.

### `ShellSkillMount.__init__`

```text
__init__(name: str, description: str, source_path: str) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Skill name, shown to the model. |
| `description` | `str` | required | What the skill does, shown to the model. |
| `source_path` | `str` | required | Local directory that holds the skill. |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
