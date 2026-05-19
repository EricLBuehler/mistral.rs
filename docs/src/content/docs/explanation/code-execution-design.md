---
title: Code execution design
description: The subprocess model, stdio protocol, and session lifecycle of the Python executor.
sidebar:
  order: 9
---

The code-execution feature runs user-requested Python in a dedicated subprocess. The model never touches the mistralrs process directly.

## Session model

A session is created lazily on the first code-execution call for a given session id. Each session owns:

- One Python subprocess, started from `python` on Windows and `python3` elsewhere by default (`--code-exec-python`).
- A working directory: a fresh temp dir by default, or a shared directory if `--code-exec-workdir` is set.

Sessions idle for more than 1 hour are reaped (the reaper runs every 5 minutes). A reaped session's subprocess is killed; the next call against the session id starts a fresh one.

## Protocol

The engine sends one request per code-execution call to the subprocess and reads one response. Requests are either `Execute { code }` or `Reset`. Responses include `stdout`, `stderr`, any exception raised, and any images or video frames produced during execution (base64 PNG).

Execution state persists between `Execute` requests: globals, imports, open file handles. `Reset` tears down user state while keeping the subprocess alive.

## Output capture

The subprocess redirects stdout and stderr to per-request buffers during `exec`. Matplotlib's display backend is replaced with a hook that captures `plt.show()`, `plt.savefig()`, and animation writes as image and video-frame data on the response. `sys.stdin` reads raise `EOFError` so user code calling `input()` fails rather than blocking.

## Timeouts

Each call has a per-call timeout (`--code-exec-timeout`, default 30 seconds). On expiry, the engine sends `SIGINT` to the subprocess and waits briefly for a graceful response. A responding subprocess keeps the session alive; a non-responding one is killed and replaced.

Non-Unix platforms do not have the graceful path and go straight to kill on timeout.

## Working directory

Per-session default: a fresh temp dir, deleted when the session ends. With `--code-exec-workdir /path`, every session shares that directory. A shared directory makes outputs inspectable across sessions but removes one layer of isolation.

## Isolation boundary

The subprocess always runs as the same user as mistralrs. What constrains it on top of that depends on the entry point and platform.

### CLI and TOML default

`mistralrs serve` and `mistralrs from-config` default to `--sandbox auto`, which enables the [OS-level sandbox](/mistral.rs/reference/sandbox/) on Linux and macOS and is a no-op with a warning on other platforms. On Linux that means env scrubbing, user/IPC/UTS (and optional NET) namespaces, a Landlock FS allowlist, `setrlimit` caps, a seccomp deny-list, and optional cgroup v2 limits. On macOS it means Seatbelt (`sandbox-exec`) with a deny-by-default profile plus env scrubbing.

`--sandbox on` promotes any missing sandbox layer (no Landlock, no seccomp, no namespaces) into a hard error at code-execution init, instead of falling back to whatever layers are available.

`--sandbox off` and `MISTRALRS_SANDBOX=off` disable all sandbox layers: the subprocess then has the same filesystem, network, and syscall access as any Python process running as the mistralrs user. A startup warning is logged so the choice is visible in logs.

### Python and Rust API behavior

The programmatic `CodeExecutionConfig` defaults to no sandbox. Passing `sandbox_policy=None` (Python) or `sandbox_policy: None` (Rust) is equivalent to `--sandbox off`: the subprocess inherits the host environment unchanged. The sandbox engages only when a `SandboxPolicy` is constructed and attached to the config. This matters when embedding mistral.rs as a library: the safer CLI default is not inherited, so an embedding application is responsible for choosing a policy.

### Threat model and limitations

The sandbox targets **model misbehavior**: a confused or jailbroken model emitting Python that reads `~/.ssh/id_rsa`, runs `rm -rf ~`, exfiltrates over the network, forks unbounded processes, or pivots through `subprocess`/`ptrace`. It is not a substitute for OS-level isolation against a determined attacker who can choose arbitrary code.

Known limitations:

- **macOS resource caps.** `max_memory_mb`, `max_cpu_secs`, `max_procs`, `max_open_fds`, and `max_file_sz_mb` are accepted on macOS for configuration parity but are not enforced by Seatbelt. Filesystem and network isolation do apply. Hard CPU/memory/process caps on macOS require an outer container or VM.
- **PID namespace.** On Linux, PID namespace isolation is not applied: `unshare(CLONE_NEWPID)` only affects future children of the calling thread, and the Python process is past that fork.
- **Missing kernel features.** Without unprivileged user namespaces, Landlock (kernel 5.13+), or seccomp filter install, the corresponding layers are skipped under `--sandbox auto`. Use `--sandbox on` to make that a hard error instead of a silent fallback.
- **Shared workdirs.** `--code-exec-workdir <path>` is made writable inside the sandbox and shared across sessions. Anything written there persists and is visible to subsequent sessions.

For high-assurance deployments (multi-tenant, untrusted prompts, regulated data), also run the mistralrs process itself inside a container or VM, as a dedicated low-privilege user, with constrained network egress. `--tool-dispatch-url` is the alternative when you want code execution to leave the mistralrs host entirely.

## See also

- Reference: [sandbox](/mistral.rs/reference/sandbox/) for the per-layer details and tuning knobs.
- Guide: [enable code execution](/mistral.rs/guides/agents/enable-code-execution/).
