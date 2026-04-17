---
title: Code execution design
description: The subprocess model, session lifecycle, and the stdin/stdout protocol used by the Python executor.
sidebar:
  order: 9
---

When code execution is enabled, mistral.rs runs Python in a dedicated subprocess and communicates over JSON-over-stdio. This page documents the design.

## Why a subprocess

Three rejected alternatives:

**Embedded Python in the Rust process.** PyO3 allows linking Python into a Rust binary. The SDK uses this; the code executor does not. A crash in user code (segfault in a C extension, OOM, stack overflow) takes down the entire mistral.rs process. For a feature running model-generated arbitrary code, that blast radius is unacceptable.

**Container-per-request.** A Docker container per tool call provides strong isolation but adds seconds of startup latency per call. For agentic workflows with many short calls, latency dominates. Containers also need infrastructure (Docker daemon, image builds) that adds friction for local-binary users.

**Sandboxed interpreters.** WebAssembly Python (Pyodide) or restricted Python (RestrictedPython) provide safety without container overhead but have limited library ecosystem support. Code execution users typically want matplotlib, pandas, and requests to work.

A subprocess sits between: cheap startup (~few hundred ms), crash-isolated from the engine, full library access. The tradeoff: not a security sandbox. Address that with OS-level isolation (user accounts, container deployments) rather than the Python layer.

## The subprocess lifecycle

Subprocesses start lazily. The first code execution in a session spawns a fresh Python; subsequent ones reuse it.

Sessions are keyed by `session_id`. Different sessions get different subprocesses, so cross-session globals are invisible. Within a session, globals persist across calls.

Subprocesses are reaped in two cases:

- **Idle TTL.** A subprocess with no calls in 1 hour is killed. The owning session is marked as needing a fresh subprocess on next call.
- **Explicit reset.** The `reset_python_session` tool or the corresponding SDK method tears down the subprocess and starts a new one. The model's escape hatch from a bad session state.

Killed subprocesses leave no orphaned state. The working directory (default: a fresh temp dir per session) is cleaned up if it was the default temp location.

## The stdio protocol

The executor script runs a loop reading JSON requests from stdin and writing JSON responses to stdout. Each message is one line.

Request types:

```json
{"type": "Execute", "code": "print('hello')"}
{"type": "Reset"}
```

Response types:

```json
{
  "type": "ExecuteResult",
  "stdout": "hello\n",
  "stderr": "",
  "images": [ ... ],
  "video_frames": [ ... ],
  "exception": null
}
```

```json
{"type": "ResetResult"}
```

`images` and `video_frames` carry base64-encoded PNGs produced during execution. See the [vision-and-video guide](/mistral.rs/guides/models/use-vision-input/) for how they reach the model.

The protocol is intentionally simple: no multiplexing, no streaming, no nested messages. Each request blocks until its response. Cross-session parallelism uses multiple concurrent subprocesses; within a session, calls serialize.

## What is in the subprocess

The executor script sets up a few things before the request loop:

**Matplotlib hooks.** The default backend is intercepted to capture figures as PNGs rather than opening a display. `plt.show()`, `plt.savefig()`, and `FuncAnimation.save()` route through hooks adding to the image or video-frame list.

**stdin blocker.** Real subprocess stdin is reserved for the protocol. `sys.stdin` is swapped with a class raising `EOFError` on read attempts so user code calling `input()` fails cleanly instead of deadlocking the protocol.

**Output capture.** stdout and stderr redirect to buffers during execution so the subprocess's real stdout remains available for protocol responses. Captured buffers go into the response's `stdout` and `stderr` fields.

**Signal handling.** SIGINT is used for timeout handling (below). The executor masks SIGINT during protocol I/O so timeouts do not corrupt the protocol stream.

## Timeouts

Each `Execute` request has a timeout (30 seconds default, configurable). If execution does not finish in the window, the engine sends SIGINT.

SIGINT is the gentle path: it raises `KeyboardInterrupt` in the Python interpreter, which user code usually does not catch. Execution stops, the executor returns a timeout response, and the subprocess remains alive. The session continues.

If SIGINT does not take effect within a 3-second grace period, the engine escalates to SIGKILL. The subprocess dies and the session is marked as needing a fresh one.

## Working directory

Every subprocess has a working directory. By default, a fresh temp directory per session (`mistralrs-code-<random>`), cleaned up at session end.

With `--code-working-dir /path`, all sessions share the directory. Useful for debugging (model outputs are inspectable) and persisted-output workflows, but removes one isolation layer between sessions.

The working directory is also where `import` finds extra packages, e.g., for an executor-specific virtual environment. Setting `PYTHONPATH` in the subprocess environment is the cleanest way to extend the import path.

## What the model sees

From the model's perspective, the code executor is a tool with a schema: takes a `code` string, returns `stdout`, `stderr`, and optionally images. The model knows nothing of subprocesses, stdio, or any of the rest.

This separation is deliberate. The model's mental model of "what happens when I write Python" should be "it runs," not a leaky abstraction involving session state or protocols. When the model needs to understand session state (for incremental builds), it is documented in the tool's description.

## The safety story

The subprocess is not a sandbox. It runs as the same user as mistral.rs and can do anything a Python process can — file system, network, syscalls, all available.

For deployments with untrusted users invoking code execution, isolation boundaries:

- **User account.** Run mistral.rs as a dedicated user with minimal filesystem privileges.
- **Container.** Use Docker or Kubernetes with filesystem and network policies.
- **Separate host.** For maximum isolation, the tool-dispatch URL feature runs tools on a separate machine.

These are standard techniques for any service running user-submitted code. mistral.rs does not reimplement them.

## Where this lives

- `mistralrs-code-exec/src/session.rs` — per-subprocess driver.
- `mistralrs-code-exec/src/lib.rs` — manager mapping session IDs to subprocesses.
- `mistralrs-code-exec/python/executor.py` — Python-side request loop.

These three together make up the code-execution feature. Separated from the main engine crate so the feature can be compiled in or out cleanly.
