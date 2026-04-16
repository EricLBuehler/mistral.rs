---
title: Code execution design
description: The subprocess model, session lifecycle, and the stdin/stdout protocol used by the Python executor.
sidebar:
  order: 9
---

When code execution is enabled, mistralrs runs Python in a dedicated subprocess and talks to it over a simple JSON-over-stdio protocol. This page documents that design: what is in the subprocess, how it communicates with the engine, and why we chose a subprocess instead of the obvious alternatives.

## Why a subprocess

Three alternatives we considered and rejected:

**Embedded Python in the Rust process.** Python can be linked into a Rust binary via PyO3. We use this for the SDK bindings but not for the code executor. The problem is that a crash in user code (segfault in a C extension, OOM, stack overflow) takes the entire mistralrs process with it. For a feature that lets a model run arbitrary code, that blast radius is unacceptable.

**Container-per-request.** A Docker container per tool call gives strong isolation but adds seconds of startup latency per call. For an agentic workflow with many short calls, that latency dominates. Additionally, containers need specific infrastructure (Docker daemon, image builds) that is friction for users running mistralrs as a local binary.

**Sandboxed interpreters.** WebAssembly-based Python (Pyodide) or restricted Python (RestrictedPython) give safety without container overhead but have limited library ecosystem support. Users who want code execution usually want matplotlib, pandas, and requests to work.

A subprocess sits between these: cheap to start (a few hundred ms), crash-isolated from the engine, with full library access. The tradeoff is that it is not a security sandbox. We address that with OS-level isolation (user accounts, container deployments) rather than by trying to solve it in the Python layer.

## The subprocess lifecycle

A subprocess is started lazily. The first code execution in a given session spawns a fresh Python, the second reuses it, and so on.

Sessions are keyed by `session_id`. Different sessions get different subprocesses, so one session's globals are invisible to another. Within a session, globals persist; you can define a variable in one call and reference it in the next.

Subprocesses are reaped in two cases:

- **Idle TTL.** A subprocess that has not received a call in 1 hour is killed. The session that owned it is marked as needing a fresh subprocess on its next call.
- **Explicit reset.** The `reset_python_session` tool or the corresponding SDK method tears down the subprocess and starts a new one. This is the model's escape hatch when it has gotten the session into a bad state.

Killed subprocesses leave no orphaned state behind. The working directory (default: a fresh temp dir per session) gets cleaned up too if it was the default temp location.

## The stdio protocol

The executor script runs a loop that reads JSON requests from stdin and writes JSON responses to stdout. Each message is a single line.

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

The `images` and `video_frames` fields carry base64-encoded PNGs produced during execution. See the [vision-and-video guide](/mistral.rs/guides/models/use-vision-input/) for how they reach the model.

The protocol is intentionally simple. No multiplexing, no streaming, no nested messages. Each request blocks until its response arrives. For parallel execution across sessions, multiple subprocesses run concurrently; within a session, calls are serialized.

## What is in the subprocess

The executor script sets up a few things before dropping into the request loop:

**Matplotlib hooks.** The default matplotlib backend is intercepted so that figures produced during execution are captured as PNGs rather than opening a display window. `plt.show()`, `plt.savefig()`, and `FuncAnimation.save()` all route through hooks that add to the image or video-frame list.

**stdin blocker.** The real subprocess stdin is reserved for the protocol. We swap `sys.stdin` with a class that raises `EOFError` on read attempts, so user code that calls `input()` fails cleanly instead of deadlocking the protocol.

**Output capture.** stdout and stderr are redirected to buffers during code execution, so the subprocess's real stdout is still available for protocol responses. The captured buffers go into the response's `stdout` and `stderr` fields.

**Signal handling.** SIGINT is used for timeout handling (more below). The executor carefully masks SIGINT during protocol I/O so that a timeout does not corrupt the protocol stream.

## Timeouts

Each `Execute` request has a timeout (30 seconds by default, configurable). If the execution does not finish in that window, the engine sends SIGINT to the subprocess.

SIGINT is the gentle path: it interrupts the Python interpreter with a `KeyboardInterrupt`, which user code usually does not catch. Execution stops, and the executor returns a response indicating a timeout. The subprocess is still alive; the session continues.

If SIGINT does not take effect within a grace period (3 seconds), the engine escalates to SIGKILL. The subprocess dies and the session is marked as needing a fresh one.

## Working directory

Every subprocess has a working directory. By default it is a fresh temporary directory per session (`mistralrs-code-<random>`), which gets cleaned up when the session ends.

If the user configured `--code-working-dir /path`, all sessions share that directory. This is useful for debugging (you can look at what the model produced) and for workflows where you want outputs to persist, but it removes one layer of between-session isolation.

The working directory is also where `import`s can find extra packages, if you want a Python virtual environment specific to the executor's use. Setting `PYTHONPATH` in the subprocess environment is the cleanest way to extend the import path.

## What the model sees

From the model's perspective, the code executor is a tool with a particular schema: it takes a `code` string and returns `stdout`, `stderr`, and optionally images. The model does not know about subprocesses, stdio, or any of the rest.

This separation is deliberate. The model's mental model of "what happens when I write Python" should be just "it runs," not a leaky abstraction involving session state or protocols. When we need the model to understand session state (for building things up across calls), we document it in the tool's description.

## The safety story

To repeat because it is important: the subprocess is not a sandbox. It runs as the same user as mistralrs and can do anything a Python process could. File system access, network calls, arbitrary system calls, all available.

For deployments where untrusted users can invoke code execution, the right isolation boundaries are:

- **User account.** Run mistralrs as a dedicated user with minimal file system privileges.
- **Container.** Put everything in Docker or Kubernetes with filesystem and network policies.
- **Separate host.** For the highest isolation, the tool-dispatch URL feature lets you run tools in an entirely separate service on a different machine.

These are the same techniques you would use for any service that runs user-submitted code. mistralrs does not try to reimplement them itself.

## Where this lives

- `mistralrs-code-exec/src/session.rs` is the per-subprocess driver.
- `mistralrs-code-exec/src/lib.rs` is the manager that maps session IDs to subprocesses.
- `mistralrs-code-exec/python/executor.py` is the Python-side request loop.

The three together make up the code-execution feature. They are a separate crate from the main engine so that the feature can be compiled in or out cleanly.
