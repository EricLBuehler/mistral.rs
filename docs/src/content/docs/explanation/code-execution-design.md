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

The subprocess runs as the same user as mistralrs. It is not a sandbox: file system, network, and syscall access match any Python process on the host.

For untrusted users, isolation needs to come from the OS: container, low-privilege user, or egress firewalling. `--tool-dispatch-url` also exists as a way to offload execution to an external service.

## See also

- Guide: [enable code execution](/mistral.rs/guides/agents/enable-code-execution/).
