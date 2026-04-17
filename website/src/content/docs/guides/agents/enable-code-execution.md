---
title: Enable code execution
description: Turn on the Python executor.
sidebar:
  order: 2
---

`--enable-code-execution` registers a code execution tool with the model. The tool runs Python in an isolated subprocess.

## Turning it on

```bash
mistralrs serve --enable-code-execution -m <model>
```

The `code-execution` Cargo feature is in the default feature set. Binaries built with `--no-default-features` need it added explicitly.

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `--code-exec-python <path>` | `python3` | Python interpreter. |
| `--code-exec-timeout <secs>` | 30 | Per-call timeout in seconds. |
| `--code-exec-workdir <path>` | per-session temp dir | Working directory. |

## Sessions and state

Each session gets its own Python subprocess on first call. Subsequent calls reuse it; variables and imports persist within the session. Without a session id, each request gets a fresh interpreter.

Subprocesses idle for more than 1 hour (`SESSION_TTL_SECS = 3600`) are reaped. Reap interval is 5 minutes.

## stdin / stdout / stderr

stdout and stderr from user code are captured and returned in the tool result. stdin reads raise `EOFError`.

## Working directory

Without `--code-exec-workdir`, each session gets a unique `mistralrs-code-<random>` temp directory.

With `--code-exec-workdir /path`, all sessions share the directory.

## Isolation

The subprocess runs as the same user as mistral.rs. It is not a sandbox. For untrusted users, run mistral.rs in a container, with a dedicated low-privilege user, and constrain network egress.

## Implementation

The executor lives in the `mistralrs-code-exec` crate. The Python side is `mistralrs-code-exec/python/executor.py`.

## What to read next

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- [Code execution design](/mistral.rs/explanation/code-execution-design/).
