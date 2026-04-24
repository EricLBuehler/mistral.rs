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

HTTP requests must also opt into the tool:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": "Use Python to calculate and plot the first 20 primes."
    }
  ],
  "enable_code_execution": true,
  "max_tool_rounds": 4
}
```

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `--code-exec-python <path>` | `python` on Windows, `python3` elsewhere | Python interpreter. |
| `--code-exec-timeout <secs>` | 30 | Per-call timeout in seconds. |
| `--code-exec-workdir <path>` | per-session temp dir | Working directory. |

## Sessions and state

Each session gets its own Python subprocess on first call. Subsequent calls reuse it; variables and imports persist within the session. Without a session id, each request gets a fresh interpreter.

Subprocesses idle for more than 1 hour are reaped. The reaper runs every 5 minutes.

## stdin / stdout / stderr

stdout and stderr from user code are captured and returned in the tool result. stdin reads raise `EOFError`.

When streaming chat completions, code execution progress is emitted as `agentic_tool_call_progress` SSE events. Complete events can include `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `working_directory`, and `execution_time_ms`.

## Working directory

Without `--code-exec-workdir`, each session gets a unique `mistralrs-code-<random>` temp directory.

With `--code-exec-workdir /path`, all sessions share the directory.

## Isolation

The subprocess runs as the same user as mistral.rs. It is not a sandbox. For untrusted users, run mistral.rs in a container, with a dedicated low-privilege user, and constrain network egress.

## Implementation

The executor lives in the `mistralrs-code-exec` crate. The Python side is `mistralrs-code-exec/python/executor.py`.

## See also

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).
- [Code execution design](/mistral.rs/explanation/code-execution-design/).
