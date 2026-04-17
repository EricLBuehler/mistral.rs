---
title: Enable code execution
description: Turn on the Python executor. Understand session lifecycle, working directories, and the isolation model.
sidebar:
  order: 2
---

Code execution gives the model access to a Python interpreter. When enabled, mistral.rs registers two tools:

- `execute_python` — runs arbitrary Python and returns stdout, stderr, and any matplotlib figures or animations.
- `reset_python_session` — clears variables and imports in the current session.

The model runs code in a subprocess on the host. The isolation model is described below.

## Turning it on

```bash
mistralrs serve --enable-code-execution -m <model>
```

The flag is compiled in by default. A binary built with `--no-default-features` and without the `code-execution` feature does not recognize the flag.

After this, the model can invoke either tool whenever appropriate.

## Execution environment

Each session gets a fresh Python subprocess with:

- A working directory: a fresh temp directory by default, or a startup-specified directory.
- A standard Python namespace. No magic modules are imported except the matplotlib hooks below.
- Access to packages importable from the system `python3` (or whatever interpreter mistralrs was built against).

stdin, stdout, and stderr are routed through mistralrs. `print(...)` ends up in the tool result. `input(...)` is blocked — reading from stdin raises an error.

## Sessions and state

Each request gets a session id. By default it is generated automatically and returned in the response. Reusing the same id continues the same Python subprocess; variables and imports persist:

```python
# First request, session_id = abc123
x = 42
print(x)  # 42

# Second request, also session_id = abc123
print(x * 2)  # 84
```

Without a session id, each request gets a fresh interpreter.

Sessions idle for more than an hour are cleaned up. Cleanup runs every few minutes; sessions near the threshold are safe, well past it are gone. State persistence across restarts requires manual export (pickle, database, etc.).

## Working directory

Without configuration, each session gets a unique temp directory named `mistralrs-code-<random>`. Files created by the model land there and are cleaned up at session end.

To pin a directory:

```bash
mistralrs serve --enable-code-execution --code-working-dir /srv/mistral-work -m <model>
```

All sessions share that directory. Useful for debugging and post-hoc output inspection, at the cost of one isolation layer between requests.

## Matplotlib integration

The executor installs hooks that capture matplotlib figures automatically. `plt.savefig("whatever.png")` adds the PNG to the `images` portion of the tool result, which the model sees as visual feedback. `fig.show()` and `plt.show()` behave the same.

For animations, the hook captures `FuncAnimation.save(...)` output as a video the model can view (when video input is supported). This enables iterative animation construction with model observation.

The tool description presented to the model documents this interaction. No prompting is required.

## Timeouts

Default execution timeout is 30 seconds. On timeout, mistralrs sends SIGINT (graceful interrupt) and reports a timeout result, prompting the model toward a less expensive approach.

Raise the timeout for long-running tasks:

```bash
mistralrs serve --enable-code-execution --code-timeout-secs 120 -m <model>
```

A timeout of zero disables the feature.

## Isolation model and safety

The subprocess runs as the same user as mistralrs. It can read any files mistralrs can read, make network connections, and otherwise do anything a Python process can. This is not a sandbox.

For untrusted users, the recommended pattern is:

1. Run mistralrs as a dedicated user with limited filesystem access.
2. Place the working directory on a volume free of sensitive data.
3. Firewall outgoing connections from the Python subprocess.
4. Run mistralrs in a Docker or Kubernetes container with network policies and least-privilege filesystem mounts for maximum isolation.

The subprocess does not escape Python-level sandboxing, but it is not in a Linux container of its own without explicit setup.

## What to read next

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/) — agent state across requests.
- [Code execution design](/mistral.rs/explanation/code-execution-design/) — internals.
