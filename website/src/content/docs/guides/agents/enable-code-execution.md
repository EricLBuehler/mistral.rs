---
title: Enable code execution
description: Turn on the Python executor. Understand session lifecycle, working directories, and the isolation model.
sidebar:
  order: 2
---

Code execution gives the model access to a Python interpreter. When turned on, mistral.rs registers two tools with the model:

- `execute_python` runs arbitrary Python and returns stdout, stderr, and any matplotlib figures or animations the code produced.
- `reset_python_session` clears variables and imports in the current session.

This is powerful but intrinsically less safe than plain chat: you are letting a model run code in a subprocess on your machine. The isolation model matters; this guide covers what that model is and how to work within it.

## Turning it on

```bash
mistralrs serve --enable-code-execution -m <model>
```

The flag is compiled in by default, so you do not need anything special at build time. If you built with `--no-default-features` and do not get the `code-execution` feature back, the flag will not be recognized.

That is all you need. From here the model can invoke either tool whenever it decides doing so is useful.

## What the execution environment looks like

Each session (more on those below) gets a fresh Python subprocess with:

- A working directory, either a fresh temp directory (the default) or a directory you specified at startup.
- A standard Python namespace. No magic modules are imported for you except the matplotlib hooks described below.
- Access to any Python packages that are importable from the interpreter mistralrs was built against. By default, the system `python3` is used.

stdin, stdout, and stderr are routed through mistralrs, which means `print(...)` works and ends up in the tool result, and `input(...)` is blocked (attempting to read from stdin raises an error).

## Sessions and state

Each request gets a session id. By default, that id is generated automatically and returned in the response. Passing the same id on a subsequent request continues the same Python subprocess, so variables and imports from the previous call are still available:

```python
# First request, session_id = abc123
x = 42
print(x)  # 42

# Second request, also session_id = abc123
print(x * 2)  # 84
```

If you do not pass a session id, each request gets a fresh interpreter.

Sessions idle for more than an hour are cleaned up automatically. The code that runs the cleanup happens every few minutes; sessions close to the threshold are safe, sessions well past it are gone. If you need state persistence across restarts, you have to export it yourself (pickle to a file, store in a database, and so on).

## Working directory

Without configuration, each session gets its own unique temp directory named `mistralrs-code-<random>`. Files the model creates (plots saved to disk, CSV outputs, notebooks) go there and are cleaned up when the session ends.

To pin a specific directory:

```bash
mistralrs serve --enable-code-execution --code-working-dir /srv/mistral-work -m <model>
```

All sessions then share that directory. This is useful for debugging (you can look at what the model produced) and for workflows where you want to inspect outputs after the fact, but it removes one layer of isolation between requests.

## Matplotlib integration

The executor installs hooks so that matplotlib figures are captured automatically. Calling `plt.savefig("whatever.png")` inside a tool call adds the PNG to the `images` portion of the tool result, which the model then sees as visual feedback. `fig.show()` and `plt.show()` do the same thing.

For animations, the hook captures `FuncAnimation.save(...)` output as a video that the model can view (when the model supports video input). This lets the model build animations step by step and observe the result.

The [code execution tool description](/mistral.rs/guides/agents/enable-code-execution/) presented to the model includes documentation for this interaction, so the model knows what is available. You do not need to prompt for it.

## Timeouts

Each execution call has a 30-second default timeout. If the code does not finish in that window, mistralrs sends a SIGINT to the subprocess (graceful interrupt) and reports a timeout result to the model, which usually prompts it to try a less expensive approach.

To raise the timeout (for example, for long-running data-processing tasks):

```bash
mistralrs serve --enable-code-execution --code-timeout-secs 120 -m <model>
```

A timeout of zero disables the feature entirely, which is probably not what you want.

## Isolation model and safety

The subprocess runs as the same user as mistralrs itself. That means it can read any files mistralrs can read, make network connections, and otherwise do anything a Python process can do. This is not a sandbox.

If you are exposing code execution to untrusted users, the right pattern is:

1. Run mistralrs as a dedicated user with limited filesystem access.
2. Put the working directory on a volume that does not contain anything sensitive.
3. Firewall the machine so outgoing connections from the Python subprocess go where you expect.
4. For the highest isolation, run mistralrs inside a Docker or Kubernetes container with network policies and filesystem mounts tuned for least privilege.

The subprocess cannot escape Python-level sandboxing, but it is not in a Linux container of its own unless you put it in one.

## What to read next

- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/) for keeping agent state across requests.
- [Code execution design](/mistral.rs/explanation/code-execution-design/) for the internals.
