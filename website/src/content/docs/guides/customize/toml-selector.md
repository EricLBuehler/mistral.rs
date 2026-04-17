---
title: TOML selector
description: Build requests from TOML instead of JSON. Useful for config-driven deployments and repeatable benchmarks.
sidebar:
  order: 6
---

The TOML selector describes a mistral.rs request in a file rather than as JSON in code. The selector takes a TOML file and runs it once against the loaded model.

This is not a substitute for the main API. It applies to:

- Reproducible benchmarks with saved prompt and sampling parameters.
- Shell scripts running the same completion across different models.
- Simple integrations preferring declarative request specification.

## Running a selector

```bash
mistralrs from-config -f models.toml --selector request.toml
```

`-f models.toml` is the main mistralrs config (which models to load). `--selector request.toml` is the selector describing what to do after load.

Minimal selector:

```toml
model = "default"
prompt = "Summarize the theory of general relativity."
max_tokens = 200
```

mistralrs loads models from `models.toml`, runs the prompt against `default`, prints the response, and exits.

## Full request shape

The selector supports the same fields as a JSON request:

```toml
model = "qwen"

[[messages]]
role = "system"
content = "You are a concise assistant."

[[messages]]
role = "user"
content = "What is the answer to life, the universe, and everything?"

max_tokens = 100
temperature = 0.3
top_p = 0.9
stream = false
```

Each `[[messages]]` block is one entry in the messages array, applied in order.

For multimodal requests, content can be an inline array:

```toml
[[messages]]
role = "user"

[[messages.content]]
type = "image_url"

[messages.content.image_url]
url = "file:///path/to/image.jpg"

[[messages.content]]
type = "text"
text = "Describe this image."
```

## Chaining multiple requests

For sequential requests (a prompt, then a follow-up), use the `[[request]]` array:

```toml
[[request]]
model = "default"
prompt = "Write a first draft of a poem about autumn."
max_tokens = 200

[[request]]
model = "default"
prompt = "Now revise that draft to be more concise."
max_tokens = 200
```

Each request runs after the previous finishes. This is a one-shot convenience, not a chain of thought or agent loop. For those, use the agentic tools.

## When to use it

The selector fits:

- One-off benchmarks.
- Demos requiring reliable conversation reproduction.
- Cron-driven completion batches.

Do not use it when:

- Streaming is required. The selector runs to completion before emitting output.
- Intermediate state inspection is required. There is no loop hook.
- The use case is an interactive service. Use the HTTP API.

## Limitations

The selector does not expose every field in the full request shape. Some fields (custom logit bias tables, grammar strings with embedded quotes) are awkward in TOML. Drop down to the HTTP API or an SDK when the selector cannot express the request.
