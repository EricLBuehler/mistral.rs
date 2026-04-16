---
title: TOML selector
description: Build requests from TOML instead of JSON. Useful for config-driven deployments and repeatable benchmarks.
sidebar:
  order: 6
---

The TOML selector is a small convenience feature for when you want to describe a mistralrs request in a file rather than building it as JSON in code. The selector takes a TOML file and turns it into a request that runs once against the loaded model.

This is not a substitute for the main API. It is useful for:

- Reproducible benchmarks where you want to save the exact prompt and sampling parameters.
- Shell scripts that run the same completion repeatedly with different models.
- Simple integrations that want to spell a request out declaratively.

## Running a selector

```bash
mistralrs from-config -f models.toml --selector request.toml
```

`-f models.toml` is the main mistralrs config (which models to load). `--selector request.toml` is the selector file that describes what to do after loading.

A minimal selector:

```toml
model = "default"
prompt = "Summarize the theory of general relativity."
max_tokens = 200
```

mistralrs loads the models from `models.toml`, runs the prompt against the model named `default`, prints the response, and exits.

## Full request shape

The selector file supports the same fields a JSON request does:

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

Each `[[messages]]` block is one entry in the messages array. They are applied in order.

For multimodal requests, the content can be an inline array:

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

For workflows that run several requests in sequence (a prompt, then a follow-up based on the previous response), the selector format handles this with a `[[request]]` array:

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

Each request runs after the previous finishes. This is a one-shot convenience, not a proper chain of thought or agent loop. For that, use the agentic tools.

## When to use it, when not to

The selector is at its best for:

- One-off benchmarks.
- Demos where you want to show the same conversation reliably.
- Scripts that run completion batches from a cron job.

Do not use it when:

- You need streaming. The selector runs to completion before emitting output.
- You need to inspect intermediate state. There is no hook into the loop.
- Your use case is an interactive service. Use the HTTP API.

## Limitations

The selector does not expose every field of the full request shape. Some rarely-used fields (custom logit bias tables, grammar strings with embedded quotes) are awkward in TOML. When a selector cannot express what you need, drop down to the HTTP API or an SDK.
