---
title: Serve multiple models from one process
description: Load several models into one mistralrs server. Route requests by the model field, unload models to free memory, reload them on demand.
sidebar:
  order: 2
---

A single `mistralrs serve -m <model>` starts a server with exactly one loaded model. If you want to offer several models through the same endpoint (a small fast one for autocomplete, a large accurate one for chat, plus a multimodal model for image questions), you can load all of them into one server and route between them using the standard `model` field in each request.

The pattern is the same one OpenAI uses: every request specifies the model it wants, and the server dispatches to the loaded model with that id.

## Starting a multi-model server

Multi-model uses a configuration file. Create something like `models.toml`:

```toml
[models.qwen]
alias = "qwen"
in_situ_quant = "4"
[models.qwen.Plain]
model_id = "Qwen/Qwen3-4B"

[models.gemma]
alias = "gemma"
in_situ_quant = "4"
[models.gemma.Plain]
model_id = "google/gemma-4-E4B-it"

[models.gemma-vision]
alias = "gemma-vision"
in_situ_quant = "4"
[models.gemma-vision.MultimodalPlain]
model_id = "google/gemma-4-E4B-it"
```

Then start the server with `from-config`:

```bash
mistralrs from-config -f models.toml
```

Each `[models.<name>]` block is one loaded model. The name inside the brackets (`qwen`, `gemma`, `gemma-vision`) is a label for human use in the config; the id clients use in requests is whatever you put in `alias`. If you do not set an alias, the canonical Hugging Face repository id is used instead.

Mixed model types work. A text model, a multimodal model, a GGUF-quantized model, and a speech model can all coexist in one server. Each one runs on its own engine thread, so they do not interfere with each other's latency.

## Routing a request to a specific model

The `model` field in the request selects which loaded model handles it:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Either the alias or the canonical repository id works. The alias is usually shorter and more stable across model updates.

If you omit the `model` field entirely, or pass `"default"`, the server picks a default. That default is either the model you specified as `--default-model-id` when starting the server, or, if you did not, whichever model is listed first in the config file.

## Listing loaded models

```bash
curl http://localhost:1234/v1/models
```

You get back a list with every model plus a status field:

```json
{
  "object": "list",
  "data": [
    { "id": "default", "object": "model" },
    { "id": "qwen", "object": "model", "status": "loaded" },
    { "id": "gemma", "object": "model", "status": "loaded" },
    { "id": "gemma-vision", "object": "model", "status": "unloaded" }
  ]
}
```

## Unloading and reloading on demand

Loaded models hold weights in GPU memory. When you are running several large ones, you can hit memory limits fast. The unload endpoint releases a model's memory while keeping its configuration around for a later reload:

```bash
curl -X POST http://localhost:1234/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gemma-vision"}'
```

Once unloaded, a subsequent request that targets that model will trigger an automatic reload before the request runs. That is a lazy-loading pattern: you can keep a dozen models in the config, leave most of them unloaded, and let the server materialize them when someone actually asks.

To reload explicitly (so the first request after does not pay the loading latency):

```bash
curl -X POST http://localhost:1234/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gemma-vision"}'
```

## Multi-model from code

Both SDKs have equivalent APIs. From Rust:

```rust
use mistralrs::{IsqType, MultiModelBuilder, TextModelBuilder, MultimodalModelBuilder};

let model = MultiModelBuilder::new()
    .add_model_with_alias(
        "qwen",
        TextModelBuilder::new("Qwen/Qwen3-4B").with_isq(IsqType::Q4K),
    )
    .add_model_with_alias(
        "gemma-vision",
        MultimodalModelBuilder::new("google/gemma-4-E4B-it").with_isq(IsqType::Q4K),
    )
    .with_default_model("qwen")
    .build()
    .await?;

let response = model
    .send_chat_request_with_model(messages, Some("qwen"))
    .await?;
```

Every request method on `Model` has a `_with_model` variant that takes an optional id. Pass `None` to hit the default model.

The Python Runner works similarly; the details are in the [Python API reference](/mistral.rs/reference/python-api/).

## Practical notes

Memory usage is roughly the sum of what each loaded model would use on its own, plus their respective KV caches. Unloading is the lever you use to stay inside a memory budget. A common pattern is to configure many models in the TOML but set `unload_on_start = true` so the server boots fast and loads each model only on first access.

If you are hosting models that serve different user populations (for example, a public chat model and an internal research model), running them in separate server processes is often cleaner than multi-model in one process. Multi-model is at its best when all the loaded models share the same trust boundary.
