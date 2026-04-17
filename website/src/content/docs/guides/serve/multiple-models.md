---
title: Serve multiple models from one process
description: Load several models into one mistralrs server. Route requests by the model field, unload models to free memory, reload them on demand.
sidebar:
  order: 2
---

`mistralrs serve -m <model>` loads exactly one model. To host multiple models behind one endpoint, load them all into one server and route by the standard `model` field on each request.

This matches OpenAI's pattern: each request specifies a model, and the server dispatches to the loaded model with that id.

## Starting a multi-model server

Multi-model uses a configuration file. Create `models.toml`:

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

Start with `from-config`:

```bash
mistralrs from-config -f models.toml
```

Each `[models.<name>]` block defines one loaded model. The bracket name is a config-internal label; the request id is the value of `alias`. Without an `alias`, the canonical Hugging Face repository id is used.

Mixed model types coexist. Text, multimodal, GGUF-quantized, and speech models can all be loaded in one server. Each runs on its own engine thread, so they do not interfere with each other's latency.

## Routing a request to a specific model

The `model` field selects the target:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Either the alias or the canonical repository id works. The alias is shorter and stable across model updates.

Omitting `model` or passing `"default"` selects the default model: either `--default-model-id` from server startup, or the first model in the config file.

## Listing loaded models

```bash
curl http://localhost:1234/v1/models
```

Returns the model list with status:

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

Loaded models occupy GPU memory. The unload endpoint releases the memory while preserving the configuration for later reload:

```bash
curl -X POST http://localhost:1234/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gemma-vision"}'
```

Subsequent requests targeting an unloaded model trigger an automatic reload before the request runs. This enables a lazy-loading pattern: configure many models, leave most unloaded, materialize on first access.

Explicit reload (avoids first-request loading latency):

```bash
curl -X POST http://localhost:1234/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gemma-vision"}'
```

## Multi-model from code

From Rust:

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

Every request method on `Model` has a `_with_model` variant taking an optional id. `None` selects the default.

The Python Runner mirrors this API; details in the [Python API reference](/mistral.rs/reference/python-api/).

## Practical notes

Memory usage is approximately the sum of each loaded model plus its KV cache. Unloading is the lever for staying within a memory budget. A common pattern is to define many models in TOML with `unload_on_start = true` so the server boots quickly and loads each model only on first access.

When models serve different user populations (e.g., public chat and internal research), separate server processes are usually cleaner. Multi-model is best when all loaded models share a trust boundary.
