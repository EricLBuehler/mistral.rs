---
title: Serve multiple models from one process
description: Load several models into one mistralrs server. Route requests by the model field, unload models to free memory, reload them on demand.
sidebar:
  order: 2
---

`mistralrs serve -m <model>` loads exactly one model. To host multiple models in one server, use a TOML config and `mistralrs from-config`.

## Starting a multi-model server

Create `models.toml`:

```toml
command = "serve"
default_model_id = "Qwen/Qwen3-4B"

[server]
host = "0.0.0.0"
port = 1234

[[models]]
kind = "text"
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"

[[models]]
kind = "multimodal"
model_id = "google/gemma-4-E4B-it"

[models.quantization]
in_situ_quant = "4"
```

Start with `from-config`:

```bash
mistralrs from-config -f models.toml
```

Each `[[models]]` entry is one loaded model. The request id is the entry's `model_id`. `default_model_id` (if set) must match a `model_id` and is used when a request omits `model` or sends `"default"`.

Mixed `kind` values coexist. Each model runs on its own engine.

## Routing a request to a specific model

The `model` field selects the target:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Omitting `model` or passing `"default"` selects the `default_model_id` (or fails if none is configured).

## Listing loaded models

```bash
curl http://localhost:1234/v1/models
```

Each entry includes `id`, `object`, `created`, `owned_by`, plus optional `status` (`loaded`/`unloaded`/`reloading`), `tools_available`, `mcp_tools_count`, `mcp_servers_connected`.

## Unloading and reloading on demand

```bash
curl -X POST http://localhost:1234/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "google/gemma-4-E4B-it"}'
```

Reload:

```bash
curl -X POST http://localhost:1234/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "google/gemma-4-E4B-it"}'
```

## Multi-model from code

Rust:

```rust
use mistralrs::{IsqType, MultiModelBuilder, TextModelBuilder, MultimodalModelBuilder};

let model = MultiModelBuilder::new()
    .add_model(TextModelBuilder::new("Qwen/Qwen3-4B").with_isq(IsqType::Q4K))
    .add_model_with_alias(
        "gemma-vision",
        MultimodalModelBuilder::new("google/gemma-4-E4B-it").with_isq(IsqType::Q4K),
    )
    .with_default_model("Qwen/Qwen3-4B")
    .build()
    .await?;
```

Every request method on `Model` has a `_with_model` variant taking an optional id. `None` selects the default.

## Notes

`cpu` in `[models.device]` must be consistent across every model entry.
