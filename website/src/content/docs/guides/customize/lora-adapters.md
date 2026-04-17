---
title: LoRA and X-LoRA adapters
description: Attach fine-tuned adapters to a base model without re-downloading the full weights.
sidebar:
  order: 1
---

LoRA adapters add task-specific fine-tuning on top of a base model without modifying the base weights. X-LoRA loads several adapters at once and lets the model select among them per request. Both work in mistral.rs.

## Why adapters

- Adapters are small (10–100 MB vs. tens of GB for a full checkpoint). Cheap to download and swap.
- The base model is unchanged. Multiple adapters can share one loaded base.
- Many community adapters are available on Hugging Face, removing the need to train.

## Loading a single LoRA

Pass the LoRA's repository id:

```bash
mistralrs run \
  --format plain \
  -m Qwen/Qwen3-4B \
  --adapter-model-id <lora-repo> \
  --adapter-type lora
```

The base model loads normally; LoRA weights merge at inference time. The combined model behaves like the base plus the adapter's specialization.

LoRA adapters on Hugging Face usually include `adapter_config.json` specifying targeted modules and rank. mistral.rs reads it automatically.

## X-LoRA: many adapters, dynamic routing

X-LoRA loads several adapters concurrently. A small scaling model decides per-step adapter weights. The result is one model fluidly using any loaded adapter.

```bash
mistralrs run \
  -m Qwen/Qwen3-4B \
  --xlora-model-id <xlora-repo> \
  --adapter-type xlora \
  --order <ordering-file.json>
```

X-LoRA is overkill for single-adapter use. It pays off on tasks benefiting from multi-adapter composition. The X-LoRA paper (linked from the repository README) covers details.

## Switching adapters at runtime

For per-request adapter selection, load adapters as separate models and route by `model` field — see [multiple models](/mistral.rs/guides/serve/multiple-models/). Each model entry points at a different adapter on the same base.

## Non-granular scalings

`--adapter-type xlora` exposes `--tgt-non-granular-index`, controlling X-LoRA scaler recompute frequency. Default: every token. Setting a target index recomputes every N tokens, trading adaptability for speed.

For most workloads, the default is fine. If scaler compute is a high-throughput bottleneck, try `--tgt-non-granular-index 4` first.

## Adapters in the SDKs

The Python and Rust SDKs use the same `Which` / `ModelBuilder` surface. Python:

```python
from mistralrs import Runner, Which

runner = Runner(
    which=Which.Lora(
        model_id="Qwen/Qwen3-4B",
        adapters_model_id="<lora-repo>",
        order="<ordering-file.json>",
    ),
)
```

Rust: `ModelBuilder::from_lora(base, lora)` and `ModelBuilder::from_xlora(base, xlora, order)`.
