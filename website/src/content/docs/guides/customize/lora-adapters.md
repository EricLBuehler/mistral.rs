---
title: LoRA and X-LoRA adapters
description: Attach fine-tuned adapters to a base model without re-downloading the full weights.
sidebar:
  order: 1
---

LoRA adapters are small files that add task-specific fine-tuning on top of a base model without changing the base weights. X-LoRA is an extension that loads several adapters at once and lets the model pick which one to use for each request. Both work in mistral.rs.

## Why adapters

The main reasons to use an adapter instead of a full fine-tune:

- Adapters are tiny. A typical LoRA is 10 to 100 MB, compared to tens of gigabytes for a full checkpoint. Downloading one, and swapping between many, is cheap.
- The base model is unchanged. Several adapters can share one loaded base, saving memory.
- Adapters from the Hugging Face community are available for a lot of tasks out of the box, so you often do not have to train one.

## Loading a single LoRA

Start mistralrs with the LoRA's repository id:

```bash
mistralrs run \
  --format plain \
  -m Qwen/Qwen3-4B \
  --adapter-model-id <lora-repo> \
  --adapter-type lora
```

The base model loads normally; the LoRA weights merge in at inference time. From the client's perspective the model behaves like the base plus whatever the adapter was trained for.

LoRA adapters on Hugging Face typically include an adapter config file (`adapter_config.json`) that specifies which modules the adapter targets and at what rank. mistralrs reads that config automatically; you do not need to configure it manually.

## X-LoRA: many adapters, dynamic routing

X-LoRA lets you load several adapters at once. At each inference step, a small scaling model decides how much weight each adapter gets. The result is a single model that can fluidly use any of the loaded adapters as the input demands.

```bash
mistralrs run \
  -m Qwen/Qwen3-4B \
  --xlora-model-id <xlora-repo> \
  --adapter-type xlora \
  --order <ordering-file.json>
```

X-LoRA is overkill for simple "I want this one adapter's behavior" workloads; it pays for itself on tasks that benefit from multi-adapter composition. The X-LoRA paper (linked from the repository README) has details on when this helps.

## Switching adapters at runtime

For serving workloads where you want to load different adapters on different requests, load them as separate models and route by the `model` field (see [multiple models](/mistral.rs/guides/serve/multiple-models/)). Each model entry in the config points at a different adapter on top of the same base.

## Non-granular scalings

The `--adapter-type xlora` path has a `--tgt-non-granular-index` flag that controls when the X-LoRA scaler recomputes its weights. By default it recomputes on every token, which is the most accurate but the slowest. Setting a target index causes it to recompute every N tokens instead, trading off some adaptability for speed.

For most X-LoRA workloads the default is fine. If you are serving at high throughput and the scaler compute shows up as a bottleneck, try `--tgt-non-granular-index 4` first.

## Adapters in the SDKs

The Python and Rust SDKs expose adapter loading through the same `Which` / `ModelBuilder` surface. From Python:

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

From Rust, `ModelBuilder::from_lora(base, lora)` and `ModelBuilder::from_xlora(base, xlora, order)` do the equivalent.
