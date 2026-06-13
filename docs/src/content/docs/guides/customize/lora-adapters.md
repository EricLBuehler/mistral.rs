---
title: LoRA and X-LoRA adapters
description: Attach fine-tuned adapters to a base model without re-downloading the full weights.
---

LoRA adapters add task-specific fine-tuning on top of a base model without modifying the base weights. X-LoRA loads several adapters at once and lets the model select among them per request.

## Loading a LoRA

```bash
mistralrs run -m <base-model> --lora <lora-repo>
```

Multiple adapters: semicolon-separated.

```bash
mistralrs run -m <base-model> --lora "<lora-repo-1>;<lora-repo-2>"
```

mistral.rs reads `adapter_config.json` from the LoRA repo for targeted modules and rank.

## X-LoRA

X-LoRA loads multiple adapters with a learned scaling head selecting per-token weighting.

```bash
mistralrs run \
  -m <base-model> \
  --xlora <xlora-repo> \
  --xlora-order <ordering-file.json>
```

`--xlora` conflicts with `--lora`. `--xlora-order` and `--tgt-non-granular-index` are only valid together with `--xlora`.

`--tgt-non-granular-index <n>` controls X-LoRA scaler recompute frequency. Without it, the scaler recomputes every token.

## Adapters in the SDKs

Python uses `Which.Lora` / `Which.XLora`; Rust uses `LoraModelBuilder` / `XLoraModelBuilder` wrapped around a `TextModelBuilder`. Full examples: [lora-zephyr](/mistral.rs/examples/python/lora-zephyr/) and [xlora-zephyr](/mistral.rs/examples/python/xlora-zephyr/) (Python), [lora](/mistral.rs/examples/rust/advanced/lora/) and [xlora](/mistral.rs/examples/rust/advanced/xlora/) (Rust).

## AnyMoE

AnyMoE goes a step further than adapters: it composes several fine-tunes of the same base model into a mixture-of-experts configuration at inference time, training only a small per-layer router.

- It is exposed through the Rust SDK (`AnyMoeModelBuilder`) and the Python SDK (`AnyMoeConfig`, `AnyMoeExpertType`); it is not configurable via the CLI.
- Expert checkpoints must share the base model architecture, and a small JSON calibration dataset is required to train the router.
- The `AnyMoeConfig` docstrings in the [Python reference](/mistral.rs/reference/python/) cover finding the `prefix`/`mlp` values from a model's `model.safetensors.index.json`.

Full examples: [anymoe](/mistral.rs/examples/python/anymoe/) (Python), [anymoe](/mistral.rs/examples/rust/advanced/anymoe/) and [anymoe-lora](/mistral.rs/examples/rust/advanced/anymoe-lora/) (Rust).
