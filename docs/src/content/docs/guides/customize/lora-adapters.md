---
title: LoRA and X-LoRA adapters
description: Attach fine-tuned adapters to a base model without re-downloading the full weights.
sidebar:
  order: 1
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

`--xlora` requires `--xlora-order` and conflicts with `--lora`.

`--tgt-non-granular-index <n>` controls X-LoRA scaler recompute frequency. Without it, the scaler recomputes every token.

## Adapters in the SDKs

Python:

```python
from mistralrs import Runner, Which

runner = Runner(
    which=Which.Lora(
        adapter_model_ids=["<lora-repo>"],
        model_id="<base-model>",
    ),
)
```

Rust: `LoraModelBuilder` and `XLoraModelBuilder`.
