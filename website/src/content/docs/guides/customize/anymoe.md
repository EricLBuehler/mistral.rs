---
title: AnyMoE
description: Compose a mixture-of-experts model at inference time from several existing fine-tunes.
sidebar:
  order: 2
---

AnyMoE combines several fine-tuned models into one mixture-of-experts configuration without retraining. Existing variants of the same base are loaded as experts; the engine routes each token through the combination.

Unlike a traditional MoE where experts were trained jointly, AnyMoE composes arbitrary fine-tunes into an ad-hoc ensemble. Output quality depends on how well the experts complement each other.

## When to use it

- Several fine-tunes of the same base model exist (math, code, creative writing).
- Combined strengths are desired without sequential model runs.
- VRAM is sufficient to keep all experts loaded.

Skip when:

- Only one fine-tune is needed — use a regular LoRA or standalone fine-tune.
- VRAM is insufficient.
- Experts were trained on very different data — the mixture muddles.

## Configuration

AnyMoE requires a config listing experts and a small routing model. Minimal example:

```toml
[anymoe]
dataset_json = "dataset.json"
prefix = "transformer.h"
target_modules = ["mlp"]
model_ids = [
    "user/fine-tune-math",
    "user/fine-tune-code",
    "user/fine-tune-creative",
]
layers = [0, 4, 8, 12, 16, 20, 24, 28]
```

Pass at startup:

```bash
mistralrs run -m <base-model> --anymoe-config anymoe.toml
```

Three config controls:

- `model_ids` — expert checkpoints. Must share the base model architecture.
- `target_modules` — modules inside each transformer block that get routed. `mlp` is typical; `attention` is possible but expensive.
- `layers` — layer indices receiving AnyMoE routing. Every fourth or eighth layer is a reasonable start.

`dataset_json` points to a small calibration dataset used to train the router (a small linear layer per routed module). A few dozen representative examples suffice.

## How the routing works

At each routed module, the input hidden state passes through a learned router producing per-expert weights. The output is a weighted combination of each expert's output for that module.

Router weights are learned from the calibration dataset and then frozen. Inference applies them without retraining.

## Cost

AnyMoE multiplies the model footprint roughly by expert count for the routed modules. It adds a small per-routed-module computation for the weighted combination.

A three-expert AnyMoE on a 7B base takes about 2.5× the VRAM of the base alone at the same quantization, and runs about 1.2× slower per token.

## Further reading

The AnyMoE paper (linked from the mistral.rs README) details router training and calibration dataset sizing.
