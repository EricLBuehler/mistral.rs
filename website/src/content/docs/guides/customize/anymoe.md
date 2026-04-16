---
title: AnyMoE
description: Compose a mixture-of-experts model at inference time from several existing fine-tunes.
sidebar:
  order: 2
---

AnyMoE is a technique for combining several fine-tuned models into one mixture-of-experts configuration, without retraining. You pick a set of existing checkpoints that are variants of the same base, tell mistralrs to use them as experts, and the engine routes each token through the combination.

This is distinct from a traditional MoE where the experts were trained together. AnyMoE lets you compose arbitrary fine-tunes into an ad-hoc ensemble. The output quality depends heavily on how well the experts complement each other, so it is a niche tool but a useful one when you have several specialized fine-tunes you want to use together.

## When to use it

Reach for AnyMoE when:

- You have several fine-tunes of the same base model (a math fine-tune, a code fine-tune, a creative-writing fine-tune).
- You want the combined strengths without running three models in sequence and picking the best.
- You have enough VRAM to keep all the experts loaded at once.

Skip it when:

- You have one fine-tune and want to use it: use a regular LoRA or a standalone fine-tuned model.
- You do not have enough VRAM for all experts.
- The experts were trained on very different data; they will produce a muddled mixture.

## Configuration

AnyMoE needs a config file listing the experts and a small routing model. A minimal config:

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

Pass it at startup:

```bash
mistralrs run -m <base-model> --anymoe-config anymoe.toml
```

Three things the config controls:

- `model_ids`: the expert checkpoints. Must be the same architecture as the base model.
- `target_modules`: which modules inside each transformer block get routed. `mlp` is the usual choice; `attention` is also possible but expensive.
- `layers`: which layer indices apply the AnyMoE routing. Every fourth or eighth layer is a reasonable starting point.

The `dataset_json` field points to a small calibration dataset. This is used to train the router's weights (a small linear layer per routed module). The dataset can be a few dozen examples of the kind of prompts you expect at inference time.

## How the routing works

At each routed module, an input token's hidden state goes through a small learned router. The router produces a weight per expert, and the output is a weighted combination of each expert's output for that module.

The weights are learned from your calibration dataset and then frozen. Inference does not retrain the router; it just applies it.

## Cost

AnyMoE doubles (or triples, or quadruples) the memory footprint of the model, because each expert's weights for the routed modules have to live in GPU memory. It adds a small computational overhead per routed module for the weighted combination.

In practice, a three-expert AnyMoE on a 7B base model takes about 2.5x as much VRAM as the base alone at the same quantization, and runs about 1.2x slower per token. Neither multiplier is catastrophic but both matter for serving workloads.

## Further reading

The AnyMoE paper (linked from the mistral.rs README) describes the math in detail, including how the router is trained and why the calibration dataset size matters. If you are tuning AnyMoE for a specific workload, that paper is worth reading.
