---
title: Online calibration
description: Improve a served quantized model from its own traffic, with no restart.
sidebar:
  order: 13
---

Online calibration observes the activations of a live, ISQ-quantized model, then requantizes
every layer from the original weights using an importance matrix derived from that traffic. The
layers are hot-swapped in place: no restart, and the model keeps serving throughout.

Quantized this way, a model is measurably closer to its full-precision outputs on the
distribution it actually serves, at the same bit width and speed.

## Usage

Serve any model with ISQ:

```bash
mistralrs serve -m <model> --isq q4k
```

Then drive the lifecycle over HTTP:

```bash
# begin observing live traffic (~15% decode overhead while on)
curl -X POST localhost:1234/calibration/start

# check per-layer collection progress
curl localhost:1234/calibration/status

# requantize from the source weights with the collected statistics and hot-swap
curl -X POST localhost:1234/calibration/apply \
  -H "Content-Type: application/json" \
  -d '{"save_cimatrix": "traffic.cimatrix"}'
```

`status` reports how many layers are collecting and the token rows seen per layer. `apply`
harvests the statistics, requantizes (seconds for small models, about a minute for a 12B), and
returns the pre-apply status. The optional `save_cimatrix` writes the collected importance
matrix for reuse with `--imatrix`.

Collection costs nothing until started, and decode returns to full speed after `apply`.

## Requirements and behavior

- The model must have been loaded with `--isq` from source weights (safetensors). Models loaded
  `--from-uqff` have no source weights on disk and cannot apply.
- Importance weighting applies to the K-quant types (`Q2K`-`Q6K`); other types requantize from
  source without it, which still avoids double quantization.
- Layers whose weights cannot be re-read exactly (matformer slicing, rank-sharded fused expert
  halves) requantize from the resident weights instead, logged at apply time.
- MoE expert stacks are rebuilt from the checkpoint in any supported layout.

## See also

- [Quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/#imatrix) for imatrix
  background.
