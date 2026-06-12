---
title: Online calibration
description: Improve a served quantized model from its own traffic, with no restart.
sidebar:
  order: 13
---

Online calibration observes the activations of a live, ISQ-quantized model, then requantizes
every layer from the original weights using an importance matrix derived from that traffic. The
layers are hot-swapped in place with no restart: the model serves normally while collecting, and
requests received during the apply step queue until it finishes.

Quantized this way, a model is measurably closer to its full-precision outputs on the
distribution it actually serves, at the same bit width and speed.

## Usage

Serve any model with ISQ:

```bash
mistralrs serve -m <model> --isq q4k
```

Then drive the lifecycle over HTTP:

```bash
# begin observing live traffic (~15% decode overhead while on; on CUDA, MoE models
# additionally run their reference expert path during collection)
curl -X POST localhost:1234/calibration/start

# check per-layer collection progress
curl localhost:1234/calibration/status

# requantize from the source weights with the collected statistics and hot-swap
curl -X POST localhost:1234/calibration/apply \
  -H "Content-Type: application/json" \
  -d '{"save_cimatrix": "traffic.cimatrix"}'
```

`status` reports how many layers are collecting and the token rows seen per layer. `apply`
harvests the statistics, requantizes, and returns the pre-apply status. The optional
`save_cimatrix` writes the collected importance matrix for reuse with `--imatrix`.

Collection costs nothing until started, and decode returns to full speed after `apply`.

## Rust SDK

The same lifecycle is exposed on [`Model`](https://docs.rs/mistralrs/latest/mistralrs/struct.Model.html):

```rust
model.begin_calibration().await?;
// ... serve traffic ...
let status = model.calibration_status().await?;
model.apply_calibration(Some("traffic.cimatrix".into())).await?;
```

Each method has a `_with_model` variant for multi-model setups. See
[`examples/quantization/online_calibration`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/quantization/online_calibration/main.rs)
for a runnable example.

## Python SDK

```python
runner.begin_calibration()
# ... serve traffic ...
status = runner.calibration_status()
runner.apply_calibration(save_cimatrix="traffic.cimatrix")
```

`calibration_status` returns a `CalibrationStatus` with `collecting`, `layers`,
`layers_tracking`, `total_rows`, `min_rows`, and `max_rows` fields. See
[`examples/python/online_calibration.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/online_calibration.py).

## Requirements and behavior

- The model must have been loaded with `--isq` from source weights (safetensors); `start` errors
  otherwise (including models loaded `--from-uqff`).
- Importance weighting applies to the K-quant types (`Q2K`-`Q6K`). GGUF-family and AFQ types
  collect and requantize; HQQ and FP8 ISQ types do not support collection, so `start` errors.
- Pre-quantized source checkpoints (FP8, GPTQ, BnB) requantize from the resident weights, not
  the source files.
- Layers whose weights cannot be re-read exactly (matformer slicing, rank-sharded fused expert
  halves) requantize from the resident weights instead, logged at apply time.
- MoE expert stacks are rebuilt from the checkpoint in any supported layout.

## See also

- [Quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/#imatrix) for imatrix
  background.
