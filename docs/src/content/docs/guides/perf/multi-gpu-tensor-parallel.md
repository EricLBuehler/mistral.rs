---
title: Split a model across multiple GPUs
description: Multi-GPU layer placement on a single machine.
sidebar:
  order: 5
---

When a model exceeds one GPU's memory after quantization, mistral.rs can split it across multiple GPUs on the same host.

## Auto-detection

The CLI detects available GPUs and splits across them when more than one is present:

```bash
mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

To restrict the device set, use the CUDA convention:

```bash
CUDA_VISIBLE_DEVICES=0,1 mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

## Per-device layer counts

`-n`/`--device-layers` specifies layers per GPU. Format: `ORD:NUM;ORD:NUM;...`.

```bash
mistralrs serve -n "0:32;1:32" -m <model>
```

For per-tensor or per-layer placement, see the [topology guide](/mistral.rs/guides/perf/topology/).

## Multi-machine

For cross-machine splitting, see the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).
