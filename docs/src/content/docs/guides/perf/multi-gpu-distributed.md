---
title: Multi-GPU and distributed inference
description: Choose between NCCL tensor parallelism, layer/P2P mapping, multi-node NCCL, and the ring backend.
sidebar:
  order: 6
---

mistral.rs has four ways to spread one model beyond a single GPU. They solve different problems and use different communication paths.

| Mode | Scope | Transport | Use when |
|---|---|---|---|
| NCCL tensor parallelism | One machine, multiple CUDA GPUs | NCCL collectives | You have similar local GPUs and want the default high-throughput CUDA path. |
| Layer mapping with CUDA P2P | One machine, multiple devices | CUDA peer access when available, CPU staging otherwise | NCCL is unavailable, disabled, or the model/layout needs contiguous layer placement. |
| Multi-node NCCL | Multiple machines, each with local CUDA GPUs | NCCL across all ranks, with mistral.rs coordinating node startup | You want tensor parallelism across machines and each node contributes one or more local CUDA ranks. |
| Ring backend | Multiple machines | mistral.rs ring transport from `RING_CONFIG` | You explicitly want the ring backend instead of NCCL. |

## Selection Flow

With a CUDA build and no manual mapping:

1. One visible GPU runs the model on that GPU.
2. Multiple visible GPUs use NCCL tensor parallelism when the binary has `cuda nccl` and `MISTRALRS_NO_NCCL` is not set.
3. If NCCL is unavailable or disabled, mistral.rs uses layer mapping. CUDA pairs use P2P when the driver allows it; otherwise transfers stage through CPU.
4. If `MISTRALRS_MN_GLOBAL_WORLD_SIZE` is set, NCCL tensor parallelism is extended across nodes.
5. If `RING_CONFIG` is set and the binary has `ring`, the ring backend is available. If the binary also has NCCL, set `MISTRALRS_NO_NCCL=1` to force ring.

## Start Here

For one machine with multiple CUDA GPUs, start with [single-machine multi-GPU](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/).

For multiple machines using NCCL tensor parallelism, use [multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/).

For multiple machines using the ring backend, use [ring backend inference](/mistral.rs/guides/perf/multi-machine-ring/).

For exact layer or tensor placement on one host, use the [topology guide](/mistral.rs/guides/perf/topology/).
