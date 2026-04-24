---
title: Environment variables
description: All environment variables mistralrs reads at build time or runtime.
sidebar:
  order: 15
---

Every environment variable `mistralrs` (or its build scripts) reads. Variables are grouped by purpose.

## Hugging Face

| Variable | Purpose |
|---|---|
| `HF_HOME` | Root of the Hugging Face cache. Default `~/.cache/huggingface`. |
| `HF_TOKEN` | Auth token. Overrides any token saved by `mistralrs login` at `$HF_HOME/token`. |
| `HF_HUB_CACHE` | Hugging Face hub cache location. |

## Logging

| Variable | Purpose |
|---|---|
| `RUST_LOG` | `tracing` log filter. Examples: `info`, `mistralrs_core=debug,tower_http=info`. |
| `MISTRALRS_DEBUG` | `MISTRALRS_DEBUG=1` enables extra debug-level engine tracing. |

## Quantization and loading

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_MMAP` | `MISTRALRS_NO_MMAP=1` loads safetensors without mmap. |
| `MISTRALRS_ISQ_SINGLETHREAD` | If set, runs ISQ quantization single-threaded. |

## Attention kernels

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_MLA` | `MISTRALRS_NO_MLA=1` disables the MLA-specific attention path for DeepSeek V2/V3. Generic attention is used instead. |

## Multi-GPU and multi-node

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_NCCL` | `MISTRALRS_NO_NCCL=1` disables NCCL. Falls back to the ring backend. |
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total world size across nodes. Presence of this variable enables multi-node mode. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Local TP size override on a single node. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Set on the head node: number of worker nodes. |
| `MISTRALRS_MN_HEAD_PORT` | Set on the head node: listening port for worker connections. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Set on worker nodes: address of the head node. |
| `MISTRALRS_MN_WORKER_ID` | Set on worker nodes: worker index (0-based). |

See the [multi-machine ring guide](/mistral.rs/guides/perf/multi-machine-ring/) for use.

## GPU memory

| Variable | Purpose |
|---|---|
| `MISTRALRS_IGPU_MEMORY_FRACTION` | Fraction of integrated GPU memory usable on CUDA systems with iGPUs. Default 0.75. |
| `CUDA_VISIBLE_DEVICES` | Standard NVIDIA variable. Restricts which GPUs mistralrs can see. |

## Build-time

These are read by build scripts, not at runtime.

| Variable | Purpose |
|---|---|
| `MISTRALRS_METAL_PRECOMPILE` | `MISTRALRS_METAL_PRECOMPILE=0` skips Metal kernel precompilation at build time; kernels are compiled at runtime on first use. |
| `MISTRALRS_GIT_REVISION` | Overrides the git revision embedded in the binary. |

## Internal

Not intended for direct use.

| Variable | Purpose |
|---|---|
| `MISTRALRS_DAEMON_INTERNAL` | Set by the engine on spawned worker processes. |
