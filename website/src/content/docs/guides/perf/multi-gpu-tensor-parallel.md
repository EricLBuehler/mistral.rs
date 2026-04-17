---
title: Split a model across multiple GPUs
description: Tensor parallelism with NCCL on a single machine. Useful for models too large for any single GPU in your box.
sidebar:
  order: 5
---

When a model does not fit on one GPU after quantization, mistral.rs can split it across multiple GPUs on the same machine via NCCL tensor parallelism. Each layer's weights are sharded across devices and matrix multiplies run in parallel with an all-reduce per layer.

Use it when:

- Multiple GPUs are present on one host.
- The model is large enough that splitting is necessary or faster than fitting it on one card with memory pressure.
- The PCIe topology is reasonable. NVLink helps noticeably on Hopper.

For cross-machine splitting, see the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).

## Prerequisites

NCCL is the NVIDIA Collective Communications Library. On Linux it is usually present with the CUDA toolkit. Verify:

```bash
ldconfig -p | grep libnccl
```

If `libnccl.so` is missing, install via package manager (`libnccl2` on Debian/Ubuntu, `libnccl` on Fedora) or NVIDIA directly.

On WSL2, NCCL is supported but slower due to Windows host PCIe virtualization overhead. Native Linux is meaningfully better for multi-GPU work.

## Basic usage

The CLI detects available GPUs and splits automatically when more than one is present:

```bash
mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

A two-GPU host uses both; an eight-GPU host uses all eight. To restrict the device set, use the CUDA convention:

```bash
CUDA_VISIBLE_DEVICES=0,1 mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

Sharding is uniform by default. Layer counts and tensor dimensions not divisible by GPU count use padding; odd GPU counts are supported.

## Forcing specific device behavior

`--num-device-layers` specifies layers per GPU, useful for asymmetric memory:

```bash
# Two GPUs, more memory on the first
mistralrs serve --num-device-layers "32;16" -m <model>
```

Semicolon-separated, one entry per GPU.

For per-tensor or per-layer placement, see the [topology guide](/mistral.rs/guides/perf/topology/).

## When tensor parallelism does not help

- **Small models.** A 7B model runs fine on one GPU. Splitting adds communication overhead without saving memory.
- **Single-request latency.** Tensor parallelism helps throughput more than latency. A smaller model on one GPU is sometimes faster than a larger one split across two.
- **Slow interconnect.** Without NVLink, inter-GPU bandwidth is the bottleneck. PCIe 4.0 x16 is usable; PCIe 3.0 x8 is painful.

## Observability

Startup logs report per-GPU layer ranges:

```
[tensor-parallel] GPU 0: layers 0-31
[tensor-parallel] GPU 1: layers 32-63
```

`mistralrs doctor` reports total memory across visible GPUs. For per-GPU memory, use `nvidia-smi`.

## What to read next

- [Ring backend](/mistral.rs/guides/perf/multi-machine-ring/) — splitting across machines.
- [Topology](/mistral.rs/guides/perf/topology/) — manual per-layer placement.
