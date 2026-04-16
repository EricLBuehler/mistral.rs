---
title: Split a model across multiple GPUs
description: Tensor parallelism with NCCL on a single machine. Useful for models too large for any single GPU in your box.
sidebar:
  order: 5
---

If a model does not fit on one GPU even after quantization, you can split it across several GPUs on the same machine. mistral.rs supports tensor parallelism through NCCL, which shards every layer's weights across the available devices and runs matrix multiplies in parallel with an all-reduce at the end of each layer.

This is the right tool when:

- You have multiple GPUs on the same host.
- The model is big enough that splitting it is either necessary (it does not fit on any one card) or faster (memory pressure on one card would push generation toward slow paths).
- You are not constrained by NVLink topology; any reasonable PCIe setup works, though NVLink helps noticeably on Hopper.

For cross-machine splitting, see the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).

## Prerequisites

NCCL is the NVIDIA Collective Communications Library. On Linux it is almost always already present if you have the CUDA toolkit installed. Verify with:

```bash
ldconfig -p | grep libnccl
```

If you do not see `libnccl.so`, install it from your package manager (`libnccl2` on Debian/Ubuntu, `libnccl` on Fedora) or from NVIDIA directly.

On WSL2, NCCL is supported but can be slower due to the Windows host's PCIe virtualization overhead. For serious multi-GPU work, a native Linux install is a meaningful upgrade.

## Basic usage

The CLI detects available GPUs and splits the model automatically when you have more than one:

```bash
mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

On a two-GPU box, that command uses both cards. On an eight-GPU box, it uses all eight. If you want to restrict it to a specific subset, use the CUDA convention:

```bash
CUDA_VISIBLE_DEVICES=0,1 mistralrs serve -m Qwen/Qwen3-32B --isq 4
```

The engine shards the model equally by default. Layers and tensor dimensions that are not divisible by the GPU count are handled with padding, so odd GPU counts work correctly.

## Forcing specific device behavior

Sometimes you want more control than "use everything." The `--num-device-layers` flag lets you specify how many layers to place on each GPU, useful when the GPUs have different memory sizes:

```bash
# Two GPUs, but the first has more free memory
mistralrs serve --num-device-layers "32;16" -m <model>
```

The semicolon-separated list has one entry per GPU. Each entry is the number of layers to assign to that GPU.

For more fine-grained control (individual tensor placements, specific layer offsets) the [topology guide](/mistral.rs/guides/perf/topology/) covers the config file option.

## When tensor parallelism does not help

A few situations where splitting is not the right move:

- **Small models.** A 7B model runs fine on a single GPU. Splitting it adds communication overhead without saving memory, so you lose throughput.
- **Single-request latency workloads.** Tensor parallelism helps most at throughput, not latency. If you are running one request at a time and every token matters, a smaller model on one GPU is sometimes faster than a larger model split across two.
- **Slow interconnect.** Without NVLink, the inter-GPU bandwidth becomes the bottleneck. PCIe 4.0 x16 is usable; PCIe 3.0 x8 is painful. The engine will still work, but the speedup shrinks.

## Observability

When tensor parallelism is active, the logs at startup include a line for each GPU and the layer range it owns:

```
[tensor-parallel] GPU 0: layers 0-31
[tensor-parallel] GPU 1: layers 32-63
```

`mistralrs doctor` reports total memory across all visible GPUs rather than per-GPU. For per-GPU memory, `nvidia-smi` is still the tool you want.

## What to read next

- [Ring backend](/mistral.rs/guides/perf/multi-machine-ring/) for splitting across machines.
- [Topology](/mistral.rs/guides/perf/topology/) for manual per-layer placement.
