---
title: Device mapping
description: How mistralrs decides where to put a model across available GPUs, and when you should override it.
sidebar:
  order: 7
---

When a model loads, mistralrs has to decide where to put each layer. With one GPU, that decision is trivial. With several, the choice affects throughput, latency, and whether the model fits at all. This page covers how the default mapping works and when overriding it makes sense.

## The single-GPU case

Every layer goes on the GPU. Embedding tables, attention weights, MLP weights, the LM head. This is the simple case and covers most usage.

If the model does not fit on one GPU, we move down the list: CPU offload for some layers, disk-based mapping for truly massive models. These work but are slow; they are last resorts when the alternative is not running the model at all.

## Multi-GPU: the layouts

With multiple GPUs, three layouts are possible.

**Tensor parallelism**: every layer is split across all GPUs. Each GPU holds a portion of every layer's weights and does a portion of every matrix multiply. At the end of each layer, an all-reduce combines partial results. This is the default when you have multiple GPUs on one machine with NCCL available.

**Pipeline parallelism**: each GPU holds a contiguous range of layers. Activations flow from GPU 0 through GPU 1 through GPU 2 sequentially. Used when tensor parallelism is not practical (slow interconnect) or when the model does not divide cleanly across the GPU count.

**Layer-level placement**: each layer is assigned to a specific GPU, with manual control over which. Used for unusual hardware (mixed GPU sizes) or for specific optimization purposes.

## Auto-detection

`mistralrs run -m <model>` with no device mapping flags:

1. Counts available CUDA (or Metal) devices.
2. If there is one, everything goes there.
3. If there are multiple on the same machine, tensor parallelism is used by default. Layers are split equally.
4. If tensor parallelism will not work (model has unusual dimensions, NCCL is missing), the engine falls back to pipeline parallelism.

The engine reports which layout was chosen in the startup logs, at `INFO` level. You can verify with `RUST_LOG=info mistralrs run ...`.

## When to override

A few situations call for manual mapping:

**Uneven GPU memory.** If you have one 24 GB GPU and one 16 GB GPU (common in home setups), equal splitting fails the smaller GPU. Use `--num-device-layers` to put fewer layers on the smaller one.

**Sharing a machine.** If other processes are using one of your GPUs, you want to mark those layers as unusable and put everything on the free GPU. `CUDA_VISIBLE_DEVICES` hides the busy GPU from mistralrs entirely.

**Specific performance needs.** Research workloads sometimes want to place specific layers on specific devices to isolate their performance. The [topology guide](/mistral.rs/guides/perf/topology/) covers this in detail.

For most users, none of this applies. The defaults are reasonable.

## What auto-detection does not handle well

**Heterogeneous hardware.** CUDA GPUs of different generations or different memory sizes can all work together, but the mapping is more complicated. Auto-detection will usually just split evenly, which wastes the bigger GPU's capacity.

**Cross-machine setups.** NCCL-based tensor parallelism is single-machine only. For multiple machines, see the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).

**NUMA effects.** On multi-socket servers with GPUs attached to different CPU sockets, there is a penalty for cross-socket data transfer. Auto-detection does not try to optimize for this; it will use any GPU it sees regardless of topology.

## Metal-specific notes

On Apple Silicon, there is no multi-GPU concept to speak of. The CPU and the GPU share unified memory, and the device mapping decision is trivial: everything is "on" the single unified device.

The implication is that `mistralrs doctor` reports a single device for Apple hardware, regardless of the CPU or GPU distinction. The engine makes its own decisions about which kernels run on CPU versus GPU at a lower level.

## The relationship to `--dtype`

Device mapping and dtype are orthogonal decisions. You can load a model in bf16 split across two GPUs, or quantized to 4-bit on a single GPU, or any other combination.

One thing worth knowing: CPU offload changes the effective dtype for offloaded layers. CPU does not have the bf16/fp16 hardware support that modern GPUs do, so CPU layers often run at f32 internally even if the on-disk weights are bf16. This is slower than you might expect. If CPU offload is in play, use `--isq` to keep the offloaded portion small.

## Observability

The logs at startup document the mapping:

```
[device] GPU 0: layers 0-31 (CUDA)
[device] GPU 1: layers 32-63 (CUDA)
```

Or for tensor parallelism:

```
[device] tensor-parallel: layers 0-63 sharded across 2 GPUs
```

At runtime, `nvidia-smi` shows per-GPU memory use. If one GPU is much fuller than another, the mapping is uneven. If one GPU is idle during inference, the parallelism scheme is not working as expected (usually a fallback to single-GPU).

## Summary

The mental model: auto-detection handles the common case. Manual overrides (`--num-device-layers`, `--topology`, `CUDA_VISIBLE_DEVICES`) are available for the uncommon cases. For anything beyond "split across these GPUs," the topology file is the tool.
