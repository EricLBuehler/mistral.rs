---
title: Device mapping
description: How mistralrs decides where to put a model across available GPUs, and when you should override it.
sidebar:
  order: 7
---

Model loading requires deciding where each layer goes. With one GPU, the choice is trivial. With several, the choice affects throughput, latency, and whether the model fits at all.

## The single-GPU case

Every layer goes on the GPU. Embedding tables, attention weights, MLP weights, LM head. Covers most usage.

If the model does not fit, fallback options apply: CPU offload for some layers, disk-based mapping for very large models. Both are slow and reserved for cases where the alternative is not running the model at all.

## Multi-GPU: the layouts

Three layouts are possible.

**Tensor parallelism.** Every layer is split across all GPUs. Each GPU holds a portion of every layer's weights and computes a portion of every matrix multiply. An all-reduce per layer combines partial results. Default with multiple GPUs on one machine and NCCL available.

**Pipeline parallelism.** Each GPU holds a contiguous range of layers. Activations flow GPU 0 → GPU 1 → GPU 2 sequentially. Used when tensor parallelism is impractical (slow interconnect) or when the model does not divide cleanly across the GPU count.

**Layer-level placement.** Each layer is assigned to a specific GPU manually. Used for unusual hardware (mixed GPU sizes) or specific optimization purposes.

## Auto-detection

`mistralrs run -m <model>` with no device mapping flags:

1. Counts available CUDA (or Metal) devices.
2. With one device, everything goes there.
3. With multiple on one machine, tensor parallelism is the default with equal layer splits.
4. If tensor parallelism is unavailable (model has unusual dimensions, NCCL missing), the engine falls back to pipeline parallelism.

The chosen layout is reported in startup logs at `INFO`. Verify with `RUST_LOG=info mistralrs run ...`.

## When to override

Cases for manual mapping:

**Uneven GPU memory.** A 24 GB and 16 GB GPU pair (common in home setups) cannot use equal splitting on the smaller GPU. Use `-n`/`--device-layers` (format `0:N1;1:N2`) to put fewer layers there.

**Sharing a machine.** When other processes use one of the GPUs, mark those layers unusable and put everything on the free GPU. `CUDA_VISIBLE_DEVICES` hides the busy GPU from mistral.rs entirely.

**Specific performance needs.** Research workloads sometimes place specific layers on specific devices for performance isolation. See the [topology guide](/mistral.rs/guides/perf/topology/).

For most users, none of this applies.

## What auto-detection does not handle well

**Heterogeneous hardware.** CUDA GPUs of different generations or memory sizes can work together, but mapping is more complex. Auto-detection splits evenly, wasting the larger GPU's capacity.

**Cross-machine setups.** NCCL-based tensor parallelism is single-machine only. For multiple machines, see the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).

**NUMA effects.** Multi-socket servers with GPUs on different sockets pay a cross-socket transfer penalty. Auto-detection does not optimize for this; it uses any visible GPU regardless of topology.

## Metal-specific notes

Apple Silicon has no multi-GPU concept. CPU and GPU share unified memory; device mapping is trivial — everything is on the single unified device.

`mistralrs doctor` reports a single device on Apple hardware regardless of CPU/GPU distinction. The engine handles CPU vs GPU kernel placement at a lower level.

## The relationship to `--dtype`

Device mapping and dtype are orthogonal. A model can be loaded in bf16 split across two GPUs, or quantized to 4-bit on a single GPU, or any other combination.

CPU offload changes the effective dtype for offloaded layers. CPU lacks the bf16/fp16 hardware support modern GPUs have, so CPU layers often run at f32 internally even with bf16 on-disk weights. Slower than expected. With CPU offload, use `--isq` to keep the offloaded portion small.

## Observability

Startup logs report the chosen layout at `INFO` level. When a topology file is in use, the logs list the device for each layer. `nvidia-smi` shows per-GPU memory at runtime.

## Summary

Auto-detection handles the common case. Manual overrides (`--device-layers`, `--topology`, `CUDA_VISIBLE_DEVICES`) cover the rest. For anything beyond splitting across GPUs, the topology file is the tool.

## See also

- Guide: [split a model across multiple GPUs](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/), [configure model topology](/mistral.rs/guides/perf/topology/).
