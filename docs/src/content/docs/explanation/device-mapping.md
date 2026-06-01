---
title: Device mapping
description: How mistralrs decides where to put a model across available GPUs, and when you should override it.
sidebar:
  order: 7
---

Device mapping controls where each model layer is placed across available GPUs.

## Single GPU

Every layer goes on the GPU: embedding tables, attention weights, MLP weights, LM head.

If the model does not fit, CPU offload places some layers on CPU, and disk-based mapping handles very large models. Both are significantly slower than GPU.

## Multi-GPU layouts

**Tensor parallelism.** Every layer is split across all GPUs. Each GPU holds a portion of every layer's weights and computes a portion of every matrix multiply. An all-reduce per layer combines partial results. Default with multiple local CUDA GPUs and NCCL available.

**Pipeline parallelism.** Each GPU holds a contiguous range of layers. Activations flow from GPU 0 to GPU 1 to GPU 2 sequentially. Used when tensor parallelism is unavailable or when the model does not divide evenly across the GPU count. CUDA transfers use peer access when available and stage through CPU otherwise.

**Layer-level placement.** Each layer is assigned to a specific GPU manually.

## Auto-detection

`mistralrs run -m <model>` with no device mapping flags:

1. Counts available CUDA (or Metal) devices.
2. With one device, everything goes there.
3. With multiple on one machine, tensor parallelism is the default with equal layer splits.
4. If tensor parallelism is unavailable (model has unusual dimensions, NCCL missing), the engine falls back to pipeline parallelism.

The chosen layout is reported in the default startup logs.

## Manual overrides

Cases for manual mapping:

**Uneven GPU memory.** A 24 GB and 16 GB GPU pair (common in home setups) cannot use equal splitting on the smaller GPU. Use `-n`/`--device-layers` (format `0:N1;1:N2`) to put fewer layers there.

**Sharing a machine.** When other processes use one of the GPUs, mark those layers unusable and put everything on the free GPU. `CUDA_VISIBLE_DEVICES` hides the busy GPU from mistral.rs entirely.

**Specific performance needs.** Per-layer device placement for performance isolation. See the [topology guide](/mistral.rs/guides/perf/topology/).

## Auto-detection limitations

**Heterogeneous hardware.** CUDA GPUs of different generations or memory sizes can work together, but mapping is more complex. Auto-detection splits evenly, wasting the larger GPU's capacity.

**Cross-machine setups.** Multi-node NCCL and the ring backend are separate distributed modes. See [multi-GPU and distributed inference](/mistral.rs/guides/perf/multi-gpu-distributed/).

**NUMA effects.** Multi-socket servers with GPUs on different sockets pay a cross-socket transfer penalty. Auto-detection does not optimize for this; it uses any visible GPU regardless of topology.

## Metal-specific notes

Apple Silicon has no multi-GPU concept. CPU and GPU share unified memory; device mapping is a no-op.

`mistralrs doctor` reports a single device on Apple hardware regardless of CPU/GPU distinction. The engine handles CPU vs GPU kernel placement at a lower level.

## Interaction with `--dtype`

Device mapping and dtype are orthogonal. A model can be loaded in bf16 split across two GPUs, or quantized to 4-bit on a single GPU, or any combination.

CPU offload changes the effective dtype for offloaded layers. CPU lacks bf16/fp16 hardware support, so CPU layers run at f32 internally even with bf16 on-disk weights.

## Observability

Startup logs report the chosen layout at `INFO` level. When a topology file is in use, the logs list the device for each layer. `nvidia-smi` shows per-GPU memory at runtime.

## See also

- Guide: [split a model across multiple GPUs](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/), [configure model topology](/mistral.rs/guides/perf/topology/).
