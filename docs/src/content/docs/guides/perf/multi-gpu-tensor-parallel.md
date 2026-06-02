---
title: Single-machine multi-GPU
description: NCCL tensor parallelism and layer/P2P mapping on one host.
sidebar:
  order: 7
---

This page covers one machine with multiple local GPUs. For the full mode comparison, start with [multi-GPU and distributed inference](/mistral.rs/guides/perf/multi-gpu-distributed/).

**Tensor parallelism** splits each layer across all GPUs and uses NCCL collectives to combine partial results. This is the preferred CUDA multi-GPU mode when the model supports it.

**Layer mapping** places different layer ranges on different devices. It is the fallback when NCCL is unavailable, disabled, or not suitable for the selected model. CUDA layer mapping enables peer access (P2P) for GPU pairs that support it; otherwise boundary activations are staged through CPU.

## Default selection

With no manual mapping flags:

1. One visible GPU runs the whole model on that GPU.
2. Multiple visible CUDA GPUs use NCCL tensor parallelism when the binary was built with `cuda nccl` and `MISTRALRS_NO_NCCL` is not set.
3. If NCCL is unavailable or disabled, mistral.rs uses layer mapping across the visible GPUs.

The selected layout is printed in the startup logs.

## Build requirements

Linux CUDA installs enable `nccl` when the installer or wheel builder finds `libnccl`.

Manual Linux CUDA build with NCCL:

```bash
cargo install mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

If NCCL is not installed, omit `nccl`:

```bash
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

To force the installer decision, use `MISTRALRS_INSTALL_NCCL=1` or `MISTRALRS_INSTALL_NO_NCCL=1`. To disable NCCL at runtime without rebuilding:

```bash
MISTRALRS_NO_NCCL=1 mistralrs serve -m Qwen/Qwen3-32B --quant 4
```

## Select GPUs

Use `CUDA_VISIBLE_DEVICES` to restrict the GPU set before mistral.rs starts:

```bash
CUDA_VISIBLE_DEVICES=0,1 mistralrs serve -m Qwen/Qwen3-32B --quant 4
```

The ordinals in `--device-layers` are the visible ordinals after `CUDA_VISIBLE_DEVICES` is applied.

NCCL tensor parallelism uses all visible CUDA GPUs. The tensor-parallel size must be compatible with the model:

- Attention heads must divide evenly across GPUs.
- KV heads must either divide evenly across GPUs or be replicated evenly when there are fewer KV heads than GPUs.

If the visible GPU count is incompatible, mistral.rs errors instead of selecting a smaller subset.

Use `CUDA_VISIBLE_DEVICES` to choose a compatible subset.

## Manual layer mapping

`-n`/`--device-layers` assigns layer counts to devices. Format:

```bash
mistralrs serve -n "0:32;1:32" -m <model>
```

For uneven GPUs, put fewer layers on the smaller or busier GPU:

```bash
mistralrs serve -n "0:44;1:20" -m Qwen/Qwen3-32B --quant 4
```

For per-layer or per-tensor placement, use the [topology guide](/mistral.rs/guides/perf/topology/).

## Performance notes

Use NCCL when possible for single-machine CUDA tensor parallelism. It keeps collective communication on the GPU path and is the expected path for multiple similar GPUs.

Layer mapping moves activations only at layer-boundary device changes, so contiguous ranges matter. On CUDA, peer access is enabled for supported GPU pairs. If the driver reports that peer access is unavailable or cannot be enabled, those transfers stage through CPU and startup logs include a warning.

For mixed GPU memory sizes, manually set `--device-layers`; the automatic split does not optimize for heterogeneous memory or PCIe/NVLink topology.

## Next

For tensor parallelism across machines, use [multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/). For the ring transport, use the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/).
