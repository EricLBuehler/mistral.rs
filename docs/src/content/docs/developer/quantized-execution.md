---
title: Quantized execution
description: How FP8 and packed FP4 checkpoints run, and how to benchmark their kernels.
---

This page explains which CUDA kernels run FP8, MXFP8, and NVFP4 checkpoints. It's for contributors working on those kernels or comparing outputs against a reference. To load a quantized model, see the [quantization reference](/reference/quantization-types/).

In this page, W8A16 means 8-bit weights with 16-bit (BF16 or F16) activations, and W8A8 means activations are quantized to 8 bits too. W4A16 and W4A4 are the 4-bit equivalents.

## FP8

The checkpoint loader in `mistralrs-quant/src/fp8_config.rs` reads native, compressed-tensors, and NVIDIA ModelOpt FP8 checkpoints. It maps their weight names, scales, excluded layers, and tensor-parallel shards onto one shared set of linear-layer schemes.

Layers with per-tensor or per-channel weight scales (`pertensor_fp8/mod.rs`) try these kernels in order, using the first one that supports the shape:

1. cuTile W8A16, for checkpoints without activation quantization. E4M3 weights stay on the GPU, and each tile is converted to BF16 or F16 inside the GEMM.
2. cuTile W8A8, for checkpoints with static per-tensor or dynamic per-token activation scales. Activations are quantized to FP8 on the GPU, then an FP8 GEMM runs.
3. A CUDA W8A16 kernel, on compute capability 8.0 and newer.
4. A fallback that dequantizes the weight once, caches it, and runs a regular BF16 or F16 matmul.

Only the cuTile W8A8 kernel quantizes activations. For a W8A8 checkpoint, kernels 3 and 4 run the activations in BF16 or F16 and skip that rounding. Their outputs differ slightly from the cuTile W8A8 kernel, so check which kernel ran before comparing numbers.

Block-scaled W8A8 checkpoints (`blockwise_fp8/mod.rs`) pick a kernel once at load. On Hopper they use DeepGEMM when it's built in, otherwise CUTLASS. On other GPUs they use a tensor-core matrix-vector kernel for small batches and the cuTile FP8 GEMM for larger ones.

## MXFP8

MXFP8 weights stay packed as E4M3 values with one unsigned E8M0 scale per 32 values (`mxfp8/mod.rs`).

- **Prefill and other multi-row inputs:** activations are quantized into 32-value blocks with the same E8M0 scaling. On Blackwell with cuTile and CUDA 13.3 or newer, the GEMM runs on native block-scaled tensor cores (`cutile/mxfp8_gemm.rs`). Otherwise it uses the CUDA matrix-vector and tensor-core kernels in `mxfp8/cuda.rs`.
- **Single-row input on CUDA with K of 512 or more**, which covers most decode steps: activation quantization is skipped and an A16 matrix-vector kernel runs, so decode keeps full activation precision.

## NVFP4

NVFP4 has two kernel families (`nvfp4/mod.rs`):

- **Native**, on Blackwell with cuTile, CUDA 13.3 or newer, and `tileiras` 13.3 or newer. W4A4 checkpoints quantize activations to FP4 and run on FP4 tensor cores.
- **Portable** (`nvfp4/portable.rs`), on compute capability 8.0 and newer. Used everywhere else, including builds without cuTile. W4A4 checkpoints run as W4A16 here, so activations stay in BF16 or F16.

On the native path, a dense layer picks a kernel per call:

1. A single token uses a dedicated matrix-vector kernel (`nvfp4/decode.rs`). Expert layers also use it for small token counts.
2. Large W4A4 projections use CUTLASS (`nvfp4/cutlass.rs`). This needs a Linux binary built with cuTile and CUDA 13.3 or newer for SM 12.0 or 12.1, and it runs only on the compute capability it was built for. N and K must each be at least 4096, with at least 1024 rows. Decode batches of 2 to 64 rows also qualify, but only when the layer's weights are larger than the GPU's L2 cache.
3. Everything else uses the cuTile tiled GEMM.

Other details:

- W4A16 checkpoints keep input activations in BF16 or F16.
- Projections that read the same input, such as an expert stack's gate and up projections, quantize that input once only when their calibrated input scales match. Each output row keeps its own weight scale.
- The fused gate/up kernel (`cutile/nvfp4_glu.rs`) rounds the activation and the product at the same points as the unfused path, for SiLU, ReLU, and sigmoid.
- Packed weights are made contiguous at load time. CUTLASS kernels are prepared when the layer is built, which must happen before CUDA graph capture.

### Benchmark

To time decode, prefill, and expert projections:

```bash
cargo run --release -p mistralrs-quant --features cuda,cutile --example nvfp4_bench -- --graph
```

| Flag | Effect |
|---|---|
| `--suite decode`, `--suite dense`, `--suite moe` | Run one group of shapes. The default runs all of them. |
| `--f16` | Use F16 activations instead of BF16. |
| `--w4a16` | Keep activations unquantized (W4A16) instead of W4A4. |
| `--graph` | Capture the launches in a CUDA graph and time the replay. |

The benchmark calls the cuTile NVFP4 entry points (`cutile_nvfp4`, `cutile_nvfp4_gather`) directly, checks every output value, and prints one JSON line per shape with median GPU and host latency. Other flags are parsed at the top of `mistralrs-quant/examples/nvfp4_bench.rs`.

Every launch reuses the same weights, so they stay in the GPU's L2 cache. The JSON marks this as `"weight_cache": "warm"`. A real model reads different weights in each layer, so measure full-model throughput separately.
