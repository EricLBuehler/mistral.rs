---
title: Quantized execution
description: How FP8 and packed FP4 checkpoints run, and how to benchmark their kernels.
---

For loading commands and format constraints, use the [quantization reference](/reference/quantization-types/). This page describes backend behavior relevant to kernel development and numerical comparisons.

## FP8

Checkpoint adapters normalize weight names, scales, exclusions, and tensor-parallel shards into shared linear schemes. On cuTile builds, W8A16 keeps E4M3 weights resident and converts tiles to BF16/F16 inside the GEMM. Tensor/channel W8A8 uses CUDA activation quantizers followed by an FP8 GEMM. Block W8A8 has CUTLASS and cuTile providers.

When the accelerated tensor/channel path is unavailable, cached dequantized weights feed an A16 matmul. This fallback does not emulate activation quantize/dequantize rounding; numerical comparisons must account for the selected provider.

## MXFP8

Weights remain packed with unsigned E8M0 scales per 32 values. Prefill dynamically quantizes activation blocks with the same scaling convention. Single-token CUDA dense projections can preserve activation precision for decode. CUDA supplies matrix-vector and tensor-core paths; eligible Blackwell builds use native block-scaled tensor cores for prefill.

## NVFP4

Decode uses a dedicated matrix-vector kernel for small batches. Larger batches use tiled matrix multiplication. W4A4 activation quantization can be shared across projections only when normalized calibrated input scales match; each output row retains its own weight scale. W4A16 preserves input activation precision.

Eligible large dense W4A4 projections on SM 12.1 with CUDA 13.3 or newer use CUTLASS. Backend selection depends on hardware and dimensions. Fused gate/up paths preserve the unfused activation and product rounding for SiLU, ReLU, and sigmoid. Packed weights are made contiguous at load time, and kernels warm before CUDA graph capture.

Benchmark decode, prefill, and expert projections:

```bash
cargo run --release -p mistralrs-quant --features cuda,cutile --example nvfp4_bench -- --graph
```

Use `--suite decode`, `--suite dense`, or `--suite moe` to select workloads, `--f16` for FP16, and `--w4a16` to preserve activations. The benchmark checks outputs and reports median GPU and host latency as JSON. Repeated projections may benefit from GPU cache; measure full-model throughput separately.
