# `mistralrs-core`

Core crate of `mistral.rs` including the models and associated executors.

Documentation: https://docs.rs/mistralrs-core/

## Features

- `cuda` — NVIDIA GPU acceleration via CUDA
- `cudnn` — cuDNN backend (requires `cuda`)
- `metal` — Apple Metal GPU acceleration (macOS)
- `accelerate` — Apple Accelerate framework for CPU BLAS (macOS)
- `mkl` — Intel MKL acceleration
- `flash-attn` — FlashAttention V2 (requires `cuda`, CC >= 8.0)
- `flash-attn-v3` — FlashAttention V3 (requires `cuda`, CC >= 9.0)
- `nccl` — Multi-GPU tensor parallelism via NCCL (requires `cuda`)
- `ring` — Distributed inference via TCP ring topology
- `kvcache-compression` — TurboQuant KV cache compression
- `mlx` — MLX-accelerated KV cache compression for macOS Apple Silicon (implies `kvcache-compression`)
