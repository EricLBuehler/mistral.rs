# FlashInfer GDN attribution

The cooperative and pipelined K-major GDN decode kernels in `src/cuda/gdn.cu` are adapted from:

- Repository: https://github.com/flashinfer-ai/flashinfer
- Revision: `4927c0e15cb63a2abb6df09019c39a172222f0eb`
- Source: `flashinfer/gdn_kernels/gdn_decode_nontranspose.py`
- Upstream copyright: Copyright (c) 2025 by FlashInfer team
- License: Apache License 2.0, reproduced in `LICENSE-APACHE`

The adapted implementation preserves the K-major state tiling, cooperative K-lane reduction,
128-bit asynchronous state loads, padded shared-memory layouts, and vector state writeback design.
The large-workload variant also preserves the upstream 32-value tile, 256-thread block, two-stage
pipeline, and one-block-per-state traversal. Both are AOT CUDA kernels with no runtime dependency
on FlashInfer or CuTe DSL.

The fused speculative value-major GDN recurrence, RMSNorm, and SiLU gate kernel is adapted from:

- Repository: https://github.com/vllm-project/vllm
- Revision: `c8438a3d40168ce1d9eade0dc15ccbe5d27adb68`
- Source: `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`
- Upstream copyright: Copyright contributors to the vLLM project
- License: Apache License 2.0, reproduced in `LICENSE-APACHE`
