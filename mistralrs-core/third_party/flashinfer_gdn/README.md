# FlashInfer GDN attribution

The cooperative K-major GDN decode kernel in `src/cuda/gdn.cu` is adapted from:

- Repository: https://github.com/flashinfer-ai/flashinfer
- Revision: `4927c0e15cb63a2abb6df09019c39a172222f0eb`
- Source: `flashinfer/gdn_kernels/gdn_decode_nontranspose.py`
- Upstream copyright: Copyright (c) 2025 by FlashInfer team
- License: Apache License 2.0, reproduced in `LICENSE-APACHE`

The adapted implementation preserves the K-major state tiling, cooperative K-lane reduction,
asynchronous vector state load, padded shared-memory layout, and vector state writeback design.
It is implemented as an AOT CUDA kernel and has no runtime dependency on FlashInfer or CuTe DSL.
