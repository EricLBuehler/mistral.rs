# ggml GDN attribution

The CUDA tensor-core GDN prefill kernel in `src/cuda/gdn_mma.cu` is adapted from:

- Repository: https://github.com/ggml-org/llama.cpp
- Revision: `7aca0d2ea8d92b56911233b822db01f9fbb1ae50`
- Sources: `ggml/src/ggml-cuda/gated-delta-net-mma.cu` and `ggml/src/ggml-cuda/mma.cuh`
- Upstream copyright: Copyright (c) 2023-2026 The ggml authors
- License: MIT, reproduced in `LICENSE`

The adaptation retains the persistent 16-token recurrence, FP32 state, FP32
triangular solve, and BF16 high/residual tensor-core products. The prepared-input
variant consumes normalized, head-expanded inputs. The fused variant reads BF16
projections directly and computes Q/K normalization and gates within the same
kernel. Both write BF16 token-major output with in-place updates to gathered or
pooled value-major state.
