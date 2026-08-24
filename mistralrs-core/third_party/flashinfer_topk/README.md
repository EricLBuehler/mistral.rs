# FlashInfer radix top-k attribution

The multi-CTA radix top-k implementation in `src/cuda/radix_topk.cuh` is adapted from:

- Repository: https://github.com/flashinfer-ai/flashinfer
- Revision: `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`
- Sources: `include/flashinfer/topk.cuh` and `include/flashinfer/topk_common.cuh`
- Upstream copyright: Copyright (c) 2024 by FlashInfer team
- License: Apache License 2.0

The adaptation retains the ordered-value radix selection, bounded persistent multi-CTA groups,
software group barriers, and last-CTA state reset. It only implements the fixed-width Basic top-k
mode, writes mistral.rs's packed F32 output directly, and strengthens equal-value ordering by
preferring lower token IDs. The persistent kernel uses a cooperative launch after checking its
dynamic-shared-memory occupancy; unsupported grids use the non-persistent fallback.
