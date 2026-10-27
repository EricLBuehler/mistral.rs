The GGUF dot products in `kernels/gguf/vecdot.cuh` adapt llama.cpp commit
`7aca0d2ea8d92b56911233b822db01f9fbb1ae50`, principally
`ggml/src/ggml-cuda/mmvq.cu` and `ggml/src/ggml-cuda/vecdotq.cuh`.
The upstream MIT license is retained in [LICENSE](LICENSE).
Dense projections, indexed experts, prefetched matvecs, and gated projection
mixing share these helpers. Supported formats are Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K; individual dispatchers retain
their existing format capabilities.

`kernels/mmvq_gguf/prefetch_row.cuh` uses eight warps per row and prefetches
one loop iteration ahead. On SM121, dense dispatch uses packed weight size,
block density, and reduction length to retain multi-row geometry for shapes
dominated by unpacking cost. Fused QKV projections reuse the same row helper
with runtime output dimensions. Mixed GGUF/dense projection pairs share a
launch while retaining each projection's arithmetic; their prefetch choice
depends on packed row size and device occupancy.

`kernels/indexed_moe/moe_decode_prefetch.cu` interleaves two rows per warp,
prefetches one iteration ahead, and uses eight warps per block. Its tensor
shapes, expert indices, routing count, and activation are runtime parameters.
For single-token inputs on SM121, expert banks larger than the device's L2
cache can use cooperative row staging. Copy alignment selects eight- or
sixteen-row panels, bounded to 24 KiB per operand. Cache-fitting banks,
unsupported alignment, and larger panels retain the existing kernel path.
