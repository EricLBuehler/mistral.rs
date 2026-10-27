The Q8 SM121 matvec kernel in `kernels/mmvq_gguf/mmvq_q8_sm121.cu` adapts the
Q8 dot product, reduction, and GB10 L2 prefetch approach from llama.cpp commit
`7aca0d2ea8d92b56911233b822db01f9fbb1ae50`, principally
`ggml/src/ggml-cuda/mmvq.cu` and `ggml/src/ggml-cuda/vecdotq.cuh`.

The local specialization retains eight warps per row and prefetches one loop
iteration ahead, selected using cold-weight measurements on GB10.

`kernels/indexed_moe/moe_decode_sm121.cu` reuses the Q4_K, Q5_K, Q4_1, Q5_1, and Q8_0 dot products
from the same llama.cpp sources, matching the existing indexed MoE arithmetic.
Its GB10 specialization interleaves two rows per warp and prefetches one loop
iteration ahead, with eight warps per block.
