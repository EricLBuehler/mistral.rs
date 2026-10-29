The GGUF dot products in `kernels/gguf/vecdot.cuh` adapt llama.cpp commit
`7aca0d2ea8d92b56911233b822db01f9fbb1ae50`, principally
`ggml/src/ggml-cuda/mmvq.cu` and `ggml/src/ggml-cuda/vecdotq.cuh`.
The upstream MIT license is retained in [LICENSE](LICENSE).

Covered formats are Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K,
Q5_K, and Q6_K. The dense, fused, indexed-expert, and prefetching matvec
kernels in `kernels/mmvq_gguf` and `kernels/indexed_moe` call these adapted
helpers.
