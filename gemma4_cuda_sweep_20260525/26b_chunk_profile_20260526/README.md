# Gemma 4 26B-A4B Q8 Chunked Prompt Profile

Scope: mistral.rs 26B-A4B Q8 prompt with default paged prefill chunking at 4096.

NCU replay makes wall-clock T/s invalid. Use kernel-time totals and proportions only.

## Commands

```bash
MISTRALRS_CUDA_GRAPHS=0 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/26b_chunk_profile_20260526/mistral_26b_a4b_q8_p8192_chunk4096.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 8192 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

MISTRALRS_CUDA_GRAPHS=0 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/26b_chunk_profile_20260526/mistral_26b_a4b_q8_p16384_chunk4096.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 16384 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

## Kernel Groups

| Group | p8192 no chunk | p8192 chunk 4096 | p16384 chunk 4096 |
| --- | ---: | ---: | ---: |
| Q8 MMQ matmul | 1330.6 | 1327.0 | 2685.7 |
| attention / rope / KV | 536.1 | 359.1 | 1034.3 |
| activation quantize | 188.6 | 171.9 | 344.8 |
| norms | 168.0 | 179.2 | 357.8 |
| MoE routing / reduce / layout | 115.7 | 116.8 | 227.5 |
| copy / cast | 67.4 | 115.6 | 249.8 |
| elementwise / GLU | 86.5 | 89.0 | 176.5 |
| Q8 matvec / lm-head | 16.5 | 20.3 | 28.2 |
| dense GEMM | 9.8 | 11.1 | 20.1 |
| other | 9.0 | 9.5 | 10.8 |
| total GPU kernel time | 2528.1 | 2399.7 | 5135.5 |

## Top Kernels

### p8192 chunk 4096

| Time | Launches | Kernel |
| ---: | ---: | --- |
| 688.0 | 350 | `mul_mat_q<8, 128, 0>` |
| 639.0 | 240 | `mul_mat_q<8, 128, 1>` |
| 139.2 | 5 | `flashinfer::BatchPrefillWithPagedKVCacheKernel<..., 32, 32, ...>` |
| 136.8 | 663 | `rmsnorm_bf16` |
| 114.0 | 470 | `quantize_mmq_q8_1<__nv_bfloat16, 0>` |
| 107.1 | 10 | `flash_fwd_kernel<...512...>` |
| 100.6 | 60 | `moe_weighted_reduce_flat_kernel<__nv_bfloat16>` |
| 86.2 | 180 | `ucopy_bf16` |

### p16384 chunk 4096

| Time | Launches | Kernel |
| ---: | ---: | --- |
| 1385.8 | 700 | `mul_mat_q<8, 128, 0>` |
| 1299.9 | 480 | `mul_mat_q<8, 128, 1>` |
| 706.6 | 15 | `flashinfer::BatchPrefillWithPagedKVCacheKernel<..., 32, 32, ...>` |
| 271.9 | 1085 | `rmsnorm_bf16` |
| 225.5 | 940 | `quantize_mmq_q8_1<__nv_bfloat16, 0>` |
| 219.3 | 420 | `ucopy_bf16` |
| 205.7 | 120 | `moe_weighted_reduce_flat_kernel<__nv_bfloat16>` |
| 119.3 | 120 | `quantize_mmq_q8_1_glu_f32<0>` |

## Readout

Chunking fixes the p8192 attention blowup: `attention / rope / KV` drops from 536.1 ms to 359.1 ms,
and the old single hd512 FlashAttention kernel drops from 425.5 ms to 107.1 ms. The work shifts to
FlashInfer paged prefill for later chunks plus mostly linear model work.

The remaining dominant cost is Q8 MMQ matmul. At p8192 it is 55.3% of GPU kernel time; at p16384 it is
52.3%. The next visible overhead from chunking is copy/cast: it rises from 67.4 ms unchunked p8192 to
115.6 ms chunked p8192, and reaches 249.8 ms at p16384.

The most likely next levers are reducing per-chunk copy/cast and metadata/input setup overhead, then
looking at Q8 MMQ and activation quantization throughput. Attention is no longer the only long-context
problem, but FlashInfer prefill is still the second-largest group at p16384.
