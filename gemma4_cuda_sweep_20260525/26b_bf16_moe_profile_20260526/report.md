# 26B A4B BF16 MoE Profile

Model: `../hf_models/gemma4_26b_a4b`

Settings:

- mistral.rs: graphs off, FlashInfer decode on, PagedAttention on, context limit 20000
- vLLM: BF16, language model only, prefix cache off, max model len 20000, GPU memory utilization 0.60
- Profiles: Nsight Compute `gpu__time_duration.sum`, request-range capture after model load and warmup
- Decode profiles include the prompt fill plus generated tokens

Artifacts:

- `mistralrs_26b_bf16_p512_ncu_range.csv`
- `vllm_26b_bf16_p512_ncu_range.csv`
- `mistralrs_26b_bf16_d128_g16_ncu_range.csv`
- `vllm_26b_bf16_d128_g16_ncu_range.csv`
- `mistralrs_26b_bf16_p512.nsys-rep`

## Prompt p512

| Runtime | Total kernel time | Kernels | Unique kernels |
| --- | ---: | ---: | ---: |
| mistral.rs | 778.128 ms | 3764 | 39 |
| vLLM | 92.358 ms | 593 | 37 |

### mistral.rs top kernels

| Kernel | Time | Launches |
| --- | ---: | ---: |
| `moe_gemm_grouped_transposed_kernel<__nv_bfloat16>` | 592.054 ms | 60 |
| `fast_sum_bf16` | 64.320 ms | 30 |
| `ucopy_bf16` | 36.178 ms | 330 |
| `nvjet_sm121_tst_mma_128x192x64_2_64x48x64_tmaAB_bz_TNNN` | 14.303 ms | 110 |
| `rmsnorm_bf16` | 12.069 ms | 211 |
| `bitonic_sort_kernel<unsigned int, 1>` | 8.729 ms | 2340 |

### vLLM top kernels

| Kernel | Time | Launches |
| --- | ---: | ---: |
| `fused_moe_kernel` | 27.597 ms | 60 |
| `nvjet_sm121_tst_mma_128x128x64_3_32x64x64_tmaAB_bz_TNNN` | 12.794 ms | 55 |
| `nvjet_sm121_tst_mma_128x176x64_2_32x88x64_tmaAB_bz_TNNN` | 11.621 ms | 30 |
| `cublasGemvParamsEx` | 9.284 ms | 1 |
| `kernel_unified_attention` | 8.800 ms | 30 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` | 5.117 ms | 30 |

## Decode d128

| Runtime | Total kernel time | Kernels | Unique kernels |
| --- | ---: | ---: | ---: |
| mistral.rs, 16 generated tokens | 1790.744 ms | 21793 | 49 |
| vLLM, 17 generated tokens | 843.757 ms | 10111 | 42 |

### mistral.rs top kernels

| Kernel | Time | Launches |
| --- | ---: | ---: |
| `moe_gemv_transposed_kernel<__nv_bfloat16, 256>` | 762.971 ms | 900 |
| `moe_gemm_grouped_transposed_kernel<__nv_bfloat16>` | 506.876 ms | 120 |
| `gemv_kernel_batched<__nv_bfloat16, __nv_bfloat162, 256, 1>` | 261.440 ms | 3075 |
| `mmvq_gguf_q8_0_bf16_plain_cuda1` | 61.651 ms | 17 |
| `fast_sum_bf16` | 42.620 ms | 510 |
| `rmsnorm_bf16` | 28.284 ms | 4037 |

### vLLM top kernels

| Kernel | Time | Launches |
| --- | ---: | ---: |
| `cublasGemvParamsEx` | 363.463 ms | 1840 |
| `fused_moe_kernel` | 250.945 ms | 1020 |
| `cublasGemvParamsEx` | 171.063 ms | 577 |
| `kernel_unified_attention` | 7.780 ms | 510 |
| `nvjet_sm121_tst_mma_192x128x64_2_96x32x64_tmaAB_bz_TNNN` | 6.158 ms | 25 |
| `triton_red_fused_add_mul_rms_norm_2` | 5.242 ms | 510 |

## Readout

The gap is not attention. Attention/cache kernels are small in both traces.

The main gap is unquantized MoE expert compute:

- Prompt: mistral.rs spends 592.054 ms in `moe_gemm_grouped_transposed_kernel`; vLLM spends 27.597 ms in `fused_moe_kernel` for the same 60 expert GEMM launches.
- Decode: mistral.rs spends 1269.847 ms in MoE expert GEMV/GEMM kernels; vLLM spends 250.945 ms in `fused_moe_kernel`.

vLLM's path is not a simple C++ CUDA kernel drop-in. The expert compute is a Triton `fused_moe_kernel` in `vllm/model_executor/layers/fused_moe/fused_moe.py`; the helper token alignment and sum kernels are C++ CUDA in `vllm/csrc/moe/moe_align_sum_kernels.cu`.

The mistral.rs WMMA MoE kernel has the wrong parallelization shape for prefill: it launches by expert and output tile, then loops over the expert's token rows inside the CTA. vLLM launches over padded token blocks and output tiles, so token rows are parallelized directly and weights get better L2 reuse.

## Next target

Implement a new unquantized CUDA MoE path that follows the vLLM layout:

- Build padded `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded`.
- Launch expert GEMM over `(token_block, output_block)` instead of `(expert, output_block)` with a serial token loop.
- Use dynamic tile config like vLLM: small M uses M tile 16 or 32, p512 uses M tile 64, larger prompt uses M tile 128.
- Use the same kernel for prefill and decode, with the small-batch `naive_block_assignment` fast path.
- Replace the current bitonic sort routing path with a vLLM-style alignment helper where applicable.
