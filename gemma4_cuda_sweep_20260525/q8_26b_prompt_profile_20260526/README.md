# Gemma 4 26B-A4B Q8 Prompt Profile

Goal: compare prompt-side kernel composition for quantized Gemma 4 26B-A4B in mistral.rs and llama.cpp.

Method: Nsight Compute with CUDA graph capture disabled for kernel attribution. Absolute T/s under NCU replay is not meaningful; use kernel-time totals and proportions.

Models:
- mistral.rs: `mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8`
- llama.cpp: `../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf`

## p512 kernel groups

| Group | mistral.rs ms | llama.cpp ms |
| --- | ---: | ---: |
| q8 prompt matmul/MMQ | 185.660 | 188.777 |
| copy/cast/layout | 39.599 | 23.516 |
| norm | 18.073 | 21.449 |
| moe routing/reduce | 17.243 | 3.644 |
| single-token/lm-head matvec | 16.331 | 4.100 |
| attention/rope/cache | 10.010 | 8.915 |
| other | 9.403 | 1.130 |
| elementwise/GLU | 9.401 | 4.884 |
| dense/cutlass | 2.702 | 1.915 |

At 512 tokens, the main q8 prompt matmul time is effectively tied. The short-prompt gap is overhead: cast/layout, MoE routing/reduce, and single-token/lm-head work.

Top mistral.rs kernels:

| Time | Count | Kernel |
| ---: | ---: | --- |
| 94.504 ms | 120 | `mul_mat_q<8, 128, 1>` |
| 73.752 ms | 175 | `mul_mat_q<8, 128, 0>` |
| 22.143 ms | 33 | `cast_bf16_f32` |
| 15.157 ms | 482 | `rmsnorm_bf16` |
| 12.064 ms | 72 | `mmvq_gguf_q8_0_bf16_plain_cuda1` |
| 9.730 ms | 120 | `ucopy_bf16` |
| 7.681 ms | 64 | `cast_f32_bf16` |
| 6.223 ms | 30 | `moe_weighted_reduce_flat_kernel` |
| 6.219 ms | 30 | `moe_gemv_fused_gate_up_q8_0_q8_1` |
| 6.125 ms | 235 | `quantize_mmq_q8_1<__nv_bfloat16, 0>` |

Top llama.cpp kernels:

| Time | Count | Kernel |
| ---: | ---: | --- |
| 161.481 ms | 202 | `mul_mat_q<8, 128, 0>` |
| 11.661 ms | 30 | `k_bin_bcast<&op_mul, ...>` |
| 10.772 ms | 260 | `quantize_mmq_q8_1<0>` |
| 8.330 ms | 58 | `mul_mat_q<8, 128, 1>` |
| 8.265 ms | 89 | `rms_norm_f32<1024, 1, 1>` |
| 7.305 ms | 122 | `rms_norm_f32<1024, 1, 0>` |
| 6.081 ms | 30 | `k_bin_bcast<&op_add, ...>` |
| 4.884 ms | 60 | `unary_gated_op_kernel<&op_gelu, float>` |
| 4.506 ms | 25 | `flash_attn_ext_f16<256, 256, 32, 2, 0, 0>` |
| 3.971 ms | 5 | `mul_mat_vec_q<8, 1, 0, 0>` |

## p4096 kernel groups

| Group | mistral.rs ms | llama.cpp ms |
| --- | ---: | ---: |
| q8 prompt matmul/MMQ | 772.842 | 1521.977 |
| copy/cast/layout | 93.101 | 188.800 |
| norm | 91.046 | 179.539 |
| attention/rope/cache | 163.261 | 123.745 |
| moe routing/reduce | 66.848 | 26.471 |
| elementwise/GLU | 46.601 | 39.526 |
| single-token/lm-head matvec | 16.560 | 8.283 |
| dense/cutlass | 7.146 | 14.119 |
| other | 9.195 | 8.581 |

At 4096 tokens, mistral.rs wins because the q8 prompt MMQ path is much faster. The remaining opportunity is not the main MMQ kernel; it is overhead around MoE reduce/routing, layout/cast/copy, and attention/rope.

Top mistral.rs kernels:

| Time | Count | Kernel |
| ---: | ---: | --- |
| 363.246 ms | 175 | `mul_mat_q<8, 128, 0>` |
| 320.187 ms | 120 | `mul_mat_q<8, 128, 1>` |
| 106.353 ms | 10 | `flash_fwd_kernel<Flash_fwd_kernel_traits<512, ...>` |
| 75.545 ms | 482 | `rmsnorm_bf16` |
| 58.730 ms | 235 | `quantize_mmq_q8_1<__nv_bfloat16, 0>` |
| 53.448 ms | 120 | `ucopy_bf16` |
| 53.364 ms | 30 | `moe_weighted_reduce_flat_kernel` |
| 30.419 ms | 30 | `quantize_mmq_q8_1_glu_f32<0>` |
| 25.093 ms | 60 | `qk_rms_norm_rope_positions_kernel<__nv_bfloat16, 1>` |
| 22.717 ms | 120 | `badd_bf16` |

Top llama.cpp kernels:

| Time | Count | Kernel |
| ---: | ---: | --- |
| 1310.226 ms | 1616 | `mul_mat_q<8, 128, 0>` |
| 96.135 ms | 234 | `k_bin_bcast<&op_mul, ...>` |
| 84.040 ms | 2080 | `quantize_mmq_q8_1<0>` |
| 68.860 ms | 700 | `rms_norm_f32<1024, 1, 1>` |
| 66.020 ms | 464 | `mul_mat_q<8, 128, 1>` |
| 62.148 ms | 946 | `rms_norm_f32<1024, 1, 0>` |
| 60.725 ms | 200 | `flash_attn_ext_f16<256, 256, 32, 2, 0, 0>` |
| 49.863 ms | 234 | `k_bin_bcast<&op_add, ...>` |
| 43.217 ms | 40 | `flash_attn_ext_f16<512, 512, 8, 8, 0, 0>` |
| 39.526 ms | 468 | `unary_gated_op_kernel<&op_gelu, float>` |

## Commands

```bash
MISTRALRS_CUDA_GRAPHS=0 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_prompt_profile_20260526/mistralrs_26b_q8_p512_ncu_full.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 512 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

GGML_CUDA_DISABLE_GRAPHS=1 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_prompt_profile_20260526/llamacpp_26b_q8_p512_ncu_full.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 512 -n 0 -r 1 -ngl 99 -fa 1 -ctk f16 -ctv f16 --no-warmup -o json

MISTRALRS_CUDA_GRAPHS=0 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_prompt_profile_20260526/mistralrs_26b_q8_p4096_ncu_full.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 4096 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

GGML_CUDA_DISABLE_GRAPHS=1 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_prompt_profile_20260526/llamacpp_26b_q8_p4096_ncu_full.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 4096 -n 0 -r 1 -ngl 99 -fa 1 -ctk f16 -ctv f16 --no-warmup -o json
```
