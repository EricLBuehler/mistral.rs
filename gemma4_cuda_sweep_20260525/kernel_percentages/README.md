# Gemma 4 CUDA Kernel Percentages

Model: `google/gemma-4-E4B-it`, Q8 UQFF in mistral.rs and Q8_0 GGUF in llama.cpp.

The useful profiler path is Nsight Compute. Nsight Systems saw the NVTX request range but did not collect mistral.rs CUDA kernel rows for this workload.

## Commands

Build mistral.rs with CUDA and FlashAttention before profiling:

```bash
cargo build --release --package mistralrs-cli --features cuda,flash-attn
```

Full mistral.rs kernel mix:

```bash
MISTRALRS_PROFILE_CUDA=1 MISTRALRS_PROFILE_NVTX=1 MISTRALRS_CUDA_GRAPHS=0 \
ncu --profile-from-start off --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/kernel_percentages/mistral_e4b_q8_p2048_flash_ncu_full.csv \
  --force-overwrite \
  target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 \
  --prompt-len 2048 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000
```

Full llama.cpp kernel mix:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
ncu --target-processes all --set none --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/kernel_percentages/llamacpp_e4b_q8_p2048_ncu_full.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench \
  -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf \
  -p 2048 -n 0 -fa 1 -ngl 99 -b 2048 -ub 512 -r 1 --no-warmup -o jsonl
```

Gemma 4 range profiles use push/pop NVTX ranges, so the Nsight Compute filter needs the trailing slash:

```bash
MISTRALRS_PROFILE_NVTX=1 MISTRALRS_CUDA_GRAPHS=0 \
ncu --target-processes all --nvtx --nvtx-include "gemma4.mlp/" \
  --set none --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/kernel_percentages/ranges_gemma4/mistral_e4b_q8_p2048_gemma4_mlp.csv \
  --force-overwrite \
  target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 \
  --prompt-len 2048 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000
```

## Full Profile

NCU profiling overhead makes reported T/s unusable here. Use GPU kernel time and percentages only.

| Runtime | Kernel rows | Total kernel time |
| --- | ---: | ---: |
| mistral.rs | 1507 | 327.546 ms |
| llama.cpp | 5921 | 632.176 ms |

### mistral.rs groups

| Group | Time | Share |
| --- | ---: | ---: |
| MMQ main | 161.029 ms | 49.16% |
| copy/cast/metadata elementwise | 61.888 ms | 18.89% |
| RMS norm | 34.353 ms | 10.49% |
| attention/cache kernels | 20.169 ms | 6.16% |
| activation quantize | 17.731 ms | 5.41% |
| MMVQ main | 13.875 ms | 4.24% |
| GLU/GELU elementwise | 13.078 ms | 3.99% |
| MMQ stream-k fixup | 4.683 ms | 1.43% |
| other elementwise | 0.657 ms | 0.20% |

Top exact mistral.rs kernels:

| Kernel | Time | Share |
| --- | ---: | ---: |
| `mul_mat_q<8, 128, 0>` | 161.029 ms | 49.16% |
| `cast_f32_bf16` | 47.699 ms | 14.56% |
| `rms_norm_residual_kernel<bf16>` | 19.007 ms | 5.80% |
| `quantize_mmq_q8_1<bf16>` | 17.731 ms | 5.41% |
| `flash_fwd_kernel<...512...>` | 13.850 ms | 4.23% |
| `fused_glu_kernel_vec4<bf16>` | 13.078 ms | 3.99% |
| `ucopy_bf16` | 11.620 ms | 3.55% |
| `rmsnorm_bf16` | 9.924 ms | 3.03% |
| `mmvq_gguf_q8_0_bf16_plain_cuda1` | 8.175 ms | 2.50% |
| `mmvq_gguf_q8_0_bf16_fused_glu_cuda1` | 5.451 ms | 1.66% |

### llama.cpp groups

| Group | Time | Share |
| --- | ---: | ---: |
| MMQ main | 381.292 ms | 60.31% |
| RMS norm | 72.565 ms | 11.48% |
| GLU/GELU elementwise | 43.406 ms | 6.87% |
| attention/cache kernels | 38.343 ms | 6.07% |
| activation quantize | 35.486 ms | 5.61% |
| MMQ stream-k fixup | 23.302 ms | 3.69% |
| dense/cuBLAS/CUTLASS | 16.010 ms | 2.53% |
| other | 7.235 ms | 1.14% |
| RoPE / qk norm | 5.158 ms | 0.82% |
| MMVQ main | 3.832 ms | 0.61% |
| other elementwise | 3.001 ms | 0.47% |
| copy/cast/metadata elementwise | 2.546 ms | 0.40% |

Top exact llama.cpp kernels:

| Kernel | Time | Share |
| --- | ---: | ---: |
| `mul_mat_q<8, 128, 0>` | 381.292 ms | 60.31% |
| `rms_norm_f32<1024, 1, 1>` | 45.015 ms | 7.12% |
| `unary_gated_op_kernel<&op_gelu, float>` | 42.603 ms | 6.74% |
| `quantize_mmq_q8_1<0>` | 35.486 ms | 5.61% |
| `mul_mat_q_stream_k_fixup<8, 128, 0>` | 23.302 ms | 3.69% |
| `flash_attn_ext_f16<256, ...>` | 19.788 ms | 3.13% |
| `rms_norm_f32<1024, 1, 0>` | 19.022 ms | 3.01% |
| `flash_attn_ext_f16<512, ...>` | 11.578 ms | 1.83% |

## Attention Dispatch Check

The normal Gemma 4 E4B and 26B-A4B text configs use `head_dim=256` for sliding layers and `global_head_dim=512` for full-attention layers. The eager-attention guard is `head_dim > 512`, so these models do not force the custom-mask eager path for the 512-dim full-attention heads.

The corrected flash-attn profile confirms that dispatch:

| Kernel family | Launches | Time |
| --- | ---: | ---: |
| `flash_fwd_kernel<...512...>` | 7 | 13.850 ms |
| `flash_fwd_kernel<...256...>` | 35 | 4.202 ms |
| `softmax_f32` | 0 | 0.000 ms |

## Readout

The practical direction is not simply copying llama.cpp MMQ. In this p2048 profile, mistral.rs is already spending less absolute GPU time in the shared `mul_mat_q<8, 128, 0>` kernel than llama.cpp. The remaining mistral.rs-specific opportunity is avoiding dtype/layout/metadata elementwise churn around attention and MLP, especially `cast_f32_bf16` and `ucopy_bf16`.

The next concrete optimization pass should start with the cast/copy churn around `gemma4.attn.paged` and `gemma4.mlp`, not the raw MMQ kernel body.
