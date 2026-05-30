# Gemma 4 26B-A4B Q8 decode profile

Goal: explain the Q8 decode gap between mistral.rs and llama.cpp.

Benchmark baseline:

| Depth | mistral.rs Q8 T/s | llama.cpp Q8 T/s |
| ---: | ---: | ---: |
| 128 | 45.5 | 47.96 |
| 512 | 44.6 | 47.03 |
| 2048 | 43.6 | 45.04 |
| 4096 | 43.1 | 45.33 |
| 8192 | 42.2 | 43.79 |
| 16384 | 40.5 | 42.86 |

## Method

Graph-enabled Nsight Systems captures were taken first at `depth=4096, gen=128`.
They reproduced the benchmark shape, but graph-node/kernel attribution was incomplete on this system.

For kernel attribution, CUDA graphs were disabled and Nsight Compute was run at `depth=128, gen=16`.
This keeps prefill small enough that the decode path is visible without making NCU replay too large.

NCU replay makes wall-clock T/s invalid. Use the kernel-time percentages only.

## NCU kernel groups

| Group | mistral.rs time | llama.cpp time |
| --- | ---: | ---: |
| q8 decode matvec / GEMV | 354.804 ms | 355.050 ms |
| copy/cast/layout | 37.548 ms | 17.704 ms |
| RMS norm | 35.096 ms | 38.876 ms |
| attention/cache decode | 4.638 ms | 13.374 ms |
| rope/qk norm | 5.513 ms | 3.622 ms |
| logits/sampling | 6.207 ms | 2.584 ms |
| dense/cutlass | 10.432 ms | 0.629 ms |

Prefill/MMQ path time is present in both profiles because the decode benchmark first fills the KV cache:

| Runtime | Prefill/MMQ time |
| --- | ---: |
| mistral.rs | 225.264 ms |
| llama.cpp | 113.363 ms |

Do not read that as decode time. It is useful only as a reminder that this NCU profile contains both the setup prefill and decode.

## Top exact kernels

### mistral.rs

| Kernel | Time |
| --- | ---: |
| `mul_mat_q<8, 128, 1>` | 128.976 ms |
| `mmvq_gguf_q8_0_bf16_plain_cuda1` | 122.886 ms |
| `moe_gemv_fused_gate_up_q8_0_q8_1` | 94.403 ms |
| `mul_mat_q<8, 128, 0>` | 82.470 ms |
| `mmvq_gguf_q8_0_bf16_fused_qkv_cuda1` | 55.426 ms |
| `moe_gemv_down_aggregate_q8_0_q8_1` | 52.899 ms |
| `rmsnorm_bf16` | 30.184 ms |
| `mmvq_gguf_q8_0_bf16_fused_glu_cuda1` | 29.190 ms |
| `cast_bf16_f32` | 23.666 ms |
| `cast_f32_bf16` | 8.587 ms |
| `ucopy_bf16` | 5.095 ms |

### llama.cpp

| Kernel | Time |
| --- | ---: |
| `mul_mat_vec_q<8, 1, 0, 0>` | 308.553 ms |
| `mul_mat_q<8, 128, 0>` | 106.054 ms |
| `mul_mat_vec_q<8, 1, 0, 1>` | 46.497 ms |
| `rms_norm_f32<1024, 1, 0>` | 16.492 ms |
| `rms_norm_f32<1024, 1, 1>` | 13.229 ms |
| `quantize_q8_1` | 13.228 ms |
| `flash_attn_ext_vec<256, 1, 1, 1, 0>` | 8.112 ms |
| `mul_mat_vec_f<float, float, 1, 256, 0, 0>` | 5.084 ms |
| `unary_gated_op_kernel<&op_gelu, float>` | 4.239 ms |
| `k_set_rows<float, long, __half>` | 3.297 ms |

## Readout

The raw q8 decode matvec total is not the obvious gap:

- mistral.rs q8 mmvq + MoE q8 GEMV: `354.804 ms`
- llama.cpp q8 matvec: `355.050 ms`

The more actionable differences are:

- mistral.rs has about `20 ms` more copy/cast/layout work in this profile.
- mistral.rs has extra small elementwise/logits work around decode and sampling.
- mistral.rs attention/cache decode is already lower than llama.cpp here, so head-dim 512 attention is not the first place to optimize for this gap.
- The 26B-A4B MoE path is a major decode component, but its q8 GEMV time combines with mmvq to roughly match llama.cpp q8 matvec time.

## Commands

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
ncu --target-processes all --set none --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_llamacpp_compare/profiles/llamacpp_26b_q8_d128_g16_ncu_full.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 0 -n 16 -d 128 -r 1 -ngl 99 -fa 1 --no-warmup -o json

MISTRALRS_CUDA_GRAPHS=0 \
ncu --target-processes all --set none --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/q8_26b_llamacpp_compare/profiles/mistralrs_26b_q8_d128_g16_ncu_full.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 128 --gen-len 16 --depth 128 \
  --iterations 1 --warmup 0 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

Raw graph-enabled Nsight Systems traces:

- `profiles/llamacpp_26b_q8_d4096_g128.nsys-rep`
- `profiles/mistralrs_26b_q8_d4096_g128.nsys-rep`

Raw NCU CSVs:

- `profiles/llamacpp_26b_q8_d128_g16_ncu_full.csv`
- `profiles/mistralrs_26b_q8_d128_g16_ncu_full.csv`
