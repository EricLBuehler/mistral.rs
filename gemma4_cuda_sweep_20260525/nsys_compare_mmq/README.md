# E4B Q8 MMQ profile comparison

Date: 2026-05-25

Goal: compare mistral.rs and llama.cpp around Gemma 4 E4B Q8 prompt MMQ behavior.

## Commands

### mistral.rs nsys, 8192 prompt

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o gemma4_cuda_sweep_20260525/nsys_compare_mmq/mistral_e4b_q8_p8192 \
  target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 \
  --prompt-len 8192 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000
```

### llama.cpp nsys, 8192 prompt

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o gemma4_cuda_sweep_20260525/nsys_compare_mmq/llamacpp_e4b_q8_p8192_nographs \
  ../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf \
  -p 8192 -n 0 -fa 1 -ngl 99 -b 2048 -ub 512 -r 1 --no-warmup -o jsonl
```

### NCU sampled MMQ, 2048 prompt

```bash
MISTRALRS_CUDA_GRAPHS=0 \
ncu --target-processes all --kernel-name 'regex:.*mul_mat_q.*' --launch-count 12 \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/nsys_compare_mmq/mistral_ncu_mmq_time12_p2048.csv \
  --force-overwrite \
  target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 \
  --prompt-len 2048 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000
```

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
ncu --target-processes all --kernel-name 'regex:.*mul_mat_q.*' --launch-count 12 \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/nsys_compare_mmq/llamacpp_ncu_mmq_time12_p2048.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf \
  -p 2048 -n 0 -fa 1 -ngl 99 -b 2048 -ub 512 -r 1 --no-warmup -o jsonl
```

## llama.cpp nsys attribution

8192 prompt, no CUDA graphs, profiler run:

| Group | Total ms | Share | Count |
| --- | ---: | ---: | ---: |
| MMQ main | 1182.381 | 62.2% | 4080 |
| GELU/GLU elementwise | 169.408 | 8.9% | 1316 |
| FlashAttention | 168.385 | 8.9% | 1344 |
| RMS norm | 141.327 | 7.4% | 4772 |
| MMQ activation quantize | 105.883 | 5.6% | 4080 |
| dense/cuBLAS/CUTLASS | 44.668 | 2.3% | 1984 |
| MMQ stream-k fixup | 30.166 | 1.6% | 2768 |

The default graph-node run showed the same ordering but with fewer visible graph-node instances.

## NCU MMQ sampled launch stats

Main Q8_0 MMQ kernels use the same core launch configuration in both systems:

| Runtime | Kernel | Example grid | Block | Regs/thread | Dynamic shared memory |
| --- | --- | ---: | ---: | ---: | ---: |
| mistral.rs | `mul_mat_q<8, 128, 0>` | `(320, 1, 1)` | `(32, 8, 1)` | 255 | 57856 B |
| llama.cpp | `mul_mat_q<8, 128, 0>` | `(320, 1, 1)` | `(32, 8, 1)` | 255 | 57856 B |

Sampled `gpu__time_duration.sum` for comparable 2048-prompt MMQ launches:

| Runtime | Launch | Grid | Duration |
| --- | ---: | ---: | ---: |
| mistral.rs | 7 | `(320, 1, 1)` | 397696 ns |
| llama.cpp | 8 | `(320, 1, 1)` | 509280 ns |
| llama.cpp | 9 | `(320, 1, 1)` | 510848 ns |

Do not overfit the individual launch order. The useful signal is that the main MMQ template and launch shape match, and sampled same-grid launches are not obviously worse in mistral.rs.

## Notes

- `nsys` did not expose mistral.rs full inference kernel composition reliably in this setup; `ncu` did capture MMQ launches.
- The main Q8_0 MMQ kernel itself is not the obvious gap. The likely remaining areas are activation quantization, BF16/F32 casts, GLU/GELU elementwise work, prompt scheduling/shape differences, and stale non-main kernel pieces such as fixup/register differences.
