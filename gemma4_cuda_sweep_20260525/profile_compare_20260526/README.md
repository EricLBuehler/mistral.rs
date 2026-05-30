# Gemma 4 26B-A4B Q8 Prompt Profile

Scope: 26B-A4B Q8 prompt only. E4B was intentionally excluded because it is already fast enough for this pass.

## Commands

Mistral.rs full NCU prompt captures:

```bash
MISTRALRS_CUDA_GRAPHS=0 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv \
  --log-file gemma4_cuda_sweep_20260525/profile_compare_20260526/ncu/mistral_26b_a4b_q8_p2048.csv \
  --force-overwrite \
  target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 2048 --gen-len 0 --depth 1 --iterations 1 --warmup 0 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

The p8192 command is the same with `--prompt-len 8192`.

llama.cpp bounded NCU prompt captures:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 ncu --target-processes all --set none \
  --metrics gpu__time_duration.sum --csv --launch-count 3000 --kill yes \
  --log-file gemma4_cuda_sweep_20260525/profile_compare_20260526/ncu/llama_26b_a4b_q8_p2048_first3000.csv \
  --force-overwrite \
  ../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 2048 -n 0 -r 1 -ngl 99 -fa 1 -ctk f16 -ctv f16 --no-warmup -o json
```

The p8192 captures use `-p 8192`; one uses the first 3000 launches and one uses
`--launch-skip 6000 --launch-count 3000`.

## NCU Results

Mistral.rs full prompt captures:

| Group | p2048 | p8192 |
| --- | ---: | ---: |
| Q8 MMQ matmul | 393.9 ms | 1330.6 ms |
| attention / rope / KV | 55.2 ms | 536.1 ms |
| activation quantize | 39.5 ms | 188.6 ms |
| norms | 52.8 ms | 168.0 ms |
| MoE routing / reduce / layout | 37.7 ms | 115.7 ms |
| copy / cast | 39.9 ms | 67.4 ms |
| elementwise / GLU | 23.2 ms | 84.5 ms |
| Q8 matvec / lm-head | 18.2 ms | 18.5 ms |
| dense GEMM | 4.8 ms | 9.8 ms |
| other | 8.5 ms | 9.0 ms |
| total GPU kernel time | 673.7 ms | 2528.1 ms |

llama.cpp bounded 3000-launch windows:

| Group | p2048 first3000 | p8192 first3000 | p8192 skip6000 count3000 |
| --- | ---: | ---: | ---: |
| Q8 MMQ matmul | 305.2 ms | 310.8 ms | 302.2 ms |
| attention / rope / KV | 19.7 ms | 19.7 ms | 30.5 ms |
| activation quantize | 17.9 ms | 18.3 ms | 17.7 ms |
| norms | 39.8 ms | 39.2 ms | 37.0 ms |
| MoE routing / reduce / layout | 11.6 ms | 11.8 ms | 10.8 ms |
| copy / cast | 0.0 ms | 0.0 ms | 0.1 ms |
| elementwise / GLU | 46.8 ms | 46.5 ms | 44.8 ms |
| Q8 matvec / lm-head | 0.0 ms | 0.0 ms | 4.1 ms |
| dense GEMM | 3.1 ms | 3.1 ms | 2.7 ms |
| other | 0.0 ms | 0.0 ms | 0.1 ms |
| sampled GPU kernel time | 444.2 ms | 449.5 ms | 450.1 ms |

## Readout

The biggest mistral.rs p8192 growth is `attention / rope / KV`: it rises from 55.2 ms at p2048 to
536.1 ms at p8192. The top single kernel is the head-dim-512 flash-attn prefill kernel, at 425.5 ms
in the p8192 capture.

llama.cpp's p8192 sampled windows stay roughly flat because its prompt path is chunked. The first
3000 launches at p2048 and p8192 are nearly identical, and a later p8192 window after 6000 skipped
launches is also similar. That points to chunked prefill or equivalent chunk-level attention as the
main lever for reducing long-prompt degradation in mistral.rs.

The remaining linear costs in mistral.rs are Q8 MMQ, activation quantization, norms, MoE reduction,
and elementwise/GLU. Copy/cast is no longer the primary p8192 limiter after the recent BF16 fixes.

## Nsys Note

Nsys artifacts were also collected under `nsys/` and `nsys_clean/`, but on this machine the exports
did not produce reliable CUDA kernel tables for this workload. NCU is the source of kernel attribution
above.
