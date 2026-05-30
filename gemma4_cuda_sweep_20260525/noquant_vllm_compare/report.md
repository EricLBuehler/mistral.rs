# Gemma 4 BF16 mistral.rs vs vLLM

Models:
- E4B: `google/gemma-4-E4B-it`
- 26B-A4B: `../hf_models/gemma4_26b_a4b`

Configuration:
- mistral.rs: `target/release/mistralrs bench`, BF16/no quant, CUDA, FlashAttention, PagedAttention on, context length 20000, 3 iterations, 1 warmup.
- vLLM: `vllm 0.21.0`, BF16/no quant, `language_model_only=True`, prefix caching disabled, max model length 20000.
- vLLM forced `TRITON_ATTN` for Gemma 4 because of mixed `head_dim=256` and `global_head_dim=512`.
- vLLM E4B used `gpu_memory_utilization=0.85`.
- vLLM 26B-A4B used `gpu_memory_utilization=0.60`; `0.85` and `0.75` both OOMed after CUDA graph capture. The resulting KV capacity was still 202,433 tokens.
- vLLM prompt T/s is approximated from `input_len / latency(output_len=1)`, so it includes one generated token.
- vLLM decode T/s is computed as `256 / (latency(output_len=257) - latency(output_len=1))`.

## E4B Prompt

| Prompt tokens | mistral.rs BF16 T/s | vLLM BF16 approx T/s |
| ---: | ---: | ---: |
| 128 | 2277.8 | 2368.2 |
| 512 | 5380.4 | 6106.4 |
| 2048 | 7119.4 | 7017.9 |
| 4096 | 7144.2 | 6992.2 |
| 8192 | 7023.7 | 6585.2 |
| 16384 | 6610.9 | 5644.9 |

## E4B Decode

| Depth | mistral.rs BF16 T/s | vLLM BF16 delta T/s |
| ---: | ---: | ---: |
| 128 | 26.1 | 19.63 |
| 512 | 25.9 | 19.49 |
| 2048 | 25.7 | 19.25 |
| 4096 | 25.5 | 19.06 |
| 8192 | 25.0 | 18.71 |
| 16384 | 24.2 | 18.25 |

## 26B-A4B Prompt

| Prompt tokens | mistral.rs BF16 T/s | vLLM BF16 approx T/s |
| ---: | ---: | ---: |
| 128 | 385.5 | 895.2 |
| 512 | 677.4 | 2466.0 |
| 2048 | 677.9 | 4378.6 |
| 4096 | 604.3 | 5046.8 |
| 8192 | 571.9 | 5132.6 |
| 16384 | 542.9 | 4321.3 |

## 26B-A4B Decode

| Depth | mistral.rs BF16 T/s | vLLM BF16 delta T/s |
| ---: | ---: | ---: |
| 128 | 15.8 | 23.84 |
| 512 | 15.6 | 23.53 |
| 2048 | 15.4 | 23.10 |
| 4096 | 15.4 | 22.99 |
| 8192 | 15.2 | 22.65 |
| 16384 | 15.0 | 22.19 |

## Plots

- E4B prompt throughput: `e4b_bf16_prompt_tps.png`
- E4B decode throughput: `e4b_bf16_decode_tps.png`
- E4B prompt relative to peak: `e4b_bf16_prompt_relative.png`
- E4B decode relative to peak: `e4b_bf16_decode_relative.png`
- 26B-A4B prompt throughput: `26b_a4b_bf16_prompt_tps.png`
- 26B-A4B decode throughput: `26b_a4b_bf16_decode_tps.png`
- 26B-A4B prompt relative to peak: `26b_a4b_bf16_prompt_relative.png`
- 26B-A4B decode relative to peak: `26b_a4b_bf16_decode_relative.png`

## Commands

```bash
target/release/mistralrs bench -m google/gemma-4-E4B-it \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --gen-len 0 --depth 1 --iterations 3 --warmup 1 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

target/release/mistralrs bench -m google/gemma-4-E4B-it \
  --prompt-len 128 --gen-len 256 \
  --depth 128,512,2048,4096,8192,16384 \
  --iterations 3 --warmup 1 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

target/release/mistralrs bench -m ../hf_models/gemma4_26b_a4b \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --gen-len 0 --depth 1 --iterations 3 --warmup 1 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000

target/release/mistralrs bench -m ../hf_models/gemma4_26b_a4b \
  --prompt-len 128 --gen-len 256 \
  --depth 128,512,2048,4096,8192,16384 \
  --iterations 3 --warmup 1 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

vLLM was run once in-process per model and saved raw latencies to:
- `gemma4_cuda_sweep_20260525/noquant_vllm_compare/vllm_e4b_bf16_latency_delta.json`
- `gemma4_cuda_sweep_20260525/noquant_vllm_compare/vllm_26b_a4b_bf16_latency_delta.json`
