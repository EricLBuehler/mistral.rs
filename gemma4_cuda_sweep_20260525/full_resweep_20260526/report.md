# Gemma 4 Q8 Full Resweep

Configuration: current branch after removing the output-dtype experiment; CUDA, Q8, FlashAttention, default CUDA graph behavior, 3 repetitions/iterations, mistral.rs warmup 1, llama.cpp default warmup.

mistral.rs uses PagedAttention on with `--pa-context-len 20000 --max-seq-len 20000`. llama.cpp uses `-ngl 99 -fa 1 -ctk f16 -ctv f16`.

## E4B Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 3429.1 | 2575.8 |
| 512 | 6716.1 | 4318.4 |
| 2048 | 8879.1 | 4388.8 |
| 4096 | 8593.3 | 4342.3 |
| 8192 | 8405.0 | 4265.9 |
| 16384 | 7804.4 | 4031.0 |

## E4B Decode T/s

| Depth | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 41.3 | 39.9 |
| 512 | 41.1 | 39.6 |
| 2048 | 40.6 | 39.1 |
| 4096 | 40.2 | 38.6 |
| 8192 | 39.2 | 37.7 |
| 16384 | 37.3 | 36.1 |

## 26B-A4B Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 1184.2 | 1099.7 |
| 512 | 2408.1 | 2463.9 |
| 2048 | 3393.1 | 2441.3 |
| 4096 | 3445.2 | 2426.7 |
| 8192 | 2974.3 | 2402.5 |
| 16384 | 2717.0 | 2310.5 |

## 26B-A4B Decode T/s

| Depth | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 48.3 | 49.0 |
| 512 | 47.3 | 48.2 |
| 2048 | 46.0 | 46.4 |
| 4096 | 45.5 | 46.7 |
| 8192 | 44.6 | 45.2 |
| 16384 | 42.6 | 43.9 |

## Artifacts

- `e4b_q8_prompt_tps.png`
- `e4b_q8_decode_tps.png`
- `26b_a4b_q8_prompt_tps.png`
- `26b_a4b_q8_decode_tps.png`

Raw outputs are in this directory as `mistralrs_*.txt` and `llamacpp_*.json`.

## Commands

```bash
../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf -p 128,512,2048,4096,8192,16384 -n 0 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf -p 0 -n 256 -d 128,512,2048,4096,8192,16384 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf -p 128,512,2048,4096,8192,16384 -n 0 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf -p 0 -n 256 -d 128,512,2048,4096,8192,16384 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
target/release/mistralrs bench -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
target/release/mistralrs bench -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```
