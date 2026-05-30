# Gemma 4 Q8 Resweep vs llama.cpp

Configuration: current branch HEAD, CUDA, Q8, FlashAttention, PagedAttention on for mistral.rs, CUDA graphs and FlashInfer decode explicitly enabled, 3 repetitions/iterations, mistral.rs warmup 1, llama.cpp default warmup.

mistral.rs uses `--pa-context-len 20000 --max-seq-len 20000`. llama.cpp uses `-ngl 99 -fa 1 -ctk f16 -ctv f16`.

## E4B Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 3523.5 | 2593.6 |
| 512 | 6815.3 | 4347.4 |
| 2048 | 9129.8 | 4419.6 |
| 4096 | 8777.4 | 4393.9 |
| 8192 | 8454.6 | 4255.2 |
| 16384 | 7858.1 | 4035.6 |

## E4B Decode T/s

| Depth | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 42.2 | 39.5 |
| 512 | 41.6 | 39.3 |
| 2048 | 41.2 | 38.7 |
| 4096 | 40.7 | 38.2 |
| 8192 | 39.7 | 37.4 |
| 16384 | 37.7 | 35.8 |

## 26B-A4B Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 1238.7 | 1108.3 |
| 512 | 2747.8 | 2406.8 |
| 2048 | 3671.5 | 2406.4 |
| 4096 | 3699.7 | 2402.3 |
| 8192 | 3636.6 | 2366.0 |
| 16384 | 3363.6 | 2285.4 |

## 26B-A4B Decode T/s

| Depth | mistral.rs | llama.cpp |
| ---: | ---: | ---: |
| 128 | 48.0 | 48.1 |
| 512 | 47.2 | 47.3 |
| 2048 | 45.9 | 45.4 |
| 4096 | 45.5 | 45.6 |
| 8192 | 44.6 | 44.2 |
| 16384 | 42.7 | 42.9 |

## Artifacts

- `e4b_q8_prompt_tps.png`
- `e4b_q8_decode_tps.png`
- `26b_a4b_q8_prompt_tps.png`
- `26b_a4b_q8_decode_tps.png`
- `results.json`

Raw outputs are in this directory as `mistralrs_*.txt` and `llamacpp_*.json`.

## Commands

```bash
../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf -p 128,512,2048,4096,8192,16384 -n 0 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../llama.cpp/gemma-4-E4B-it-Q8_0.gguf -p 0 -n 256 -d 128,512,2048,4096,8192,16384 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf -p 128,512,2048,4096,8192,16384 -n 0 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
../llama.cpp/build/bin/llama-bench -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf -p 0 -n 256 -d 128,512,2048,4096,8192,16384 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -o json
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```
