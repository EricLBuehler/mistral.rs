# Gemma 4 26B-A4B Q8 mistral.rs vs llama.cpp

Models:
- mistral.rs: `mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8` (`q8_0` UQFF).
- llama.cpp: `../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf`.

Configuration:
- mistral.rs: `target/release/mistralrs bench`, CUDA, BF16 activations, Q8_0 UQFF weights, FlashAttention, PagedAttention on, context length 20000, 3 iterations, 1 warmup.
- llama.cpp: `../llama.cpp/build/bin/llama-bench`, CUDA, Q8_0 GGUF weights, `-ngl 99`, `-fa 1`, `-ctk f16`, `-ctv f16`, default CUDA graph behavior, 3 repetitions.
- mistral.rs cannot currently load this GGUF because its GGUF parser rejects architecture `gemma4`, so this compares native mistral.rs UQFF Q8_0 against llama.cpp GGUF Q8_0 rather than the same file.

## Prompt

| Prompt tokens | mistral.rs Q8 T/s | llama.cpp Q8 T/s |
| ---: | ---: | ---: |
| 128 | 1207.6 | 1111.6 |
| 512 | 2671.3 | 2439.6 |
| 2048 | 3636.1 | 2457.3 |
| 4096 | 3639.9 | 2415.9 |
| 8192 | 3235.1 | 2389.6 |
| 16384 | 2721.3 | 2273.8 |

## Decode

| Depth | mistral.rs Q8 T/s | llama.cpp Q8 T/s |
| ---: | ---: | ---: |
| 128 | 45.5 | 47.96 |
| 512 | 44.6 | 47.03 |
| 2048 | 43.6 | 45.04 |
| 4096 | 43.1 | 45.33 |
| 8192 | 42.2 | 43.79 |
| 16384 | 40.5 | 42.86 |

## Notes

- mistral.rs prompt is ahead at every tested length, especially from 2048 through 8192 tokens.
- llama.cpp decode is ahead at every tested depth; this is the optimization target.
- Both decode curves are fairly flat, so the gap looks mostly like per-token model/kernel overhead rather than long-context attention degradation.

## Plots

- Prompt throughput: `26b_a4b_q8_prompt_tps.png`
- Decode throughput: `26b_a4b_q8_decode_tps.png`
- Prompt relative to peak: `26b_a4b_q8_prompt_relative.png`
- Decode relative to peak: `26b_a4b_q8_decode_relative.png`

## Commands

```bash
../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 128,512,2048,4096,8192,16384 -n 0 -r 3 -ngl 99 -fa 1 -o json

../llama.cpp/build/bin/llama-bench \
  -m ../hf_models/gemma4_26b_a4b_gguf/gemma-4-26B-A4B-it-Q8_0.gguf \
  -p 0 -n 256 -d 128,512,2048,4096,8192,16384 -r 3 -ngl 99 -fa 1 -o json

target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 \
  --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000

target/release/mistralrs bench \
  -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff 8 \
  --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 \
  --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```
