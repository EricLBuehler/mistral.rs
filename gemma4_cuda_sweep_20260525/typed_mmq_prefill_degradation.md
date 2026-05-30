# Gemma 4 E4B Q8 Typed MMQ Prefill Degradation

Command:

```bash
target/release/mistralrs bench -m google/gemma-4-E4B-it --quant 8 \
  --prompt-len 128,512,2048,4096,8192,16384 \
  --gen-len 0 --depth 1 --iterations 3 --warmup 1 \
  --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

## Current typed-MMQ prefill

| Prompt tokens | Prefill T/s | TTFT | Drop from 2048-token peak |
| ---: | ---: | ---: | ---: |
| 128 | 3657.1 | 35.00 ms | 59.3% |
| 512 | 6957.2 | 73.67 ms | 22.6% |
| 2048 | 8982.8 | 228.00 ms | 0.0% |
| 4096 | 8739.7 | 468.67 ms | 2.7% |
| 8192 | 7508.7 | 1091.00 ms | 16.4% |
| 16384 | 5765.0 | 2842.00 ms | 35.8% |

## Compared with previous sweep

Previous sweep source: `gemma4_cuda_sweep_20260525/summary.md`.

| Prompt tokens | Previous mistral.rs T/s | Current typed-MMQ T/s | Gain |
| ---: | ---: | ---: | ---: |
| 128 | 3398.8 | 3657.1 | 7.6% |
| 512 | 5981.7 | 6957.2 | 16.3% |
| 2048 | 7314.5 | 8982.8 | 22.8% |
| 4096 | 7009.8 | 8739.7 | 24.7% |
| 8192 | 5766.7 | 7508.7 | 30.2% |
| 16384 | 4860.3 | 5765.0 | 18.6% |

## Compared with llama.cpp from previous sweep

| Prompt tokens | llama.cpp T/s | Current mistral.rs T/s |
| ---: | ---: | ---: |
| 128 | 2600.1 | 3657.1 |
| 512 | 4324.2 | 6957.2 |
| 2048 | 4391.2 | 8982.8 |
| 4096 | 4352.6 | 8739.7 |
| 8192 | 4236.7 | 7508.7 |
| 16384 | 4006.5 | 5765.0 |

## Notes

The absolute long-context prefill performance improved substantially, but the curve still degrades after the
2048-token peak. The 16k point is now 5765.0 T/s, up from 4860.3 T/s, while the drop from the current peak is
35.8%.
