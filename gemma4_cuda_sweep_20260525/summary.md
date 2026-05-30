# Gemma 4 CUDA Sweep

Artifacts: `/tmp/mistralrs_gemma4_sweep_20260525_075155`

Configuration: Q8, CUDA, flash attention enabled, 3 repetitions/iterations, 1 mistral.rs warmup, prompt lengths `128,512,2048,4096,8192,16384`, decode length `256`, decode depths `128,512,2048,4096,8192,16384`, mistral.rs PagedAttention context length `20000`.

## E4B Q8 Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
|---:|---:|---:|
| 128 | 3398.8 | 2600.1 |
| 512 | 5981.7 | 4324.2 |
| 2048 | 7314.5 | 4391.2 |
| 4096 | 7009.8 | 4352.6 |
| 8192 | 5766.7 | 4236.7 |
| 16384 | 4860.3 | 4006.5 |

## E4B Q8 Decode T/s

| Depth | mistral.rs | llama.cpp |
|---:|---:|---:|
| 128 | 42.1 | 39.4 |
| 512 | 41.6 | 39.2 |
| 2048 | 41.2 | 38.7 |
| 4096 | 40.7 | 38.2 |
| 8192 | 39.7 | 37.3 |
| 16384 | 37.7 | 35.7 |

## 26B A4B Q8 Prompt T/s

| Prompt tokens | mistral.rs | llama.cpp |
|---:|---:|---:|
| 128 | 1203.9 | 1107.4 |
| 512 | 2564.7 | 2384.5 |
| 2048 | 3413.6 | 2388.4 |
| 4096 | 3393.6 | 2364.9 |
| 8192 | 3026.3 | 2355.2 |
| 16384 | 2599.0 | 2261.5 |

## 26B A4B Q8 Decode T/s

| Depth | mistral.rs | llama.cpp |
|---:|---:|---:|
| 128 | 45.3 | 48.4 |
| 512 | 44.5 | 47.5 |
| 2048 | 43.4 | 45.6 |
| 4096 | 43.1 | 45.8 |
| 8192 | 42.0 | 44.4 |
| 16384 | 40.2 | 43.3 |

