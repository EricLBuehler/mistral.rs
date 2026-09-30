# Release benchmark tables

All cells are mean +/- sample standard deviation across five measured repetitions after two excluded warmups.

## Same GGUF: target-only serving

| Workload | Unit | mistral.rs | llama.cpp (patched) |
| --- | --- | ---: | ---: |
| pp512 | input tok/s | 1370.237 +/- 26.896 | 868.344 +/- 10.179 |
| pp2048 | input tok/s | 1376.273 +/- 8.680 | 881.836 +/- 22.467 |
| pp8192 | input tok/s | 1348.285 +/- 2.383 | 870.570 +/- 9.029 |
| python | output tok/s | 26.331 +/- 0.024 | 26.350 +/- 0.044 |
| rust | output tok/s | 26.310 +/- 0.025 | 26.292 +/- 0.033 |
| prose | output tok/s | 26.452 +/- 0.036 | 26.657 +/- 0.043 |
| json | output tok/s | 26.200 +/- 0.034 | 26.468 +/- 0.060 |
| primes | output tok/s | 26.449 +/- 0.014 | 26.620 +/- 0.034 |
| math | output tok/s | 25.959 +/- 0.145 | 26.473 +/- 0.029 |
| translation | output tok/s | 26.232 +/- 0.031 | 26.508 +/- 0.129 |
| quicksort | output tok/s | 26.561 +/- 0.029 | 26.332 +/- 0.055 |
| concurrency8 | aggregate output tok/s | 74.668 +/- 0.278 | 66.236 +/- 0.147 |

The arithmetic mean of the eight ordinary-prompt means is 26.312 tok/s for mistral.rs and 26.462 for patched llama.cpp. This is not a pooled throughput measurement; no repetition uncertainty is assigned to this cross-prompt mean.

## Final Flash-Next Q4K + adaptive MTP

| Engine | Concurrency | Aggregate output tok/s | Per-active output tok/s |
| --- | ---: | ---: | ---: |
| mistral.rs | 1 | 54.703 +/- 1.232 | 54.711 +/- 1.232 |
| mistral.rs | 6 | 119.848 +/- 2.457 | 20.800 +/- 0.350 |
| mistral.rs | 8 | 131.742 +/- 1.072 | 17.321 +/- 0.218 |

## Same-checkpoint Qwen3.5 BF16, target-only

| Engine | Concurrency | Aggregate output tok/s | Per-active output tok/s |
| --- | ---: | ---: | ---: |
| mistral.rs | 1 | 31.843 +/- 0.008 | 31.847 +/- 0.008 |
| vLLM | 1 | 30.921 +/- 0.015 | 30.923 +/- 0.015 |
| mistral.rs | 6 | 84.326 +/- 0.171 | 14.062 +/- 0.027 |
| vLLM | 6 | 72.505 +/- 0.124 | 12.089 +/- 0.021 |
| mistral.rs | 8 | 93.854 +/- 0.166 | 11.737 +/- 0.021 |
| vLLM | 8 | 95.466 +/- 0.164 | 11.939 +/- 0.019 |
