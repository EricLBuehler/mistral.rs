# mistral.rs v0.9.0 CPU Benchmark Report

CPU-only comparison of mistral.rs against llama.cpp on GB10 (aarch64: 10x Cortex-X925 + 10x Cortex-A725), covering Qwen3 4B, Gemma 4 E4B, and LFM2.5 230M at Q4_K, Q6_K, and Q8_0. Values are tokens per second; speedups are mistral.rs divided by llama.cpp at the same length or depth. Both engines run pinned to the 10 big cores at their best configuration (see the affinity study).

This report reflects the post-optimization state: an initial sweep identified decode and small-model gaps, a day of optimization closed them (and uncovered a latent correctness bug, fixed below), and the final sweep below is measured on the fixed, optimized build.

## Headline Results

| Model | Quant | Prefill mean speedup | Decode mean speedup |
|---|---|---:|---:|
| gemma4-e4b | Q4_K | 2.75x | 1.31x |
| gemma4-e4b | Q6_K | 2.72x | 1.17x |
| gemma4-e4b | Q8_0 | 2.08x | 1.16x |
| qwen3-4b | Q4_K | 1.43x | 1.06x |
| qwen3-4b | Q6_K | 1.51x | 1.01x |
| qwen3-4b | Q8_0 | 1.18x | 1.03x |
| lfm2.5-230m | Q4_K | 0.92x | 1.10x |
| lfm2.5-230m | Q6_K | 0.96x | 1.03x |
| lfm2.5-230m | Q8_0 | 0.90x | 0.98x |

![CPU speedup bars](figures/cpu_speedup_bars.png)

![Gemma 4 E4B CPU throughput](figures/gemma4-e4b_cpu_throughput.png)
![Qwen3 4B CPU throughput](figures/qwen3-4b_cpu_throughput.png)
![LFM2.5 230M CPU throughput](figures/lfm2_5-230m_cpu_throughput.png)

Observations:

- Every 4B-class mean is at or above llama.cpp, prefill and decode: gemma4-e4b decodes 1.16x to 1.31x faster and prefills 2.1x to 2.75x faster; qwen3-4b decodes 1.01x to 1.06x faster and prefills 1.18x to 1.51x faster. llama.cpp runs with flash attention on (its `-fa auto` default resolves to on, verified as its faster configuration).
- Long-context prefill, the gap in earlier drafts of this report, is now won: qwen3-4b at 8192 tokens runs 1.04x (Q8_0) to 1.11x (Q4_K) ahead of llama.cpp's flash kernel after the blocked attention work below.
- lfm2.5-230m sits at parity on decode means (0.98x to 1.10x) and slightly behind on prefill means (0.90x to 0.96x), with 8192-token prefill at 0.7x the one structural residual. Profiling attributes it to per-op fixed costs that only a 230M model exposes: single-threaded elementwise chains and per-op allocation churn while pool workers idle. The scoped follow-up is parallel elementwise maps in candle plus output-buffer reuse.
- The initial sweep had decode at 0.6x to 0.7x on the 4B models, 0.4x to 0.6x on LFM2.5, and long prefill at 0.5x to 0.75x. The optimization changes are summarized below.

## Correctness fix uncovered by this work

The optimization pass exposed a latent bug shipped with the aarch64 repacking kernels: the q4k/q5k tiled (m >= 4) matmul kernels read the interleaved q8k `bsums` with a row-major index while the quantizer stores them quarter-major, corrupting the `dmin` bias term. Any prefill whose token count is a multiple of 4 (i.e., every real chunked prefill) produced garbage activations; short chat prompts happened to take the non-multiple-of-4 generic path and benchmarks used random tokens, so nothing caught it.

Fixed in candle commit `25b498cd` with a repack-vs-reference regression test across m in {1, 4, 8, 23, 512} and all repacked quant types. Verified end to end with long-context recall prompts on qwen3-4b and gemma4-e4b.

## Optimization summary (initial sweep -> this report)

All measured on qwen3-4b q4k unless noted; each change validated by unit tests plus long-context generation checks.

1. Dynamic chunked dispatch in candle's barrier pool (`execute_chunked`, atomic cursor) replacing static per-thread slices in all repacked matmul kernels. Worker spin fell from 32% to 22% of cycles; LFM2.5 decode went from 159 to 456 t/s (2.9x) - tiny models were almost pure imbalance loss.
2. Decode attention (`single_q.rs`) rewritten: kv-axis splitting with online-softmax partial merge, GQA grouping (stream K/V once per kv head for the 4 to 8 q rows that share it), adaptive unit granularity. Deep decode flipped from far behind to ahead (d8192: 8.8 -> 16.7 t/s vs llama.cpp 15.6).
3. CPU fused qkv and gate/up projections sharing one barrier region and one lhs quantization (candle `QTensor::gemv_fused_shared_lhs` + mistralrs wiring), cutting ~72 barrier crossings per token.
4. Sliding-cursor rotating KV cache: the sliding-window cache now slides through slack capacity and relocates once per slack run instead of shifting the whole window every token.
5. `fused_glu` CPU moved from rayon to the barrier pool. This was the single largest win: rayon threads were fighting the barrier workers spinning between matmuls on the same pinned cores. Gemma decode +46%, qwen decode +14%.
6. Prefill attention (`full.rs`): q-blocked K/V streaming (8 query rows share each K/V pass), barrier pool instead of rayon, and binary-searched live kv ranges per row.
7. Blocked prefill attention kernel, the long-context unlock: scores a 128-position KV tile per pass with contiguous mask-row slices, applies the online-softmax correction once per tile instead of per position, and accumulates P*V with each V row shared across the q block. The tile restructure alone reached llama.cpp flash-kernel parity at 8192 tokens (47.8 -> 64.5 t/s on qwen3-4b Q8_0); NEON micro-kernels (4-wide dot with shared q registers, vectorized polynomial exp) pushed past it (67.0 t/s, 1.04x; Q4_K 70.7 t/s, 1.11x).
8. Direct CPU kernels for the GDN path used by qwen3-next-style models (fused causal conv1d + gated-delta-rule time scan), replacing per-timestep tensor-op chains; measured via unit tests, these do not affect the three benchmark models.

## Out-of-the-Box Decode

With both engines at stock settings (no pinning anywhere; llama.cpp default `-t 20`), mistral.rs leads decode comfortably: its default thread sizing already avoids the little cores, while llama.cpp's decode loop loses over 2x to core stragglers until manually pinned. The affinity-study defaults rows below quantify this; the headline comparison above deliberately gives llama.cpp its best (pinned) configuration.

## CPU Affinity Study

Measured on qwen3-4b (prefill 2048, decode at depth 512) before the optimization pass; the structural conclusions are unchanged. `mask` is engine-internal pinning (`CANDLE_CPU_MASK=5-9,15-19` for mistral.rs; `-C 0xF83E0 --cpu-strict 1 -t 10` for llama.cpp); `taskset` is OS-level pinning; `t10` is llama.cpp with 10 threads unpinned.

![Affinity study](figures/affinity_study.png)

| Engine | Quant | Strategy | Prefill T/s | Decode T/s |
|---|---|---|---:|---:|
| mistral.rs | Q4_K | default | 103.3 | 19.4 |
| mistral.rs | Q4_K | mask | 101.8 | 28.3 |
| mistral.rs | Q4_K | taskset | 101.0 | 28.1 |
| mistral.rs | Q8_0 | default | 88.6 | 12.0 |
| mistral.rs | Q8_0 | mask | 89.5 | 18.0 |
| mistral.rs | Q8_0 | taskset | 88.8 | 18.1 |
| llama.cpp | Q4_K | default (t20) | 83.6 | 15.9 |
| llama.cpp | Q4_K | t10 | 64.3 | 34.5 |
| llama.cpp | Q4_K | mask | 65.2 | 34.5 |
| llama.cpp | Q4_K | taskset | 65.3 | 34.9 |
| llama.cpp | Q8_0 | default (t20) | 83.5 | 10.6 |
| llama.cpp | Q8_0 | t10 | 66.3 | 23.1 |
| llama.cpp | Q8_0 | mask | 66.3 | 23.2 |
| llama.cpp | Q8_0 | taskset | 66.2 | 23.1 |

Findings:

- Pinning decode to the 10 big cores is a large win for both engines: 2.2x for llama.cpp, +46% for mistral.rs (pre-optimization). The little cores actively hurt decode on this big.LITTLE part.
- The pinning mechanism does not matter: engine-internal masks and OS `taskset` are within noise; llama.cpp's thread-count alone (`t10`) matches pinning, so the win is straggler avoidance.
- llama.cpp prefill is the exception: it prefers all 20 threads (+28% over pinned).
- mistral.rs prefill is insensitive to pinning, so `CANDLE_CPU_MASK=5-9,15-19` is safe to recommend globally on this host.

## Method

- Workloads: prompt lengths and decode depths of 128, 512, 2048, 4096, and 8192 tokens (qwen3-4b Q4_K additionally has 16384); 256 generated tokens per decode depth; 1 warmup and 2 measured iterations per point.
- CPU-only builds: mistral.rs without GPU features, run with `--cpu`; llama.cpp with `GGML_CUDA=OFF GGML_NATIVE=ON` (Release).
- Quantized comparisons: mistral.rs ISQ `q4k`/`q6k`/`q8_0` (benchmarked from prequantized UQFF generated by `mistralrs quantize`, numerically identical to `--isq`) versus llama.cpp GGUF `Q4_K_M`/`Q6_K`/`Q8_0`. ISQ Q4K is uniform; GGUF Q4_K_M mixes Q4_K/Q6_K per tensor, so the 4-bit tiers are close but not bit-identical schemes.
- Affinity: mistral.rs under `CANDLE_CPU_MASK=5-9,15-19`; llama.cpp prefill at stock `-t 20`, decode under `taskset -c 5-9,15-19 -t 10` (each engine's best).
- The box was otherwise idle; any contended runs were discarded and rerun.

## Commands and Reproducibility

- `scripts/bench_cpu_sweep.py` - sweep orchestrator (`--phase affinity|full`, `--engines`, `--mrs-uqff`); one JSON row per measurement appended to `raw/results_full.jsonl` (later rows supersede earlier ones for the same point), raw engine stdout under `raw/raw_full/`.
- `scripts/plot_results.py` - regenerates all figures.
- `scripts/capture_metadata.sh` - host/commit/model metadata (`raw/metadata.txt`).

```bash
# affinity study
python3 releases/v0.9.0/scripts/bench_cpu_sweep.py --phase affinity

# full sweep at best-per-engine affinity
python3 releases/v0.9.0/scripts/bench_cpu_sweep.py --phase full \
  --mrs-mode mask --lcpp-mode default --lcpp-decode-mode taskset \
  --iters 2 --warmup 1 --gen-len 256 --lengths 128,512,2048,4096,8192
```

Engine command shapes:

```bash
# mistral.rs (ISQ from BF16 safetensors; or --from-uqff a file made by `mistralrs quantize`)
CANDLE_CPU_MASK=5-9,15-19 target/release/mistralrs bench --cpu \
  --prompt-len 128,512,2048,4096,8192 --depth 128,512,2048,4096,8192 \
  --gen-len 256 --iterations 2 --warmup 1 -m Qwen/Qwen3-4B --isq q4k

# llama.cpp prefill / decode
llama.cpp/build-cpu/bin/llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 128,...,8192 -n 0 -r 2 -o json -t 20
taskset -c 5-9,15-19 llama.cpp/build-cpu/bin/llama-bench -m Qwen3-4B-Q4_K_M.gguf \
  -p 0 -n 256 -d 128,...,8192 -r 2 -o json -t 10
```

### Model artifacts

| Artifact | HF repo id | Use |
|---|---|---|
| Qwen3 4B BF16 | Qwen/Qwen3-4B | mistral.rs `--isq` source |
| Gemma 4 E4B BF16 | google/gemma-4-E4B-it | mistral.rs `--isq` source |
| LFM2.5 230M BF16 | LiquidAI/LFM2.5-230M | mistral.rs `--isq` source |
| Qwen3 4B GGUF | Qwen/Qwen3-4B-GGUF | llama.cpp Q4_K_M / Q6_K / Q8_0 |
| Gemma 4 E4B GGUF | unsloth/gemma-4-E4B-it-GGUF | llama.cpp Q4_K_M / Q6_K / Q8_0 |
| LFM2.5 230M GGUF | LiquidAI/LFM2.5-230M-GGUF | llama.cpp Q4_K_M / Q6_K / Q8_0 |

### Versions and host

| Component | Commit or version |
|---|---|
| mistral.rs | 4f6042b41 (master) + CPU perf changes from this campaign (see report body; commits pending) |
| candle | 78e1d851 + 25b498cd (bsums correctness fix) + CPU perf changes (commits pending) |
| llama.cpp | 2d973636e292ee6f75fadcf08d29cb33511f509f |
| rustc | 1.96.1 |

Host: GB10 (spark-4ec2), Linux 6.17.0-1021-nvidia, 20 cores (10x Cortex-X925 3.9 GHz on CPUs 5-9/15-19, 10x Cortex-A725 2.8 GHz on CPUs 0-4/10-14), 1 NUMA node. Full details in `raw/metadata.txt`.

## Appendix: Full Tables

All values are tokens per second; speedup is mistral.rs divided by llama.cpp in the same row.

#### qwen3-4b

##### Q4_K Prefill

| Length | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 143.7 | 90.0 | 1.598x |
| 512 | 180.4 | 90.6 | 1.992x |
| 2048 | 132.5 | 83.6 | 1.585x |
| 4096 | 100.7 | 76.0 | 1.325x |
| 8192 | 72.3 | 64.2 | 1.127x |
| 16384 | 46.2 | 48.9 | 0.945x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 39.9 | 36.9 | 1.081x |
| 512 | 37.9 | 34.4 | 1.103x |
| 2048 | 29.4 | 27.6 | 1.064x |
| 4096 | 24.2 | 22.0 | 1.098x |
| 8192 | 16.6 | 15.6 | 1.067x |
| 16384 | 8.8 | 9.3 | 0.946x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 108.3 | 69.1 | 1.567x |
| 512 | 127.9 | 70.3 | 1.819x |
| 2048 | 104.2 | 66.0 | 1.580x |
| 4096 | 84.4 | 61.4 | 1.375x |
| 8192 | 63.8 | 53.3 | 1.197x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 29.6 | 29.2 | 1.015x |
| 512 | 28.0 | 28.0 | 0.999x |
| 2048 | 23.7 | 23.2 | 1.024x |
| 4096 | 19.6 | 19.1 | 1.027x |
| 8192 | 14.2 | 14.1 | 1.007x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 94.0 | 89.4 | 1.051x |
| 512 | 123.0 | 91.6 | 1.343x |
| 2048 | 109.2 | 83.7 | 1.304x |
| 4096 | 87.9 | 76.5 | 1.148x |
| 8192 | 67.3 | 64.4 | 1.045x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 24.8 | 24.1 | 1.028x |
| 512 | 23.7 | 23.2 | 1.023x |
| 2048 | 20.6 | 19.8 | 1.040x |
| 4096 | 17.4 | 16.6 | 1.048x |
| 8192 | 12.9 | 13.0 | 0.990x |

#### gemma4-e4b

##### Q4_K Prefill

| Length | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 195.2 | 72.4 | 2.697x |
| 512 | 243.9 | 75.2 | 3.244x |
| 2048 | 211.2 | 72.3 | 2.921x |
| 4096 | 176.7 | 70.5 | 2.505x |
| 8192 | 159.2 | 67.1 | 2.374x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 32.9 | 25.1 | 1.311x |
| 512 | 31.8 | 24.3 | 1.311x |
| 2048 | 30.0 | 23.4 | 1.284x |
| 4096 | 29.3 | 22.6 | 1.295x |
| 8192 | 26.1 | 19.7 | 1.325x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 152.2 | 57.2 | 2.659x |
| 512 | 184.0 | 59.7 | 3.082x |
| 2048 | 168.3 | 58.4 | 2.880x |
| 4096 | 147.8 | 57.2 | 2.584x |
| 8192 | 133.2 | 55.2 | 2.411x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 23.9 | 20.4 | 1.171x |
| 512 | 23.4 | 20.1 | 1.161x |
| 2048 | 22.7 | 19.5 | 1.167x |
| 4096 | 21.9 | 18.9 | 1.160x |
| 8192 | 20.1 | 16.7 | 1.202x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 131.8 | 72.2 | 1.826x |
| 512 | 164.9 | 75.6 | 2.183x |
| 2048 | 165.7 | 73.2 | 2.263x |
| 4096 | 146.9 | 71.5 | 2.055x |
| 8192 | 141.8 | 68.5 | 2.071x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 20.1 | 17.5 | 1.149x |
| 512 | 19.7 | 17.2 | 1.147x |
| 2048 | 19.3 | 16.8 | 1.152x |
| 4096 | 18.7 | 16.2 | 1.151x |
| 8192 | 17.4 | 14.6 | 1.190x |

#### lfm2.5-230m

##### Q4_K Prefill

| Length | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1229.6 | 1255.5 | 0.979x |
| 512 | 1678.9 | 1624.3 | 1.034x |
| 2048 | 1506.4 | 1484.5 | 1.015x |
| 4096 | 1223.7 | 1359.5 | 0.900x |
| 8192 | 816.7 | 1181.6 | 0.691x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 469.0 | 509.5 | 0.920x |
| 512 | 457.5 | 482.6 | 0.948x |
| 2048 | 447.4 | 404.4 | 1.106x |
| 4096 | 444.5 | 337.6 | 1.317x |
| 8192 | 299.0 | 252.3 | 1.185x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1080.6 | 1278.7 | 0.845x |
| 512 | 1480.4 | 1363.5 | 1.086x |
| 2048 | 1439.6 | 1287.3 | 1.118x |
| 4096 | 1138.2 | 1183.5 | 0.962x |
| 8192 | 786.4 | 1027.1 | 0.766x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 471.5 | 429.1 | 1.099x |
| 512 | 475.4 | 414.7 | 1.146x |
| 2048 | 305.1 | 359.2 | 0.849x |
| 4096 | 316.4 | 301.8 | 1.048x |
| 8192 | 233.7 | 232.6 | 1.005x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1356.4 | 1744.8 | 0.777x |
| 512 | 1876.7 | 1749.1 | 1.073x |
| 2048 | 1663.7 | 1622.3 | 1.026x |
| 4096 | 1318.9 | 1462.0 | 0.902x |
| 8192 | 860.9 | 1230.0 | 0.700x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 299.8 | 368.9 | 0.813x |
| 512 | 324.3 | 353.7 | 0.917x |
| 2048 | 312.3 | 308.6 | 1.012x |
| 4096 | 298.7 | 267.6 | 1.116x |
| 8192 | 224.9 | 211.2 | 1.065x |
