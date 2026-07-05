# mistral.rs v0.9.0 CPU Benchmark Report

CPU-only comparison of mistral.rs against llama.cpp on GB10 (aarch64: 10x Cortex-X925 + 10x Cortex-A725), covering Qwen3 4B, Gemma 4 E4B, and LFM2.5 230M at Q4_K, Q6_K, and Q8_0. Values are tokens per second; speedups are mistral.rs divided by llama.cpp at the same length or depth. Both engines run pinned to the 10 big cores at their best configuration (see the affinity study).

This report reflects the post-optimization state: an initial sweep identified decode and small-model gaps, a day of optimization closed them (and uncovered a latent correctness bug, fixed below), and the final sweep below is measured on the fixed, optimized build.

## Headline Results

| Model | Quant | Prefill mean speedup | Decode mean speedup |
|---|---|---:|---:|
| gemma4-e4b | Q4_K | 2.71x | 1.31x |
| gemma4-e4b | Q6_K | 2.67x | 1.17x |
| gemma4-e4b | Q8_0 | 2.01x | 1.16x |
| qwen3-4b | Q4_K | 1.18x | 1.08x |
| qwen3-4b | Q6_K | 1.31x | 1.02x |
| qwen3-4b | Q8_0 | 1.01x | 1.03x |
| lfm2.5-230m | Q4_K | 0.81x | 1.07x |
| lfm2.5-230m | Q6_K | 0.83x | 1.01x |
| lfm2.5-230m | Q8_0 | 0.78x | 1.00x |

![CPU speedup bars](figures/cpu_speedup_bars.png)

![Gemma 4 E4B CPU throughput](figures/gemma4-e4b_cpu_throughput.png)
![Qwen3 4B CPU throughput](figures/qwen3-4b_cpu_throughput.png)
![LFM2.5 230M CPU throughput](figures/lfm2_5-230m_cpu_throughput.png)

Observations:

- Decode is at or above llama.cpp on every model/quant mean: gemma4-e4b 1.16x to 1.31x, qwen3-4b 1.02x to 1.08x, lfm2.5-230m 1.00x to 1.07x. The weakest single decode points are shallow LFM depths at 0.88x to 0.92x, within that tiny model's run-to-run noise.
- Gemma 4 E4B prefill is 2.0x to 2.7x across all quants (peak 3.19x).
- Known gap: full-attention prefill at long context (8192+) runs 0.5x to 0.75x on qwen3 and lfm2.5. llama.cpp's blocked fp16 attention kernels win there; a blocked SIMD CPU attention kernel is the scoped follow-up. Gemma is unaffected (sliding windows keep its prefill attention small).
- These decode results are a large shift from the initial sweep, which had decode at 0.6x to 0.7x on the 4B models and 0.4x to 0.6x on LFM2.5. The optimization changes are summarized below.

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
6. Prefill attention (`full.rs`): q-blocked K/V streaming (8 query rows share each K/V pass), barrier pool instead of rayon, fast polynomial exp for the online softmax, and binary-searched live kv ranges per row.

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
| 128 | 137.8 | 90.0 | 1.532x |
| 512 | 168.6 | 90.6 | 1.861x |
| 2048 | 109.1 | 83.6 | 1.305x |
| 4096 | 75.8 | 76.0 | 0.997x |
| 8192 | 49.4 | 64.2 | 0.770x |
| 16384 | 29.9 | 48.9 | 0.612x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 40.5 | 36.9 | 1.097x |
| 512 | 38.1 | 34.4 | 1.109x |
| 2048 | 31.0 | 27.6 | 1.122x |
| 4096 | 24.6 | 22.0 | 1.116x |
| 8192 | 16.6 | 15.6 | 1.067x |
| 16384 | 8.8 | 9.3 | 0.946x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 105.4 | 69.1 | 1.525x |
| 512 | 121.3 | 70.3 | 1.726x |
| 2048 | 89.9 | 66.0 | 1.363x |
| 4096 | 66.4 | 61.4 | 1.082x |
| 8192 | 45.8 | 53.3 | 0.859x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 29.8 | 29.2 | 1.022x |
| 512 | 28.2 | 28.0 | 1.006x |
| 2048 | 23.8 | 23.2 | 1.028x |
| 4096 | 19.7 | 19.1 | 1.032x |
| 8192 | 14.3 | 14.1 | 1.014x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 91.6 | 89.4 | 1.025x |
| 512 | 117.1 | 91.6 | 1.278x |
| 2048 | 93.5 | 83.7 | 1.116x |
| 4096 | 68.6 | 76.5 | 0.896x |
| 8192 | 47.8 | 64.4 | 0.742x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 24.7 | 24.1 | 1.024x |
| 512 | 23.7 | 23.2 | 1.023x |
| 2048 | 20.6 | 19.8 | 1.040x |
| 4096 | 17.5 | 16.6 | 1.054x |
| 8192 | 13.0 | 13.0 | 0.997x |

#### gemma4-e4b

##### Q4_K Prefill

| Length | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 199.2 | 72.4 | 2.753x |
| 512 | 239.5 | 75.2 | 3.185x |
| 2048 | 208.0 | 72.3 | 2.877x |
| 4096 | 173.2 | 70.5 | 2.455x |
| 8192 | 151.3 | 67.1 | 2.256x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 32.6 | 25.1 | 1.299x |
| 512 | 31.9 | 24.3 | 1.315x |
| 2048 | 30.1 | 23.4 | 1.288x |
| 4096 | 29.2 | 22.6 | 1.291x |
| 8192 | 26.3 | 19.7 | 1.336x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 153.6 | 57.2 | 2.683x |
| 512 | 180.8 | 59.7 | 3.029x |
| 2048 | 164.1 | 58.4 | 2.808x |
| 4096 | 143.4 | 57.2 | 2.507x |
| 8192 | 128.2 | 55.2 | 2.321x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 23.9 | 20.4 | 1.171x |
| 512 | 23.3 | 20.1 | 1.156x |
| 2048 | 22.7 | 19.5 | 1.167x |
| 4096 | 21.8 | 18.9 | 1.155x |
| 8192 | 20.1 | 16.7 | 1.202x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 129.2 | 72.2 | 1.790x |
| 512 | 161.2 | 75.6 | 2.134x |
| 2048 | 160.6 | 73.2 | 2.193x |
| 4096 | 140.6 | 71.5 | 1.967x |
| 8192 | 135.0 | 68.5 | 1.972x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 20.3 | 17.5 | 1.161x |
| 512 | 19.6 | 17.2 | 1.142x |
| 2048 | 19.3 | 16.8 | 1.152x |
| 4096 | 18.7 | 16.2 | 1.151x |
| 8192 | 17.5 | 14.6 | 1.197x |

#### lfm2.5-230m

##### Q4_K Prefill

| Length | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1241.6 | 1255.5 | 0.989x |
| 512 | 1597.8 | 1624.3 | 0.984x |
| 2048 | 1292.0 | 1484.5 | 0.870x |
| 4096 | 936.5 | 1359.5 | 0.689x |
| 8192 | 601.5 | 1181.6 | 0.509x |

##### Q4_K Decode

| Depth | mistral.rs ISQ q4_k | llama.cpp GGUF Q4_K_M | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 447.6 | 509.5 | 0.878x |
| 512 | 471.1 | 482.6 | 0.976x |
| 2048 | 455.2 | 404.4 | 1.125x |
| 4096 | 407.4 | 337.6 | 1.207x |
| 8192 | 297.1 | 252.3 | 1.178x |

##### Q6_K Prefill

| Length | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1106.4 | 1278.7 | 0.865x |
| 512 | 1428.5 | 1363.5 | 1.048x |
| 2048 | 1209.3 | 1287.3 | 0.939x |
| 4096 | 894.3 | 1183.5 | 0.756x |
| 8192 | 581.6 | 1027.1 | 0.566x |

##### Q6_K Decode

| Depth | mistral.rs ISQ q6_k | llama.cpp GGUF Q6_K | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 470.0 | 429.1 | 1.095x |
| 512 | 430.8 | 414.7 | 1.039x |
| 2048 | 318.7 | 359.2 | 0.887x |
| 4096 | 304.0 | 301.8 | 1.007x |
| 8192 | 233.2 | 232.6 | 1.003x |

##### Q8_0 Prefill

| Length | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 1320.9 | 1744.8 | 0.757x |
| 512 | 1842.6 | 1749.1 | 1.053x |
| 2048 | 1429.1 | 1622.3 | 0.881x |
| 4096 | 1036.7 | 1462.0 | 0.709x |
| 8192 | 640.7 | 1230.0 | 0.521x |

##### Q8_0 Decode

| Depth | mistral.rs ISQ q8_0 | llama.cpp GGUF Q8_0 | mistral.rs speedup |
|---:|---:|---:|---:|
| 128 | 328.8 | 368.9 | 0.891x |
| 512 | 324.3 | 353.7 | 0.917x |
| 2048 | 318.1 | 308.6 | 1.031x |
| 4096 | 296.3 | 267.6 | 1.107x |
| 8192 | 227.3 | 211.2 | 1.076x |
