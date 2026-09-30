# Qwen3.5-35B-A3B BF16 engine comparison

Five measured trials after two warmups per concurrency; values are mean +/- sample standard deviation.

| Engine | C | Aggregate tok/s | Per-active tok/s | Mean active requests | Mean request latency, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| mistralrs | 1 | 31.843 +/- 0.008 | 31.847 +/- 0.008 | 1.000 +/- 0.000 | 4.019 +/- 0.001 |
| mistralrs | 6 | 84.326 +/- 0.171 | 14.062 +/- 0.027 | 5.997 +/- 0.001 | 9.103 +/- 0.018 |
| mistralrs | 8 | 93.854 +/- 0.166 | 11.737 +/- 0.021 | 7.996 +/- 0.001 | 10.905 +/- 0.020 |
| vllm | 1 | 30.921 +/- 0.015 | 30.923 +/- 0.015 | 1.000 +/- 0.000 | 4.139 +/- 0.002 |
| vllm | 6 | 72.505 +/- 0.124 | 12.089 +/- 0.021 | 5.997 +/- 0.000 | 10.588 +/- 0.018 |
| vllm | 8 | 95.466 +/- 0.164 | 11.939 +/- 0.019 | 7.996 +/- 0.001 | 10.721 +/- 0.017 |

| Comparison | C1 | C6 | C8 |
| --- | ---: | ---: | ---: |
| mistral.rs / vLLM aggregate rate | 1.030x | 1.163x | 0.983x |
| mistralrs, relative to own C1 | 1.000x | 2.648x | 2.947x |
| vllm, relative to own C1 | 1.000x | 2.345x | 3.087x |

Operating ranges below cover whole phase commands, including warmups. Swap counters are system-wide, not GPU page-in measurements.

| Engine | Phase | SM MHz range | Power W range | GPU C range | Global swap-in/out MiB | Process VmSwap before/after MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mistralrs | c1 | 2476.0-2522.0 | 19.1-37.1 | 56.0-61.0 | 44.17 / 0.00 | 198.04 / 195.75 |
| mistralrs | c6 | 2411.0-2522.0 | 13.3-39.5 | 56.0-63.0 | 81.44 / 0.00 | 195.75 / 195.62 |
| mistralrs | c8 | 2411.0-2502.0 | 17.4-39.9 | 60.0-65.0 | 12.71 / 0.00 | 195.62 / 195.56 |
| vllm | c1 | 2405.0-2535.0 | 11.6-35.0 | 47.0-60.0 | 28.06 / 0.00 | 0.00 / 0.00 |
| vllm | c6 | 2411.0-2522.0 | 14.7-32.9 | 56.0-61.0 | 206.17 / 0.00 | 0.00 / 0.00 |
| vllm | c8 | 2496.0-2522.0 | 22.5-33.0 | 58.0-61.0 | 4.43 / 0.00 | 0.00 / 0.00 |

vLLM process VmSwap refers to its container init process and can omit model-worker children. The precision contract is BF16 weights/attention KV/convolution state and F32 GDN recurrent state, supported by the checkpoint and pinned implementation sources.

All request bodies, ordered prompts, prompt counts, checkpoint config, tokenizer, and frozen harness hashes match. Output text agreement is recorded in comparison.json without requiring identical continuations.

- Five measured closed-loop trials follow two excluded warmups at each concurrency; finite trial startup and drain are included.
- Per-active throughput is output tokens divided by summed request latency, not aggregate throughput divided by requested concurrency.
- Reported variability is sample standard deviation across five trial rates; engine and scaling ratios divide means and have no inferred confidence interval.
- The ordered prompts and request bodies match, but engine arithmetic, generated text, expert routes, and scheduling can differ.
- Physical KV-cache allocation differs despite the same 16384-token limit and eight-sequence cap; report both capacities and memory observations.
- Runs are sequential, not randomized; clocks, cache state, and memory pressure may differ.
- These target-only BF16 results do not measure Flash-Next low-bit experts or adaptive MTP performance.
- Counter snapshots include warmups and are engine-specific; absent counters are unavailable, not zero.
- Clock/power/temperature ranges use each whole command window including warmups; frozen trial records lack absolute per-trial boundaries.
- Swap deltas are global system counters; process VmSwap snapshots do not identify GPU paging or its timing cost.

Physical cache settings and observed startup capacities, engine identities, raw trial samples, and input hashes are recorded in comparison.json.
