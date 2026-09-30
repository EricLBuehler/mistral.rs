# Qwen3.5-35B-A3B BF16: mistral.rs and vLLM on GB10

On the same Qwen3.5-35B-A3B BF16 checkpoint, mistral.rs reaches 93.85 tok/s at concurrency eight and vLLM reaches 95.47 tok/s. Relative to each engine's concurrency-one result, that is 2.95x and 3.09x. Mistral.rs is 3.0% faster at C1, 16.3% faster at C6, and 1.7% slower at C8 in these runs. Limited C8 scaling therefore appears in both tested engines on this BF16 MoE workload; it is not unique to the Flash-Next low-bit path. This comparison does not establish a hardware ceiling or predict vLLM's Flash-Next/MTP performance.

The September 30, 2026 runs use one NVIDIA GB10 and the same local checkpoint, tokenizer, ordered raw completion prompts, and frozen client. MTP and quantization are disabled. Both engines use BF16 weights, attention KV, and convolution state, with F32 GDN recurrent state established from the checkpoint and pinned implementation sources. The configured context limit is 16,384 tokens, maximum active sequences eight, and maximum batched tokens 4,096. Prefix caching is disabled. Each request asks for 128 output tokens with temperature zero, the same seed, and `ignore_eos=true`.

Five measured closed-loop trials follow two excluded warmups at each concurrency. C1 trials contain eight requests; C6 and C8 trials contain 24. Values below are mean +/- sample standard deviation across the five measured trials. Trial startup, prefill, and drain are included.

| Engine | Concurrency | Aggregate output tok/s | Per-active output tok/s | Mean request latency, s |
| --- | ---: | ---: | ---: | ---: |
| mistral.rs | 1 | 31.843 +/- 0.008 | 31.847 +/- 0.008 | 4.019 +/- 0.001 |
| vLLM | 1 | 30.921 +/- 0.015 | 30.923 +/- 0.015 | 4.139 +/- 0.002 |
| mistral.rs | 6 | 84.326 +/- 0.171 | 14.062 +/- 0.027 | 9.103 +/- 0.018 |
| vLLM | 6 | 72.505 +/- 0.124 | 12.089 +/- 0.021 | 10.588 +/- 0.018 |
| mistral.rs | 8 | 93.854 +/- 0.166 | 11.737 +/- 0.021 | 10.905 +/- 0.020 |
| vLLM | 8 | 95.466 +/- 0.164 | 11.939 +/- 0.019 | 10.721 +/- 0.017 |

Per-active throughput divides output tokens by summed request latency. Measured mean active counts are approximately 1.000, 5.997, and 7.996 for both engines, so the comparison is not explained by one engine receiving substantially less concurrent work. Ratios divide trial means; no confidence interval is inferred from the sample standard deviations.

| Engine | C6 / C1 aggregate throughput | C8 / C1 aggregate throughput |
| --- | ---: | ---: |
| mistral.rs | 2.648x | 2.947x |
| vLLM | 2.345x | 3.087x |

The native executable reports mistralrs 0.9.4 and contains the runtime changes committed as `74a469b6d402bf3795bb3b8eb40920d242185ae9`. It was built before that commit with embedded revision `ec33e3a75`; exact source snapshots and build-time changes are retained. Its SHA256 is `92c050e15bbfe15fc99ef11077366a8fb40db5af96223c67e0f9299fce23f2fa`. vLLM 0.28.0 runs from immutable image `sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f`, whose source commit is `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Its startup selects FlashInfer CUTLASS unquantized MoE, FlashAttention 2, Triton/FLA GDN prefill, and CUDA GDN decode. Those vLLM backend labels are startup selection evidence, not a kernel trace.

A separate native diagnostic records `fused_moe_kernel` inside CUDA graphs at both C1 and C8: 19,840 and 2,480 observed graph-kernel launches respectively. This confirms execution of the BF16 cuTile MoE path during graph work. The diagnostic uses eight requests with 32 outputs each and is excluded from the throughput table. Its startup captured three graph shapes and deferred five when reported free device memory reached 2,028 MiB; the unprofiled native run captured all eight. The [shape-specific proof](raw/model_scaling/qwen3.5_bf16/kernel_trace/analysis/graph_shape_proof.json) identifies 30 B8 graph steps, one B1 graph step, and one eager B7 tail in the C8 phase; B8 was captured lazily during warmup. The trace establishes kernel execution, not identical graph coverage or profiling-free performance.

All 280 measured requests per engine completed with the expected prompt counts and exactly 128 output tokens. Ordered request bodies, checkpoint config and index, tokenizer, and harness hashes match. vLLM omits cached-token counts in responses; its effective server configuration establishes that prefix caching is disabled.

Raw decoded strings differ in all 280 pairs, but that is substantially a presentation difference. Stripping leading whitespace makes 77 pairs identical; additionally removing only visible `<|endoftext|>`, `<|im_start|>`, and `<|im_end|>` markers makes 134 identical. Under that explicit normalization, 244 pairs share their first 128 characters. Later content can differ. This character comparison neither establishes generated-token equality nor measures expert-route agreement; differing continuations and engine arithmetic remain comparison limits.

Decode graph policies also differ: the native unprofiled run captures exact batch sizes 1 through 8, while vLLM uses full-decode buckets 1, 2, 4, and 8; its mixed-prefill graph sizes also include 16. C6 is not an exact vLLM graph bucket, while C8 is present in both engines. That policy difference is a possible contributor to the C6 result, not a demonstrated cause of the 16.3% gap.

Physical cache allocation differs: native logs 321 MB of attention KV with 513 blocks of 32 tokens, while vLLM uses an explicit 4 GiB hybrid cache budget and reports 170,738 cache tokens. Those are engine-specific capacities, not matching physical layouts. The runs are sequential, not randomized. Whole-phase clock/temperature monitoring includes warmups: observed SM clocks span 2,411-2,522 MHz for native and 2,405-2,535 MHz for vLLM; GPU temperatures span 56-65 C and 47-61 C respectively. System-wide swap-in occurs in both runs, with no swap-out; it does not establish GPU paging or its timing cost. The detailed phase ranges and process-swap limitations are retained in the evidence.

The [compact evidence archive](raw/model_scaling/qwen3.5_bf16/README.md) contains raw trials, exact commands and identities, operating-condition snapshots, validated summaries, and the separate native kernel proof. Its [manifest](raw/model_scaling/qwen3.5_bf16/SHA256SUMS.json) covers every archived file except itself. The first vLLM attempt failed a port preflight before Docker/GPU launch and is preserved separately, outside the statistics. Large caches, tokenizer copies, model weights, binaries, and profiler databases remain external with identity records.
