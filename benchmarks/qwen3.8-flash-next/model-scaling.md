# Current Flash-Next model scaling

The current whole-model profile localizes the weak C1-to-C8 batching gain mainly to expert projections: their recorded time per output improves 1.54x, while all other kernels together improve 4.13x. This complements the [selected-weight and route controls](optimization.md#remaining-expert-kernel-headroom); it does not establish a hardware ceiling or predict another engine's throughput.

The September 30 capture uses the same `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d` binary as the [final serving rerun](optimization.md#final-serving-rerun): Q4K ISQ, adaptive MTP, masked MMQ loads, and the scoped 24-31-row MMQ dispatch. It precedes the depth-four graph-policy change. Each phase has one warmed closed-loop trial, with eight requests at C1 and 24 at C8, each generating 128 tokens. The finite envelopes include prefill, request startup, and drain. This instrumented diagnostic is separate from the repeated unprofiled benchmark.

| Recorded kernel group | C1 ms/output | C8 ms/output | Cost reduction |
| --- | ---: | ---: | ---: |
| Expert projections | 7.712 | 5.016 | 1.54x |
| Vocabulary projection | 2.250 | 0.435 | 5.18x |
| GDN projections | 1.985 | 0.417 | 4.77x |
| Hyper-connection projections | 1.804 | 0.363 | 4.97x |
| Attention projections | 0.771 | 0.155 | 4.96x |
| GDN recurrence and convolution | 0.357 | 0.308 | 1.16x |
| All except expert projections | 9.225 | 2.233 | 4.13x |
| All kernels | 16.937 | 7.249 | 2.34x |

These are summed recovered kernel durations per completed output, not disjoint wall-time components or pure decode costs. Expert projections account for 69.19% of C8 summed kernel time. Routing, reduction, and grouped activation quantization together cost 0.074 ms/output. Projection matmuls are included explicitly; a small recurrence or attention-kernel category alone would not establish that the entire feature is cheap. [Complete accounting](raw/model_scaling/current_profile/semantic/interpretation.md) includes the remaining categories and denominators.

| Phase measure | C1 | C8 |
| --- | ---: | ---: |
| Accepted / proposed draft tokens | 77.75% | 76.44% |
| Mean proposed depth per sequence proposal | 3.359 | 3.789 |
| Target graph replays / eager dispatches | 289 / 0 | 59 / 65 |
| Target-shaped graph kernel-time coverage | 94.79% | 28.76% |
| Observed GPU activity / request-envelope time | 90.14% | 95.14% |

Acceptance does not collapse at C8. All 65 eager target dispatches report `batch_unsupported`; prefill is counted separately. Target-shaped interval counts and draft vocabulary rows independently match telemetry. Mixed target/draft intervals remain unassigned, accounting for 10.43% of C8 summed kernel time. Identified draft-shaped work remains eager and occupies 8.77%; these stage fractions do not include the unassigned work.

The C8 envelope contains 22.303 seconds of observed GPU activity in 23.442 seconds. Removing every observed gap while holding recorded GPU work fixed would give 1.051x. That conditional calculation does not bound changes to kernels, accepted work, or GPU scheduling. It shows why observed launch gaps alone cannot explain a several-fold gain. Nsight warns of potentially missing CUDA/OS runtime events and provides no unified-memory or CPU scheduling trace here; the activity fraction is not SM utilization or measured DRAM bandwidth.

Commit `74a469b6d402bf3795bb3b8eb40920d242185ae9` enables depth-four verification graphs through batch eight. Policy, 35/40-row changed-route graph replay, and Q5 GDN prefix-replay checks pass, including the MMQ graph sanitizer check. A Flash-Next full-model benchmark of that change has not yet run, so no speedup is claimed. A separate same-checkpoint Qwen3.5-35B-A3B BF16 comparison between mistral.rs and vLLM at C1/C6/C8, with MTP disabled, is being measured; its results are not inferred from this Flash-Next trace.

The [compact archive](raw/model_scaling/current_profile/README.md) retains requests, metrics, alignment checks, profiler warnings, analysis scripts, and hashes. Large profiler reports and SQLite databases remain external with recorded paths and SHA256 hashes. Its [manifest](raw/model_scaling/current_profile/SHA256SUMS.json) covers every archived file except itself.
