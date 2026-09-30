# Current C1/C8 kernel accounting

Current binary SHA256: `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d`. Snapshot: `current_d2c85e_masked_mmq24_autotuner`.

The current trace localizes the weak batching gain to expert projections. Their recorded cost per output improves only 1.54x; all other kernels together improve 4.13x. At C8, expert projections occupy 69.19% of summed recovered kernel time. Acceptance remains similar, so the observed scaling loss is not explained by an acceptance collapse.

The table uses the full finite request envelope, including prefill and drain. It reports kernel duration per completed output, not wall-time shares or pure decode. C1 completed 1,024 output tokens and C8 completed 3,072.

| Recorded kernel group | C1 ms/output | C8 ms/output | C1/C8 cost ratio |
| --- | ---: | ---: | ---: |
| Expert projections | 7.712 | 5.016 | 1.54x |
| Vocabulary projection | 2.250 | 0.435 | 5.18x |
| GDN projections | 1.985 | 0.417 | 4.77x |
| Hyper-connection projections | 1.804 | 0.363 | 4.97x |
| Attention projections | 0.771 | 0.155 | 4.96x |
| Shared expert projections | 0.393 | 0.101 | 3.89x |
| GDN recurrence and convolution | 0.357 | 0.308 | 1.16x |
| Attention kernels and cache | 0.257 | 0.056 | 4.55x |
| Other kernels | 1.407 | 0.398 | 3.53x |
| All kernels | 16.937 | 7.249 | 2.34x |
| All except expert projections | 9.225 | 2.233 | 4.13x |

The vocabulary head, GDN projections, hyper-connection projections, and attention projections each amortize roughly 4.8-5.2x. GDN recurrence/convolution amortizes only 1.16x, but costs 0.308 ms/output at C8, compared with 5.016 ms/output for experts. Expert routing, reduction, and grouped activation quantization together are 0.074 ms/output at C8. They are not a hidden large share comparable with the three expert projections.

| Current phase measure | C1 | C8 |
| --- | ---: | ---: |
| Draft token acceptance | 77.75% | 76.44% |
| Mean proposed depth per sequence proposal | 3.359 | 3.789 |
| Accepted draft tokens per sequence proposal | 2.612 | 2.896 |
| Target graph replays / eager dispatches | 289 / 0 | 59 / 65 |
| Target-shaped graph kernel time coverage | 94.79% | 28.76% |
| Draft-shaped graph kernel time coverage | 0% | 0% |
| Draft-shaped share of summed kernel time | 13.68% | 8.77% |
| Draft kernel ms per proposed token | 2.514 | 0.660 |
| Draft vocabulary projection ms per proposed token | 1.644 | 0.309 |
| Proposed draft + sequence rows per output | 1.196 | 1.217 |

All 65 C8 eager target dispatches report `batch_unsupported`; prefill is counted separately (8 C1 and 23 C8). The 289/124 target-shaped intervals exactly match the target dispatch totals. Draft vocabulary projections contain exactly 944/2,959 rows, matching the proposed-token counters. Mixed target/draft intervals remain unassigned: 1.46% of C1 and 10.43% of C8 summed kernel time. The graph-cap expansion is therefore a concrete next control, but these counts alone do not predict its gain.

| Whole request envelope | C1 | C8 |
| --- | ---: | ---: |
| Envelope seconds | 19.267136 | 23.442392 |
| Observed GPU activity union seconds | 17.368099 | 22.302646 |
| Observed no-GPU-activity seconds | 1.899037 | 1.139745 |
| Observed GPU busy fraction | 90.14% | 95.14% |
| Remove all observed gaps, fixed GPU work | 1.109x | 1.051x |

At C8, the recovered trace has 22.303 seconds of GPU activity in a 23.442-second envelope. Removing its entire 1.140 seconds of observed gaps would give at most 1.051x under fixed recorded GPU work. This conditional calculation is not an overall hardware ceiling: kernels, accepted work, or the GPU schedule could change, and missing events can inflate apparent idle. It does show that merely removing the observed launch gaps cannot explain a several-fold gain on this run.

The next supported check is the narrow depth-4 graph expansion, measured with matched homogeneous prompts and explicit acceptance/graph counters. It tests the 65 observed batch-unsupported dispatches without assuming they cost 65 full forward passes of CPU idle. The draft path remains eager, but eliminating all of its recorded kernels would remove only 8.77% of the summed kernel work; optimizing it alone cannot recover near-linear C8 scaling. The small Q6K hyper-injection projection is a possible focused follow-up (0.268 seconds total, including 0.185 seconds of MMQ fixup), but its entire category is only 1.21% of C8 summed kernels.

C1/C8 instrumented finite-phase throughput was 53.139/131.031 output tokens/s (2.466x). This is diagnostic trace context, not a new unprofiled benchmark. Capture-off/profiled wall times were 19.393/19.270 seconds for C1 and 23.393/23.445 seconds for C8, with only 4/8 and 14/24 text-identical outputs.

The sampled clock-offset envelopes are 2.992 us for C1 and 2.704 us for C8. Switching from the whole request envelope to its trimmed interior changes recorded kernel duration by only 2,416 ns and 0 ns respectively. No full-phase token counts are assigned to the separately exported completion-boundary interior.

Nsight retained these warnings: not all CUDA events might have been collected; not all OS runtime events might have been collected; no NVTX events; absent scheduling data; unified-memory tracing unavailable; cuBLAS symbol lookup failures. This report describes recovered kernel timelines, not guaranteed complete hardware activity.

Reproduce the arithmetic with `python3 current_analysis/write_interpretation.py` after running the gated adapter. The exact source/trace/export/input hashes are in `results/provenance.json`; `results/manifest.json` and the parent `SHA256SUMS` cover the derived artifacts. No GPU execution or model requests are involved in this report generation.
