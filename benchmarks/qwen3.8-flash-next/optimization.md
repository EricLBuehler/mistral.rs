# MoE kernel optimization

This continues the [scaling investigation](scaling.md) using the exact captured Flash-Next layer-8 weights, inputs, and routes on GB10. These are warmed layer replays, not updated serving benchmarks. The existing serving results remain the baseline until a full-model rerun is recorded.

## Skip padded activation-column loads

Grouped MMQ used to read every column of its activation tile, including columns past an expert's final assignment. The kernel now fills those unused shared-memory columns with zero without reading them from global memory. Valid columns, tile selection, routing, and accumulation are unchanged. The optimization applies to grouped tiles of at most 64 columns.

The initial implementation increased register spilling in larger tiles. The bounded implementation compiles active Q4K/Q4_1 tiles above 64 columns to identical SASS as the original kernel; small tiles match the measured candidate. Q2K/Q6K larger tiles also retain identical SASS. This protects the larger-tile path from that observed regression.

Two comparison passes ran in opposite order: baseline, candidate, candidate, baseline. Each input has three warmups and seven alternating GEMV/MMQ rounds of ten forwards. The table reports the median across five inputs of the mean of each input's two baseline/candidate ratios. GEMV is unchanged and provides a separate observation of timing drift; it is not used to normalize the result.

| Input rows | Route source | Grouped MMQ speed ratio | GEMV control ratio, first / second pass |
| --- | --- | ---: | ---: |
| 24 | Derived per-sequence prefix | 1.007x | 1.017x / 1.010x |
| 32 | Derived per-sequence prefix | 1.023x | 1.014x / 1.028x |
| 40 | Derived per-sequence prefix | 1.047x | 1.013x / 1.024x |
| 42 | Native C6 verification | 1.030x | 1.011x / 1.030x |
| 56 | Native C8 verification | 1.038x | 1.005x / 1.025x |

The native 56-row median is 1.041x in the first pass and 1.036x in the reversed pass; every input improves in both. Smaller shapes have mixed results relative to the control drift, including one native 42-row input with a slight regression across the two passes. All 25 candidate outputs match the original grouped outputs byte for byte in both passes and in the bounded implementation's replay. These results establish a modest measured layer-level gain, not a serving-throughput prediction. [Paired samples and statistics](raw/optimization/masked_y/masked_y/abba_summary.json), [bounded replay](raw/optimization/masked_y/masked_y_bounded/replay/replay.json), and [generated-code comparison](raw/optimization/masked_y/masked_y_bounded/code_equivalence.json) retain the evidence.

Nsight Compute on the same native 56-row sample records 48.1% fewer L2 read sectors in each gate/up projection and 53.4% fewer in down. These are cache read requests, not DRAM traffic or bandwidth utilization. Profiling timings are excluded from the speed ratios above. [Counters](raw/optimization/masked_y/masked_y/ncu_comparison.json) and the raw per-kernel reports record the exact launches and metrics.

Independent regression tests cover empty experts, partial and exact column tiles, the last valid column, multiple tiles, and forced stream-K fixup. They exercise Q4K, Q4_1, Q6K, and Q2K with BF16, F16, and F32 inputs. The CPU oracle dequantizes identical weight bytes; power-of-two block scales and exact-Q8 activations isolate indexing from intermediate scale rounding. The original general-weight fixture failed identically under both kernels because native MMQ rounds intermediate scale products to FP16; that failed test and baseline reproduction are retained. The corrected fixture keeps the existing tolerance. CUDA graph replay tests cover changed inputs and routes; both regression binaries pass Compute Sanitizer with zero reported errors.

[Final validation commands and hashes](raw/optimization/masked_y/masked_y_bounded/validation.json) record the release build, scoped CUDA cargo check and Clippy, formatting, test runs, and sanitizer results. The [archive manifest](raw/optimization/masked_y/manifest.json) hashes the raw text evidence and records the saved output tensor hashes.

## Native GGUF cuTile prototype

The first cuTile prototype reads original Q4K/Q4_1 compressed weight bytes, decodes tiles to BF16 inside the kernel, and accumulates tensor-core products in FP32. It does not allocate a full dequantized weight copy. Gate/up outputs remain FP32 through SiLU and multiplication; the intermediate is rounded to BF16 for down. The existing runtime dispatch remains selected while this implementation is evaluated.

Four CUDA tests pass against independent CPU dequantization, including every nibble/scale position, column/K tails, routed projections, nonlocal-expert zeros, and changed-route CUDA graph replay. They also pass Compute Sanitizer with zero errors. The release build, scoped CUDA cargo check, and Clippy pass. [Validation and source hashes](raw/optimization/cutile_initial/projection.validation.json) preserve the exact implementation.

A first probe used one native C8 verification capture (56 rows, layer 8). Each configuration has three warmups and seven alternating rounds of ten complete expert forwards. Timings include GPU routing, three projections, activation, and weighted reduction. Medians in milliseconds:

| Tile BM/BN/BK | Existing path, eager | cuTile, eager | Existing path, graph | cuTile, graph |
| --- | ---: | ---: | ---: | ---: |
| 16/64/128 | 3.323 | 17.725 | 3.319 | 17.685 |
| 8/32/64 | 3.415 | 19.297 | 3.409 | 19.150 |
| 8/64/256 | 3.428 | 296.114 | 3.429 | 296.358 |

The direct decoder is substantially slower on this input. Graph replay does not remove the gap. Baseline output matches the saved native capture exactly, and each cuTile configuration's graph output matches its eager output exactly. The default cuTile/MMQ output difference is 2.23% relative RMS, with cosine 0.999752; this compares different activation/weight arithmetic and is not an answer-quality evaluation. The separate complete-FFN FP32 and rounded-weight reference harness has not been run at this checkpoint. [Raw samples, configuration, and output hashes](raw/optimization/cutile_initial/summary.json) record the measured scope; this single-input probe does not establish general performance.
