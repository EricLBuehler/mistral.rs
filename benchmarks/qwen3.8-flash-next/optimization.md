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

Nsight Compute profiles the same captured input after warmup. The default Q4K gate/up kernels reach 255 registers per thread and generate substantial local-memory traffic, consistent with register spilling. The wider tile makes that traffic much larger. The existing masked MMQ projections record zero local-memory sectors on this input.

| Kernel and tile | Registers/thread | Local load sectors | Local store sectors | Tensor activity |
| --- | ---: | ---: | ---: | ---: |
| Masked MMQ gate, 64 columns | 255 | 0 | 0 | 15.18% |
| cuTile gate/up, each, 16/64/128 | 255 | 10,664,000 | 5,792,800 | 1.37% |
| cuTile down, 16/64/128 | 184 | 0 | 0 | 1.92% |
| cuTile gate, 8/64/256 | 255 | 583,561,656 | 527,490,588 | 0.00% |

Tensor activity is the profiler's active tensor-pipeline cycles as a percentage of sustained elapsed cycles. The wide gate also reports only 0.03 eligible warps per scheduler cycle and a long-scoreboard stall ratio of 27.09, versus 0.30 and 4.47 for the default gate. These observations support reducing the live decoded tiles and repeated metadata loads. They do not explain every cost: the default down projection is slow without local-memory traffic.

Local and L2 sectors count cache requests, not DRAM bytes or bandwidth. These profiles use kernel replay with ten counter passes, unchanged caches/clocks, and explicit graph-node profiling. The profiler intentionally terminates each target after the selected launches: completion means the requested counter records were exported, not that the test finished or produced a replay summary. Profiled durations are excluded from the timing table. [Counters and interpretation](raw/optimization/cutile_profile/ncu_comparison.json), [exact commands](raw/optimization/cutile_profile/ncu_native56/metadata.json), and the [archive manifest](raw/optimization/cutile_profile/manifest.json) preserve the evidence.

### Rejected blocked decoder

A second layout loads each Q4K block's metadata in smaller arrays, unpacks its 256 values, and selects the requested K subtile. All four expanded projection tests pass, covering BK 32/64/128/256 with unchanged tolerances, and Compute Sanitizer reports zero errors. Correctness did not translate into performance on the same native 56-row capture:

| Tile BM/BN/BK | Existing path, eager | cuTile, eager | Existing path, graph | cuTile, graph |
| --- | ---: | ---: | ---: | ---: |
| 16/64/128 | 3.440 | 927.615 | 3.482 | 927.241 |
| 8/32/256 | 3.435 | 93.658 | 3.429 | 93.571 |

Times are median milliseconds over the same seven-round protocol. At BK 128, this layout decodes the entire block twice and dynamically extracts each half. BK 256 eliminates the repeated decode but remains substantially slower. The source also introduces concatenation and tile-layout conversion; no counter profile of this version was collected, so the particular cause of its regression is not established. This version is rejected. Baseline outputs still match the native capture exactly, and candidate graph/eager outputs match each other exactly. [Results](raw/optimization/cutile_blocked/summary.json), [validation](raw/optimization/cutile_blocked/validation.json), and [source archive](raw/optimization/cutile_blocked/manifest.json) retain the experiment.

## Compact grouped-MMQ scheduling

An external prototype builds an expert-tile prefix on the GPU and uses a fixed grid of persistent blocks to visit only nonempty tiles. It keeps the existing default tile selection and full column coverage, runs each tile's complete K dimension, and reuses the bounded masked-Y math. It requires no CPU readback. Freshly linked baseline and candidate executables use identical Rust dependencies and replay source; only the Q4K/Q4_1 CUDA objects differ.

The prototype is rejected because it is substantially slower. Each row below contains five captured inputs, each with three warmups and seven alternating GEMV/MMQ rounds of ten forwards. Times are medians of ordinary CUDA-event FFN measurements; ratios are medians of the paired input ratios.

| Input rows | Route source | Masked baseline, ms | Compact, ms | Baseline / compact |
| --- | --- | ---: | ---: | ---: |
| 24 | Derived per-sequence prefix | 2.026 | 3.575 | 0.560x |
| 32 | Derived per-sequence prefix | 2.311 | 4.829 | 0.478x |
| 40 | Derived per-sequence prefix | 2.597 | 5.572 | 0.468x |
| 42 | Native C6 verification | 2.552 | 5.438 | 0.467x |
| 56 | Native C8 verification | 3.427 | 7.383 | 0.467x |

All 25 outputs are bit-exact against the masked baseline. Independent CPU-oracle tail/stream-K tests and changed-route CUDA graph tests pass, including Compute Sanitizer with zero errors. The unchanged GEMV control has native 42/56-row ratios of 0.988x/0.991x, so ordinary drift does not explain the roughly doubled grouped cost. This rejects this persistent scheduling implementation; it does not establish that every compact schedule would lose. No smaller-tile followup or production adoption was made, and these warmed single-layer replays make no serving-throughput claim.

[Paired measurements and controls](raw/optimization/compact_schedule/summary.json), [correctness commands](raw/optimization/compact_schedule/validation/metadata.json), [source patch](raw/optimization/compact_schedule/sources/compact_schedule.patch), and [provenance](raw/optimization/compact_schedule/final_provenance.json) retain the result. The [manifest](raw/optimization/compact_schedule/manifest.json) excludes executables, model weights, and saved output tensors while retaining their hashes in the lifecycle records.
