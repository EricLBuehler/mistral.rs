# MoE kernel optimization

This continues the [scaling investigation](scaling.md) on GB10. The full-model serving rerun below measures the adopted changes together. Later sections use warmed replays of exact captured Flash-Next layer-8 weights, inputs, and routes to distinguish kernel measurements from serving results.

## Final serving rerun

The September 30 rerun uses the same safetensors revision, Q4K ISQ, adaptive MTP, prompts, BF16 KV cache, 16,384-token context, and eight-sequence scheduler as the preceding serving baseline. All 48 layers remain on the GPU, with Q4 PLE storage, 513 KV blocks, and 34 CUDA graphs. Two warmups precede five measured trials; decode and concurrency requests generate 128 tokens. The binary SHA256 is `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d`.

| Concurrency | Before, aggregate tok/s | After, aggregate tok/s | Change | Before, per-active tok/s | After, per-active tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.86 +/- 1.16 | 53.78 +/- 1.43 | -1.98% | 54.87 +/- 1.16 | 53.78 +/- 1.43 |
| 6 | 114.94 +/- 1.21 | 119.95 +/- 0.71 | +4.36% | 20.12 +/- 0.29 | 20.75 +/- 0.13 |
| 8 | 126.56 +/- 1.96 | 131.94 +/- 0.71 | +4.25% | 16.64 +/- 0.17 | 17.10 +/- 0.08 |

These closed-loop trials replenish completed requests and include the finite trial's startup and drain. C1 has eight requests per trial; C6/C8 have 24. Per-active throughput divides output tokens by summed request latency. Mean active requests are 5.78 at C6 and 7.71 at C8; mean request latencies are 6.17 and 7.48 seconds. Variability is sample standard deviation. C8/C1 aggregate scaling is 2.45x, versus 2.31x previously; part of that ratio increase comes from lower C1 throughput. The result remains far below 60 tok/s per request or 360 aggregate tok/s at C6.

In the separate serving suite, four-request bursts improve from 89.98 to 95.16 tok/s (+5.76%) and eight-request bursts from 124.70 to 128.99 (+3.45%). The arithmetic mean of the eight ordinary single-prompt rates changes from 54.15 to 53.88 (-0.49%). Prefill 512/2048/8192 records 1288.83/1293.90/1255.58 tok/s, changes of -2.24%/-0.78%/-2.28%; synthetic 16-token-input decode records 48.41 tok/s (+2.99%). These measurements do not establish a single-request or prefill improvement. The earlier cross-engine GGUF comparison is unchanged; no new same-GGUF versus llama.cpp speedup is claimed.

This is a staged comparison of masked activation loads, the scoped small-batch dispatch, and the intervening logging fix, not a randomized causal estimate of either optimization. Adaptive MTP behavior also changes. Across the combined C6/C8 command, acceptance rises from 66.4% to 74.5%, mean proposed depth falls from 5.18 to 3.94, and target graph/eager dispatches change from 720/1065 to 1400/557. Those counters include warmups and cannot separate C6 from C8; dispatch counts are not kernel-time coverage. Different routes, rounding, generated text, and adaptive depth can affect the measured workload.

No system swap-out is recorded during the new benchmark phases. The C6/C8 command records 19.91 MiB of system swap-in versus 151.03 MiB previously, with model process `VmSwap` approximately 477 MiB at both boundaries in both stages. Reported CUDA free memory at startup also differs, approximately 8,106 MiB versus 5,039 MiB previously. These are not matched memory-pressure conditions. Global counters do not identify model page-ins or their timing impact. The server runs without a profiler and stops after all checks.

Validation confirms matching settings, tokenizer, prompt hashes and token counts, complete trials, no reported prefix reuse, and finite validation logprobs. The 8,128-token repetitive-prompt chat returns the requested coherent sentence and stops normally. Eight concurrent mixed-context requests spanning 1,024 and 2,304 input tokens also pass. These checks do not establish model quality or exact MTP equivalence.

[Validated comparison and all per-prompt results](raw/optimization/final_serving/comparison/comparison.md), [machine-readable summary](raw/optimization/final_serving/comparison/comparison.json), [run metadata](raw/optimization/final_serving/run/metadata.json), and the [archive manifest](raw/optimization/final_serving/manifest.json) retain the raw samples, commands, counters, memory records, source/build provenance, and validators. The historical baseline files remain intact.

The [current whole-model profile](model-scaling.md) measures the same binary at C1/C8: expert kernel time per output improves 1.54x, versus 4.13x for all other kernels together, with similar MTP acceptance. It also records 59 graph and 65 unsupported-batch eager target dispatches at C8. The subsequent depth-four graph expansion passes focused CUDA checks but has no Flash-Next full-model speed measurement yet.

## Remaining expert-kernel headroom

The serving gain does not resolve the scaling gap. A followup control tests whether removing more MMQ computation has enough measured headroom to plausibly explain the requested jump. It reads the exact compressed bytes of every selected expert, with no model computation. Coalesced vector loads feed four observable XOR checksums; independent CPU checksums validate every selected byte range. The control uses the same 8 KiB chunks for all inputs, seven rounds of ten repetitions, and frozen production-FFN replays before and after the scan.

| Native captured rows | Captures | Selected weight payload, median MiB | Read-only scan, median ms | Complete FFN, median ms | Median paired FFN/scan ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42, B6xQ7 | 5 | 448.44 | 1.905 | 2.541 | 1.359x |
| 56, B8xQ7 | 5 | 582.42 | 2.466 | 3.403 | 1.380x |

FFN times use each capture's midpoint of the before/after medians. The ratios are paired per capture, not ratios of table medians. Other scan chunk sizes give 2.491-2.548 ms at M56; a separate 96 MiB cache-flush control gives 2.559 ms at the fixed 8 KiB size. The device reports 24 MiB of L2. The flush is outside the event window and is an eviction attempt, not proof of cold DRAM. M56 FFN drift between bracketing runs ranges from -1.37% to +0.63% per capture.

The arithmetic padding is substantial: the current M56 token tiles execute a median 24.34x the useful token-column work, or 25.97x after including down-projection K padding. But a separate Nsight Compute profile of the median-selected-expert M56 input records 628,068,608 bytes of L2 refill equivalents against 610,713,600 bytes of selected compressed weights, only 2.84% extra. All observed fills use the system-memory aperture. This is data returned to GPU L2, not measured memory-controller traffic or total LPDDR bandwidth. The profile uses five replay passes; its timings are excluded from the unprofiled scan comparison.

Matched route accounting also distinguishes MoE from dense weight reuse. Within each saved B8xQ7 batch, compare the union of all selected experts with the sum of expert sets selected by those exact eight sequences separately. The median payload reuse factor is only 1.469x, versus ideal 8x logical cross-sequence weight reuse for a dense projection. B6xQ7 gives 1.384x. For the profiled B8 sample, the separate sequences select 313 expert payloads in total, versus 213 distinct payloads for the batch: 897.43 MB versus 610.71 MB. These are paired logical byte counts, not predicted throughput.

Together, these controls show why padded arithmetic alone does not imply a similarly large time saving. The read-only control takes much of the complete FFN latency, and batching these routes provides limited weight reuse. The remaining gap is worth investigating, but this evidence does not support expecting the 3.00x full-model gain needed to reach 360 tok/s at C6 from 119.95 solely by removing MMQ padding. The scan includes checksum/launch overhead and differs in access pattern and cache history; it is neither a physical lower bound nor an attainable-FFN guarantee. These fixed-Q7 layer-8 captures also differ from the adaptive-depth full-model run, so no model-wide ceiling is inferred.

All 312 scan checksum comparisons pass, and focused Compute Sanitizer reports zero errors. Both 71-case FFN replays pass all 26 native source-path guards. Six pre-existing numerical diagnostic failures comparing GEMV with alternate MMQ remain visible in small native/derived cases; those alternate outputs are not substituted into this control's current-path comparison. The release CLI cargo check passes, and no production kernel changes result from this followup. [Scan samples and methodology](raw/optimization/headroom/selected_scan/report.md), [L2 refill counters](raw/optimization/headroom/l2_refill/README.md), [paired route reuse](raw/optimization/headroom/accounting/paired_sequence_reuse.md), and [operation accounting](raw/optimization/headroom/accounting/report.md) retain the evidence and an unimplemented integer-MMA design with its risks.

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

## Small verification batches

The adopted dispatch selects existing grouped MMQ at 24-31 rows for the measured Flash-Next geometry on compute capability 12.1. Eligibility requires BF16, SiLU, 512 experts with top-10 routing, Q4K gate/up tensors shaped `[512,640,2560]`, Q4_1 down shaped `[512,2560,640]`, and multi-token queries. Biased projections, LoRA, statistics collection, and sharded experts retain their existing dispatch. Other hardware and model geometries are unchanged.

The global 32-row threshold remains intact because it also contributes to speculative graph budgeting. The exception changes only eligible forwards. Five balanced 24-row subsets of captured C8 verification inputs previously favored existing MMQ over GEMV on every sample, with a 1.257x median paired ratio (range 1.191-1.503x). Median latencies were 2.024 and 2.612 ms. These are derived layer inputs, not naturally captured shorter-depth verification or serving measurements. [Samples and independent references](raw/scaling_investigation/real_routing/replay_sequence_query_prefix/summary.json) retain that distinction.

Expanded regression coverage replays changed inputs and routes at 24, 28, and 32 rows after growing the shared workspace. The packed-MMQ and cuTile projection tests pass normally and under Compute Sanitizer with zero reported errors. The release CLI build, cargo check, scoped quant/core/CLI Clippy, and formatting pass. A separate logging regression verifies that repeated messages no longer append duplicate entries to the one-time log caches. [Validation commands, logs, sources, and binary hashes](raw/optimization/small_group_candidate/validation/metadata.json) identify the exact serving candidate. One lint attempt for a different core feature combination was intentionally stopped; the final core/CLI lint uses the production feature union and passes.

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
| 16/64/128 | 3.440 | 927.615 | 3.482 | 927.337 |
| 8/32/256 | 3.435 | 93.658 | 3.429 | 93.571 |

Times are median milliseconds over the same seven-round protocol. At BK 128, this layout decodes the entire block twice and dynamically extracts each half. BK 256 eliminates the repeated decode but remains substantially slower. The source also introduces concatenation and tile-layout conversion; no counter profile of this version was collected, so the particular cause of its regression is not established. This version is rejected. Baseline outputs still match the native capture exactly, and candidate graph/eager outputs match each other exactly. [Results](raw/optimization/cutile_blocked/summary.json), [validation](raw/optimization/cutile_blocked/validation.json), and [source archive](raw/optimization/cutile_blocked/manifest.json) retain the experiment.

### Grouped metadata decoder

The third layout decodes only the requested K lanes and broadcasts metadata from one entry per 32-value group. It removes the full-block concatenation/extraction. The same native 56-row input gives these median milliseconds:

| Tile BM/BN/BK | Existing path, eager | cuTile, eager | Existing path, graph | cuTile, graph |
| --- | ---: | ---: | ---: | ---: |
| 8/32/64 | 3.428 | 8.663 | 3.415 | 8.605 |
| 16/64/128 | 3.362 | 9.468 | 3.328 | 9.526 |
| 16/64/32 | 3.419 | 8.774 | 3.423 | 8.680 |
| 16/128/32 | 3.435 | 17.164 | 3.423 | 19.529 |
| 16/32/64 | 3.336 | 7.931 | 3.404 | 7.944 |

This improves on the earlier decoders but still loses to MMQ. The best configuration takes 2.38x the baseline's eager time. No native cuTile GGUF path is enabled in production; the experiment lives in test support with its replay harness and CPU-oracle tests.

Nsight Compute for 8/32/64 reports 151/151/153 registers in gate/up/down, zero local-memory load/store sectors, and about 25% achieved warp occupancy. Its tensor activity is zero. Offline SASS inspection of corresponding unit-test cubins confirms that the 8-row shape lowers to ordinary arithmetic, whereas the examined 16-row shapes contain HMMA instructions. Thus removing spills and using tensor cores are insufficient by themselves to establish a faster complete forward. The matched 16/32/64 timing is an additional negative probe, not a profiled run.

The independent full-FFN reference ran separately for 8/32/64. Relative RMS against original decoded FP32 weights is 0.2785% for cuTile and 2.2111% for existing MMQ. Against a reference with BF16-rounded weights, intermediate, and final output, cuTile differs by 0.00859% RMS; accumulation, nonlinear evaluation, and reduction order still differ. These are one-layer numerical comparisons, not language-model quality measurements. All five replay configurations preserve exact baseline/capture and candidate eager/graph agreement.

[Timing samples](raw/optimization/cutile_grouped_metadata/summary.json), [full-FFN references](raw/optimization/cutile_grouped_metadata/oracle_native56_8_32_64/oracle.json), [profiling counters](raw/optimization/cutile_grouped_metadata/ncu_native56_8_32_64/kernels.json), and [validation](raw/optimization/cutile_grouped_metadata/validation.json) retain the tested scope and provenance. Profiled durations are excluded from the timing table; the profiling process intentionally stops after the selected launches.

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

## Grouped small-row GEMV

An external CUDA prototype reuses each expert's weight fragment across two or four routed rows. It keeps the indexed GEMV Q8_1 activation format, F32 intermediate, fused gate/up/SiLU, and weighted atomic reduction. Routing construction, both quantizations, output clearing, and BF16 conversion remain inside each measured FFN call. The first layout handles two gate/up output features and four down features per warp; a second layout doubles those counts.

The first layout with two routed rows per group improves the 24-row cases, where current dispatch uses indexed GEMV, but roughly ties MMQ at larger shapes. The doubled-feature layout loses every comparison against MMQ at 32 or more rows. Neither new kernel is adopted. Each table row contains five inputs; times are medians of per-input eager CUDA-event medians, and ratios are medians of paired input ratios. The two layouts run in separate processes and each ratio uses its own contemporaneous default control.

| Input rows | Route source | Default, ms | First layout, group 2, ms | Default / first layout | Default / doubled features |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 | Derived per-sequence prefix | 2.628 | 2.028 | 1.282x | 1.207x |
| 32 | Derived per-sequence prefix | 2.326 | 2.353 | 0.992x | 0.939x |
| 40 | Derived per-sequence prefix | 2.610 | 2.612 | 1.002x | 0.927x |
| 42 | Native C6 verification | 2.547 | 2.545 | 1.001x | 0.933x |
| 56 | Native C8 verification | 3.359 | 3.359 | 0.998x | 0.918x |

The first layout's 24-row graph ratio is 1.308x; both eager and graph comparisons improve on all five inputs. Grouping four routed rows is slower than grouping two in the first layout. The two-row kernels use 55/48 registers in gate-up/down, versus 53/48 after doubling output features. All four compiled configurations have zero stack and spill bytes. These resource counts establish feasibility, not a measured occupancy or bandwidth explanation. [First-layout results](raw/optimization/grouped_gemv/summary.json) and [doubled-feature results](raw/optimization/grouped_gemv/more_features_summary.json) include both group widths, graphs, controls, and clock samples.

Both layouts pass all 25 strict comparisons against existing GEMV, all native default/capture checks, eager/graph comparisons, and one changed-routing graph case per run using the same device allocation. The first layout's maximum difference from GEMV is 0.00404% relative RMS; atomic accumulation can occur in a different order. Separate FP32 expert-FFN references cover every saved output and retain the existing GEMV quantization error. MMQ has different activation quantization and intermediate rounding, so its output difference is recorded separately. These are numerical kernel checks, not model-quality evaluations.

Two initial sanitizer attempts included repeated timing replays and were stopped without a pass/fail conclusion. Final validation uses a separate harness with no warmups or timed iterations, linked to each benchmark's exact CUDA object. Both group widths and changed-route graphs pass the focused checks with zero reported errors. Sanitizer timings are excluded from the table. [Final provenance](raw/optimization/grouped_gemv/final_provenance.json) retains the stopped attempts, successful checks, commands, object hashes, and unchanged production-source/archive/binary checks.

This experiment supports investigating the existing MMQ path for sub-32-row batches as a smaller production change. It does not validate broader eligibility for the new GEMV kernels. Repeated single-layer weights can have different cache behavior from full-model execution, and these results make no serving-throughput or DRAM-bandwidth claim. The [archive manifest](raw/optimization/grouped_gemv/manifest.json) preserves source and raw evidence while excluding executables, model weights, and output tensors; their hashes remain recorded.
