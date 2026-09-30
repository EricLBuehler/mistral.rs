# Concurrency scaling investigation

The final Flash-Next workload scales 2.31x from C1 to C8, compared with 7.44x for the v0.9.3 target-only workload and 5.99x with DFlash2. This comparison changes hardware, model, quantization, sampling, and request lengths together; it does not isolate an engine regression.

| Workload | C1 aggregate tok/s | C8 aggregate tok/s | C8/C1 | C8 tok/s per active request |
| --- | ---: | ---: | ---: | ---: |
| v0.9.3 dense target-only | 96.98 | 721.22 | 7.44x | 90.16 |
| v0.9.3 dense + DFlash2 | 216.91 | 1,299.91 | 5.99x | 167.24 |
| Flash-Next MTP before tuner exploration fix | 52.60 | 124.90 | 2.37x | 16.51 |
| Flash-Next MTP after tuner exploration fix | 54.86 | 126.56 | 2.31x | 16.64 |

The [release measurements](../../releases/v0.9.3/report.md) used Qwen3.8-27B-FP8 on GH200, FP8 KV, 64 requests with 512 outputs, stochastic sampling, and optionally seven external DFlash2 proposals. Flash-Next used ISQ q4k on GB10, BF16 KV, 24 requests at C8 with 128 outputs, greedy sampling, and adaptive built-in MTP. Release values are medians of three trials; both Flash-Next stages use means of five. The [comparison JSON](raw/scaling_investigation/release_comparison.json) preserves unrounded values, source hashes/selectors, and the old per-active-request reconstruction from mean TTFT/TPOT. Both suites include prefill and scheduling in client wall time. Mean C8 activity was approximately 8.00 for the release target-only run, 7.77 with DFlash2, 7.56 before the tuner fix, and 7.61 afterward. The [latest validated concurrency summary](concurrency.summary.json) supplies the final row. Startup/drain occupancy alone does not explain the scaling difference.

The release client also applied the model's chat template: an xhigh reasoning system instruction, role markers, and an assistant `<think>` prefix added exactly 52 tokens to every canonical prompt. CPU rendering reconciles all 64 prompts to the recorded 128-139 input-token range and 8,592-token total. The comparison JSON contains the pinned vLLM source commit and lines, all count pairs, and one rendered sample.

Limited scaling also appears without MTP: target-only Flash-Next ISQ records 36.01 tokens/s as the arithmetic mean of the eight single-request prompt rates and 84.49 aggregate tokens/s in the eight-request burst, a 2.35x ratio. This is separate finite-burst evidence, not the closed-loop scaling ratio. It motivates investigating the target model's expert path alongside MTP; the comparison alone does not establish a kernel-level cause or a hardware ceiling. [Serving summary](summary.json) preserves the exact values.

## Same-GB10 dense control

The branch's target-only dense FP8 control was measured before the final autotuner exploration change, with speculation disabled. It used the same pinned 27B checkpoint as the release, BF16 KV, greedy sampling, the first eight raw canonical prompts (77-85 input tokens), and 128 outputs. Server capacity remained eight sequences for C1 and C8. Each case had one warmup and three measured trials. C8 is one finite wave of eight requests, not sustained traffic. [Raw requests/responses](raw/scaling_investigation/dense_control/raw.json), [validated summary](raw/scaling_investigation/dense_control/summary.json), and [binary/model provenance](raw/scaling_investigation/dense_control/metadata.json) are preserved.

Only the clean rerun, performed after pausing the identified rust-analyzer compiler processes, supplies the dense figures below. An earlier run overlapped a background debug CUDA build and is [archived separately](raw/scaling_investigation/dense_control_background_build/summary.json); its [interference note](raw/scaling_investigation/dense_control_background_build/interference.json) and [process identities](raw/scaling_investigation/dense_control/paused_editor_processes.json) preserve that limitation. Stable trial rates in the earlier run did not establish CPU isolation.

| Concurrency | Aggregate tok/s | Tok/s per active request | Mean active requests | Mean request latency (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.03 +/- 0.03 | 8.03 +/- 0.03 | 1.00 | 15.944 |
| 8 | 58.97 +/- 0.05 | 7.38 +/- 0.01 | 7.99 | 17.336 |

Values are means +/- sample SD where shown. Aggregate throughput divides outputs by whole-trial wall time; per-active-request throughput divides outputs by summed request latency. C8/C1 aggregate scaling is 7.35x, retaining 92.0% of the C1 per-active-request rate. This measures current dense-model behavior on GB10. The model, prompts, speculation, and finite-wave workload differ from Flash-Next; the old release also differs in hardware, sampling, formatting, output length, and KV precision. A version regression requires matched runs under both binaries. The [control script](raw/scaling_investigation/run_dense_control.py) records binary hashes and exact server/request settings.

### Matched dense FP8 versus Q4K execution paths

A second control holds the current executable (`d245efbb`), GB10, pinned 27B FP8 checkpoint, greedy prompts, BF16 KV, and eight-sequence capacity fixed; only the Q4K arm adds `--isq q4k`. Each C1/C8 case has one warmup and three measured trials of eight requests with 128 outputs. C8 remains one finite wave. The [independently validated comparison](raw/scaling_investigation/backend_comparison/comparison.json) recomputes all rates from the saved timestamps and responses.

| Weight/execution path | C1 aggregate tok/s | C8 aggregate tok/s | C8/C1 | C8 tok/s per active request | C8 per-active retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native FP8 | 8.118 +/- 0.020 | 59.253 +/- 0.075 | 7.30x | 7.423 +/- 0.017 | 91.4% |
| FP8-to-Q4K ISQ | 13.019 +/- 0.002 | 71.009 +/- 0.105 | 5.45x | 8.891 +/- 0.019 | 68.3% |

Values are means +/- sample SD. Q4K is 60.4% faster at C1 and 19.8% faster at C8 despite its weaker relative scaling. The execution path therefore matters even for this fixed dense architecture, but dense Q4K still scales 5.45x. This control does not isolate expert routing or explain the full Flash-Next scaling gap.

This is double quantization: ISQ dequantizes the published FP8 weights before requantizing them. It also changes eligible `lm_head` precision to Q6K, activation processing, and stored projection layout; separate Q4K gate/up tensors still use fused MMVQ decode computation. The production packed-affine Marlin backend is disabled in both recorded environments. [Startup and source evidence](raw/scaling_investigation/backend_comparison/startup_verification.json) records the matching 64-layer GPU placement, cache, and graphs while distinguishing logged facts from source-derived dispatch. No quality-equivalence or single-kernel causal claim follows from these rates.

The [archive](raw/scaling_investigation/backend_comparison/README.md) includes full responses, exact commands, frozen scripts, reproducible validation, and hashes. [Process provenance](raw/scaling_investigation/backend_comparison/provenance.json) preserves stopped-editor observations and the coordinated pause in agent builds/GPU work; no continuous system-wide activity monitor was captured.

### Dense FFN dispatch control

An [isolated layer-0 FFN probe](raw/scaling_investigation/backend_comparison/dense_ffn/README.md) uses actual 27B weights, synthetic BF16 activations, and the production gate/up, SiLU, and down-projection hooks. Here M is the number of input rows to one layer, not serving concurrency. Median whole-FFN times across seven alternating rounds of ten invocations, after three warmups:

| Default execution path | M1 time (ms) | M8 time (ms) | M1-to-M8 row-throughput scaling |
| --- | ---: | ---: | ---: |
| Native FP8 | 1.261 | 1.274 | 7.92x |
| FP8-to-Q4K | 0.807 | 1.073 | 6.02x |

Scaling is `8 * time(M1) / time(M8)`. A separate process with the optional affine Marlin backend enabled reduced Q4K M8 time from 1.073 to 0.739 ms, a 1.45x ratio. Successful affine preparation was recorded for all three projections at M8 and above; the default full-model control did not enable it.

Each path passed numerical checks against FP32 computation with its own dequantized weights. This checks kernel output on synthetic inputs, not model quality or equality between quantizations. CUDA-event timing includes host launch gaps and excludes graph replay and repacking setup. Affine repacking adds weight storage; full-model capacity and end-to-end speed with it enabled were not measured. The isolated layer also omits upstream normalization fusion and model-wide cache effects, so its speedup does not predict a serving speedup.

## Expert dispatch probe

A [single-layer probe](raw/scaling_investigation/expert_dispatch/summary.json) used actual Flash-Next layer 8 weights, BF16 synthetic activations, 512 experts, top-10 routing, hidden width 2,560, and intermediate width 640. It compared forced GEMV with grouped MMQ through the full expert pipeline. This original probe also overlapped the background compiler; its timings are diagnostic evidence with possible CPU-launch interference.

A separate [clean known-occupancy probe](raw/scaling_investigation/expert_dispatch/known_occupancy_clean/summary.json) ran with the compiler processes paused, verified before and after each run. It compared the same grouped projection pipeline with the normal column bound versus a bound of eight, after verifying every synthetic expert receives at most eight rows. Representative 56-row median times and median paired speedups:

| Weights | Synthetic routing | Normal bound (ms) | Known bound 8 (ms) | Paired speedup |
| --- | --- | ---: | ---: | ---: |
| GGUF | spread | 8.157 | 7.930 | 1.03x |
| GGUF | groups of seven | 1.490 | 1.447 | 1.04x |
| ISQ | spread | 7.672 | 7.419 | 1.04x |
| ISQ | groups of seven | 1.471 | 1.322 | 1.11x |

These routes are deterministic spread controls or groups of seven sharing experts, not captured model routes. Timings cover seven paired alternating rounds of ten invocations after warmup; CUDA events include host launch gaps. Repeated single-layer weight reuse differs from a full 48-layer model. All clean-probe BF16 outputs matched both the normal bound and packed reference exactly. Observed gains were roughly 0-11%, with small regressions in some cases. This is a modest ideal-occupancy opportunity: applying a bound of eight generally can omit expert rows. The measured projection pipeline also quantizes gate/up activations separately, while the packed production pipeline shares that quantization. Neither probe establishes a memory-bandwidth ceiling or predicts serving throughput. [Original provenance](raw/scaling_investigation/expert_dispatch/provenance.json), [post-measurement review](raw/scaling_investigation/expert_dispatch/post_measurement_review.json), and [clean-probe metadata](raw/scaling_investigation/expert_dispatch/known_occupancy_clean/metadata.json) preserve the scope and implementations.

## Captured routes and GPU resource limits

A separate [actual-route experiment](raw/scaling_investigation/real_routing/README.md) captured layer-8 inputs, expert IDs, routing weights, outputs, and exact ISQ weights from Flash-Next. It used fixed MTP depth 6 and disabled CUDA graphs; capture synchronization changes execution, so this is not a sample of the final adaptive server's unperturbed behavior. All 24 capture requests completed. The replay measures the routed expert contribution before the shared expert addition, with three warmups and seven alternating rounds of ten forwards per sample.

| Native input shape | Samples | Selected experts, median | GEMV (ms) | Grouped MMQ (ms) | Paired GEMV/MMQ ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode, 1 row | 8 | 10 | 0.1162 | 0.3206 | 0.363x |
| C1 verification, 7 rows | 8 | 31 | 0.5556 | 0.5567 | 0.987x |
| C6 verification, 42 rows | 5 | 164 | 4.0690 | 2.6551 | 1.536x |
| C8 verification, 56 rows | 5 | 213 | 5.3111 | 3.5214 | 1.582x |

Times are medians across per-sample median CUDA-event intervals, which include host launch gaps. Ratios are medians of paired per-sample GEMV/MMQ ratios, so they need not equal the quotient of the displayed times. Grouped MMQ wins every native 42/56-row comparison; forcing GEMV at those observed shapes would lose in this layer replay.

Median route assignments per selected expert rise only from 2.27 at 7 rows to 2.63 at 56 rows. Larger batches select more experts rather than sharing every projection across all rows as in a dense FFN. The [pinned configurations](raw/scaling_investigation/backend_comparison/architecture_comparison.json) distinguish the dense 27B model's 64 layers and width-17,408 FFN from Flash-Next's 48 layers and 512 width-640 experts with top-10 routing. At 56 rows, the median 213 selected experts represent 582.4 MiB of logical quantized weights for one routed layer. This describes possible reuse, not measured DRAM traffic or a causal estimate of the full-model scaling gap. Every captured C6/C8 input has an expert with more than eight rows, so the earlier synthetic bound-of-eight shortcut would omit work on actual routes.

Balanced derived controls select the first 3, 4, or 5 query positions from each of the eight sequences in the captured 7-query inputs. Their 24/32/40-row paired GEMV/MMQ ratios are 1.257x/1.368x/1.433x. These are derived activations, not observed lower-depth decode runs. They motivate testing dispatch choices without establishing a serving gain or a safe policy for every model.

All 26 native same-dispatch replays passed strict equivalence checks; 22 were exact. The first replay's failure of the inherited 3% cross-kernel RMS guard is preserved, and the adjusted diagnostic records remaining disagreements rather than treating them as safe replacements. An independent FP32 computation using the same dequantized weights gives median native-56 relative RMS errors of 2.145% for GEMV and 2.155% for grouped MMQ. These measure differences from activation quantization and computation, not model quality; the [full numerical results and failure history](raw/scaling_investigation/real_routing/README.md#numerical-checks-and-preserved-failure) retain their scope.

[Nsight Compute profiling](raw/scaling_investigation/real_routing/ncu_native56/README.md) of one captured 56-row input found that the grouped 64-column kernels use 254 registers per thread, rounded to 256 allocated, with 256 threads per block. That consumes all 65,536 registers per SM for one block; the launch data also reports a one-block shared-memory limit. Achieved warp occupancy is 16.75-16.98%, versus 89.61-97.82% for GEMV; the grouped tensor-pipe activity counter is 14.74-17.00% of peak sustained activity over the measured interval. This identifies concrete resource constraints, but higher occupancy alone does not predict faster execution: grouped MMQ still wins the unprofiled native 42/56-row replay. This isolated Nsight Compute profile is separate from the historical full-model Nsight Systems traces discussed in the [main report](README.md#gpu-profiling).

The requested DRAM counters were unavailable. L2 counters and logical selected-weight sizes do not establish bandwidth saturation or a physical ceiling. Nsight Compute serializes and replays kernels; its durations are separate from unprofiled timing. These experiments cover one layer and omit shared experts, attention, GDN, hyper-connections, and full-model cache effects. They narrow the kernel investigation without proving an absence of regressions or explaining the entire 2.31x serving scaling ratio.

### Measured kernel experiments

Smaller tiles and a rewrite that loads one K fragment group at a time were tested on the same 25 inputs: five native 42-row, five native 56-row, and five each of the derived 24/32/40-row layouts. Every candidate reproduced all 25 separately saved grouped-MMQ outputs exactly. The checks establish equivalence on these inputs, not model-quality guarantees. No candidate showed a consistent pipeline gain, so production tile selection and fragment loading remain unchanged.

| Grouped-MMQ variant | Native 42-row paired ratio | Native 56-row paired ratio |
| --- | ---: | ---: |
| 16-column tile | 0.941x | 0.964x |
| 32-column tile | 0.955x | 1.014x |
| Streamed fragments, default tiles | 0.940x | 0.938x |
| Streamed fragments, 16-column tile | 0.992x | 1.011x |

Each value is the median of per-input baseline/variant CUDA-event timing ratios; greater than one means the variant was faster. The baseline and variants use uniformly monitored, warmed, repeated layer replays, with unchanged GEMV measurements retained as an observational drift control. The small positive ratios are not convincing gains. Full distributions, derived cases, and exact implementations are archived for [smaller tiles](raw/scaling_investigation/real_routing/tile_variants/README.md), [streamed fragments](raw/scaling_investigation/real_routing/register_streaming/README.md), and [their combination](raw/scaling_investigation/real_routing/register_streaming_tile16/README.md).

Smaller tiles alone reduced compiled register counts to 201-227 per thread, still permitting only one 256-thread block per SM. Streaming fragments at default tile widths lowered the counts to 164/168, but made the native 42/56-row pipeline about 6-7% slower. Combining streaming with the 16-column tile lowered counts to 112/126. The new Nsight Compute capture confirmed two resident-block capacity with a 100 KiB shared-memory carveout and achieved occupancy of 31.11-32.77%, up from 16.75-16.98%. This was a measured increase in occupancy without a convincing pipeline gain, not merely a resource estimate. The combination also quadruples the launch grid; it changes more than register use, and its profiled kernel durations are not throughput benchmarks. These bounded negative results do not rule out other kernel designs.

### Kernel provenance and dense execution

The MMQ implementation predates Flash-Next support: [commit 2d4ba4f16](https://github.com/EricLBuehler/mistral.rs/commit/2d4ba4f16f61e5e18be085d0dd137bc95cba038a) introduced the fast CUDA MMQ kernels on April 14, 2026, and [commit e07d83bf0](https://github.com/EricLBuehler/mistral.rs/commit/e07d83bf09cda2e0a733dddfcfe8076c6522e77b) extended CUDA MoE optimization on May 23. Between the branch's September 25 merge base `2370966bb` and reviewed committed head `0003cf351`, the MMQ CUDA files and [Rust MMQ wrapper](../../mistralrs-quant/src/gguf/fast_mmq.rs) have no diff. The committed expert-backend change only exposes the existing 32-row threshold within the crate. The current reviewed working tree does change the [MMQ header](../../mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh): it adds three bounds checks for expert destination-ID loads. The kernel family is therefore pre-existing, with explicit correctness changes in this work.

Dense and MoE runs do not exercise interchangeable kernels. The dense FP8 control uses the [blockwise FP8 path](../../mistralrs-quant/src/blockwise_fp8/mod.rs), not GGUF MMQ. The default dense Q4K control uses MMVQ for its eight-row decode batches; the [dense dispatch](../../mistralrs-quant/src/gguf/mod.rs) selects MMQ above eight rows when the optional affine path is disabled. Dense MMQ can share low-level arithmetic with grouped MoE MMQ, but has no expert routing or separate groups of selected rows. Neither the dense scaling result nor the kernel's age isolates the cause of Flash-Next's scaling gap.

## Tuner exploration bug

The [automatic-depth search](../../mistralrs-core/src/speculative/autotuner.rs) previously explored only neighboring depths. Costs need not be monotonic across the supported depths 2, 3, 4, and 6: starting at 6, a slow depth 4 could prevent discovery of a faster depth 3. Warmup now samples every undersampled candidate, excluding the currently preferred depth; steady-state refresh remains local. The regression test models this slower-neighbor barrier and requires discovery of depth 3. The [24 passing autotuner tests](raw/validation/tuner.autotuner_tests.log) and [old-fails/new-passes regression](raw/scaling_investigation/autotuner_red_green/results.json) cover this exploration failure.

The exploration correction did not materially close the scaling gap. The final unprofiled Flash-Next run records 54.86 tokens/s at C1 and 126.56 aggregate tokens/s at C8, with 16.64 tokens/s per active request at C8. Relative to the preserved pre-exploration stage, those aggregate rates change by +4.3% and +1.3%. The C8 change from 124.9 to 126.6 tokens/s is a small stage difference, not a demonstrated causal speedup. C8 retains 30.3% of the C1 per-active-request rate. The dense control's 7.35x scaling shows this engine can scale that dense workload on GB10; it does not by itself explain Flash-Next's different scaling or isolate a version regression. These are stage comparisons, not causal estimates of individual optimizations or proof of a hardware ceiling.

[Saved counters](raw/scaling_investigation/final_metrics/README.md) bracket each whole subprocess, including discarded warmups and all trials. They give 79.3% draft-token acceptance and mean proposed depth 3.28 at C1, versus 66.4% and 5.18 across the combined C6/C8 phase. Target dispatches were 2,025 graph replays and 0 eager at C1, versus 720 replays and 1,065 eager dispatches for unsupported batch shapes across combined C6/C8. These counts do not measure kernel-time coverage. The snapshots cannot separate C6 from C8 and do not record depth distributions or actual expert occupancy.

[Memory boundary snapshots](raw/scaling_investigation/final_memory/summary.json) recorded no system swap-out during serving or concurrency commands. System swap-in occurred, including 151 MiB across the combined C6/C8 command; model process `VmSwap` remained approximately 477 MiB at its boundaries. These global counters bracket whole subprocesses, including warmups, and cannot identify model-specific page-ins or their effect on individual timed trials.
