# Actual Flash-Next routes: isolated expert replay

The experiment captures exact target-layer-8 operands from the real Qwen3.8-Flash-Next model, then replays the same inputs, routes and post-ISQ weights through indexed GEMV and grouped MMQ. It tests kernel dispatch on actual routes; it does not measure full-model serving throughput or a hardware ceiling.

## Capture and provenance

One temporary instrumented server used the pinned `de4b8e4d43b917e7706784d8bb445c9af86a3540` snapshot, ISQ q4k, fixed MTP depth 6, maximum 8 sequences, context 16384, no prefix cache and disabled CUDA graphs. The runner submitted 8 diverse existing serving-benchmark prompts at each of C1/C6/C8, with 128 output tokens each. All 24 requests completed with 128 tokens and nonempty output. Primary requests omitted logprobs to retain ordinary greedy device verification. The CPU validator checked every saved activation, routing weight and output for finite values, shape, dtype, and route bounds/distinctness.

The hook stored BF16 expert inputs/outputs, U32 selected expert IDs, F32 routing weights, and exact QTensor bytes once in GGUF. It captured the routed expert contribution before the shared expert addition. Gate/up are Q4K; down is Q4_1. The temporary core hooks and production CLI were restored before capture began. Serving used a separately archived diagnostic CLI, then stopped before all replay timing. `build/lifecycle.json` records restoration. `runtime/runtime.metadata.json` records the server configuration, PID exit and original failed replay.

Capture coverage is 8 native M1 decode, 8 native M7 verification, 5 native M42 verification and 5 native M56 verification samples. C6/C8 use native B6xQ7/B8xQ7 layouts. Capture writes and downloads synchronize execution, so these samples do not establish an unperturbed routing or autotuner-depth distribution.

## Native results

Times are milliseconds for the routed expert pipeline of one layer. Each sample uses 3 warmups and 7 alternating-order rounds, averaging 10 forwards per round. Table times are medians across per-sample medians; the speed ratio is the median of paired per-sample GEMV/MMQ ratios, so it need not equal the ratio of the displayed medians. CUDA-event intervals include host launch gaps; host timings are retained too.

| Native shape | Samples | Selected experts, median | GEMV ms | Grouped MMQ ms | Paired GEMV/MMQ ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode, M1 | 8 | 10 | 0.1162 | 0.3206 | 0.363 |
| C1 verify, M7 | 8 | 31 | 0.5556 | 0.5567 | 0.987 |
| C6 verify, M42 | 5 | 164 | 4.0690 | 2.6551 | 1.536 |
| C8 verify, M56 | 5 | 213 | 5.3111 | 3.5214 | 1.582 |

For the native 56 samples, U ranges 132..219 selected experts out of 512. Median assignments per selected expert are 560/U=2.629; the busiest expert's occupancy ranges 16..45 rows. This is sparse reuse across many weights, not the synthetic ideal of seven identical routes for every expert. At 2,867,200 bytes per quantized expert, U=213 selects 610,713,600 logical weight bytes. These are a working-set description and potential intra-call reuse, not measured DRAM traffic or a bandwidth lower bound; caches may supply weights.

The current grouped path beats GEMV on all native 42/56 samples, so this experiment does not support forcing GEMV for those shapes. It leaves other kernels, launch overhead and dispatch choices open.

## Derived controls

`replay_adjusted/replay.json` also contains explicitly labeled prefix controls and one-query-per-sequence controls. Prefix controls take the first flattened rows and can concentrate rows from only a few sequences. The separate `replay_sequence_query_prefix/` run addresses that geometry: it takes the first 3/4/5 query positions from EACH of the 8 sequences in each original B8xQ7 capture. Exact source row indices are saved. These match the row layout of B8xQ3/4/5 but are derived from depth 6 activations, not independently observed depth 2/3/4 execution.

| Derived layout | Samples | Selected experts, median | GEMV ms | Grouped MMQ ms | Paired GEMV/MMQ ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| B8xQ3, M24 | 5 | 134 | 2.6125 | 2.0241 | 1.257 |
| B8xQ4, M32 | 5 | 151 | 3.3628 | 2.3982 | 1.368 |
| B8xQ5, M40 | 5 | 168 | 3.9923 | 2.7445 | 1.433 |

Grouped MMQ wins all 5 balanced M24 comparisons, with paired ratios 1.191..1.503. This supports evaluating a lower dispatch threshold in a full model. It does not establish an end-to-end gain or justify changing all models' dispatch policy from this one layer.

## Numerical checks and preserved failure

All 26 native same-dispatch replays satisfy the strict relative-RMS<1e-3 and cosine>.999999 gate. Of these, 22 are exactly equal; the worst remaining relative RMS is 2.75e-7. Finite checks remain mandatory.

The first replay stopped when GEMV/MMQ disagreement on one C1 verification input exceeded the inherited synthetic 3% RMS guard (3.330%). Its original source, executable hash, log and failed lifecycle are preserved. That failure is not silently marked successful. The adjusted diagnostic records the same 3%/cosine>.999 guard as a pass/fail flag while retaining strict native equivalence and finite checks. It reports 4 flagged native and 2 flagged derived samples out of 71 total. A flagged alternate kernel is not established as a safe replacement by timing it. All 15 additional balanced-query controls pass that existing cross-kernel guard.

A separate oracle test ran AFTER timings on all 26 native and both flagged derived samples, then all 15 balanced-query samples. It dequantizes the same dumped QTensors to F32, computes gate/up, SiLU product and down with F32 cuBLAS and TF32/reduced-F32 explicitly disabled, and accumulates routing-weighted contributions on CPU in F64 before F32 comparison. It has no activation quantization. Its transposes, transfers and reduction are excluded from timings.

Median RMS errors against this reference are:

| Shape | GEMV RMS | Grouped MMQ RMS |
| --- | ---: | ---: |
| Native M1 | 1.509% | 2.284% |
| Native M7 | 1.772% | 2.372% |
| Native M42 | 2.075% | 2.407% |
| Native M56 | 2.145% | 2.155% |
| Derived B8xQ3 | 2.117% | 2.199% |
| Derived B8xQ4 | 2.168% | 2.255% |
| Derived B8xQ5 | 2.067% | 2.174% |

These are measured numerical differences, not new accuracy tolerances or model-quality guarantees. The two production implementations differ in activation quantization/correction and rounding, in addition to reduction order. Neither these results nor the original cross-guard failure alone identify a broken kernel.

## Files and limitations

- `runtime/captures/`: all native metadata and activation tensors; `capture.validation.json` checks/hashes include the exact GGUF.
- `runtime/requests.json`, `server.log`, `server.memory.jsonl`: bounded capture evidence. Capture request times are not serving benchmarks.
- `replay_adjusted/{replay,oracle,summary,lifecycle}.json`: 71 ordinary/derived comparisons and 28 reference comparisons.
- `replay_sequence_query_prefix/{replay,oracle,summary,lifecycle}.json`: 15 balanced-query comparisons and 15 reference comparisons.
- Both replay directories include exact source snapshots, scripts, build/check/Clippy logs and executable hashes. The extra run includes a passing row-layout regression.
- `base_tree/`, `patch_tree/`, `capture.patch`, `replay.patch`, `source_manifest.json`: reversible diagnostic-hook provenance. Those patches describe the initial capture build; later replay source snapshots are authoritative for each replay.
- `external_artifacts.json`: exact local paths/hashes of the 1.37 GiB dumped weight file and executable archives, omitted from this repository package.
- `SHA256.json`: verifies every packaged file except itself.

Repeated isolated-layer work can retain selected weights in cache differently from a full model. This experiment excludes shared experts, attention, GDN, hyper-connections and CUDA graphs, and samples only layer 8. No claim of bandwidth saturation, complete absence of a regression, or physical scaling ceiling follows from it.
