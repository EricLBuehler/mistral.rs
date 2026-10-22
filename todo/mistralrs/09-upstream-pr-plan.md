# Upstream PR plan

- **Repo:** mistral.rs
- **Status:** the decomposition below is settled; PR 1 is pushed and ready to open
- **Impact:** the whole of `cpu-decode-bandwidth` that is worth upstreaming

## What is upstreamable and what is not

The branch is 112 commits ahead of master and 2.1M insertions, but most of that is not code:

```
todo/artifacts/llamacpp-oracle/kolibri1_layer_taps.tsv      1,280,000   our dumps
todo/artifacts/llamacpp-oracle/kolibri1-llamacpp-layerdump.tsv  783,360   our dumps
todo/artifacts/llamacpp-oracle/kolibri1_goldens_llamacpp.json   22,802   our dumps
mistralrs-core/src/models/kolibri1.rs                            2,444   the model
todo/... (the tracker)                                             ~3,000  our notes
```

The upstreamable set is about twenty files, and the rest is the tracker and the measurement
dumps, which stay here. The llama.cpp side of the work - the per-layer taps, the router dump -
is in the other tree and is not in scope for these PRs.

## The existing stack, and how it reconciles

There is already a six-PR stack, built earlier in the session:

| branch | delta vs master | what it is |
|---|---|---|
| `pr/01-rmsnorm-scale-f32` | 1 file, 62 lines | the RMSNorm f32-scale fix |
| `pr/02-cpu-decode-engine` | 19 files, 2,568 | the decode GEMV, the bf16 dot, the NEON attention |
| `pr/03-kolibri1-model` | 31 files, 5,233 | the Kolibri-1 model |
| `pr/04-bfmmla-reduction-order` | 31 files, 5,393 | the BFMMLA dot variant |
| `pr/05-moe-residency-cache` | 32 files, 5,634 | the expert residency cache |
| `pr/06-layer-taps-and-bench` | 32 files, 5,860 | the per-layer taps and the bench harness |

The stack is cumulative: each PR takes the previous ones' paths, so the diff against master
grows and the *delta* is what gets reviewed. That is the convention to keep, and it means
"small" is a property of the delta rather than of the branch.

**`pr/01` should be dropped, not opened.** It is the RMSNorm f32-scale fix, and
huggingface/candle#4057 fixes the same defect upstream. Once that lands the local
reimplementation is redundant and slower than candle's, so opening it would be asking a
maintainer to take a change that the next candle release makes unnecessary.

**The work since the stack was built needs new PRs on top of it**, and they are the ones this
session added: the per-expert gather, the scoped dequants on both paths, the fp8 dense-weight
cache, the expert cache wiring, the `kolibri1` GGUF path, and the fp8 dequant's f32 form.

**One PR does not belong in the stack at all**: the fp8 dequant's f32 form touches only
`blockwise_fp8/ops.rs` and depends on nothing, so it is standalone and off master. It is
pushed as `pr/01-fp8-dequant-f32` and is the first thing that can be opened.

## The decomposition

Each PR takes master's final state for the paths it owns, so they are independent of each other
except where one needs another's new type.

| # | PR | files | lines | needs | measured effect |
|---|---|---|---|---|---|
| 1 | fp8 dequant keeps f32 instead of widening to f64 | `blockwise_fp8/ops.rs`, `Cargo.toml`, the probe | 144 | - | 4.86 -> 5.18 GB/s of bf16 output on the fp8 checkpoint, bit-identical |
| 2 | the per-expert CPU gather | `moe_gather_cpu.rs`, `per_expert.rs`, `unquantized/mod.rs`, `lib.rs` | ~1,100 | - | the per-expert gather the next two need; bit-identical to the fused path by construction |
| 3 | the fp8 MoE gathers only the routed experts | `per_expert_quantized.rs`, `blockwise_fp8/mod.rs` | ~560 | 2 | **70.8 -> 2.4 s per token** (29.5x) |
| 4 | the GGUF MoE gathers only the routed experts | `gguf/cpu.rs` | 151 | 2 | the same fix on the GGUF path; the Q2_K route's footprint |
| 5 | the expert residency cache | `moe_cache.rs` + the wiring in 3 | ~330 | 3 | the fp8 path's expert cache: **3.5x** on the routed phase |
| 6 | the dense fp8 weight is dequantised once | in 3's file | - | 3 | the attention and output projections: 957 -> 61 ms per step |
| 7 | Kolibri-1 support | `models/kolibri1.rs`, the registry, the config | ~2,500 | - | a new model |
| 8 | the Kolibri-1 GGUF path | `gguf/normal_bindings.rs`, `gguf_tokenizer.rs`, `gguf/normal_config.rs`, `gguf/normal_registry.rs` | 122 | 7 | the GGUF names, the `kolibri1` pre-tokenizer profile, the four sandwich norms |
| 9 | the CPU decode GEMV, the bf16 dot, the NEON attention | `gemv_cpu.rs`, `dot_cpu.rs`, `cpu_threads.rs`, `attention/` | ~1,300 | - | the throughput core; needs its own benchmarks upstream |

**6 rides with 3** because both live in `blockwise_fp8/mod.rs`, and taking that file's final
state wholesale is the convention that keeps the PRs independent. Splitting it would mean
hand-editing one file into two states, which is how a stack stops being reviewable.

## The order to open them in

1, then 7 and 8 (independent of everything else, and 8 is the one that makes a published
GGUF loadable at all), then 2, then 3 and 4, then 5, then 9 last. 9 is the largest and the
most likely to need a conversation, because it changes the kernels rather than adding to them.

## What is deliberately not in the set

- **The RMSNorm f32-scale fix.** `mistralrs-core/src/layers.rs` reimplements the norm with the
  scale in f32, because candle's fused CPU kernel rounds the scale to the storage dtype. That
  is huggingface/candle#4052, and huggingface/candle#4057 fixes it upstream. Once that lands,
  the local fix is redundant and *slower* than candle's, so it should not be upstreamed at all -
  the right outcome is to drop it and take the upstream fix.
- **The measurement dumps and the tracker.** Ours.
- **The llama.cpp side.** The other tree.

## Notes for the reviewer

- Every PR in the set is bit-identical to the code it replaces, and says so in a test. The
  per-expert gather is bit-identical to the fused gather, the scoped dequant is bit-identical to
  the whole-stack dequant, and the f32 dequant is bit-identical to the f64 one. The upstream
  property this work rests on is that parallelisation never reorders a reduction, and each of
  these PRs is a case of that.
- The numbers in the table are from one box with a shared-host caveat, and the ones quoted in
  each PR's description carry the conditions they were taken under.
