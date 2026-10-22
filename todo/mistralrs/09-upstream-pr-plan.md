# Upstream PR plan

- **Repo:** mistral.rs
- **Status:** the chain is rebased onto `origin/master` and every branch compiles; the work
  since the chain was built is not in any of them yet, and the IQ and Flint stacks (below) are
  separate branches on `flint_by_flint` with nothing pushed
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

The upstreamable set is about twenty files. The dumps, the tracker, and the llama.cpp side of
the work stay here.

## The chain, after the rebase

The upstream moved 7 commits while the stack was open, so everything was rebased onto
`origin/master` (`f512686b3`). Three conflicts, all of them the upstream adding code next to
ours:

- `pipeline/mod.rs`: the upstream added `ActivationMemoryContext`, `ActivationMemoryEstimate`
  and `ActivationModelConfig` to the loader import list.
- `layers.rs`: the upstream added `fused_add_rms_norm_applies` next to our
  `rms_norm_f32_scale`. The conflict was *inside* the function, so the two sides could not be
  concatenated; the rebased stack holds the correct merge and the PR branches take that file
  from it.
- `normal_loaders.rs`: the upstream added a required `activation_model_config` method to the
  device-mapped loader trait, which the Kolibri-1 loader has to implement.

**The chain's order was wrong and is now right.** `per_expert.rs` calls `crate::moe_cache`, so
the residency cache has to arrive *with* the per-expert path rather than after it, and
`pr/05-moe-residency-cache` duplicated that. The cache is now part of the decode engine PR and
the separate one is gone. Two branches could not compile before this, independent of the
rebase: `pr/02` declared `pub mod moe_cache` without the file, and the cache PR came after the
PR that uses it.

| branch | what | lines | needs |
|---|---|---|---|
| `pr/01-fp8-dequant-f32` | the fp8 dequant keeps f32 instead of widening to f64, and the probe that measures it | 136 | - |
| `pr/02-cpu-decode-engine` | the decode GEMV, the bf16 dot, the NEON attention, the per-expert gather, the residency cache | 2,907 | - |
| `pr/03-kolibri1-model` | the Kolibri-1 model | 5,595 | 2 |
| `pr/04-bfmmla-reduction-order` | the BFMMLA dot variant | 5,755 | 3 |
| `pr/05-layer-taps-and-bench` | the per-layer taps and the bench harness | 5,981 | 4 |

The chain is cumulative: each PR takes the previous ones' paths, so the diff against master
grows and the *delta* is what gets reviewed.

**`pr/01-rmsnorm-scale-f32` is not in the set and its branch is deleted.** The RMSNorm
f32-scale fix is huggingface/candle#4052, and huggingface/candle#4057 fixes the same defect
upstream. Once that lands the local reimplementation is redundant and slower than candle's,
so it should not be upstreamed at all.

## The work since, which is in none of them

| what | files | effect |
|---|---|---|
| the scoped fp8 gather | `per_expert_quantized.rs`, `blockwise_fp8/mod.rs` | **70.8 -> 2.4 s per token** |
| the scoped GGUF gather | `gguf/cpu.rs` | the same fix on the GGUF path |
| the dense fp8 weight cache | `blockwise_fp8/mod.rs` | the attention and output projections: 957 -> 61 ms per step |
| the fp8 expert cache wiring | `blockwise_fp8/mod.rs` | the fp8 path's expert cache: **3.5x** on the routed phase |
| the Kolibri-1 GGUF path | `gguf/normal_bindings.rs`, `gguf/gguf_tokenizer.rs`, `gguf/normal_config.rs`, `gguf/normal_registry.rs` | the published GGUF loads at all |
| the f32 dequant | `blockwise_fp8/ops.rs` | 1.07x, bit-identical |

These are the next PRs. The scoped fp8 gather is the one that carries the 29.5x, and it
depends on the per-expert gather that `pr/02` already carries, so it is the natural first
addition to the chain.

## The IQ and Flint stacks, 2026-10-22

Two stacks that are not part of the chain above. The IQ one is generic - it is what makes any
IQ GGUF load - so it can land ahead of the model work and does not touch the chain's paths.

| branch | commits | what | needs |
|---|---|---|---|
| `pr/iq-block-formats` | `af82d4bf6` | the nine IQ block layouts, the grids/sign tables extracted from `ggml-common.h`, and golden vectors from real files decoded by gguf-py | - |
| `pr/iq-linear` | `37a9e461d` | `IqTensor` (blocks stay in the mapping, or in a buffer for the sharded case) and `IqGgufMatMul` (`QuantMethod`: dense f32 cache, expert-scoped gather), wired through `load_iq_linear` | the first |
| `pr/gguf-eog-stop` | `2a2a12b9c` | `tokenizer.ggml.eog_token_ids` joins the stop set | - |
| `pr/flint-model` | `86a985579`, `dce303a5c` | the Flint model, registry/config/bindings, its tests | the stop-token PR, `pr/iq-linear` for the published files |
| `fix/flint-window` | `831fea537` | the sliding window reaches `SdpaParams` on the fused sinks path | `pr/flint-model` |

The IQ pair rests on the same property as the rest of the set: the expert gather is bit-identical
to the whole-stack path, and the tests say so over every shape the MoE hands it, including the
small-shape matmul fallback. The pointer test in `pr/iq-linear` pins the other invariant that
matters there, which is that a loaded tensor is a view of the mapping and not a copy.

The `pr/iq-linear` branch name is a placeholder: if the reviewer prefers, the two IQ commits can
land as one PR, since the block formats have no consumer on their own.

## The order to open them in

`pr/01-fp8-dequant-f32` is standalone and can go at any time. Then the chain, in order. Then
the work above, starting with the scoped fp8 gather. The decode engine PR is the largest and
the one most likely to need a conversation, because it changes the kernels rather than adding
to them.

## Notes for the reviewer

- Every PR in the set is bit-identical to the code it replaces, and says so in a test. The
  per-expert gather is bit-identical to the fused gather, the scoped dequant is bit-identical
  to the whole-stack dequant, and the f32 dequant is bit-identical to the f64 one. The
  upstream property this work rests on is that parallelisation never reorders a reduction, and
  each of these PRs is a case of that.
- The numbers are from one box with a shared-host caveat, and the ones quoted in each PR's
  description carry the conditions they were taken under. The same measurement moves 2.7x on
  page placement alone, so a bare tok/s figure would mislead.
