# Flint (RecursiveSI) is unsupported, and it is the same MoE family as Kolibri-1

- **Repo:** mistral.rs
- **Components:** `mistralrs-core/src/models/`, `mistralrs-core/src/gguf/normal_registry.rs`,
  `mistralrs-core/src/gguf/normal_bindings.rs`, `mistralrs-core/src/gguf/gguf_tokenizer.rs`
- **Status:** surveyed, nothing implemented. The port is a shape change to a model we already
  have, not a new model family
- **Impact:** `flint` is rejected by the GGUF registry, and the safetensors path has no
  `FlintForCausalLM` at all

## What

Flint is a 40B-parameter MoE with 2.5B active, from RecursiveSI (not Mistral; the
`unsloth/flint-GGUF` and `RecursiveSI/flint` repos are the same model). It was released
2026-10-21 and is too new for mistral.rs to know about.

It is **the same MoE family as Kolibri-1**, which is why it matters here:

| | Kolibri-1 | Flint |
|---|---|---|
| layers | 50 | 49 |
| hidden | 2560 | 2048 |
| heads / kv heads | 48 / 4 | 32 / 4 |
| head dim | 128 | 128 |
| experts | 384 | 256 |
| experts per token | 6 | 6 |
| shared experts | 1 (512) | 2 (1024) |
| moe intermediate | 512 | 512 |
| dense ff (lead layers) | 512 | 6144 |
| sliding window | 513 | 513 |
| vocab | 128,000 | 150,144 |
| router | sigmoid, no renorm, with `exp_probs_b.bias` | sigmoid, **renormalised**, no bias |
| attention sinks | no | **yes** (`attn_sinks.weight`, 32 per layer) |
| rope | every layer | **14 layers have no rope** (`no_rope_layer_indices`) |
| router score function | - | `sigmoid` |
| softmax type | - | `learnable` |

So Flint is Kolibri-1 with four changes: attention sinks, a per-layer no-rope pattern, two
shared experts instead of one, and a renormalised router without the expert bias.

## The four things a port needs

1. **The model.** `FlintForCausalLM` is a new `mistralrs-core/src/models/flint.rs`. The MoE
   half is nearly `kolibri1.rs`: the same `layer_types` pattern, the same shared-expert
   shape, the same QK-norms. The genuinely new parts are the sinks (a `sinks: Some(...)` on
   the SDPA params; the backend already exists at
   `mistralrs-core/src/attention/backends/sinks.rs` and is wired for gpt-oss), the
   `no_rope_layer_indices` list (the per-layer `rope_pattern` in the GGUF), and the router
   renormalisation (`norm_topk_prob: true`).

2. **The GGUF registry.** `CanonicalGgufArchitecture` needs a `Flint` variant, the
   `flint` string mapping, and the layout. The tensor names follow the same convention as
   Kolibri-1's (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_inp`,
   `ffn_*_shexp`) with two additions: `attn_sinks.weight` per layer, and `ffn_gate`/
   `ffn_up`/`ffn_down` for the two leading dense layers.

3. **The tokenizer.** The GGUF carries `tokenizer.ggml.pre = "flint"`. Upstream llama.cpp
   has a `LLAMA_VOCAB_PRE_TYPE_FLINT` with its own regex (it splits `\p{N}{1,3}` and adds a
   CJK range, and sets `clean_spaces = false`). We have a `BpePreTokenizerKind` per family
   already, so this is one more variant of the Kolibri-1 one.

4. **The dense lead.** `leading_dense_block_count = 2` and `mlp_only_layers = [0, 1]`.
   Kolibri-1 has none, so the layer loop has to branch on it.

## The quant situation

This is the part that decides whether Flint is usable here at all.

**candle has no IQ types.** `GgmlDType::from_u32` covers `Q4_0`..`Q8_K` and stops there -
no `IQ2_XXS`, `IQ3_S`, `IQ1_M`. Our own `mistralrs-quant/src/gguf/archive.rs` has a *name*
table for the IQ types but no dequantiser for them, and it names `IQ2_XXS` as "a better fit"
for a smaller Kolibri-1. So the IQ family is a real hole, and it is the same hole that
`todo/gaps/04` already names.

The two GGUFs on the box are both unusable as they stand:

| file | size | experts | readable by us |
|---|---|---|---|
| `flint-preview-UD-IQ1_M.gguf` | 10.8 GB | IQ1_M | no |
| `flint-preview-UD-Q4_K_XL.gguf` | 23.3 GB | IQ3_S | no |

Both are verified against their `.sha256`. The Q4_K_XL's `ffn_gate_exps` and
`ffn_up_exps` are IQ3_S, and `ffn_down_exps` is Q5_K, so it is not even uniform.

**But a K-quant-only Flint exists.** `lmstudio-community/flint-GGUF` publishes
`flint-Q4_K_M.gguf` (24.2 GB), `flint-Q6_K.gguf` (32.7 GB) and `flint-Q8_0.gguf` (42.4 GB).
The Q4_K_M's tensor mix is `Q4_K` 412, `Q6_K` 74, `F32` 293 - entirely inside what candle
can read. `terelabs/flint-GGUF` has the same file in the same layout.

**The safetensors route is better still.** `HawkBearPig/flint-FP8-BF16` uses the *same*
blockwise fp8 recipe as Kolibri-1: `weight_block_size [128, 128]`, `float8_e4m3fn` weights,
`float32` scales, `weight_scale_inv` suffix, `float32(weight) * weight_scale_inv[row // 128,
col // 128]`. The tensor names are identical to Kolibri-1's too:
`model.layers.N.mlp.experts.I.{gate,up,down}_proj.weight` with the matching
`.weight_scale_inv`. Our `blockwise_fp8` module and the per-expert quantised gather in
`mistralrs-quant/src/per_expert_quantized.rs` expect exactly this layout.

So the fp8 Flint checkpoint is a **drop-in for the fast path we already built**: the scoped
gather, the dense-weight cache, and the expert residency cache all apply unchanged. That is
the interesting part: Flint is the first model that exercises them on a checkpoint we did
not build them for.

## Why this is worth doing

- It is a **second MoE family** to test the per-expert gather against, with a different
  expert count (256 vs 384), two shared experts, and a renormalised router. The
  Kolibri-1 logit-gap work needed a second implementation to compare against; Flint gives
  us a *third* family, and upstream llama.cpp already implements it, so there is an oracle
  for free.
- The **attention sinks** path exists in our backend but is only wired for gpt-oss. A second
  model on it is worth having.
- The **per-layer no-rope** pattern is a shape we do not have anywhere else, and it is the
  same kind of per-layer pattern as the sliding window, which we already handle per layer.
- The fp8 checkpoint is the **first test of the fp8 fast path on a checkpoint we did not
  author**, which is the strongest available check that the path is general.

## What it does not fix

Flint does not move the Kolibri-1 logit gap. That gap is a router tie-break and a bf16
activation width, and it is measured against the Kolibri-1 oracle dumps. Flint is a
different model and a different reference.

## Suggested order

1. The **GGUF route first**: `lmstudio-community/flint-GGUF/flint-Q4_K_M.gguf` is K-quants
   only, so it needs no new dequantiser and no new quant support - only the architecture,
   the bindings, the sinks, the no-rope list, and the tokenizer variant. Compare against
   upstream llama.cpp, which supports `flint` today (`src/models/flint.cpp`).
2. The **fp8 route second**, on `HawkBearPig/flint-FP8-BF16`, which should reuse the
   Kolibri-1 model code almost wholesale and put the scoped gather on a model it was not
   written for.
3. The **IQ types** are a separate, larger piece of work and are worth doing for their own
   sake - they are what makes the *published* Flint GGUFs readable, and `todo/gaps/04`
   already wants `IQ2_XXS` for a smaller Kolibri-1.
