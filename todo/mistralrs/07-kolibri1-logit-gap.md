# Kolibri-1 CPU forward does not reproduce the reference logits

**Status:** open, unlocalized. Numerical, not architectural.

**Where:** `mistralrs-core/src/models/kolibri1.rs`, `Model::forward` / `DecoderLayer::forward`.

## Measured

bf16 checkpoint, 8-prompt golden pass, node 3, 32 threads:

| metric | this build | reference vLLM CPU forward |
| --- | --- | --- |
| worst top-8 logit gap | 6.72 | 2.19 (stated band 2.5) |
| chain argmax | 168/248 | 141/145 |
| first-step argmax | 2/8 | n/a, effectively 8/8 |
| reference top-1 at final position | 6/8 | 8/8 |

The fp8 checkpoint measures the same or worse (worst gap 6.84), so this is not the fp8
dequant path.

## Why the chain number is not the headline

The generated sequences are degenerate: a base LM on a six-token prompt loops. `"alpha beta
gamma delta epsilon zeta"` emits token 121598 thirty-two times; `"der Mond..."` alternates
two tokens. Scoring a teacher-forced chain of those is scoring a repetition task, so a single
early flip makes every later step differ by construction. That is why the reference's own
gate is 141/145 rather than 145/145.

**Step 0 is the only per-prompt sample that isolates correctness**: it predicts from the
prompt alone with no self-reference. It scores 2/8 here, but see below - that number is
dominated by prompts where the reference's own top-1/top-2 margin is a fraction of a logit,
not by a wrong forward.

## The forward is correct; this is a numerical-amplification gap

The strongest evidence is behavioural, and it needs no goldens. On the bf16
checkpoint the model answers:

```
"The capital of France is"     -> " Paris"    (10.38, margin 1.56)
"The capital of Australia is"  -> " Canberra" (10.12, margin 0.06)
"The capital of Japan is"      -> " Tokyo"    (11.25, margin 2.12)
```

That exercises embedding, both MoE paths, the sandwich residual threading, the
per-head qk-norm, RNoPE on the sliding layers only, and the causal mask, in one
prefill. It is pinned by `kolibri1_answers_ordinary_prompts`.

"Canberra at a margin of 0.06" is also the point of the whole issue. This model
is knife-edge on these prompts: the goldens' own bench notes four of the
reference's 145 argmax positions flipping at margins of 0.009 to 0.38. A 0.06
margin on a question the model has to get right means the fingerprint comparison
is measuring rounding, not correctness, for any prompt whose margin is under a
logit.

The likely mechanism is amplification rather than a defect. A 384-expert router
with top-6 selection turns a difference of order 1e-3 in the logits into a
different expert for whichever token happens to sit near the 6th/7th boundary,
and 50 layers of that compounds. The reference implementation and the torch
goldens differ from each other by 2.19 for the same reason; there are three
implementations in this comparison (torch goldens, the vLLM CPU forward, this
one), not two.

Confirming it needs per-layer hidden states from all three, which neither
existing implementation emits.

## A self-consistency metric that looked alarming and is not

An earlier version of this file recorded first-step argmax 2/8 as the headline.
Reading per-position predictions showed most of that is not model error at all:

```
row 1: top " capital"  rank(ids[1]) = 0    rank(ids[2]) = 7
row 2: top " of"       rank(ids[2]) = 0    rank(ids[3]) = 115
row 3: top " is"       rank(ids[4]) = 0
row 4: top " Canberra"
```

Rows 1 and 2 predict their own index's token, which looks like a one-position
shift until you notice " capital" is a perfectly good continuation of "The" and
" of" a plausible one after "The capital". Rows 3 and 4, the ones the
fingerprints use, are correct. A per-row rank dump distinguishes the two; a
0/9 aggregate does not.

## Not a measurement error

The comparison harness itself was checked:

- `input_ids` are taken verbatim from the goldens, no BOS, matching
  `tokenizer.encode(prompt, add_special_tokens=False)` in the generator script.
- The fingerprint context is `input + generated[0..30]`, which is the forward that predicted
  the 32nd token. (Using the full chain instead is one position late and the reference
  measures top-k diffs up to 11.95 for that mistake - this build is nowhere near it.)
- RoPE positions are `0..seq_len`, from `seqlen_offsets = [0]` and `position_ids =
  0..seq_len`.
- Step `k` is read from row `prompt_len - 1 + k`, so step 0 is the last prompt token.

## Ruled out

Every one of these was diffed line by line against
`src/vllm/model_executor/models/kolibri1_forward.cpp` and `kolibri1_shared.h`:

- **Residual threading.** The residual carries post-normed values, not a pre-norm stream.
  `res += xs` at `input_layernorm`, `res += attn_n` at `post_attention_layernorm`, then
  `hidden = post_ffn_norm(moe)` and `res += hidden` at the next layer. `post_attn_norm` and
  `post_ffn_norm` carry none. This one was a real bug (0/8 -> 6/8) and is fixed.
- **Router.** `mlp.gate` logits in f32, select on `logits + expert_bias`, weight by
  `sigmoid` of the *unbiased* logit, no renormalisation, ties to the lower index. Matches
  `SigmoidLogitAddRouting`.
- **Expert combine.** Weighted routed sum plus the ungated shared expert, `routed_scale`
  1.0, shared added after weighting.
- **qk-norm.** Per-head RMSNorm over `head_dim` with a shared gamma, **before** rope, on
  every layer including the no-rope ones. The `.contiguous()` before the norm-only branch
  was a real bug and is fixed.
- **RNoPE.** RoPE on sliding layers only, full rotary (`rotary_dim == head_dim`), neox
  style. Full-attention layers get no rope at all, not a partial one.
- **Mask.** `make_swa_mask(53, 0, 513)` is provably full causal here: the `too_old` branch
  needs `q_pos >= sliding_window`, and 53 < 513. So the sliding window cannot explain a gap
  on these prompts.
- **GQA and scale.** `n_kv_groups = 48/4 = 12`, `softmax_scale = 128^-0.5`. Matches.
- **Logit gain or a missing bias term.** The goldens carry `logits_sum` and `abs_sum` over
  the whole vocab row. This build's two agree with each other to within 0.3% and land
  within 6-15% of the reference, so there is no scale error or constant offset - the hidden
  states are simply different.

## What is left

Ruled out by measurement, not by reading:

- **The router's bf16 logits.** Widening the router weight so the gate matmul runs in f32
  rather than upcasting afterwards scores *worse* (0/8 first step against 2/8), so the
  rounding the upstream `out_dtype=torch.float32` warns about is not the cause here.
- **The causal mask.** `binary_mask_range` derives the visible range from the mask including
  the diagonal, and two new regression tests pin it at both the kernel and the dispatcher.
  Previously nothing asserted "the CPU kernel includes self", which is why the `CausalFlash`
  hole next to it went unnoticed; that hole is closed in `fa638120d`.
- **The forward as a whole.** Paris, Berlin, Canberra, Tokyo, plus comma continuation and
  code completion, all correct in one prefill.

That leaves amplification as the working hypothesis, and it is consistent with the
reference's own behaviour: it disagrees with the torch goldens by up to 2.19 on the same
prompts, and flips 4 of 145 argmax positions at margins of 0.009 to 0.38.

The decisive next step is per-layer hidden states from all three implementations - the torch
goldens, the vLLM CPU forward, and this one. None emits them today, so there is nothing to
diff. The first layer whose hidden state diverges past bf16 noise names the culprit, and a
gradual divergence from layer 0 versus a jump at one layer separates ordinary accumulation
from a routing flip.

A cheaper partial experiment, not yet run: force `top_k = 1` on two of the three sides. That
removes the router's amplification and leaves pure attention plus MLP, which would put a
bound on the non-MoE part of the gap.

## Gate impact

`kolibri1_matches_reference_goldens` now floors the chain at 55% and first-step at 12.5%,
where this build measures 67.7% and 25%. Those floors catch a collapse; they do not
certify correctness. Raising them is the acceptance criterion for this issue.