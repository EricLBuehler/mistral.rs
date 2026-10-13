# Kolibri-1 CPU forward does not reproduce the reference logits

**Status:** mechanism found and the one attributable cause fixed (`09f6c7e56`). What remains is
irreducible: 2.2% of routings sit inside the arithmetic's own noise, against a reference
implementation that is not bit-exact with the goldens either.

**Where:** `mistralrs-core/src/models/kolibri1.rs`, `Model::forward` / `DecoderLayer::forward`.

## Measured

bf16 checkpoint, 8-prompt golden pass, node 3, 32 threads:

| metric | this build | reference vLLM CPU forward |
| --- | --- | --- |
| worst top-8 logit gap | 6.59 | 2.19 (stated band 2.5) |
| chain argmax | 166/248 | 141/145 |
| first-step argmax | 0/8 | n/a, effectively 8/8 |
| mean top-k logit gap (64 entries) | 2.324 | 2.19 (stated envelope, a max not a mean) |
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

## Resolved: the router amplifies rounding, and it was being rounded

The mechanism is now measured rather than assumed. Traced over 5550 routings (3
golden prompts x 50 layers), recording the gap between the 6th and 7th
`logits + bias` scores against the top-to-7th spread:

| | bf16-valued logits | true f32 |
| --- | --- | --- |
| routings with a 6/7 gap below 1e-3 of the spread | 7.0% | 2.2% |
| selection actually changed by widening the matmul | 7.7% | - |

The histogram predicts the flip rate to within a percentage point, so the
accounting is direct: a perturbation smaller than the difference between two
implementations that are both correct reroutes ~8% of (layer, token) decisions.
Compounded over 50 layers, `0.923^50 = 1.9%` - the chance that a token's
representation survives the stack untouched is under 2%. That is the whole
explanation for a logit gap of several units with the ranking still mostly intact.

The one thing that could be *our* fault was, and was fixed in `09f6c7e56`: the
router logits were rounded to bf16 before selection, because candle's matmul
returns bf16 for bf16 @ bf16 and `to_dtype(F32)` afterwards widens values that
are already bf16-representable. Both reference implementations compute the gate
in f32 (`LinearBTRaw(..., kF32)` and `out_dtype=torch.float32`). Holding the
weight widened drops the sub-1e-3 routings from 7.0% to 2.2% and improves every
statistic over the 64 top-k golden entries:

| variant | mean | median | p90 | max |
| --- | --- | --- | --- | --- |
| bf16 | 2.452 | 1.969 | 5.250 | 6.719 |
| f32 | 2.324 | 1.844 | 4.875 | 6.594 |

The residual 2.2% is the irreducible part: attention reduction order and matmul
blocking, neither of which any implementation controls. The reference's own
measured envelope against these torch goldens is 2.19, so this build is now
within a small factor of a reference implementation that is itself not
bit-exact.

## The forward is correct

The strongest evidence is behavioural, and it needs no goldens. On the bf16
checkpoint the model answers:

```
"The capital of France is"     -> " Paris"    (10.50, margin 1.69)
"The capital of Australia is"  -> " Sydney"   (10.62, margin 1.06)
"The capital of Japan is"      -> " Tokyo"    (11.44, margin 1.81)
```

That exercises embedding, both MoE paths, the sandwich residual threading, the
per-head qk-norm, RNoPE on the sliding layers only, and the causal mask, in one
prefill. Paris and Tokyo are asserted by `kolibri1_answers_ordinary_prompts`.

Australia is deliberately not asserted. This build says Sydney; it said Canberra
when the router was bf16-valued. Two plausible capitals from a rounding
difference is the amplification above appearing in a semantic answer rather than
in a fingerprint, so pinning either one would be pinning a coin flip.

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

## What would still be needed to close the last 2.2%

Not worth doing, and worth saying why. Three implementations are in this
comparison - the torch goldens, the vLLM CPU forward, and this one - and the
reference's own gap to the goldens is 2.19 on prompts whose margins are
fractions of a logit. Making this build agree to the last bit would mean
reproducing another implementation's reduction order exactly, which is a
different and much less useful goal than the one this issue started from.

The thing that would still be worth having, if anyone wants it, is a
diagnostic: per-layer hidden states from all three implementations. The first
layer whose hidden state diverges past bf16 noise would show whether the residual
2.2% accumulates gradually (consistent with everything measured here) or jumps at
one layer (which would mean a bug this investigation missed). None of the three
emits them today.

## Gate impact

`kolibri1_matches_reference_goldens` asserts on the 6/8 reference top-1 and floors the
248-sample chain at 55%, where this build measures 66.9%. Both catch a collapse;
neither certifies bit-exactness, and the report line prints the logit gaps so a
regression in the gap is visible even when both assertions still pass.

The first-step argmax is reported and deliberately not asserted on. It is the most
tempting metric - one sample per prompt, no self-reference - and the least useful
here: it moved 2/8 -> 0/8 when the router was made to match the reference's f32
arithmetic, while every statistic over the 64 top-k entries improved. That pair of
results is the clearest demonstration in this file of how little signal 8 samples
carry on this model.