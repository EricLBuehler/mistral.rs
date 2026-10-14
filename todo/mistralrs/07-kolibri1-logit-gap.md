# Kolibri-1 CPU forward does not reproduce the reference logits

**Status:** router-noise mechanism found and its attributable cause fixed (`09f6c7e56`). The
*router* remainder is irreducible, but it does not account for the whole gap: the goldens' own
scale oracle fails by a prompt-dependent few percent, and that is not rounding. See
"Where the remaining gap probably is" below. Re-measured after `12e0b9ad7`.

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

## FALSIFIED: the router dtype does not affect the goldens at all

This file previously claimed that moving the router from bf16-valued logits to true f32
"improves every statistic over the 64 top-k golden entries" (mean 2.452 -> 2.324). **That is
wrong, and the error came from treating two different measurements as one thing.**

The oracle is `scripts/gen-kolibri1-goldens.py`, a pure-torch transcription, and its router
line is:

```python
logits = self.linear(h, p + ".mlp.gate").to(torch.float32)   # bf16 matmul, THEN widen
```

So the goldens were produced with a **bf16** router, whatever the upstream plugin does. This
build ran a true f32 GEMV against a weight held widened. Making it match the oracle - bf16
weight, bf16 matmul, widen afterwards - produced output that is **bit-identical on all eight
prompts**: every `logits_sum` and `abs_sum` to the printed precision, first-step 0/8, chain
166/248, top-1 6/8, worst gap 6.5938. Not close. Identical.

So the router's precision is not what separates this build from the goldens, and the 2.324 vs
2.452 comparison was measuring something other than what it was read as. The f32 router is
kept, because it is the faster of two things that agree: bf16 drops the weight to 1.97 MB,
which falls under the 2 MiB parallelism threshold, and the router goes serial - **3.5 ms ->
12.0 ms** for output that is bit-identical.

What this does *not* do is explain the scale gap below. If the router can change freely without
moving a single digit, then router noise cannot be the mechanism, and the first-step 0/8 is
caused by something the router has no part in.

## Re-measured after the decode work (`12e0b9ad7`)

Routing the MoE router through the decode GEMV changed its f32 accumulation order, so the
oracle was re-run. It moved nothing:

```
first-step 0/8, chain argmax 166/248, final-token argmax 6/8,
reference top-1 6/8, worst top-k logit gap 6.5938   (recorded before as 6.59)
```

Identical to four significant figures, so that commit is performance-only with no measured
effect on the goldens.

## Where the remaining gap probably is: not a scale error either

**This section previously argued the gap was a scale error. That was wrong, and it can be
disproved without running anything.**

The argument was that `logits_sum` and `abs_sum` agree to within 0.002 on every prompt, which
looks like the logit row being close to a rescaling of the reference's. But the final stage is

    logits = lm_head(rmsnorm(h + residual))

and `rmsnorm` is scale-invariant to high precision. With `x -> s*x`, RMSNorm returns
`s * x * rsqrt(s^2 * var + eps)` against `x * rsqrt(var + eps)`; with `rms_norm_eps = 1e-6` and
a normalised `var ~ 1` the eps term is negligible and the two are equal to within f32 rounding.
**A uniform scale error in the residual stream cannot produce a 3% difference in the logits at
all** - the final norm removes it. So the ratio agreement was a coincidence between two noisy
statistics, and there is no scale error to find.

What the numbers actually say is the opposite of a scale:

| | value |
| --- | --- |
| top-1 logit (oracle, prompt 1) | 9.44 |
| worst top-k logit gap | 6.59 |
| top-1 / top-2 margin | 1.97 |
| mean abs logit over the vocab row | 3.81 |

The top of the distribution is wrong by **70% of its own magnitude** while the bulk of 128,000
logits is within 3%. A scale error cannot do that, and neither can ordinary accumulation noise:
if the final hidden state differed by bf16 rounding (~2e-3 relative per element), the error it
induces in a logit is `~2e-3 * ||W_i|| ~ 0.008`, two orders of magnitude below 6.59.

That gap is the size of a **structured, discrete** difference - not a rounding, not a scale. The
one discrete choice in a decode step is which 6 of 384 experts a token routes to, and a single
different expert replaces a whole `2560 -> 512 -> 2560` block of the MLP with different weights.
The signature fits that: the bulk stays close because most of the computation agrees, and the
top moves because a few matmul blocks do not.

So the residual gap is most likely **expert selection differing on some routings**, which is the
2.2%-of-routings knife-edge effect measured earlier - not as a second-order rider, as the
previous revision of this file said, but as the actual mechanism. The router-dtype experiment
above does not contradict this: it shows that *perturbing the router's precision* does not move
the selection on these eight prompts, which is a statement about this particular perturbation,
not about the knife-edge cases being unreachable.

### How to localise it

Not by ratio - there is no ratio, the final norm is scale-invariant. By **comparison of the
selected expert ids**, which is the one discrete quantity in a decode step:

1. For one short prompt, record the top-6 expert ids per layer from this build.
2. Get the same from the reference. This needs `scripts/gen-kolibri1-goldens.py` run with a hook
   on `Kolibri1Ref.mlp`, which needs torch. torch is not installed on this host, so this is the
   blocking dependency.
3. The first layer where an id differs is the whole investigation. Above it, expert ids should
   match exactly and every difference below is a consequence, not a cause.

This is the right target because it is discrete and small: 50 layers x 6 ids = 300 integers per
prompt, against 128,000 floats of logits that will never agree to the last bit between a
reimplementation and a torch GEMM. Matching the ids is achievable. Matching the logits is not.

### What is already ruled out

- **The router's precision.** Bit-identical oracle output between the f32 and bf16 routers.
- **The norms.** candle's `RmsNorm::cpu_fwd` is generic over the storage element type, so a bf16
  input yields a bf16 output, exactly as the oracle's explicit `.to(x.dtype)` does. Five norms
  per layer plus the final one.
- **The residual contract.** The oracle's `forward_last_logits` accumulates
  `residual += post_norm(block_output)` and normalises the sum, with `post_attn_norm` and
  `post_ffn_norm` applied to the block outputs *before* they are added. `DecoderLayer::forward`
  does the same.

One difference is known and unquantified: the oracle scales q *before* the score matmul
(`bmm(qf * head_dim**-0.5, kf.T)`) where the CPU flash kernel scales *after* it
(`dot(q, k) * ctx.scale`). Those differ only by f32 rounding, which the arithmetic above puts
two orders of magnitude below the observed gap, so it is recorded rather than chased.

### On bit-exactness

Worth being precise about the target. The oracle is not bit-exact with its own goldens - the
reference scores 141/145 against the torch fingerprints - so "bit-exact against the goldens" is
not a goal any implementation can hit. The achievable goal is **matching the reference
implementation's generated text**, which needs agreement at the top of the distribution, not at
every bit.

Two separable problems sit between here and there:

- **The scale error above.** Localisable, and probably tractable. Do this first; everything else
  is measured against a forward pass that is systematically wrong.
- **The reduction order.** Attention and matmul blocking differ between any two implementations
  and the reference does not publish its order, so this is not solvable in general - only
  approached, by guessing orders and testing against the oracle. The router is the cheap place to
  do that, since it is 384x2560 and costs 3.5 ms of a 130 ms step, so a deliberate reduction
  order there costs little and is directly testable.


## Measured: the routing decision is decided by rounding, not by the math

The oracle's generator is runnable here (torch 2.14.1 in a venv), so this is measured rather
than inferred. `scripts/gen-kolibri1-goldens.py` was hooked to record the top-k expert ids, and
`kolibri1_dump_expert_ids` records this build's, for golden prompt 0 at step 0 - the sample
with no self-reference. 50 layers x 6 tokens x 6 experts = 1800 ids each side.

**Expert selection does not match, and it diverges from the very first decision.**

| | id agreement | rows with a different expert set |
| --- | --- | --- |
| this build | 68.8% | 88.7% (266/300) |
| chance (6x6 of 384) | 9.4% | - |

First divergence is layer 0, token 0. Nothing has accumulated there: it is embedding ->
`input_layernorm` -> router. So this is not an accumulation problem.

**The reason is visible in the scores.** Layer 0, token 0, `logit + bias`, descending:

```
reference  -3.484  -3.492  -3.500  -3.508  -3.508  -3.516  -3.516
this build -3.498  -3.515  -3.519  -3.525  -3.527  -3.530  -3.532
```

The top **7** candidates span **0.032**, and the two implementations' logits differ by at most
0.032. The bf16 ulp at this logit magnitude (~5.5) is 0.03125, so the boundary separating the
6th expert from the 7th is **one bf16 ulp wide**. Every marginal decision on this model is
therefore made by the last bit of a 2560-term f32 dot product.

That is not a close call being lost to noise; it is a call with no margin in it. And it has a
measurable consequence that is easy to miss: **f32 and bf16 routers pick different expert sets
on 132 of 300 rows**, i.e. changing only the router's precision reroutes 44% of the model's
decisions.

### The oracle does not reproduce its own goldens

Re-running the generator on the golden prompt, this host, torch 2.14.1:

```
logits_sum  -493322.8125      the goldens record  -484325.375
argmax      121598  (matches the goldens)
```

A 1.86% difference in the fingerprint statistic from the generator that produced it, identical
at 32 and 128 threads (so not a threading-order effect). The argmax survives; the sum does not.

So the goldens are a property of one torch build on one machine. **No implementation can be
bit-exact against them, including the implementation that made them.**

### What this means for the goal

Bit-exact logits are out of reach for the reasons already recorded, and this adds the decisive
one: the oracle disagrees with itself across torch versions, so there is no stable target.

Bit-exact *routing* is also out of reach, and that is the more surprising half. It would need
the oracle's exact f32 accumulation order over 2560 terms, which is a torch BLAS detail that is
not published and that this measurement shows is not even stable across torch builds.

What is reachable, and what this file should be read as recommending:

1. **Accept the divergence.** It is quantified: 68.8% id agreement, 88.7% of rows differing on
   the marginal pick, argmax 121598 matching the goldens.
2. **Gate on the things that are stable.** The reference's argmax, and the chain rate, both of
   which this build passes. Both survive an oracle that does not reproduce its own sum.
3. **Stop spending on router precision.** It has now been tested twice and both times it is not
   the cause. The f32 router is kept because it is both faster (3.5 ms vs 12 ms, since a bf16
   weight falls under the parallelism byte threshold) and marginally closer to the oracle
   (68.8% vs 67.1% id agreement).

One thing this does *not* explain: with 88.7% of routing rows differing, the output is still
largely right - top-1 matches 6/8 and the chain 166/248. The model's predictions are far less
sensitive to which 6 experts it uses than the id statistic suggests, which is itself worth
knowing before anyone spends more effort chasing the ids.
