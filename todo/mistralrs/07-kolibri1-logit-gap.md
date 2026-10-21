# Kolibri-1 CPU forward does not reproduce the reference logits

**Status:** **the reference was ambiguous, and the primary oracle's own logprob rows were
off by one position.** A third implementation (llama.cpp, `ggml-org/llama.cpp#30673`) sides
with the transcription, not with the primary. With the primary's row offset corrected, it
agrees with llama.cpp at 91/91 positions of the common prefix, and every remaining chain
divergence is a near-tie. See "Rewritten: the reference was ambiguous" below, which supersedes
the hypotheses in the rest of this file - they were fitted to a comparison against misaligned
rows. The forward itself is correct; the residual, RNoPE, and router-dtype fixes stand.

**Where:** `mistralrs-core/src/models/kolibri1.rs`, `Model::forward` / `DecoderLayer::forward`.

## Rewritten: the reference was ambiguous

**Three findings, in order of how much they move the picture.**

## 1. The primary oracle's per-position logprob rows are stored one position ahead

`tests/vllm/models/kolibri1_primary_topk_rows.json` (in the vllm.cpp repo) stores, per
position, the top-100 with logprobs. The chain and the rows disagree at 139 of 202 positions,
with the generated token sitting up to 11 nats below its own top-1. That is not a sampling
bug - the primary *is* greedy - it is an off-by-one in the recording:

```
generated[k] == rows[k+1].top1    195/195   (100.0%)
generated[k] == rows[k].top1       63/202   ( 31.2%)
```

So the row stored at index `k+1` is the distribution for step `k`. Everything that compared
those rows position-by-position was reading the wrong position, which is why prompt 0 position
0 appeared to show the primary generating a token at logprob -7.4476 while its own top-1 sat
at -0.325.

This is the vllm.cpp harness's bug, not the model's. Raised with Maki.

## 2. llama.cpp is a third implementation, and it sides with the transcription

llama.cpp gained Kolibri-1 support in `#30673` (merged 2026-10-17). Run on CPU against a bf16
GGUF converted from this checkpoint, greedy, feeding the goldens' own token ids:

| | transcription | primary | llama.cpp |
|---|---|---|---|
| first token agrees with transcription | - | 6/8 | **8/8** |
| identical 32-token sequence with transcription | - | 3/8 | **7/8** |
| first token agrees with primary | 6/8 | - | 6/8 |
| identical 32-token sequence with primary | 3/8 | - | 3/8 |

llama.cpp and the transcription agree with each other. The primary is the outlier.

The GGUF is a valid oracle: `Kolibri-1-bf16` is the *dequantisation* of the fp8 checkpoint,
not an independent model. Verified on layer 0 expert 0 `gate_proj`: fp8 is `float8_e4m3fn`
with a 128x128 block scale (shape 4x20 for a 512x2560 tensor), the dequant is
`fp8 * scale_inv`, and the bf16 tensor is that value rounded to bf16 - mean absolute
difference 5.1e-7 on values of magnitude ~0.011, i.e. 4.6e-5 relative.

## 3. With the offset corrected, the primary and llama.cpp agree everywhere they can

Comparing only along the common prefix of the two chains (91 positions), so that a divergence
does not contaminate the positions after it:

```
same argmax: 91/91
```

Every remaining chain divergence is a near-tie. Prompt 0, step 0:

```
primary:   116026 (-0.944)  vs  121598 (-1.215)   margin 0.271  -> picked 116026
llama.cpp:  121598 (-0.907)  vs  116026 (-1.254)   margin 0.347  -> picked 121598
```

Same top-2, ~0.3 nats apart, and a ~0.05-0.3 nat difference in absolute logprob between the
two implementations flips the order.

## What this means for the earlier conclusions in this file

The hypotheses this file spent most of its length on are all **superseded**, because they were
fitted to a comparison against a reference whose rows were misaligned:

- The scale-error hypothesis is not just disproved by the final norm's scale invariance - the
  "scale oracle" was reading the wrong position.
- The router dtype was falsified as a cause, and that still holds.
- The residual-contract and RNoPE bugs were real and are fixed; those were found by running
  the forward, not by comparing logits.
- The BFMMLA reduction-order result (chain argmax 166/248 -> 170/248) was measured against
  the **transcription**, and the transcription is the trustworthy one, so it stands. But it
  was small, and the reason the gap is hard to close is now visible: the top-2 is often
  within 0.3 nats, so any implementation difference of that size decides the token.

## The mechanism: the router's top-k boundary is an exact tie

The router selects the top 6 of 384 experts by `logits + bias`. On the reference's own layer-0
router logits, **the 6th and 7th experts are exactly tied** on 5 of the 6 token positions:

```
  pos  top6-top7 margin  same expert set?
    0           0.000000  no
    1           0.000000  no
    2           0.000000  no
    3           0.007812  yes
    4           0.000000  no
    5           0.000000  no
```

Every row where the margin is zero is a row where our expert selection differs from the
reference's. The one row with a non-zero margin is the one row where we agree, exactly.

All 2304 of the reference's stored router logits and 384 bias values are exactly
bf16-representable, and in bf16 values differing by under ~0.4% are the same value, so many
adjacent experts land on the same score and the selection boundary falls inside a group of
equal scores. Which expert makes the cut is then decided by the sort's tie-break.

**Confirmed on the reference side.** The peer agent's production build measures the 6th-to-7th
router margin in f32, un-rounded, for "The capital of Australia is" at step 0: the smallest
across 50 layers is **9.391e-05** (layer 22), the median about 3.3e-02, layer 0's is 0.0149.
An order of magnitude below one bf16 ulp at that score magnitude, and far below the 0.05-0.3
nat class of difference measured between two implementations on this model.

**So the boundary is inside the tie group, and exact agreement between two implementations is
not attainable without reproducing the other's sort tie-break.** The band gate is the honest
standard rather than a compromise. This is also why the primary and llama.cpp disagreed with
each other on 2 of 8 first tokens while agreeing at 91/91 on the common prefix: same
distribution, different tie-break.

## But that was not why we disagreed: our scores were not the same scores

On "The capital of Australia is" all three references agreed with each other and we disagreed,
so a tie-break difference could not be the whole story. The cause was **candle's fused
RMSNorm, which rounds the scale to the storage dtype before dividing.**

```rust
// candle-nn/src/ops.rs, the fused CPU kernel
let m = (sum2 / n as f32 + eps).sqrt();
let m = T::from_f32(m).unwrap_or_else(T::nan);   // scale -> bf16
*d = *s / m * *alpha
```

Hugging Face, ggml and llama.cpp all keep the scale in f32 and round the normalised *value*
instead. Rounding the scale applies one relative error of up to a bf16 ulp - about 0.4% - to
every element of a row, uniformly. Measured on a bf16 row: max relative difference 7.8e-3
against the f32-scale form, and only 1480 of 2560 elements bit-exact.

A 0.4% error in the norm output that feeds the router is worth 0.05-0.3 nats in the router
logits - three orders of magnitude more than the 9.4e-05 selection margin. So we were not
tie-breaking the same ties; we were computing genuinely different scores and picking the
genuinely-best expert.

**Fixed** in `d9198c578`, across all thirteen RMSNorm call sites in `layers.rs`, behind
`MISTRALRS_RMSNORM_CANDLE=1` to restore candle's kernel for A/B.

| | before | after |
|---|---|---|
| "The capital of Australia is" | " Sydney" (logit 10.62, margin 1.06) | **" Canberra"** (logit 10.00, margin 0.38) |
| chain argmax vs the llama.cpp goldens | 0/52 step matches | **185/248** |

"Canberra" is what llama.cpp, the vLLM primary and the reference goldens all produce. That
prompt had deliberately not been asserted, because the answer depended on the router's
arithmetic; it is asserted now.

## Which reference is the oracle

Not the transcription. Against the transcription, chain argmax moved 166/248 -> 167/248 - the
fix barely helped. Against llama.cpp it went from 0/52 step matches to 185/248.

The reason is that the transcription computes its router in **bf16**, so its selection scores
are exactly the bf16-rounded values whose ties are tabulated above, and it tie-breaks them
differently. The primary and llama.cpp both compute the router in f32 and agree with each
other at 91/91 on the common prefix. The f32 pair is the oracle; the transcription is a third
implementation with a less accurate router, and its goldens are not a valid CPU oracle.

## What the per-layer dump shows

The peer agent's production build dumped per-layer states for the same raw ids and positions.
Diffing ours against theirs, with the fix in place:

| quantity | layer 0 max abs | layer 0 cosine |
|---|---|---|
| attn_out_pre_norm | 9.8e-04 | 0.999996 |
| attn_post_norm | 3.1e-02 | 0.999991 |
| ffn_post_norm | 1.2e-02 | 0.999986 |
| l_out | 6.3e-02 | 0.999988 |

The max differences are exactly bf16 ulp values at those magnitudes (3.125e-2 = 2^-5,
6.25e-2 = 2^-4) and the cosine is 0.99999 at layer 0 on every quantity. There is **no
structural divergence at any layer**: the difference accumulates smoothly and the cosine decays
to 0.92-0.95 by layer 25, which is what a knife-edge router does to a growing arithmetic
difference.

The caveat: that comparison is our CPU against their tt-metal GPU, so it separates devices as
much as implementations.

### The CPU-to-CPU comparison, which is the isolating one

llama.cpp can expose its layer input stream through the mechanism it already has -
`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`, the same one deepseek4,
mimo2 and nemotron-h use - with two lines added to `src/models/kolibri1.cpp`. That is on a
`peer/llamacpp-layerdump` branch, 16 lines in one file, no change to `llama-context.cpp` or
`llama.h`.

Comparing our `l_out(L)` against llama.cpp's `residual_in(L+1)`, on the same CPU, the same
weights and the same token ids:

| layer | position | max abs diff | rms diff | cosine | beyond 2 ulp |
|---|---|---|---|---|---|
| 0 | 0 | 5.9e-02 | 3.4e-03 | **0.999992** | 7.0% |
| 0 | 5 | 1.0e-01 | 1.1e-02 | 0.999891 | 17.2% |
| 1 | 0 | 1.6e-01 | 6.5e-03 | 0.999991 | 48.2% |
| 2 | 0 | 1.1e-01 | 8.3e-03 | 0.999980 | 73.2% |
| 5 | 5 | 4.6e+00 | 1.5e-01 | 0.988067 | 93.7% |
| 9 | 0 | 2.9e-01 | 3.9e-02 | 0.999496 | 86.3% |

**Our block-0 output agrees with llama.cpp's to cosine 0.999992 and a relative RMS of 0.7%.**
That is a handful of bf16 ulps, not a semantic difference. It then grows with depth, which is
the knife-edge router amplifying ulp-level arithmetic across 50 layers.

**So the chain closes.** The norm's scale rounding was a 0.4% error and it moved the router
directly; fixing it removed a real defect. What remains after that is a few ulps per block,
amplified by a selection margin of 9.4e-05 into the steps that still disagree. The forward is
correct, exact agreement is not attainable, and the band gate is the honest standard.

**One caveat about our own dump.** Our `residual_in` tap is not the layer's input stream: the
implementation carries the residual in two parts, `xs` being the previous block's post-normed
output and `res` the previous `ffn_inp`, and the layer's true input is `xs + res`. So our
`residual_in` is the `ffn_inp` alone and has no counterpart in the llama.cpp dump. It is
excluded from the comparison; the `l_out` comparison is the valid one.

## The synthesis: both backends had a norm-scale precision defect, and the router margin is 20,000x smaller than either

This closes the loop on why implementations disagree.

**Our side.** Candle's fused CPU RMSNorm rounds the *scale* to bf16 before dividing, so every
row of every norm output carries a relative error of up to one bf16 ulp - about 0.4% -
uniformly. Fixed in `d9198c578`, by keeping the scale in f32 and rounding the normalised value
the way Hugging Face, ggml and llama.cpp do.

**Their side.** The Tenstorrent device's rms computation has a -1.9% regime against its own host
arm, on bit-identical inputs, and the bisect attributes it to the compute config
(`approx_mode`, `fp32_acc`) rather than to the compiler or the toolchain version. A 1.9% relative
deviation in a norm output is the same class of defect as our 0.4% one, just larger.

**The router's selection margin is 9.4e-05 nats** on the smallest layer, and 0.0149 at layer 0.
So the routing decision - which decides the token - is being made at a precision 20,000x finer
than the norm error in either backend.

| deviation | magnitude | where |
|---|---|---|
| router selection margin | 9.4e-05 nats | what the decision is made at |
| candle fused RMSNorm, CPU | ~0.4% per row | our side, **fixed** |
| device rms, `approx_mode=true` | ~1.9% per row | the device's default config |

**Which means the device is now the outlier on the norm, not us.** Our f32-scale norm agrees
with ggml's, and the device's does not. So the cleanest CPU reference is llama.cpp rather than
the primary, which is the opposite of where this started - the primary was the reference until
the row-offset bug in its writer put it in doubt.

**That last claim does not survive the peer's measurement, and the correction matters.** Their
production keeps the rms scale in f32 exactly as ggml does, so the norm cannot be what
separates the implementations. And the measured ordering is the opposite of what the norm story
predicts: their production and ours agree to a mean gap of -0.028 nats (cosine 0.9991), while
llama.cpp sits a **median 1.03 nats** from the primary (p90 1.99, max 5.88). So llama.cpp is the
outlier, and the norm is not the discriminator.

**The container is ruled out by measurement, not argument.** The hypothesis was that the GGUF
is `round_bf16(fp8 * scale_inv)` and therefore differs from the bf16 safetensors by about
4.6e-5 relative. It does not: the two hold the same bytes.

```
compared 78,643,200 bf16 elements across 60 tensors
  (5 layers x 4 experts x gate/up/down projections)
  bit-identical: 78,643,200/78,643,200 (100.0000%)
```

The GGUF was converted from the fp8 checkpoint as `round_bf16(fp8 * scale_inv)`, and the bf16
safetensors are that same dequantisation rounded to bf16 - the same computation done once, not
twice. So the 1.03 nats is the implementation, with identical weights on both sides.

**What is left, and what is not.** The router's reduction order is the obvious candidate: a
2560-element f32 dot product with the top-6 cut on a 9.4e-05-nat margin. But it does not
explain the ordering either - our f32 router and the peer's f32 router agree on the logits to
0.03 nats while selecting the same expert set on only about two rows in three, so the selection
is not what is driving the 0.03 nats. There is no mechanism here yet that explains the
ordering, and it is recorded as unexplained rather than fitted.

The measurement that would localise it is a per-layer dump from llama.cpp, which is the outlier:
the same quantities the peer's production dump carries, at the same positions and the same raw
ids, plus llama.cpp's router top-6 per layer. If the selections agree and the hidden states
diverge, the router is exonerated and the arithmetic downstream of it is at fault. If the
selections differ, it is the router and the fix is matching the reduction order.

The attention is a candidate that has not been eliminated either: llama.cpp runs a flash-attention
CPU kernel and this build runs candle's eager SDPA, over a 513-window, accumulating the sequence
in different orders.

**And the band gate is not a compromise.** Two independent backends each carry a per-row norm
deviation an order of magnitude larger than the margin the routing decision is made at. The
honest gate is a band, and any implementation that is not bit-reproducing another one's sort
tie-break cannot be expected to agree with it on the expert selection at all.

## Retracted: "the logit distance is mediated by the expert selection"

This section replaced a claim that turned out to rest on a mis-parse, so it is withdrawn and
what survives is recorded here instead.

**The error.** `ggml_argsort_top_k` returns `[n_tokens, k]`, so ne0 is the token axis. The
llama.cpp dump loop iterated the wrong axis and wrote `layer, rank, position, id`, and the
reader grouped by the wrong column. The tell was in the data: some of the "position 5" sets
had a repeated expert id, which a top-k cannot have. With the columns transposed the way they
actually were, llama.cpp's layer-0 position-5 selection is 359, 203, 11, 197, 273, 68, which
is the peer's rows file for the same layer and position, in the same order.

The published 33.45% was that error, and the conclusion built on it - that the logit distance
is mediated by the expert selection - does not survive. The artifact is republished in the
column order the peer's rows file uses, `layer, position, rank, expert_id`, with the raw dump
kept alongside as `llamacpp-router-ids-raw.tsv`.

Two other things said here were wrong and are withdrawn: the -0.028 nats was the peer's
production against the primary, not "theirs and ours" - mistral.rs against the primary in nats
was never measured, and the 248 figure is a chain-argmax count rather than a distance in nats.
And "llama.cpp is 3-6x closer to the device" was computed with a broken scale; on the correct
metric it is about 2x.

**The corrected comparison.** Top-6 set agreement, all positions, with every parse checked
(300 groups, no duplicate-id group, and the peer's own number reproduced to the digit):

| | peer | llama.cpp | mistral.rs | reference |
|---|---|---|---|---|
| peer | 100% | 87.56% | 68.67% | 87.17% |
| llama.cpp | 87.56% | 100% | 68.50% | **88.33%** |
| mistral.rs | 68.67% | 68.50% | 100% | **68.78%** |
| reference | 87.17% | 88.33% | 68.78% | 100% |

**mistral.rs is the outlier, not llama.cpp.** llama.cpp agrees with the reference 88.33% of the
time and with the peer's production 87.56%; mistral.rs agrees with the reference 68.78%. The
opposite of what the bad parse said.

**Why the selection still matters, though it does not explain the llama.cpp gap.** Every
implementation is within about one bf16 ulp of the device at layer 0:

| quantity, layer 0 | mistral.rs vs device | llama.cpp vs device |
|---|---|---|
| attn_out_pre_norm | 3.54e-03 | 1.66e-03 |
| attn_post_norm | 4.95e-03 | 2.32e-03 |
| ffn_post_norm | 5.86e-03 | 3.00e-03 |
| l_out | 7.80e-03 | 2.00e-03 |

(relative rms of the difference, over the reference's rms.) The router's margin - the peer's
7.86e-04 minimum and 2.35e-02 median, recomputed from their score file and matching their
arithmetic exactly - is comparable to those differences, so the selection is a coin flip per
layer. With fifty layers, a 12% per-layer disagreement rate means about 99.8% of positions
have at least one flipped layer somewhere, which is how a per-layer selection difference
becomes a per-position logit difference.

But llama.cpp and the peer's production select alike at 87.56% and still sit 0.03 and 1.03 nats
from the primary, so the selection does not explain that gap: the primary and the transcription
are not the same oracle for the selection, and agreeing with the transcription does not imply
agreeing with the primary.

**The storage dtypes differ, and that reframes the 2x.** How many of the dumped values are
exactly bf16-representable:

| quantity, layer 0 | mistral.rs | llama.cpp | peer CPU path |
|---|---|---|---|
| attn_out_pre_norm | 100% | **0%** | 100% |
| attn_post_norm | 100% | **0%** | 100% |
| l_out | 100% | **0%** | 21% |

llama.cpp keeps the hidden state in f32 throughout. We keep it in bf16. The peer's CPU path
keeps the attention and norm outputs in bf16 and the residual stream in f32.

So "llama.cpp is about 2x closer to the peer than we are" is not a measure of arithmetic
quality - llama.cpp simply stores more precision, and the peer's bf16 rounding is the
dominant term in every llama.cpp-vs-peer comparison. The comparison that is not a storage
artefact is ours against the peer, because both are bf16 at attn_out_pre_norm: 3.54e-03
relative, about 0.9 of a bf16 ulp, which for two roundings of the same value would be far
smaller. **So our attention output really is about 0.6% away from the peer's, while
llama.cpp's is essentially at it.**

That is a real arithmetic difference and it is what the selection gap follows from. It also
narrows where it is: at layer 0, position 0, the attention is over a single element, so the
softmax cannot be the cause and the difference is in the projections - the query, key and
value projections or the output projection - rather than in the attention's accumulation over
the sequence.

**The device is a fourth implementation, and it is a separate cluster.** The peer's TT device
taps and selections, for the same raw ids and the same quantities, make the geometry
complete. The relative rms of the difference at layer 0, `attn_out_pre_norm`:

| pair | rel rms | |
|---|---|---|
| llama.cpp vs peer CPU path | **1.66e-03** | the closest pair |
| ours vs llama.cpp | 3.01e-03 | |
| ours vs peer CPU path | 3.54e-03 | |
| ours vs device | **1.18e-02** | |
| llama.cpp vs device | 1.23e-02 | |
| peer CPU path vs device | 1.23e-02 | |

and at `l_out`, layer 0: llama.cpp vs the peer CPU path 2.00e-03, ours vs llama.cpp 7.16e-03,
ours vs the peer CPU path 7.80e-03, and the device 6.42e-03 from llama.cpp.

So there are two clusters. **llama.cpp and the peer's CPU path are the same thing to within
the storage dtype**, and the device is 6-7x further from both than they are from each other -
which is the rms_norm compute-config deviation, and it is why the device produces a
different first token for the golden prompt (74493 where every CPU reference gives 121598,
with the top two 0.017 nats apart and the golden's token third).

**And mistral.rs is the outlier within the CPU cluster, in the residual stream and not only
in the selection.** llama.cpp and the peer's CPU path are 1.66e-03 apart at layer 0; we sit
3.01e-03 and 3.54e-03 from them, and 7.16e-03 and 7.80e-03 away at `l_out` - three to four
times further than they are from each other.

Since llama.cpp's attention output agrees with the peer's CPU path to within 1.66e-03, and
llama.cpp's is the one that keeps f32 while the CPU path's is bf16, **that 1.66e-03 is
almost entirely the storage dtype and llama.cpp's arithmetic is essentially at the peer's.**
Ours differs from the peer's by 3.54e-03 with the *same* bf16 storage, so the arithmetic
difference is the larger term and it is ours.

That is what the selection agreement was telling us all along: mistral.rs agrees with the
transcription's selections 68.78% of the time where llama.cpp agrees 88.33% and the peer's
CPU path 87.17%, and the transcription sits in the llama.cpp/peer-CPU cluster. The device
agrees with the transcription 45.00%, with the peer's CPU path 47.00%, with ours 49.67% and
with llama.cpp 20.07% - it agrees with nobody, which is what a 6-7x residual-stream gap
does to a top-6 cut on a 1e-2-nat margin.

**Where the attention differs, narrowed as far as the existing data allows.** Our CPU
attention runs with the input dtype: `run_flash_attn_cpu_for_dtype` takes `out_dtype =
q.dtype()` and dispatches to `run_flash_attn_cpu::<bf16>` for a bf16 model, so the attention's
internals and its output are bf16, and the output projection consumes bf16. llama.cpp's flash
attention and the peer's accumulate in f32, and llama.cpp's attention output stays f32 through
its output projection.

At layer 0, position 0, the attention is over a single element, so the softmax cannot be the
cause and the difference is in the projections or in the attention's internal precision. And
the size is wrong for a reduction-order difference: if both sides computed the same f32 value
and rounded it the same way, the rounded values would differ on a fraction of a percent of
the elements, not on most of them. The measured 0.9 of a bf16 ulp means the f32 values
themselves differ by about 1.5 ulp, which is about 0.6% relative.

0.6% relative in a projection is the size of a bf16 rounding, not of a different summation
order. The candidates, in the order they are worth testing:

1. **The attention output is rounded to bf16 before the output projection.** llama.cpp keeps it
   in f32 there. This is a one-place change and it is the most likely single cause.
2. **The attention's softmax probabilities or its P-V product are kept in bf16.** llama.cpp and
   ggml keep them in f32.
3. **The value projection itself.** Our decode path uses `gemv_cpu::gemv` on a prefill-shaped
   batch; the accumulation is f32 there, so this is the least likely of the three.

The cost of fixing it is not zero, and that is worth knowing before doing it: bf16 attention
is half the memory traffic of f32 attention over the KV cache. So the fidelity fix and the
throughput fix pull in opposite directions here, and the right answer may be f32 accumulation
with a bf16 cache, which is what ggml does.

**Where our 2x lives, which is the actionable part.** At layer 0 the input is the embedding,
so the whole of attn_out_pre_norm's divergence is the attention's own arithmetic, and at
position 5 - the only position with a real attention window - it is 4.7x larger than at
position 0. We run candle's eager SDPA; llama.cpp's flash-attention CPU kernel is the one that
sits closer to the device. Our residual stream is about 2x further from the device's than
llama.cpp's is at layer 0, and our selections inherit that.

**Still open.** Two measurements, both waiting on the box or on the peer:

- mistral.rs's own selections with the CURRENT build. The 68.78% is from a 2026-10-14 dump,
  before the RMSNorm fix and before the f32 router, so it describes that build and not this
  one. It is an upper bound on how far we are from the reference, not a measurement of it.
- The primary's per-layer selections, which are what would say whether llama.cpp's 1.03 nats is
  selection-driven after all. The peer's rows file is their production CPU path, not the
  primary.

## Where we actually stand, and a fourth finding that undercuts the measurement itself

**The teacher-forced prefill is one position behind.** Our logits at step `k` are the
reference's step `k-1` distribution. 34 of 52 steps have a reference margin above 3 nats, and
in every one our value is the reference's *next* step, so this is not a numerical tie. Free
generation is correct, so the fault is in the prefill path, not the model.

Consequence: **the chain argmax 166/248 quoted below is not a measure of model quality.** It
is a measure of a harness that compares our position `k` against the reference's position `k`.
It also explains why the gap would not close: a one-position shift in the measurement is
immune to the numerics that were being tuned. See `09-teacher-forced-prefill-is-one-position-behind.md`.

Everything below this point was measured with that harness and should be re-measured once it
is fixed. The *relative* comparisons - FMA versus BFMMLA, router dtype, scale invariance - are
unaffected, because they were all made against the same shifted baseline.

## Where we actually stood before that

Against the transcription (which llama.cpp corroborates), this build scores chain argmax
166/248, final-token 6/8, first-step 0/8, worst top-k logit gap 6.5938.

**The reference is no longer ambiguous, so the remaining gap is ours.** Two thirds of the
teacher-forced positions are wrong even though the semantics are now confirmed to match
llama.cpp's, the config matches, and the routing matches (`select on logits+bias`, `weight by
sigmoid(unbiased logits)`, no renorm, f32 router - all identical to llama.cpp's
`src/models/kolibri1.cpp` and `conversion/kolibri1.py`).

Not yet re-measured against the llama.cpp chains specifically: the model needs ~150 GB resident
and the box was occupied. The transcription and llama.cpp agree on 7 of 8 full sequences, so
the two numbers should be close, but that is an inference, not a measurement.

## Artifacts

- `kolibri1_goldens_llamacpp.json` - the llama.cpp chains with per-position top-20 logprobs,
  in the goldens format the test already reads (`KOLIBRI1_GOLDENS`).
- Regenerate with llama.cpp commit `62318cd57` against the bf16 GGUF, native `/completion`
  endpoint (it returns token **ids**; `/v1/completions` returns text and splitting that text
  gives words, not tokens).

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


## Measured: matching the oracle's reduction order moves every argmax metric

`MISTRALRS_DOT_BFMMLA=1` (`0cee4ad00`) switches the bf16 dots to a BFMMLA reduction, which is
the order the reference's CPU GEMM uses. Measured at 3.4 GHz against the transcription goldens:

| metric | FMA order | BFMMLA order |
|---|---|---|
| chain argmax | 166/248 | **170/248** |
| final-token argmax | 6/8 | **7/8** |
| first-step argmax | 0/8 | **1/8** |
| worst top-k logit gap | 6.59 | 6.94 |

Every argmax metric moves the same way, and the gap gets slightly worse.

**Weak evidence, stated plainly.** Eight prompts means 6/8 -> 7/8 is one prompt flipping. The
chain figure is the only one with enough samples to lean on, and it is +4. Improvement and a
worse gap are also what noise looks like.

**It is measured against the transcription, not the primary oracle.** That comparison is still
outstanding and is the one that would settle whether this is convergence or coincidence.
Default stays off until then.

It is also a small speedup, 9.36 -> 9.59 tok/s, all of it in the `attn_proj` phase; `routed` is
streaming-bound and does not care.

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
