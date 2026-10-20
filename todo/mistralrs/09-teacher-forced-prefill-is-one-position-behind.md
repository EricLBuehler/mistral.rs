# The teacher-forced prefill is one position behind, so the golden test has been measuring the harness

- **Repo:** mistral.rs
- **Component:** `mistralrs-core/src/models/kolibri1.rs`, the teacher-forced prefill path
- **Status:** open. Cause not yet identified. **This invalidates the golden test's chain-argmax
  number, and with it the logit-gap investigation that was built on it.**
- **Impact:** every teacher-forced comparison this repo makes against a reference chain

## What

When the golden test teacher-forces a reference chain, our logits at step `k` are the
reference's step `k-1` distribution. The model is not one token behind - **free generation is
correct** - so the fault is in the prefill path specifically.

Measured against the llama.cpp chains (bf16 GGUF, commit `62318cd57`), 52 teacher-forced steps
over 2 prompts, before the run was OOM-killed on a shared box:

| prompt | step | ours | llama.cpp at k | llama.cpp at k+1 | reference margin at k |
|---|---|---|---|---|---|
| der Mond... | 2 | 46252 | 948 | **46252** | 6.31 nats |
| der Mond... | 3 | 948 | 46252 | **948** | 6.29 nats |
| der Mond... | 4 | 46252 | 948 | **46252** | 6.26 nats |
| der Mond... | 6 | 46252 | 948 | **46252** | 6.31 nats |
| der Mond... | 7 | 948 | 46252 | **948** | 6.29 nats |
| der Mond... | 10 | 46252 | 948 | **46252** | 6.30 nats |

**34 of the 52 steps have a reference margin above 3 nats.** A 6-nat margin cannot be flipped
by any numerical difference between two implementations, so this is not a tie and not a
rounding question. In every one of those steps our value is exactly the reference's *next*
step.

## Why this matters more than the number it replaces

The golden test's headline number - chain argmax 166/248 - is measured by this comparison. If
our logits are systematically one position behind, that number is not a measure of model
quality at all.

It also explains a long-standing puzzle. The chain argmax barely moved across a week of work
(166/248 with the FMA reduction order, 170/248 with BFMMLA) and the logit gap would not close
however the numerics were tuned. A one-position shift in the measurement is exactly the kind of
thing that is immune to both.

## Where the fault most likely is

Not yet identified. The prefill path and the incremental path differ in:

- **the causal mask.** A prefill builds an explicit `[1, 1, q, k]` mask; incremental decoding
  with one token does not need one. If the prefill mask made query `p` attend to keys `0..p-1`
  rather than `0..p`, then logits `p` would be the distribution given a context missing its own
  token, and step `k` would reproduce the reference's step `k-1` - which is exactly the
  observed pattern. The mask comes from candle's shared `CausalMasker::make_causal_mask`, so
  the bug would be in how it is called or configured here, not in candle.
- **position ids.** `teacher_forced_logits` passes `position_ids = 0..seq_len`. A uniform
  offset would not explain this, because RoPE is relative and the sliding window is indexed
  separately, but a per-layer mismatch could.
- **the residual threading.** The pre-norm residual is carried across layers explicitly
  (`Model::forward` moves it to the right device between layers). An off-by-one there would
  show up in free generation too, so it is less likely.

## The experiment that would settle it

Cheapest: in `teacher_forced_logits`, after the prefill, check that `argmax(logits[p])` is the
reference token at `p + 1` rather than at `p`. Even cheaper, and it needs no reference at all:
feed a sequence whose next token is known, and check that `logits[p]` peaks at `ids[p+1]`.

That single check distinguishes the mask hypothesis from the position-ids hypothesis from the
residual hypothesis, and it does not need the 150 GB checkpoint if it is run on a small model
or a synthetic one.

Then, separately, verify that the incremental path and the prefill path agree: generate one
token incrementally from a prefix, and check that the prefill's last row gives the same
argmax. If they disagree, the fault is in whichever path is wrong, and the mask is the first
thing to look at.

## Caveats

- The measurement was cut short by an OOM kill on a shared box, so this covers 52 steps over
  2 of the 8 prompts. The pattern is already unambiguous, but the full set should be re-run.
- The shift is established for the teacher-forced path. Free generation is correct, which is
  why the model produces sensible chains and why this was not obvious.
- The llama.cpp chains were produced on this box by peer-Ce; see
  `../artifacts/llamacpp-oracle/`.
