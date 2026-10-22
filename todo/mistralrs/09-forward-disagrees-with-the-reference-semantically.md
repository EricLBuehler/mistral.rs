# The teacher-forced prefill disagrees with the reference semantically, not by a position shift

- **Repo:** mistral.rs
- **Component:** `mistralrs-core/src/models/kolibri1.rs`, the forward path
- **Status:** open. The first attempt at this diagnosis was wrong and is recorded below, because
  the way it went wrong is itself a warning about the evidence in this directory.
- **Impact:** every teacher-forced comparison against a reference chain

## Retraction: the "one position behind" finding was wrong

An earlier revision of this file claimed our teacher-forced prefill was one position behind the
reference, with our logits at step `k` being the reference's step `k-1` distribution, and that
the golden test's chain-argmax number was therefore measuring the harness rather than the model.

**That was wrong, and it was checked and disproved.** The test that settles it needs no
reference oracle: it generates 24 tokens through the incremental path, teacher-forces that same
chain, and scans offsets -2..+2 to report the direction and size of any misalignment.

```
[align] chain: [20733, 46, 543, 9255, 279, 5240, 735, 20733, 46, 543, ...]
offset -2: 0/21
offset -1: 0/22
offset  +0: 23/23
offset +1: 0/23
offset +2: 0/22
best offset 0: 23/23
```

Our prefill and our incremental path agree perfectly, and there is no position shift anywhere
in our own code. The mask is fine.

**How the wrong conclusion was reached**, because it is a trap worth recording:

- The primary's logprob rows are stored one position ahead of its chain. Reading them as
  unlagged makes any comparison against them look displaced.
- Several of the golden chains are periodic - "der Mond" alternates two tokens - so a
  one-step displacement is partly invisible and the pattern reads as systematic.
- The 34-of-52 "margin above 3 nats" arithmetic was correct. The conclusion drawn from it was
  not, because the quantity it was computed on was misaligned.

A shift in the *measurement* and a shift in the *model* produce the same signature here. The
only way to tell them apart is a self-consistency check, and there was not one.

## What is actually true

Our step-0 answer against llama.cpp's step-0 distribution - same context (the prompt alone),
same weights, both greedy:

| prompt | ours | llama.cpp's rank for ours | llama.cpp's top-1 | gap |
|---|---|---|---|---|
| alpha beta gamma delta epsilon zeta | 116026 | 1 (-1.254) | 121598 (-0.907) | 0.35 nats |
| der Mond scheint hell über den Bergen | 1178 | 3 (-2.448) | 948 (-1.294) | 1.15 nats |
| The capital of Australia is | 20733 | 1 (-2.318) | 75361 (-0.127) | 2.19 nats |
| Wissen ist Macht, aber Zeit ist Geld | 4771 | not in top-20 | 44 (-0.586) | > 5 nats |
| x1 = 3, x2 = 7, x3 = 11, x4 = | 576 | not in top-20 | 32 (-0.015) | > 6 nats |
| Translation to German: the weather is beautiful today | 4484 | 2 (-1.444) | 46 (-1.105) | 0.34 nats |

Two of the six have llama.cpp's top-1 ahead by more than 5 nats, and for "x1 = 3, x2 = 7, x3 =
11, x4 =" llama.cpp's top-1 is the space token at -0.015 with its runner-up at -6.359. **A
6-nat margin is not flipped by any numerical difference between two implementations.** So this
is a semantic disagreement, not a rounding one.

That is a much better-conditioned problem than the one it replaced: there is a definite answer
to find, and it is findable.

## How to localise it

The comparison to make is per layer, not per token. A per-layer residual-stream dump from
llama.cpp and from this build, for the same position, gives the first divergent layer directly
instead of by bisection.

Positions to compare: **0** and the last prompt position. Position 0 is the sharp probe -
its context is a single token, so if the divergence appears there, the fault is in the
attention half of the first block; if position 0 matches at every layer, the fault is later and
the last prompt position becomes the probe.

The quantities: the residual stream at the end of each block, and the two post-norm outputs, so
a block can be split in two.

A cross-check that is free once the reference rows are re-emitted: the primary's step-0 answer
for "The capital of Australia is" is 75361 on llama.cpp's side at p=0.88, and 20733 on ours
with a 2.19-nat gap. If the re-emitted primary rows show the primary agreeing with llama.cpp
there, the primary is a usable CPU oracle and the disagreement is ours. If the primary agrees
with us, it is not.

## Caveats

- Our step-0 answer for "Wissen ist Macht" and "x1 = 3, ..." is outside llama.cpp's top-20.
  How far outside is not yet measured; the top-100 would settle it.
- The reference is llama.cpp at commit `62318cd57` on a bf16 GGUF, and the chain is in
  `../artifacts/llamacpp-oracle/`.
