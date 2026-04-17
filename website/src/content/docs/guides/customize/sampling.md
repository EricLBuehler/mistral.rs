---
title: Sampling parameters
description: Temperature, top-k, top-p, min-p, DRY, and how they interact. Which knobs actually matter, and when.
sidebar:
  order: 5
---

Sampling parameters control how the engine selects the next token from the model's probability distribution. Most cases require only one or two.

## The core three

**Temperature** scales the logit distribution before sampling. Higher temperature flattens it (more variety, more random); lower temperature sharpens it (more conservative, more repetitive).

- `temperature: 0.0` — greedy. Always picks the most likely token. Suitable for deterministic code generation or tests.
- `temperature: 0.7` — common chat default. Varied but coherent.
- `temperature: 1.0` — matches the model's training distribution.
- Above `1.5` — output quality degrades fast.

**Top-p (nucleus sampling)** keeps the smallest set of tokens whose cumulative probability exceeds `p`, then renormalizes. Useful with non-zero temperature to filter low-probability tokens.

- `top_p: 0.9` — common default.
- `top_p: 0.95` — more permissive.
- `top_p: 1.0` — disables nucleus sampling.

**Top-k** caps the candidate count at `k` regardless of probabilities. Less principled than top-p but useful as a hard ceiling.

- `top_k: 40` — common default.
- `top_k: 1` — equivalent to greedy.
- `top_k: 0` — disabled (default when unset).

For most workloads, `temperature` plus `top_p` is sufficient. Top-k is usually redundant with top-p but useful as a hard limit.

## Min-p

Min-p scales with the most likely token's probability. The threshold is `min_p` × top-token-probability; everything below is dropped.

When the model is confident, min-p filters aggressively. When uncertain, min-p filters less. This adapts better than top-p across prompt types.

- `min_p: 0.1` — reasonable default.
- Applied alongside `temperature`; top-p is usually disabled with min-p.

If top-p is inconsistent across prompt types, try min-p.

## DRY (Don't Repeat Yourself)

DRY penalizes sequences reproducing spans from preceding text. Useful for long-form generation to prevent loops.

Parameters:

- `dry_multiplier` — penalty strength. Higher pushes harder. `0.8` is a starting point.
- `dry_base` — exponent base for penalty scaling. `1.75` is typical.
- `dry_allowed_length` — match length before the penalty kicks in. `2` triggers on two-token repeats; `5` only on longer.
- `dry_sequence_breakers` — tokens that reset matching. Commonly newlines and quotes.

Off by default. Enable when long-context output becomes repetitive.

## Repetition penalty and frequency penalty

Simpler repetition tools:

- `presence_penalty` — flat penalty on tokens that appeared at all. Encourages topic diversity.
- `frequency_penalty` — penalty proportional to occurrence count. Discourages loops.

Both are OpenAI-compatible. Values 0.5–1.0 are reasonable; higher produces erratic output.

DRY usually outperforms both, but these are simpler to reason about.

## Interaction order

When multiple filters are active, application order is:

1. Temperature (logit scaling)
2. Top-k (hard candidate cap)
3. Top-p (or min-p, not both)
4. Repetition/frequency/presence/DRY penalties

In practice the order rarely matters because filters tend to agree on which tokens to drop.

## Where to set them

All parameters work on the HTTP API, in SDK request types, and in interactive-mode slash commands (`/temperature`, `/topk`, `/topp`). Slash-command values persist between requests; per-request API values override them.

For deployment-wide defaults, the [CLI TOML config](/mistral.rs/reference/cli-toml-config/) has a `[sampling]` section applied to requests not specifying a parameter.

## What rarely matters in practice

**Decimal precision past two places.** `temperature: 0.7` and `0.71` are indistinguishable to a human observer.

**Stacking every knob.** Combining temperature, top-p, top-k, min-p, DRY, and frequency penalties produces worse output than any one alone.

**Randomness control at non-zero temperature.** Seeds are supported (`seed` in the request), but output at temperature > 0 is dominated by the prompt. Identical seeds with identical prompts produce identical output; the same seed across different prompts shows no meaningful correlation.

For 95% of cases, setting `temperature` and nothing else is correct.
