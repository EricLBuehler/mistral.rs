---
title: Sampling parameters
description: Temperature, top-k, top-p, min-p, DRY, and how they interact.
sidebar:
  order: 5
---

Sampling parameters control how the engine selects the next token from the model's probability distribution.

## Temperature, top-p, top-k

**Temperature** scales the logit distribution before sampling. Higher temperature flattens it; lower temperature sharpens it.

- `temperature: 0.0`: greedy. Always picks the most likely token.
- `temperature: 1.0`: matches the model's training distribution.

**Top-p (nucleus sampling)** keeps the smallest set of tokens whose cumulative probability exceeds `p`, then renormalizes.

- `top_p: 1.0`: disables nucleus sampling.

**Top-k** caps the candidate count at `k`.

- `top_k: 1`: equivalent to greedy.
- `top_k: 0`: disabled (default when unset).

## Min-p

Min-p scales with the most likely token's probability. The threshold is `min_p` times the top-token probability; everything below is dropped.

When the model is confident, min-p filters more tokens. When uncertain, it filters fewer.

## DRY (Don't Repeat Yourself)

DRY penalizes sequences reproducing spans from preceding text.

Parameters:

- `dry_multiplier`: penalty strength.
- `dry_base`: exponent base for penalty scaling.
- `dry_allowed_length`: match length before the penalty applies.
- `dry_sequence_breakers`: tokens that reset matching.

Off by default.

## Repetition penalty and frequency penalty

- `presence_penalty`: flat penalty on tokens that appeared at all.
- `frequency_penalty`: penalty proportional to occurrence count.

Both are OpenAI-compatible.

## Interaction order

When multiple filters are active, application order is:

1. Temperature (logit scaling)
2. Top-k (hard candidate cap)
3. Top-p or min-p
4. Repetition/frequency/presence/DRY penalties

## Setting parameters

All parameters work on the HTTP API, in SDK request types, and in interactive-mode slash commands (`/temperature`, `/topk`, `/topp`). Slash-command values persist between requests; per-request API values override them.

For deployment-wide defaults, the [CLI TOML config](/mistral.rs/reference/cli-toml-config/) has a `[sampling]` section applied to requests not specifying a parameter.

## Seeds

`seed` in the request controls randomness. Identical seeds with identical prompts produce identical output.
