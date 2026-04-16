---
title: Sampling parameters
description: Temperature, top-k, top-p, min-p, DRY, and how they interact. Which knobs actually matter, and when.
sidebar:
  order: 5
---

Sampling parameters control how the engine picks the next token from the model's probability distribution. A lot of them exist; most of the time only one or two matter. This guide is a practical rundown of which knobs are worth reaching for and when.

## The core three

**Temperature** scales the logit distribution before sampling. Higher temperature flattens it (more variety, more random); lower temperature sharpens it (more conservative, more repetitive).

- `temperature: 0.0` is greedy sampling. The model always picks the single most likely token. Good for deterministic code generation or tests.
- `temperature: 0.7` is a common default for chat. Produces varied but coherent output.
- `temperature: 1.0` matches the model's training distribution.
- Above `1.5`, output quality degrades fast.

**Top-p (nucleus sampling)** keeps only the smallest set of tokens whose cumulative probability exceeds `p`, then renormalizes. Useful with any non-zero temperature to cut off low-probability garbage tokens.

- `top_p: 0.9` is a common default.
- `top_p: 0.95` is a little more permissive.
- `top_p: 1.0` disables nucleus sampling (everything is fair game).

**Top-k** caps the number of candidate tokens at `k`, regardless of their probabilities. Less principled than top-p but can be useful as a hard ceiling.

- `top_k: 40` is a common default.
- `top_k: 1` is equivalent to greedy sampling.
- `top_k: 0` (or leaving it unset) disables the cap.

For most workloads, setting `temperature` and `top_p` is enough. Top-k is usually redundant with top-p but occasionally useful when you want a hard limit.

## Min-p

Min-p is an alternative to top-p that scales with the most likely token's probability. Threshold is `min_p` times the probability of the top token; everything below that threshold is dropped.

The intuition: when the model is confident (one token has very high probability), min-p filters aggressively. When the model is uncertain (probabilities are flat), min-p filters less. This adapts better than top-p across a variety of prompts.

- `min_p: 0.1` is a reasonable default.
- Applied alongside `temperature`; typically top-p is disabled when using min-p.

If you find top-p inconsistent across prompt types, min-p is worth trying.

## DRY (Don't Repeat Yourself)

DRY reduces repetition by penalizing sequences that would reproduce spans found in the preceding text. Useful for long-form generation where you do not want the model to loop.

The DRY parameters are:

- `dry_multiplier`: penalty strength. Higher values push harder against repetition. `0.8` is a good starting point.
- `dry_base`: exponent base for the penalty scaling. `1.75` is the usual value.
- `dry_allowed_length`: how many tokens can match before the penalty kicks in. `2` means two-token repeats trigger it; `5` only penalizes longer repetitions.
- `dry_sequence_breakers`: tokens that reset the matching. Commonly newlines and quotes.

DRY is off by default. Turn it on when you see the model producing repetitive output on long contexts.

## Repetition penalty and frequency penalty

Two simpler tools for discouraging repetition:

- `presence_penalty`: a flat penalty on tokens that have appeared at all. Encourages topic diversity.
- `frequency_penalty`: a penalty proportional to how often a token has appeared. Discourages loops.

Both are OpenAI-compatible and straightforward. Values between 0.5 and 1.0 are reasonable; higher values produce more erratic output.

For most workloads, DRY works better than these two. But they are simpler to reason about and occasionally preferred for that reason.

## Interaction order

When several filters are active, they apply in this order:

1. Temperature (scales logits)
2. Top-k (hard cap on candidate count)
3. Top-p (or min-p, not usually both)
4. Repetition/frequency/presence/DRY penalties

This means that, for example, top-k removes low-probability candidates before top-p considers cumulative probability. In practice the order rarely matters because the filters tend to agree on which tokens to drop.

## Where to set them

All these parameters work on the HTTP API, in the SDK request types, and in the interactive-mode slash commands (`/temperature`, `/topk`, `/topp`). The values carry between requests when set through slash commands; each API request can override them.

For deployment-wide defaults, the [CLI TOML config](/mistral.rs/reference/cli-toml-config/) has a `[sampling]` section that applies to requests that do not specify the parameter themselves.

## What rarely matters in practice

**Exact values past the second decimal place.** `temperature: 0.7` and `temperature: 0.71` behave identically to a human observer.

**Combining every knob.** Stacking temperature, top-p, top-k, min-p, DRY, and frequency penalties all at once produces output that is worse than any one of them on its own.

**Randomness control at non-zero temperature.** Seeds are supported (`seed` in the request), but model output at temperature > 0 is still dominated by the prompt. Identical seeds with identical prompts give identical output; different prompts with the same seed do not produce any meaningful correlation.

For 95% of use cases, setting `temperature` and nothing else is the right move.
