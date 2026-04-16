---
title: Use speculative decoding
description: Pair a small draft model with a larger target model for higher generation throughput.
sidebar:
  order: 8
---

Speculative decoding is a trick where a small, fast model drafts a few tokens at a time and a larger target model verifies them in a single forward pass. When the draft tokens are accepted, you get several tokens for the cost of one target forward, which can be a substantial speedup. When they are rejected, you pay a modest overhead but still make progress.

mistral.rs supports speculative decoding natively. Two models load together; the engine handles the drafting and verification in the request path.

## When it helps

Speculative decoding helps most when:

- The target model is large and expensive per forward pass.
- The draft model is small and fast.
- The draft model agrees with the target model reasonably often. Models from the same family (different sizes of the same series) work well; models from different families often do not.
- You are latency-bound, not throughput-bound. The speedup comes from finishing individual generations faster, not from serving more of them.

## When it does not help

- Small target models (7B and under). The overhead of drafting and verifying is a larger fraction of total time.
- Target models where there is no obvious smaller sibling. Pairing random draft and target models usually gives low acceptance rates, negating the benefit.
- High-batch throughput workloads. The bookkeeping cost is proportional to batch size, and the acceptance rate tends to be lower at high concurrency.

## Configuration

Pass both models on the CLI:

```bash
mistralrs serve \
  -m Qwen/Qwen3-32B \
  --draft-model-id Qwen/Qwen3-0.6B \
  --speculative-gamma 4 \
  --isq 4
```

- `-m` is the target model (the one you want the output of).
- `--draft-model-id` is the smaller model that drafts.
- `--speculative-gamma` is the number of tokens to draft at a time. The default is 4.

Both models run in the same process and share the same GPU budget.

## Tuning gamma

Gamma is the knob that controls throughput vs. overhead. Higher values let the draft model produce more tokens per target forward, but with a higher chance that at least one gets rejected, which then wastes the others.

Typical values:

- Gamma 2 is conservative. Low overhead when acceptance is poor, small gains when acceptance is good.
- Gamma 4 is the default. A reasonable middle ground for most model pairings.
- Gamma 8 or higher is aggressive. Works when the draft agrees with the target very often (e.g., same architecture at 10x size ratio).

You can benchmark gamma values by repeating the same generation and watching tokens per second.

## Acceptance rate as a diagnostic

The server logs an acceptance rate line per request when speculative decoding is enabled:

```
[spec] accepted 3 / 4 draft tokens
```

A rate above 60% is a good pairing; anything near the maximum gamma is excellent. A rate below 30% usually means the draft and target are too different, and you are better off running the target alone.

## Cost model

Speculative decoding costs:

- Memory for both models, since they are both loaded.
- An extra copy of the draft model's KV cache.
- Some CPU overhead per request for the coordination between draft and target.

On a memory-constrained system, the memory cost alone can make speculative decoding impractical. On a memory-rich system with latency-sensitive workloads, it is one of the biggest levers you have.

## Alternatives

If the goal is higher throughput per GPU rather than lower latency, paged attention at higher concurrency usually wins, because it amortizes per-request overhead across more requests. Speculative decoding and paged attention compose correctly (you can have both on at once), but they target different failure modes.
