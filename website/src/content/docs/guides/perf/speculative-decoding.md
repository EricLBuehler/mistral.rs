---
title: Use speculative decoding
description: Pair a small draft model with a larger target model for higher generation throughput.
sidebar:
  order: 8
---

Speculative decoding pairs a small fast model that drafts tokens with a large target model that verifies them in a single forward pass. Accepted draft tokens give multiple tokens per target forward; rejected ones cost a modest overhead but still produce one token.

mistral.rs supports speculative decoding natively. Two models load together and the engine handles drafting and verification in the request path.

## When it helps

- The target model is large and expensive per forward pass.
- The draft model is small and fast.
- Draft and target agree often. Same family at different sizes works well; different families usually do not.
- Latency-bound rather than throughput-bound. The win is faster individual generations, not higher concurrency.

## When it does not help

- Small target models (7B and under). Drafting and verification overhead dominates.
- No obvious smaller sibling for the target. Random pairings give low acceptance rates.
- High-batch throughput. Bookkeeping cost scales with batch size and acceptance falls at high concurrency.

## Configuration

Pass both models on the CLI:

```bash
mistralrs serve \
  -m Qwen/Qwen3-32B \
  --draft-model-id Qwen/Qwen3-0.6B \
  --speculative-gamma 4 \
  --isq 4
```

- `-m` — target model.
- `--draft-model-id` — draft model.
- `--speculative-gamma` — tokens drafted per round. Default 4.

Both models run in the same process and share GPU memory.

## Tuning gamma

Gamma trades throughput against overhead. Higher values produce more tokens per target forward but increase the chance of mid-sequence rejection, which wastes the rest.

Typical values:

- Gamma 2 — conservative. Low overhead with poor acceptance, small gains with good acceptance.
- Gamma 4 — default. Reasonable for most pairings.
- Gamma 8+ — aggressive. Works when draft and target agree very often (same architecture at 10× size ratio).

Benchmark by repeating the same generation and measuring tokens/sec.

## Acceptance rate as a diagnostic

The server logs an acceptance rate per request:

```
[spec] accepted 3 / 4 draft tokens
```

>60% indicates a good pairing. Near-maximum gamma is excellent. <30% means draft and target are too different — running the target alone is faster.

## Cost model

Speculative decoding costs:

- Memory for both models.
- Extra KV cache for the draft model.
- CPU overhead per request for draft/target coordination.

On memory-constrained systems the memory cost can be prohibitive. On memory-rich systems with latency-sensitive workloads, it is one of the largest available levers.

## Alternatives

For per-GPU throughput rather than per-request latency, paged attention at higher concurrency usually wins by amortizing per-request overhead. Speculative decoding and paged attention compose, but target different failure modes.
