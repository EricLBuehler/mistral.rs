---
title: Use speculative decoding
description: Pair a draft model with a target model.
sidebar:
  order: 8
---

Speculative decoding pairs a small fast model that drafts tokens with a large target model that verifies them in a single forward pass.

## Availability

Speculative decoding is exposed through the SDKs only. It is not configurable via the CLI or the TOML config.

## Python

`Runner` accepts `which_draft` (a `Which`) and `speculative_gamma` (default 32):

```python
from mistralrs import Runner, Which

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-32B"),
    which_draft=Which.Plain(model_id="Qwen/Qwen3-0.6B"),
    speculative_gamma=4,
    in_situ_quant="4",
)
```

## Rust

`TextSpeculativeBuilder` is the entry point.

## Notes

Throughput gain depends on acceptance rate, the fraction of draft tokens the target accepts. Same-family draft/target pairings tend to have higher acceptance rates than cross-family pairings.

Both models load together and share GPU memory.
