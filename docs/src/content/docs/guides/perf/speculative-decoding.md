---
title: Use speculative decoding
description: Pair a draft model with a target model.
sidebar:
  order: 8
---

Speculative decoding pairs a small fast model that drafts tokens with a large target model that verifies them in a single forward pass.

## Availability

Classic target/draft speculative decoding is exposed through the SDKs. Gemma 4 MTP is exposed through the CLI for paged-attention runs.

## Gemma 4 MTP

Gemma 4 MTP attaches the assistant module to the target Gemma 4 model and reuses the target paged KV cache:

```bash
mistralrs run --paged-attn on -m google/gemma-4-E4B-it --quant 8 \
  --mtp-model ./gemma4-mtp \
  --mtp-n-predict 6
```

`--mtp-model` currently expects a local assistant model directory. If `--mtp-n-predict` is omitted, mistral.rs reads `num_assistant_tokens` from the assistant `generation_config.json` and falls back to 6.

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
