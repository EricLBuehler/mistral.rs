---
title: Block-diffusion models
description: Run diffusion text generation models like DiffusionGemma.
sidebar:
  order: 8
---

Block-diffusion models generate text by iteratively *denoising* whole blocks of tokens in parallel instead of sampling one token at a time. A causal encoder fills the KV cache with the prompt; the model then refines a block (a "canvas") of mask tokens over a handful of bidirectional passes, commits it, and repeats. Because each pass produces many tokens at once, throughput is typically around 2x an equivalent autoregressive model.

Currently supported: 
- **DiffusionGemma** (`google/diffusiongemma-26B-A4B-it`), a 26B-A4B mixture-of-experts model with vision input, built on the Gemma 4 architecture.

## Quick start

No special flags or APIs, as block-diffusion models are detected automatically and served through the standard endpoints:

```bash
mistralrs run -m google/diffusiongemma-26B-A4B-it
```

Or as an OpenAI-compatible server:

```bash
mistralrs serve -p 1234 -m google/diffusiongemma-26B-A4B-it
```

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 1024
  }'
```

The Rust and Python SDKs work unchanged — see [`mistralrs/examples/models/diffusion_gemma`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/diffusion_gemma) (streaming, shows block-at-a-time output) and [`examples/python/diffusion_gemma.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/diffusion_gemma.py) (vision input). Any other chat completion example applies as-is.

## What behaves differently

- **Streaming is bursty.** Output arrives one block (up to 256 tokens) at a time, after that block's denoising loop converges, rather than token by token.
- **Sampling is the diffusion schedule.** The temperature ramp, entropy-bound acceptance, and stopping thresholds come from the checkpoint's `generation_config.json`. Request-level `temperature`, `top_p`, and penalties are ignored. `max_tokens` still caps output length.
- **Stats split differently.** Prompt T/s measures the encoder prefill alone; decode T/s is the effective denoising throughput (committed tokens over denoising time).
- **Thinking is on by default.** DiffusionGemma's channel-tag reasoning is parsed into the reasoning field, like other thinking models.
- **Tool calling** works through the model's native format, including calls spanning block boundaries. Grammar-constrained generation (`tool_choice: required`/named tools, JSON schema outputs) is not enforced during denoising — parallel token refinement is incompatible with per-token grammars — so the model's trained formatting is relied upon.

See also: [model notes](/mistral.rs/reference/model-notes/) and the [supported models reference](/mistral.rs/reference/supported-models/).
