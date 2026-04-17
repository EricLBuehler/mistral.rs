---
title: Multi-head Latent Attention
description: What MLA changes relative to standard attention, and why DeepSeek introduced it.
sidebar:
  order: 6
---

Multi-head Latent Attention (MLA) is an attention variant introduced in DeepSeek V2 and used in DeepSeek V3. Its main contribution is significantly smaller KV caches, which matters for serving large models at long context.

For full mathematical details, the DeepSeek V2 paper is authoritative.

## The problem MLA addresses

Standard attention caches K and V projections per past token. With `n_heads` attention heads and `head_dim` per head, each cached token uses `2 * n_heads * head_dim` floats (the 2 covers K and V).

For a 128k-context sequence, this is significant. A 32-head, 128-head-dim model: 8 KB per token, 1 GB for 128k tokens, per layer. A 60-layer model: 60 GB of KV cache per sequence at max context. Even at moderate contexts, KV cache dominates memory for long conversations.

Two common mitigations: grouped-query attention (GQA) reduces KV head count without reducing query heads; multi-query attention (MQA) goes further with one KV head for all query heads. Both reduce cache size at some quality cost.

MLA takes a different approach.

## What MLA does

The insight: cache a smaller latent vector instead of K and V directly, then reconstruct K and V on demand.

Inference:

1. Compute a low-dimensional latent vector from the input (much smaller than `n_heads * head_dim`).
2. Cache the latent.
3. At attention time, project the cached latent back up to K and V for the current step.

The latent is typically 4×–8× smaller than what GQA/MQA would save, producing a much smaller cache per token. The quality tradeoff is noticeably smaller than GQA or MQA because the model can learn to use the full K and V space at attention time, even though only the compressed latent is persisted.

Attention computation is slightly more expensive due to the projection step. The cost is a small fraction of attention work, more than offset by memory savings.

## Where it shows up in mistral.rs

Models using MLA:

- DeepSeek V2 (`deepseek-ai/DeepSeek-V2-Chat`)
- DeepSeek V3 (`deepseek-ai/DeepSeek-V3`)
- Derivative fine-tunes of the above

These models load normally. mistral.rs detects MLA from the architecture and uses the appropriate cache layout and attention kernel.

## Interaction with other features

**Quantization.** MLA weights quantize the same way as other attention weights. The latent projection matrices have unremarkable numerical behavior; Q4K and similar formats work fine.

**Paged attention.** Block-based cache works with MLA — each block holds latents instead of raw K and V. Block size and memory accounting are unchanged.

**Flash attention.** Flash kernels are adapted for MLA's reconstruction step. CUDA uses a dedicated kernel path that understands the latent-to-KV projection inside the flash framework. The end-user experience is transparent: flash attention helps MLA models the same way as non-MLA models.

## Why the tradeoff is favorable

MLA cache is typically 8×–16× smaller than standard multi-head attention's cache. A massive win for long-context serving — a model needing 60 GB of cache memory needs closer to 4 GB, changing what hardware can run the model at useful concurrency.

The cost is a few extra matrix multiplications per attention step. Empirically 2–5% more compute on well-tuned kernels, a rounding error compared to the savings.

Short-context workloads see less cache benefit, and MLA offers no compute benefit over standard attention. Short contexts are not what MLA was designed for; DeepSeek's rationale was long-context serving where savings are enormous.

## What the paper covers that this page does not

- Specific projection matrices and rank choices.
- MLA interaction with rotary position embeddings.
- Training stability considerations.
- Benchmark comparisons against GQA and MQA at the same parameter count.

[DeepSeek V2](https://arxiv.org/abs/2405.04434) is the canonical reference. The V3 paper iterates the same ideas.

## What this means for running DeepSeek models

Practically, nothing. MLA is an internal architectural choice; from the outside, DeepSeek V2 and V3 behave like any other language model. Load, send requests, get responses.

The externally visible consequence: these models support longer contexts on the same hardware than comparably-sized standard-attention models. For very-long-context deployments, DeepSeek (or other MLA-based models) is worth considering for that reason.
