---
title: Multi-head Latent Attention
description: What MLA changes relative to standard attention, and why DeepSeek introduced it.
sidebar:
  order: 6
---

Multi-head Latent Attention (MLA) is a variant of the attention mechanism introduced in DeepSeek V2 and used in DeepSeek V3. Its main contribution is cutting the size of the KV cache significantly, which matters a lot for serving large models at long context.

This page is a short conceptual overview. For the full mathematical details, the DeepSeek V2 paper is the authoritative source.

## The problem MLA addresses

Standard attention caches the key and value projections for each past token. For a model with `n_heads` attention heads and `head_dim` dimensions per head, each cached token takes `2 * n_heads * head_dim` floating-point numbers (the factor of 2 is for both K and V).

For a 128k-context sequence, that is a lot of memory. On a 32-head, 128-head-dim model, each token takes 8 KB; 128k tokens takes 1 GB, per layer. For a 60-layer model, that is 60 GB of KV cache per sequence at max context. Even at moderate context lengths, KV cache dominates memory usage for long conversations.

Two mitigations are common. Grouped-query attention (GQA) reduces the number of KV heads without reducing the number of query heads. Multi-query attention (MQA) goes further, using a single KV head for all query heads. Both reduce cache size at some cost to quality.

MLA takes a different approach.

## What MLA does

The key insight: instead of caching K and V directly, cache a smaller latent vector from which K and V can be reconstructed on demand.

During inference:

1. Compute a low-dimensional latent vector from the input (much smaller than `n_heads * head_dim`).
2. Cache only this latent vector.
3. At attention time, project the cached latent back up to K and V for the current step's computation.

The latent is typically 4x to 8x smaller than what GQA/MQA would save, making the KV cache much smaller per token. The quality tradeoff is noticeably smaller than GQA or MQA, because the model can learn to use the full K and V space at attention time even though only the compressed latent is persisted.

The catch is that attention computation is slightly more expensive, because of the projection step. In practice this cost is a small fraction of the attention work, and the memory savings more than compensate.

## Where it shows up in mistralrs

Models using MLA:

- DeepSeek V2 (`deepseek-ai/DeepSeek-V2-Chat`)
- DeepSeek V3 (`deepseek-ai/DeepSeek-V3`)
- Any derivative fine-tune of the above

These models load normally. mistralrs detects MLA from the architecture and uses the appropriate cache layout and attention kernel.

## Interaction with other features

**Quantization.** MLA weights quantize the same way other attention weights do. The latent projection matrices are not unusual in their numerical behavior; Q4K and similar formats work fine.

**Paged attention.** The block-based cache still works with MLA; each block holds latents instead of raw K and V. The block size and memory accounting are the same.

**Flash attention.** Flash attention kernels have to be adapted for MLA's reconstruction step. We use a dedicated kernel path on CUDA that understands the latent-to-KV projection inside the flash framework. The end-user experience is transparent: flash attention still helps with MLA models the same way it does with non-MLA ones.

## Why the tradeoff is favorable

The MLA cache is usually 8x to 16x smaller than standard multi-head attention's cache. That is a massive win for long-context serving: a model that would have needed 60 GB of cache memory needs closer to 4 GB, which changes what hardware can run the model at useful concurrency.

The cost is a few extra matrix multiplications per attention step. Empirically these take 2-5% more compute than standard attention on well-tuned kernels, which is a rounding error compared to what you save.

For short-context workloads, the cache savings are less important, and MLA offers no compute benefit over standard attention. But short contexts are not what MLA was designed for; DeepSeek's rationale was explicitly long-context serving, where the savings are enormous.

## What the paper covers that this page does not

- The specific projection matrices and rank choices.
- How MLA interacts with rotary position embeddings.
- Training stability considerations.
- Benchmark comparisons against GQA and MQA at the same parameter count.

[DeepSeek V2](https://arxiv.org/abs/2405.04434) is the canonical reference. The V3 follow-up paper iterates the same ideas.

## What this means for running DeepSeek models

Practically, nothing changes. MLA is an internal architectural choice; from the outside, DeepSeek V2 and V3 look like any other language model. You load them, you send requests, you get responses.

The main externally visible consequence is that these models can support longer contexts on the same hardware than a comparably-sized standard-attention model could. If you are planning a deployment that needs very long contexts, DeepSeek (or other MLA-based models as they appear) is worth considering specifically for that reason.
