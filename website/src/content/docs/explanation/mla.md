---
title: Multi-head Latent Attention
description: How mistralrs handles MLA, including the optimized decode path and opt-out.
sidebar:
  order: 6
---

MLA (Multi-head Latent Attention) is the attention variant used by DeepSeek V2 and V3. Instead of caching K and V per token, it caches a low-dimensional latent vector from which K and V are reconstructed on demand.

## Cache layout

The KV cache stores the latent vector rather than full K and V. A projection step at attention time re-expands the latent into per-head K and V. The cache footprint is substantially smaller than standard multi-head attention at the same context length.

## Optimized decode path

On CUDA (Unix), mistralrs uses a specialized MLA decode kernel when all of the following hold:

- `MISTRALRS_NO_MLA` is not set to `"1"`.
- The attention mask is empty (single-token decode: prefill falls back to the generic path).
- Sequence length is 1.
- Paged attention is enabled.
- The device is a CUDA device.
- Paged KV indptr metadata is available.

Otherwise the engine uses the generic attention path with the latent reconstructed per step.

A parallel fast path exists for prefill with prefix caching (paged attention enabled, CUDA device, MLA not disabled).

## Opting out

`MISTRALRS_NO_MLA=1` forces the generic path. Use when debugging suspected MLA kernel issues.

## Models

MLA applies to DeepSeek V2, DeepSeek V3, and their derivative fine-tunes. Other models do not use it.

## See also

- DeepSeek V2 paper: [arxiv 2405.04434](https://arxiv.org/abs/2405.04434).
- Reference: [supported models](/mistral.rs/reference/supported-models/).
