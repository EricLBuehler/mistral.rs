---
title: Model notes
description: Short FAQ-style notes for the handful of models whose behavior is genuinely surprising.
sidebar:
  order: 10
---

The vast majority of supported models behave the way you would expect from their documentation: load them, send a request, get a response. This page is for the small number of models that have specific quirks worth knowing about. Everything not listed here behaves like its peers.

## Gemma 4

**Strict tool grammar.** Gemma 4 expects tool calls in a specific grammar that mistralrs enforces through constrained decoding (llguidance). This is normally invisible: you enable tools, the model emits valid tool calls. If you see output that looks like the model is trying to emit tool calls but producing malformed text, the grammar enforcement is usually the thing to look at. Disabling constrained decoding produces worse output reliability, so we keep it on by default.

**Audio and video out of the box.** Gemma 4 E4B and E2B both accept audio, image, and video content parts in the same message. You do not need a separate variant or a special flag. This is different from most other multimodal models that support only one or two modalities.

**HF license acceptance.** Gemma requires you to accept a license on the Hugging Face model page before weights are downloadable. `mistralrs login` with your HF token is sufficient after acceptance.

## Qwen3-VL

**Video frame sampling defaults.** Qwen3-VL is tuned for short-to-medium video clips. For very long videos, the default frame-sampling rate produces too many tokens; adjust with request-level video options (see the [HTTP API reference](/mistral.rs/reference/http-api/)).

**Prefix cache with mixed-modal inputs.** The prefix cache interacts with vision tokens in a non-obvious way. In multi-turn conversations that reuse prefix cache entries, the pixel inputs are narrowed per turn based on grid counts, not image counts. In practice this works correctly; it becomes visible only if you are debugging cache hits manually.

## Qwen3 Next

**Hybrid attention.** Qwen3 Next mixes linear attention layers with standard softmax attention. The cost profile is different from a pure softmax model; at very long contexts it is considerably faster, but the first few thousand tokens can be slightly slower than a same-size Qwen3 (non-Next).

## DeepSeek V2 and V3

**Multi-head Latent Attention (MLA).** DeepSeek's attention variant produces smaller KV caches than standard attention, which helps memory but interacts with paged attention differently. If you see unexpected behavior under paged attention specifically, try `--no-paged-attn` as a sanity check.

## Phi 3.5 MoE

**MoE routing caveat.** Phi 3.5 MoE routes per-token to experts, and the routing produces slightly nondeterministic outputs across different quantization levels for the same seed. This is expected; the model's sensitivity to numerical noise in the router is higher than a dense model's sensitivity to similar noise elsewhere. The [explanation on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/) covers the general principle.

## Gemma 3n

**MatFormer slice.** Gemma 3n is a MatFormer-trained model. To load a specific size variant, pass `--matformer-config`; see the [MatFormer guide](/mistral.rs/guides/customize/matformer/). Without that config, it loads the default slice, which is the full size.

## Llama 4

**Context length.** Llama 4 Scout has a very large context window (up to 10M tokens). Actually using that much context requires paged attention at a large memory budget and usually tensor parallelism across multiple GPUs. The model loads fine with smaller contexts too; you just need the infrastructure for the long-context case.

## FLUX

**License variants.** FLUX.1-schnell is permissive. FLUX.1-dev requires accepting a license on Hugging Face before downloading. Both work identically in mistralrs aside from the license step.

**Quantization quality.** Diffusion models are more sensitive to quantization than language models. At 4-bit, output quality is noticeably lower than BF16. Use `--isq 8` if you have the memory; skip quantization entirely if you can afford 30 GB of VRAM.

## Something surprising not in this list?

If you find behavior that does not match what you expected, and the model is not in this list, that usually means the behavior is a bug. An issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues) is the place to report it, with a reproducer if possible.
