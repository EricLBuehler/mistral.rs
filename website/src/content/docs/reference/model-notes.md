---
title: Model notes
description: Short FAQ-style notes for the handful of models whose behavior is genuinely surprising.
sidebar:
  order: 10
---

Most supported models behave as documented. This page covers the small number with specific quirks.

## Gemma 4

**Strict tool grammar.** Gemma 4 expects tool calls in a specific grammar enforced through constrained decoding (llguidance). Normally invisible: enable tools, the model emits valid tool calls. Output that looks like malformed tool calls usually points to grammar enforcement. Disabling constrained decoding produces worse output reliability, so it stays on by default.

**Audio and video out of the box.** Gemma 4 E4B and E2B accept audio, image, and video parts in the same message. No separate variant or flag required. This differs from most multimodal models supporting one or two modalities.

**HF license acceptance.** Gemma requires accepting a license on the Hugging Face model page before download. `mistralrs login` with an HF token suffices after acceptance.

## Qwen3-VL

**Video frame sampling defaults.** Qwen3-VL is tuned for short-to-medium clips. Long videos at the default sampling rate produce too many tokens; adjust with request-level video options (see the [HTTP API reference](/mistral.rs/reference/http-api/)).

**Prefix cache with mixed-modal inputs.** The prefix cache interacts with vision tokens non-obviously. In multi-turn conversations reusing prefix cache entries, pixel inputs are narrowed per turn by grid count, not image count. Works correctly; visible only when manually debugging cache hits.

## Qwen3 Next

**Hybrid attention.** Qwen3 Next mixes linear attention layers with standard softmax attention. The cost profile differs from a pure softmax model — significantly faster at very long contexts, slightly slower than a same-size Qwen3 (non-Next) at the first few thousand tokens.

## DeepSeek V2 and V3

**Multi-head Latent Attention (MLA).** DeepSeek's attention variant produces smaller KV caches than standard attention. This helps memory but interacts with paged attention differently. For unexpected behavior under paged attention, try `--paged-attn off` as a sanity check.

## Phi 3.5 MoE

**MoE routing caveat.** Phi 3.5 MoE routes per-token to experts. The routing produces slightly nondeterministic outputs across different quantization levels for the same seed. Expected; the model's sensitivity to numerical noise in the router is higher than a dense model's sensitivity to similar noise elsewhere. See the [explanation on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).

## Gemma 3n

**MatFormer slice.** Gemma 3n is MatFormer-trained. Specifying a particular size variant currently requires the SDK-level `matformer_config_path` and `matformer_slice_name` (no CLI flag). Without configuration, the default slice loads. See the [MatFormer guide](/mistral.rs/guides/customize/matformer/).

## Llama 4

**Context length.** Llama 4 Scout supports up to 10M tokens. Actually using that context requires paged attention with a large memory budget, usually with multi-GPU tensor parallelism. The model loads fine at smaller contexts.

## FLUX

**License variants.** FLUX.1-schnell is permissive. FLUX.1-dev requires accepting a license on Hugging Face. Both work identically in mistral.rs aside from the license step.

**Quantization quality.** Diffusion models are more sensitive to quantization than language models. Prefer higher bit widths when memory permits.

## Something surprising not in this list?

If observed behavior does not match expectations and the model is not listed, that is usually a bug. File an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues), with a reproducer when possible.
