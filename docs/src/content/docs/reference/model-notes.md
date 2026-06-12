---
title: Model notes
description: Per-model behavior that differs from the defaults.
sidebar:
  order: 10
---

## DiffusionGemma

See the [block-diffusion guide](/mistral.rs/guides/models/use-block-diffusion/) for setup, examples, and performance tuning.

**Block-diffusion text generation.** DiffusionGemma generates text in 256-token blocks: a causal encoder fills the KV cache, then each block is iteratively denoised with bidirectional attention (up to 48 steps, usually far fewer thanks to adaptive stopping). Output streams one block at a time rather than token by token, and reported stats split encoder prefill (prompt) from denoising (decode) time.

**Sampling parameters.** The denoising schedule (temperature ramp, entropy-bound acceptance, stopping thresholds) comes from the checkpoint's `generation_config.json`. Request-level `temperature`/`top_p` are ignored; `max_tokens` caps the number of generated blocks.

**Concurrency.** Concurrent requests with equal context lengths batch together and denoise their blocks in lockstep, amortizing the expert sweep; requests with different prompt lengths run as separate groups.

**Thinking and tools.** Channel-tag reasoning is enabled by default and parsed into the reasoning field. Native/auto tool calling works, including calls spanning block boundaries. Grammar-constrained generation (`tool_choice: required`/named, JSON schema outputs) is not enforced during denoising; the model's native formatting is relied upon.

## Gemma 4

**Strict tool grammar.** Gemma 4 enforces tool call format through constrained decoding (llguidance). Enabled by default.

**Multimodal inputs.** Gemma 4 E4B and E2B accept audio, image, and video parts in the same message. No separate variant or flag required.

**HF license acceptance.** Gemma requires accepting a license on the Hugging Face model page before download. `mistralrs login` with an HF token suffices after acceptance.

## Qwen3-VL

**Video frame sampling controls.** Per-request sampling overrides are not currently exposed.

**Prefix cache with mixed-modal inputs.** In multi-turn conversations reusing prefix cache entries, pixel inputs are narrowed per turn by grid count, not image count.

## Qwen3 Next

**Hybrid attention.** Qwen3 Next mixes linear attention layers with standard softmax attention. The cost profile at long contexts differs from a pure softmax model.

## DeepSeek V2 and V3

**Multi-head Latent Attention (MLA).** DeepSeek's attention variant produces smaller KV caches than standard attention. For unexpected behavior under paged attention, try `--paged-attn off` as a sanity check.

## Phi 3.5 MoE

**MoE routing.** Phi 3.5 MoE routes per-token to experts. Outputs for the same seed vary across quantization levels due to router sensitivity to numerical noise. See the [explanation on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).

## Gemma 3n

**MatFormer slice.** Gemma 3n is MatFormer-trained. Select a size variant with `--matformer-config-path` and `--matformer-slice-name` (or the matching SDK/TOML fields). Without configuration, the default slice loads. See the [MatFormer guide](/mistral.rs/guides/customize/matformer/).

## Llama 4

**Context length.** Llama 4 Scout supports up to 10M tokens. Using the full context requires paged attention with a large memory budget, generally with multi-GPU tensor parallelism.

## FLUX

**License variants.** FLUX.1-schnell is permissive. FLUX.1-dev requires accepting a license on Hugging Face.

**Quantization sensitivity.** Diffusion models are more sensitive to quantization than language models.

## Reporting issues

File an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues), with a reproducer when possible.
