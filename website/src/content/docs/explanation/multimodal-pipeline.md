---
title: The multimodal pipeline
description: How pixels, video frames, and audio reach a multimodal model and end up as tokens the transformer can attend to.
sidebar:
  order: 8
---

A text-only model's request lifecycle is simple: receive a string, tokenize, feed tokens through the transformer. Multimodal models add steps between raw input and tokens.

## The high-level shape

For a multimodal request, each non-text content part (image, audio clip, video) goes through its own preprocessing path before reaching the transformer. The paths produce "tokens" in the transformer's input space, interleaved with text tokens in the user-provided order.

From the transformer's perspective, it is one sequence of token-shaped things. The model neither knows nor cares which tokens came from text, images, or audio; it attends to all of them uniformly.

## Image path

Images flow through:

1. **Decoding.** Whatever format the user sent (JPG, PNG, base64 data URL) is decoded to a pixel buffer. URLs trigger a fetch first.
2. **Preprocessing.** Resize to the model's expected input resolution, normalize to the expected pixel range, convert to tensor. Models vary; preprocessing is coded against each model's reference implementation.
3. **Vision encoder.** A dedicated vision model (SigLIP, CLIP, or model-specific variant) turns the pixel tensor into a sequence of high-dimensional vectors per image patch. Usually the most expensive per-image step.
4. **Projection.** A small learned projection maps vision encoder output to the transformer's hidden dimension. The image is now a sequence of tokens in transformer input space.
5. **Placement.** Image tokens are inserted at the position where `{"type": "image_url", ...}` appeared in the user's content.

Per-image. Multiple images batch through the vision encoder for efficiency.

## Video path

Video is treated as an image sequence. Decoding extracts frames at a sampled rate (not every frame; typically one per second for 30fps clips), then each frame goes through the image path.

Frame sampling matters for quality and cost. Too few frames hide what is happening; too many consume the request's token budget on video.

Gemma 4 and Qwen3-VL have reasonable defaults. To override, per-request HTTP API options adjust sampling rate; see the [vision-and-video guide](/mistral.rs/guides/models/use-vision-input/).

## Audio path

Audio flows through:

1. **Decoding.** The audio file is decoded to PCM samples at the model's expected sample rate. FFmpeg handles non-native formats.
2. **Feature extraction.** Most models use spectrogram or mel-spectrogram representation rather than raw waveform. CPU-computed.
3. **Audio encoder.** A dedicated encoder (Whisper-style or model-specific) turns the spectrogram into a sequence of vectors.
4. **Projection and placement.** Same as image path — project to the transformer's hidden dimension, interleave with text at the correct position.

Audio is the most heterogeneous path across models. Architectures use very different encoders, and per-second token cost varies widely. Gemma 4 uses its own audio encoder; Voxtral uses one tuned for speech; MiniCPM-O uses yet another.

## The encoder cache

Preprocessing is expensive. For conversations referencing the same image multiple times, the encoded representation is cached so it computes once.

The cache key is a hash of raw pixel data plus a modality marker. Sessions referencing the same image in turn 1 and turn 3 reuse the computed tokens from turn 1.

The modality marker matters. An image and a video frame might have identical bytes (e.g., a GIF with frames including a still image already sent). Without the marker, the cache would return image tokens for the video frame, which have different counts, and inference would get the wrong shape. With the marker, they cache separately.

## Why it is not one unified encoder

Different modalities have very different statistics. A vision encoder optimized for natural images is bad at spectrograms. A speech encoder optimized for speech is bad at music. Unified-encoder research exists but has not converged on a clear winner; production-quality multimodal models still use separate per-modality encoders.

Encoder choice is also a training-time decision. Multimodal models train jointly across modalities with their encoders as integrated parts. Swapping encoders requires retraining.

## Token cost

Approximate planning numbers: a 1024×1024 image is 256–1024 tokens depending on the model. A 10-second audio clip at typical model resolution is ~500 tokens. A 30-second video at 1 fps is 30× the image cost.

These add up fast. For context-budget-conscious applications, multimodal token cost is usually dominant. A "128k context" model charging 500 tokens per image holds 256 images.

## When something goes wrong

The multimodal path has more failure modes than text. Common ones:

**Unsupported format.** The decoder does not recognize the format. Convert to a known format first (JPG, PNG for images; WAV for audio).

**Size limits.** Enormous images are silently downsized; non-pixel data (Exif) is ignored. Unexpected results from huge images mean the model saw a smaller version.

**FFmpeg missing.** Video support requires FFmpeg on the server. Without it, video requests fail with a clear error.

**Modality mismatch.** Sending video to an image-only model fails. The [supported models reference](/mistral.rs/reference/supported-models/) has the compatibility matrix.

## Where this lives in the code

- `mistralrs-core/src/vision_models/` — per-model vision (and often audio) encoder implementations.
- `mistralrs-core/src/paged_attention/encoder_cache.rs` — per-modality encoder cache.
- `mistralrs-vision/` — shared image preprocessing primitives.
- `mistralrs-audio/` — shared audio decoding and preprocessing primitives.

For debugging multimodal behavior, `DEBUG`-level logs report per-modality token counts and encoder cache hits.
