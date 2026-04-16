---
title: The multimodal pipeline
description: How pixels, video frames, and audio reach a multimodal model and end up as tokens the transformer can attend to.
sidebar:
  order: 8
---

A text-only model's request lifecycle is straightforward: receive a string, tokenize, feed the tokens through the transformer. Multimodal models have more steps between the raw input and the tokens. This page is a walkthrough of those steps.

## The high-level shape

For a multimodal request, each non-text content part (an image, an audio clip, a video) goes through its own preprocessing path before reaching the transformer. The paths produce "tokens" in the transformer's input space, which are then interleaved with the text tokens in the order the user provided them.

From the transformer's point of view, it is all one sequence of token-shaped things. The model does not know or care which tokens came from text, images, or audio; it attends to all of them uniformly.

## Image path

Images flow through:

1. **Decoding.** Whatever format the user sent (JPG, PNG, base64 data URL) is decoded to a pixel buffer. For URLs, a fetch happens first.
2. **Preprocessing.** Resize to the model's expected input resolution, normalize to the expected pixel range, convert to a tensor. Different models expect different things here; each model's preprocessing is coded against its reference implementation.
3. **Vision encoder.** A dedicated vision model (SigLIP, CLIP, a model-specific variant) turns the pixel tensor into a sequence of high-dimensional vectors, one per image patch. This is usually the most expensive step per image.
4. **Projection.** A small learned projection maps the vision encoder's output to the transformer's hidden dimension. At this point the image is a sequence of tokens in the transformer's input space.
5. **Placement.** The image tokens get inserted into the full token sequence at the position where `{"type": "image_url", ...}` appeared in the user's content.

The whole process is per-image. Multiple images in one request go through the vision encoder in a batch for efficiency.

## Video path

Video is treated as a sequence of images. The decode step extracts frames at a sampled rate (not every frame; for a 30fps clip, we typically sample one frame per second), then each frame goes through the image path above.

Frame sampling matters for quality and cost. Too few frames, and the model cannot see what is happening. Too many, and the token budget for the request gets consumed by video.

Gemma 4 and Qwen3-VL both have reasonable defaults. If you need to override, per-request options in the HTTP API let you adjust the sampling rate; see the [vision-and-video guide](/mistral.rs/guides/models/use-vision-input/).

## Audio path

Audio goes through:

1. **Decoding.** The audio file is decoded to PCM samples at the model's expected sample rate. FFmpeg handles non-native formats transparently.
2. **Feature extraction.** Most models use a spectrogram or mel-spectrogram representation rather than raw waveform. This is computed on CPU.
3. **Audio encoder.** A dedicated encoder (Whisper-style, or model-specific) turns the spectrogram into a sequence of vectors.
4. **Projection and placement.** Same as the image path: project to the transformer's hidden dimension, interleave with text at the right position.

Audio is the most heterogeneous path across models. Different architectures have very different encoders, and the per-second token cost varies wildly. Gemma 4 uses its own audio encoder; Voxtral uses one tuned for speech; MiniCPM-O uses yet another.

## The encoder cache

Preprocessing is expensive. For conversations that refer to the same image multiple times, we cache the encoded representation so it is only computed once.

The cache is keyed by a hash of the raw pixel data and a modality marker. Sessions that mention the same image in turn 1 and turn 3 reuse the computed tokens from turn 1 rather than re-encoding.

The modality marker matters. An image and a video frame might have identical bytes (say, a GIF whose frames include a still image you already sent). Without the modality key, the cache would return the image's tokens for the video frame, which have different token counts, and the inference would get the wrong shape. With the key, they get cached separately.

## Why it is not one unified encoder

You might wonder why each modality needs its own encoder, especially given that they all produce "tokens" for the same transformer.

The short answer: different modalities have very different statistics. A vision encoder that works well on natural images is bad at spectrograms. A speech encoder that works well on speech audio is bad at music. The research on unified encoders exists but has not converged on a clear winner; most production-quality multimodal models still use separate encoders per modality.

The longer answer: the choice of encoder is also a training-time decision. Multimodal models are trained jointly across modalities, with their encoders as integrated parts of the model. You cannot swap in a different vision encoder without retraining.

## Token cost

For rough planning, a 1024x1024 image is typically around 256 to 1024 tokens, depending on the model. A 10-second audio clip at typical models' resolution is around 500 tokens. A 30-second video clip sampled at 1 fps is 30 times the image cost.

These add up fast. For context-budget-conscious applications, the token cost of multimodal content is usually the dominant factor. Models that claim "128k context" but charge 500 tokens per image can only hold 256 images in that context.

## When something goes wrong

The multimodal path has more failure modes than the text path. A few common ones:

**Unsupported format.** The decoder does not recognize the format. Convert to a known format first (JPG, PNG for images; WAV for audio).

**Size limits.** Enormous images are resized down silently; non-pixel data (Exif) is ignored. If you are getting unexpected results from a huge image, the model is seeing a smaller version than you sent.

**FFmpeg missing.** Video support needs FFmpeg installed on the server. Without it, video requests fail with a clear error.

**Modality mismatch.** Sending video to a model that only supports images will fail. The [supported models reference](/mistral.rs/reference/supported-models/) has the compatibility matrix.

## Where this lives in the code

- `mistralrs-core/src/vision_models/` has the per-model vision (and often audio) encoder implementations.
- `mistralrs-core/src/paged_attention/encoder_cache.rs` implements the per-modality encoder cache.
- `mistralrs-vision/` has shared image preprocessing primitives.
- `mistralrs-audio/` has shared audio decoding and preprocessing primitives.

For people debugging multimodal behavior, the log at `DEBUG` level reports per-modality token counts and encoder cache hits, which is usually what you need to understand unexpected behavior.
