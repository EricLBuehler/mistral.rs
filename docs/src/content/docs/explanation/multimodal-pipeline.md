---
title: The multimodal pipeline
description: How image, video, and audio inputs reach the model.
sidebar:
  order: 8
---

Multimodal requests carry non-text content parts (`image_url`, `audio`, `video`). Each part goes through a per-modality preprocessing and encoder path before the tokens reach the transformer.

## Request shape

Content parts are ordered in the request body. The engine preserves that order when it interleaves media tokens with text tokens, so text surrounding an image appears on either side of the image tokens in the final token sequence. The transformer sees one uniform token stream.

## Image path

1. **Decode.** The URL form (`file://`, `http(s)://`, `data:image/...;base64,`) is resolved to a pixel buffer. HTTP URLs trigger a fetch.
2. **Preprocess.** Model-specific: resize to the vision encoder's expected resolution, normalize per-channel, tensorize.
3. **Encode.** The vision encoder produces a sequence of patch embeddings.
4. **Project.** A learned projection maps patch embeddings to the transformer's hidden dimension.
5. **Place.** The projected tokens are inserted at the position corresponding to the content part in the user's request.

Multiple images in one request are encoded as a batch.

## Video path

Video is decoded to frames before model preprocessing. Each selected frame then flows through the image path.

Supported containers and FFmpeg installation are covered in [Set up video input](/mistral.rs/guides/models/video-setup/).

## Audio path

1. **Decode.** The audio file is decoded to PCM at the model's expected sample rate. FFmpeg handles non-native formats.
2. **Feature extraction.** Mel-spectrogram or similar.
3. **Encode.** A model-specific audio encoder produces a sequence of vectors.
4. **Project and place.** As with images.

## Encoder cache

Encoder outputs are cached by content hash and modality. When the same image, video, or audio clip appears in a later request, or in a later turn of the same session, the encoder pass is skipped and the cached tokens are reused.

The modality is part of the key: identical bytes processed as an image versus as a video frame can yield different token counts, and the cache keeps them separate.

The cache is LRU with a fixed capacity per model. Hit and miss counters are exposed for observability.

## See also

- Guide: [work with vision and video input](/mistral.rs/guides/models/use-vision-input/).
- Setup: [set up video input](/mistral.rs/guides/models/video-setup/).
- Reference: [supported models](/mistral.rs/reference/supported-models/).
