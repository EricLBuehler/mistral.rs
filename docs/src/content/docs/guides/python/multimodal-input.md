---
title: Multimodal input from Python
description: Pass images, audio, and video to a multimodal model using the Python SDK.
sidebar:
  order: 2
---

The multimodal message format follows the OpenAI convention: `content` can be a list of typed parts rather than a string. Each part has a `type` field. The examples below use Qwen3-VL for vision and Gemma 4 for audio and video.

## Sending an image

```python
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        arch=MultimodalArchitecture.Qwen3VL,
    ),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-VL-4B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file:///path/to/photo.jpg"}},
                    {"type": "text", "text": "What do you see in this image?"}
                ]
            }
        ],
        max_tokens=256,
    )
)

print(response.choices[0].message.content)
```

`image_url` accepts three URL forms:

- `file:///absolute/path`: local files.
- `https://...`: network fetches.
- `data:image/png;base64,...`: inline base64.

Multiple images per message work, include several `image_url` parts. The model sees them in order.

## Sending audio

Gemma 4 E4B handles audio natively:

```python
from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-4-E4B-it",
        arch=MultimodalArchitecture.Gemma4,
    ),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "file:///path/to/clip.wav"}},
                    {"type": "text", "text": "Transcribe this audio."}
                ]
            }
        ],
    )
)

print(response.choices[0].message.content)
```

Audio parts use the same URL forms as images. Native formats: `.wav`, `.mp3`, `.flac`, `.ogg`. Other formats require FFmpeg; see [Set up video input](/mistral.rs/guides/models/video-setup/) for installation.

## Sending video

Gemma 4 accepts video as a sequence of sampled frames:

```python
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "file:///path/to/clip.mp4"}},
                    {"type": "text", "text": "Describe what happens in this video."}
                ]
            }
        ],
    )
)
```

Video setup, supported containers, and FFmpeg installation are covered in [Set up video input](/mistral.rs/guides/models/video-setup/). The engine decodes frames, encodes them, and passes them to the model. Per-request sampling controls are not currently exposed.

## Mixing modalities in one request

A message can include any combination of parts. Order matters for the model:

```python
messages=[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///chart.png"}},
            {"type": "audio_url", "audio_url": {"url": "file:///commentary.wav"}},
            {"type": "text", "text": "Does the commentary match what the chart shows?"}
        ]
    }
]
```

Requires a model supporting both modalities. Gemma 4 handles images, audio, and video in one message; Qwen3-VL handles images plus video.

Per-model modality support: [supported models reference](/mistral.rs/reference/supported-models/).

## Programmatic attachments

For in-memory images (bytes or PIL Image), encode as base64 and pass inline:

```python
import base64
from io import BytesIO
from PIL import Image

img = Image.open("photo.jpg")
buf = BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

messages=[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
```

The engine handles base64 decoding and image preprocessing.
