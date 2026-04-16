---
title: Multimodal input from Python
description: Pass images, audio, and video to a multimodal model using the Python SDK.
sidebar:
  order: 2
---

The multimodal message format follows the OpenAI convention: each message's `content` can be a list of typed parts rather than a plain string. Each part is a dict with a `type` field telling the engine what it is.

This guide uses Qwen3-VL for vision, Gemma 4 for audio and video, because those are the modalities each model family handles best today.

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

The `image_url` part accepts three URL forms:

- `file:///absolute/path` for local files.
- `https://...` for network fetches.
- `data:image/png;base64,...` for inline base64-encoded images.

Multiple images in the same message work; just include several `image_url` parts. The model sees them in the order they appear.

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
                    {"type": "audio", "audio": {"url": "file:///path/to/clip.wav"}},
                    {"type": "text", "text": "Transcribe this audio."}
                ]
            }
        ],
    )
)

print(response.choices[0].message.content)
```

Audio parts take the same URL shapes as images. Supported formats include `.wav`, `.mp3`, `.flac`, and `.ogg`. For everything else, the underlying audio decoder requires FFmpeg to be installed.

## Sending video

Gemma 4 accepts video input as a sequence of sampled frames:

```python
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"url": "file:///path/to/clip.mp4"}},
                    {"type": "text", "text": "Describe what happens in this video."}
                ]
            }
        ],
    )
)
```

Video support requires FFmpeg. The engine samples frames at a default rate (configurable via request fields covered in the [HTTP API reference](/mistral.rs/reference/http-api/)), encodes them, and passes them to the model.

Supported containers include `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm`, plus `.gif` for animated images.

## Mixing modalities in one request

A message can include any combination of parts. The order matters for how the model sees them:

```python
messages=[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///chart.png"}},
            {"type": "audio", "audio": {"url": "file:///commentary.wav"}},
            {"type": "text", "text": "Does the commentary match what the chart shows?"}
        ]
    }
]
```

This only works on a model that supports both modalities. Gemma 4 handles images, audio, and video in one message; Qwen3-VL does images plus video.

The [supported models reference](/mistral.rs/reference/supported-models/) has a table of what each model accepts.

## Programmatic attachments

If you have an image in memory (as bytes or a PIL Image), encode it as base64 and pass it inline:

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

The engine handles the base64 decoding and image preprocessing internally.
