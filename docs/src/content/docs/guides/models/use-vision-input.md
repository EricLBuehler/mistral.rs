---
title: Work with vision and video input
description: How to send images and video to a multimodal model. Covers Qwen3-VL and Gemma 4.
sidebar:
  order: 1
---

The two heavily tested multimodal families are Qwen3-VL (vision, video) and Gemma 4 (vision, audio, video). Both accept the OpenAI multimodal message format.

If you will send video files, install FFmpeg first. The canonical setup checklist is [Set up video input](/mistral.rs/guides/models/video-setup/).

## From the CLI

Interactive mode auto-detects file paths in prompts:

```bash
mistralrs run -m Qwen/Qwen3-VL-4B-Instruct
```

```
> Describe this: /path/to/photo.jpg
```

Or pass attachments with `-i`:

```bash
mistralrs run -m Qwen/Qwen3-VL-4B-Instruct --image photo.jpg -i "What is this?"
```

The CLI supports `--image`, `--audio`, and `--video`. Each accepts multiple values.

## From the HTTP API

Multimodal messages use typed content parts:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "file:///path/to/photo.jpg"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }]
  }'
```

URLs accept three forms: `file://` for local paths, `http(s)://` for network fetches, `data:image/png;base64,...` for inline base64.

## Video

Use a `video_url` content part:

```json
{
  "role": "user",
  "content": [
    {"type": "video_url", "video_url": {"url": "file:///clip.mp4"}},
    {"type": "text", "text": "Summarize what happens in this video."}
  ]
}
```

Video decoding requirements, supported containers, and platform install commands are covered in [Set up video input](/mistral.rs/guides/models/video-setup/). Per-request frame-sampling controls are not currently exposed.

Both Qwen3-VL and Gemma 4 accept video. Gemma 4 handles longer clips better; Qwen3-VL handles short-clip detail better.

## Audio

Audio is model-specific. Gemma 4 E4B handles audio; Voxtral is the dedicated speech-to-text model:

```json
{
  "role": "user",
  "content": [
    {"type": "audio_url", "audio_url": {"url": "file:///clip.wav"}},
    {"type": "text", "text": "Transcribe this."}
  ]
}
```

Native formats: `.wav`, `.mp3`, `.flac`, `.ogg`. Other formats use FFmpeg conversion; see [Set up video input](/mistral.rs/guides/models/video-setup/) for FFmpeg installation.

## Multiple attachments in one message

A single message can include multiple parts of any combination:

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "file:///before.jpg"}},
    {"type": "image_url", "image_url": {"url": "file:///after.jpg"}},
    {"type": "text", "text": "What changed between these images?"}
  ]
}
```

The model sees parts in order.

## Preprocessing

Default preprocessing resizes images to the model's input resolution preserving aspect ratio, uses the decoded video frames, and resamples audio to the model's expected rate. Per-request preprocessing overrides are not currently exposed.

Large images are downsized before reaching the model. Vision encoders have fixed input resolutions.
