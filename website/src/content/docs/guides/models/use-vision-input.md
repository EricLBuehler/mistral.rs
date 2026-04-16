---
title: Work with vision and video input
description: How to send images and video to a multimodal model. Covers Qwen3-VL and Gemma 4.
sidebar:
  order: 1
---

The two primary multimodal families we test heavily are Qwen3-VL (for vision and video) and Gemma 4 (for vision, audio, and video). Both accept the OpenAI multimodal message format. This guide covers the practical details of using them.

## From the CLI

Interactive mode auto-detects file paths in your prompt. Start a multimodal model and paste image, audio, or video paths directly:

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

The CLI supports `--image`, `--audio`, and `--video` flags. Each can be passed more than once to attach multiple files.

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

URLs accept three forms: `file://` for local paths, `http(s)://` for network fetches, and `data:image/png;base64,...` for inline base64 encoding.

## Video

For video, use a `video` content part:

```json
{
  "role": "user",
  "content": [
    {"type": "video", "video": {"url": "file:///clip.mp4"}},
    {"type": "text", "text": "Summarize what happens in this video."}
  ]
}
```

Video decoding requires FFmpeg to be installed on the server. The engine samples a subset of frames by default; adjust how aggressively with request fields documented in the [HTTP API reference](/mistral.rs/reference/http-api/).

Qwen3-VL and Gemma 4 both accept video. Capacities differ: Gemma 4 is the stronger choice for longer clips, Qwen3-VL tends to be better at detail extraction in short ones. In practice both are usable for most workloads.

## Audio

Audio is specific to the models that accept it. Gemma 4 E4B is the current recommendation, with Voxtral as a dedicated speech-to-text model when you only need transcription:

```json
{
  "role": "user",
  "content": [
    {"type": "audio", "audio": {"url": "file:///clip.wav"}},
    {"type": "text", "text": "Transcribe this."}
  ]
}
```

Supported formats are `.wav`, `.mp3`, `.flac`, and `.ogg` natively. For other formats, FFmpeg converts them behind the scenes.

## Multiple attachments in one message

A single message can include several parts of any combination:

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

The model sees the parts in order. For tasks where spatial relationships matter (showing several panels of the same figure, for example), sequential attachment in a single message usually works better than separate messages.

## Preprocessing

mistralrs does reasonable preprocessing by default. Images are resized to the model's preferred resolution without distorting aspect ratios; video frames are sampled at a default rate; audio is resampled to the model's expected sample rate. You can override any of this at the request level for workloads that need it, but the defaults are good enough that most users never have to.

If you are passing very large images (multi-megapixel photographs, for example), the engine will downsize them automatically before feeding them to the model. This is almost always what you want, because vision encoders have fixed input resolutions and passing 4K images does not buy you more detail.
