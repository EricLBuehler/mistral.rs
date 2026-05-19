---
title: Set up video input
description: Install FFmpeg and send video files to multimodal models.
sidebar:
  order: 2
---

Video input works with multimodal models that list video in the [supported models reference](/mistral.rs/reference/supported-models/). The commonly tested families are Qwen-VL and Gemma.

## Requirement

Non-GIF video formats require the `ffmpeg` binary on the server `PATH`. mistral.rs uses FFmpeg at request time to decode video files into frames.

GIF files are decoded natively and do not require FFmpeg.

Check the server environment:

```bash
ffmpeg -version
```

If that command fails in the same shell, service, or container that starts `mistralrs`, video requests for non-GIF files will fail.

## Install FFmpeg

Debian or Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

Fedora or RHEL:

```bash
sudo dnf install ffmpeg
```

macOS with Homebrew:

```bash
brew install ffmpeg
```

Windows:

Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html), install or extract it, and add the directory containing `ffmpeg.exe` to `PATH`.

Docker:

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

## Supported formats

FFmpeg-backed decoding supports any container FFmpeg can read, including `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, and `.m4v`.

Animated `.gif` files are supported without FFmpeg.

## Send video from the CLI

Use `--video` with one-shot input:

```bash
mistralrs run -m google/gemma-4-E4B-it --quant 8 --video clip.mp4 -i "Summarize this clip."
```

Multiple videos are allowed:

```bash
mistralrs run -m google/gemma-4-E4B-it --video clip1.mp4 --video clip2.mp4 -i "Compare these clips."
```

Interactive mode also auto-detects video file paths in prompts:

```text
> What happens in this video? /absolute/path/clip.mp4
```

## Send video over HTTP

Use a `video_url` content part:

```json
{
  "role": "user",
  "content": [
    {"type": "video_url", "video_url": {"url": "file:///absolute/path/clip.mp4"}},
    {"type": "text", "text": "What happens in this video?"}
  ]
}
```

Video URLs can be local `file://` paths or network `http(s)://` URLs.

## Frame handling

mistral.rs decodes video into frames before passing them through the model's vision path. Per-request frame-sampling controls are not currently exposed.

## Troubleshooting

If video decoding fails, first run `ffmpeg -version` from the same runtime environment as the server. For systemd, Docker, launchd, or Windows services, the service `PATH` can differ from your interactive shell.

If a local file is not found over HTTP, use an absolute `file://` URL and make sure the server process can read that path. Client-local paths only work when the client and server are on the same machine and the server can access the same filesystem.

If a container image needs video support, install FFmpeg in the image rather than only on the host.
