---
title: Set up video input
description: Install FFmpeg so multimodal models can decode video files.
---

This page covers the FFmpeg setup that video input depends on. To actually send video (CLI flags, content parts, model notes), see [Send images, audio, and video](/mistral.rs/guides/models/multimodal-input/). Video input works with multimodal models that list video in the [supported models reference](/mistral.rs/reference/supported-models/).

## You need FFmpeg on the server PATH

- Non-GIF video formats require the `ffmpeg` binary on the server `PATH`. mistral.rs invokes FFmpeg at request time to decode video files into frames.
- GIF files are decoded natively and do not require FFmpeg.

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

FFmpeg-backed decoding supports any container FFmpeg can read, including `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, and `.m4v`. Animated `.gif` files are supported without FFmpeg.

## Troubleshooting

If video decoding fails, first run `ffmpeg -version` from the same runtime environment as the server. For systemd, Docker, launchd, or Windows services, the service `PATH` can differ from your interactive shell.

If a local file is not found over HTTP, use an absolute `file://` URL and make sure the server process can read that path. Client-local paths only work when the client and server are on the same machine and the server can access the same filesystem.

If a container image needs video support, install FFmpeg in the image rather than only on the host.
