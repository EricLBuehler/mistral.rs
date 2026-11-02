---
title: Set up video input
description: Install FFmpeg for non-GIF video decoding.
---

Non-GIF video requires `ffmpeg` on the PATH of the process running mistral.rs. Animated GIFs decode natively. To send a video, see [multimodal input](/guides/models/multimodal-input/#video).

## You need FFmpeg on the server PATH

Run this in the same shell, service, or container that starts mistral.rs:

```bash
ffmpeg -version
```

A working interactive-shell installation does not guarantee that systemd, launchd, Docker, or a Windows service can find it.

## Install FFmpeg

The [installer](/reference/installer/) offers FFmpeg through Customize, `--with-ffmpeg` on macOS/Linux, or `-WithFFmpeg` on Windows. Or install it separately:

| Platform | Installation |
| --- | --- |
| Debian / Ubuntu | `sudo apt update` then `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| macOS / Homebrew | `brew install ffmpeg` |
| Windows | Install a build from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `bin` directory to PATH. |

For a Debian-based container:

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

## Supported formats

Any container the installed FFmpeg can decode, such as MP4, MOV, MKV, or WebM. The selected model must also support video. Soundtracks are not extracted automatically.

## Troubleshooting

If decoding fails, check FFmpeg from the server environment. For HTTP requests, send a public URL or inline data; local `file://` URLs are not accepted. Local paths work through the CLI and native SDKs.
