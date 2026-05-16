---
title: Install
description: Platform-specific install steps and source builds.
---

The install script in [Tutorial 1](/mistral.rs/tutorials/01-install-and-run/) works on Linux, macOS, and Windows, detects the accelerator, and selects the matching feature flags. For manual installs, specific driver versions, or source builds, use the guides below.

## Install options

| Situation | Guide |
|---|---|
| Linux with an NVIDIA GPU | [Linux with CUDA](/mistral.rs/guides/install/linux-cuda/) |
| Apple Silicon Mac | [macOS with Metal](/mistral.rs/guides/install/macos-metal/) |
| Windows (native or WSL) | [Windows](/mistral.rs/guides/install/windows/) |
| Build from source | [Build from source](/mistral.rs/guides/install/from-source/) |

The [cargo features reference](/mistral.rs/reference/cargo-features/) maps GPU generations to feature flags. For containerised or production deployment, see the [Deploy guides](/mistral.rs/guides/deploy/).

Video input uses FFmpeg at runtime. The install commands and runtime checks are in [Set up video input](/mistral.rs/guides/models/video-setup/).
