---
title: Install and deploy
description: Platform-specific install steps, Docker, and what to check before running in production.
---

The install script in [Tutorial 1](/mistral.rs/tutorials/01-install-and-run/) works on Linux, macOS, and Windows, detects the accelerator, and selects the matching feature flags. For manual installs, specific driver versions, or source builds, use the guides below.

## Install options

| Situation | Guide |
|---|---|
| Linux with an NVIDIA GPU | [Linux with CUDA](/mistral.rs/guides/install/linux-cuda/) |
| Apple Silicon Mac | [macOS with Metal](/mistral.rs/guides/install/macos-metal/) |
| Windows (native or WSL) | [Windows](/mistral.rs/guides/install/windows/) |
| Build from source | [Build from source](/mistral.rs/guides/install/from-source/) |
| Container deployment | [Docker](/mistral.rs/guides/deploy/docker/) |
| Production deployment | [Production checklist](/mistral.rs/guides/deploy/production-checklist/) |

The [cargo features reference](/mistral.rs/reference/cargo-features/) maps GPU generations to feature flags.
