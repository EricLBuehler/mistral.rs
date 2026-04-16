---
title: Install and deploy
description: Platform-specific install steps, Docker, and what to check before running in production.
---

The fastest way to install mistral.rs is the install script we show in [Tutorial 1](/mistral.rs/tutorials/01-install-and-run/). It works on Linux, macOS, and Windows, detects your accelerator, and builds with the right feature flags automatically. Most people should start there.

These guides cover the cases where the script is not enough. That includes CUDA setups with specific driver versions, macOS where you want flash-attention support, and Windows with WSL. They also cover building from source when you want a specific set of features, and the Docker images we publish for deployment.

## Which one do I need?

| Situation | Guide |
|---|---|
| Linux with an NVIDIA GPU | [Linux with CUDA](/mistral.rs/guides/install/linux-cuda/) |
| Apple Silicon Mac | [macOS with Metal](/mistral.rs/guides/install/macos-metal/) |
| Windows (native or WSL) | [Windows](/mistral.rs/guides/install/windows/) |
| I want to build from source | [Build from source](/mistral.rs/guides/install/from-source/) |
| I want a container | [Docker](/mistral.rs/guides/deploy/docker/) |
| I want to go to production | [Production checklist](/mistral.rs/guides/deploy/production-checklist/) |

If you are not sure which accelerator combination applies to your hardware, the [cargo features reference](/mistral.rs/reference/cargo-features/) has a table that maps GPU generations to feature flags.
