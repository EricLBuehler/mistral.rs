---
title: Install on Windows
description: Install mistral.rs on Windows, either natively or through WSL. Which one is right depends on what you want to run.
sidebar:
  order: 3
---

Two options exist on Windows: native install or WSL2 with Ubuntu. The install script supports both.

**Native Windows**, for running mistral.rs as a Windows process, integration with Windows-specific tooling, or when WSL is not already configured.

**WSL2**, for existing WSL development environments, Linux-only features (systemd services, Docker, shell scripting), or compatibility with Linux instructions.

The engine is identical in both. GPU support works in both; CUDA drivers for WSL ship with recent NVIDIA Windows drivers.

## Native Windows

From PowerShell:

```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

The script detects CUDA and selects features accordingly. To install manually:

```powershell
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

Prerequisites:

- Rust 1.88+ from [rustup.rs](https://rustup.rs). The `rustup-init.exe` installer can install Visual Studio build tools on request.
- The CUDA toolkit (NVIDIA GPU only). Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- Visual Studio 2022 Build Tools, if not installed by `rustup-init.exe`.
- FFmpeg for video input. See [Set up video input](/mistral.rs/guides/models/video-setup/) for the Windows `PATH` requirement.

Native Windows builds do not have full feature parity with Linux. Flash attention works on modern GPUs, but ring-backend distributed inference and some other experimental features are Linux-only. Use WSL for those.

## WSL2 with Ubuntu

Ensure WSL has GPU access:

1. From an administrator PowerShell: `wsl --install -d Ubuntu`.
2. Verify the NVIDIA driver supports WSL CUDA (recent drivers do).
3. From inside WSL Ubuntu, `nvidia-smi` should list the GPU.

Then follow the [Linux with CUDA](/mistral.rs/guides/install/linux-cuda/) guide inside WSL.

For video input inside WSL, install FFmpeg inside the WSL distribution, not only on the Windows host.

CUDA behavior under WSL matches native Linux, including flash-attention and flash-attn-v3. Throughput is within single-digit percentage points of native Linux.

## Verifying the install

```powershell
mistralrs doctor
```

Or inside WSL:

```bash
mistralrs doctor
```

Both list the GPU and compiled features.

## Common Windows-specific issues

PowerShell execution policy can block the install script. If `Invoke-RestMethod` is refused, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Long paths cause `cargo build` failures. Enable long path support: `git config --global core.longpaths true`.

The Hugging Face cache defaults to `%USERPROFILE%\.cache\huggingface`. Redirect it via the `HF_HOME` environment variable when the home drive is space-constrained.
