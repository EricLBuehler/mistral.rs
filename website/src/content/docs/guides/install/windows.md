---
title: Install on Windows
description: Install mistral.rs on Windows, either natively or through WSL. Which one is right depends on what you want to run.
sidebar:
  order: 3
---

Windows has two reasonable paths: a native install, or WSL2 with an Ubuntu distribution inside it. The install script supports both. Which one you want depends on what kind of workflow you are planning.

**Use a native Windows install** if you want to run the binary as a normal Windows process, integrate it with Windows-specific tooling, or if you do not already have WSL set up.

**Use WSL2** if you already develop inside WSL, if you need any of the features that are easier on Linux (systemd services, Docker, shell scripting), or if you are porting instructions from a Linux tutorial and do not want to translate them.

The engine itself is the same either way. GPU support works in both; CUDA drivers for WSL are maintained by NVIDIA and come with recent Windows driver releases.

## Native Windows

From PowerShell:

```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

This script detects whether you have CUDA installed and picks features accordingly. If the detection guesses wrong, install manually with:

```powershell
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

Prerequisites the script assumes you have installed:

- Rust 1.88+ from [rustup.rs](https://rustup.rs). The `rustup-init.exe` installer includes Visual Studio build tools if you tell it to.
- The CUDA toolkit, if you have an NVIDIA GPU. Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- Visual Studio 2022 Build Tools, if you did not install them as part of Rust setup.

Native Windows builds do not have feature parity with Linux. Flash attention works on modern GPUs, but some of the more experimental features (ring-backend distributed inference in particular) are Linux-only. If you need those, use WSL.

## WSL2 with Ubuntu

First, make sure WSL is set up with GPU access:

1. On Windows, install WSL2: `wsl --install -d Ubuntu` from an administrator PowerShell.
2. Make sure your NVIDIA driver is recent enough to include the WSL CUDA support (anything shipped in the last couple of years works).
3. From inside WSL Ubuntu: `nvidia-smi` should list your GPU. If it does, CUDA is working.

Then follow the [Linux with CUDA](/mistral.rs/guides/install/linux-cuda/) guide from inside WSL.

Under WSL, everything mistral.rs does from a CUDA standpoint works identically to a native Linux box, including flash-attention and flash-attn-v3. Throughput is within single-digit percentage points of native Linux performance in our testing.

## Verifying the install

```powershell
mistralrs doctor
```

Or inside WSL:

```bash
mistralrs doctor
```

Both should list your GPU and the features that were compiled in.

## Common Windows-specific issues

PowerShell execution policy sometimes blocks the install script. If `Invoke-RestMethod` refuses to run, `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell temporarily opens it up.

Long paths can bite you on Windows: if the repository clones fine but `cargo build` complains about paths, enable long path support with `git config --global core.longpaths true`.

The Hugging Face cache defaults to `%USERPROFILE%\.cache\huggingface` on Windows. If your home directory is on a drive with limited space, redirect the cache with the `HF_HOME` environment variable before running `mistralrs`.
