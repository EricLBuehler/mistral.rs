# Installation Guide

## Quick Install (Recommended)

The install script automatically detects your hardware (CUDA, Metal, MKL) and builds with optimal features.

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

## Prerequisites

1. Install required packages:
   - OpenSSL: `sudo apt install libssl-dev` (Ubuntu)
   - pkg-config (Linux only): `sudo apt install pkg-config`

2. Install Rust from https://rustup.rs/
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. (Optional) Set up HuggingFace authentication:
   ```bash
   mistralrs login
   ```
   Or use `huggingface-cli login` as documented [here](https://huggingface.co/docs/huggingface_hub/en/installation).

## Supported Accelerators

| Accelerator              | Feature Flag  | Additional Flags       |
|--------------------------|---------------|------------------------|
| NVIDIA GPUs (CUDA)       | `cuda`        | `flash-attn`, `flash-attn-v3`, `cudnn`  |
| Apple Silicon GPU (Metal)| `metal`       |                        |
| CPU (Intel)              | `mkl`         |                        |
| CPU (Apple Accelerate)   | `accelerate`  |                        |
| Generic CPU (ARM/AVX)    | _none_        | ARM NEON / AVX enabled by default |

> **Note for Linux users:** The `metal` feature is macOS-only. Use `--features "cuda flash-attn cudnn"` for NVIDIA GPUs or `--features mkl` for Intel CPUs instead of `--all-features`.

## Feature Detection

Determine which features to enable based on your hardware:

| Hardware | Features |
|----------|----------|
| NVIDIA GPU (Ampere+, compute >=80) | `cuda cudnn flash-attn` |
| NVIDIA GPU (Hopper, compute 90) | `cuda cudnn flash-attn flash-attn-v3` |
| NVIDIA GPU (older) | `cuda cudnn` |
| Apple Silicon (macOS) | `metal accelerate` |
| Intel CPU with MKL | `mkl` |
| CPU only | (no features needed) |

## Install from crates.io

```bash
cargo install mistralrs-cli --features "<your-features>"
```

Example:
```bash
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

## Build from Source

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo install --path mistralrs-cli --features "<your-features>"
```

Example:
```bash
cargo build --release --features "cuda flash-attn cudnn"
```

## Docker

Docker images are available for quick deployment:

```bash
docker pull ghcr.io/ericlbuehler/mistral.rs:latest
docker run --gpus all -p 1234:1234 ghcr.io/ericlbuehler/mistral.rs:latest \
  serve -m Qwen/Qwen3-4B
```

[Docker images on GitHub Container Registry](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs)

Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/

## Python SDK

Install the Python package:

```bash
pip install mistralrs-cuda    # For NVIDIA GPUs
pip install mistralrs-metal   # For Apple Silicon
pip install mistralrs-mkl     # For Intel CPUs
pip install mistralrs         # CPU-only
```

- [Full installation instructions](PYTHON_INSTALLATION.md)
- [SDK documentation](PYTHON_SDK.md)

## Verify Installation

After installation, verify everything works:

```bash
# Check CLI is installed
mistralrs --help

# Run system diagnostics
mistralrs doctor

# Test with a small model
mistralrs run -m Qwen/Qwen3-0.6B
```

## Getting Models

### From Hugging Face Hub (Default)

Models download automatically from Hugging Face Hub:

```bash
mistralrs run -m meta-llama/Llama-3.2-3B-Instruct
```

For gated models, authenticate first:
```bash
mistralrs login
# Or: mistralrs run --token-source env:HF_TOKEN -m <model>
```

### From Local Files

Pass a path to a downloaded model:

```bash
mistralrs run -m /path/to/model
```

### Running GGUF Models

```bash
mistralrs run --format gguf -m author/model-repo -f model-quant.gguf
```

Specify tokenizer if needed:
```bash
mistralrs run --format gguf -m author/model-repo -f file.gguf -t author/official-tokenizer
```

## Next Steps

- [CLI Reference](CLI.md) - All commands and options
- [HTTP API](HTTP.md) - Run as an OpenAI-compatible server
- [Python SDK](PYTHON_SDK.md) - Python package documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
