# Cargo Features Reference

This document provides a complete reference for all cargo features available in mistral.rs.

## Quick Reference

| Feature | Description | Platform | Requires |
|---------|-------------|----------|----------|
| `cuda` | NVIDIA GPU acceleration | Linux, Windows | CUDA toolkit |
| `cudnn` | NVIDIA cuDNN backend | Linux, Windows | `cuda`, cuDNN |
| `flash-attn` | FlashAttention V2 | Linux, Windows | `cuda`, CC >= 8.0 |
| `flash-attn-v3` | FlashAttention V3 | Linux, Windows | `cuda`, CC >= 9.0 |
| `metal` | Apple GPU acceleration | macOS | - |
| `accelerate` | Apple CPU acceleration | macOS | - |
| `mkl` | Intel MKL acceleration | Linux, Windows | Intel MKL |
| `nccl` | Multi-GPU (NVIDIA NCCL) | Linux | `cuda`, NCCL |
| `ring` | Multi-GPU/node (TCP ring) | All | - |

## GPU Acceleration Features

### `cuda`

Enables NVIDIA GPU acceleration via CUDA. This is the primary feature for running on NVIDIA GPUs.

**Requirements:**
- NVIDIA GPU
- CUDA toolkit installed
- Linux or Windows (WSL supported)

**Usage:**
```bash
cargo build --release --features cuda
cargo install mistralrs-cli --features cuda
```

**What it enables:**
- GPU tensor operations via CUDA
- PagedAttention on CUDA devices
- Quantized inference on GPU

---

### `cudnn`

Enables NVIDIA cuDNN for optimized neural network primitives. Provides faster convolutions and other operations.

**Requirements:**
- `cuda` feature
- cuDNN library installed

**Usage:**
```bash
cargo build --release --features "cuda cudnn"
```

---

### `flash-attn`

Enables FlashAttention V2 for faster attention computation. Significantly reduces memory usage and improves throughput.

**Requirements:**
- `cuda` feature (automatically enabled)
- GPU with compute capability >= 8.0 (Ampere or newer)

**Compatible GPUs:**

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|--------------|
| Ampere | 8.0, 8.6 | RTX 30 series, A100, A40 |
| Ada Lovelace | 8.9 | RTX 40 series, L40S |
| Blackwell | 10.0, 12.0 | RTX 50 series |

**Usage:**
```bash
cargo build --release --features "cuda flash-attn cudnn"
```

> Note: FlashAttention V2 and V3 are mutually exclusive. Do not enable both.

---

### `flash-attn-v3`

Enables FlashAttention V3 for Hopper architecture GPUs. Provides additional performance improvements over V2 on supported hardware.

**Requirements:**
- `cuda` feature (automatically enabled)
- GPU with compute capability >= 9.0 (Hopper)

**Compatible GPUs:**

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|--------------|
| Hopper | 9.0 | H100, H800 |

**Usage:**
```bash
cargo build --release --features "cuda flash-attn-v3 cudnn"
```

> Note: FlashAttention V2 and V3 are mutually exclusive. Do not enable both.

---

### `metal`

Enables Apple Metal GPU acceleration for macOS devices.

**Requirements:**
- macOS with Apple Silicon or AMD GPU
- macOS only (not available on Linux)

**Usage:**
```bash
cargo build --release --features metal
```

**What it enables:**
- GPU tensor operations via Metal
- PagedAttention on Metal devices (opt-in via `--paged-attn`)
- Quantized inference on Apple GPUs

> Note: PagedAttention is disabled by default on Metal. Enable with `--paged-attn` flag.

---

## CPU Acceleration Features

### `accelerate`

Enables Apple's Accelerate framework for optimized CPU operations on macOS.

**Requirements:**
- macOS

**Usage:**
```bash
cargo build --release --features accelerate
# Or combined with Metal:
cargo build --release --features "metal accelerate"
```

---

### `mkl`

Enables Intel Math Kernel Library (MKL) for optimized CPU operations.

**Requirements:**
- Intel MKL installed
- Intel CPU recommended (works on AMD but Intel-optimized)

**Usage:**
```bash
cargo build --release --features mkl
```

---

## Distributed Inference Features

### `nccl`

Enables multi-GPU distributed inference using NVIDIA NCCL (NVIDIA Collective Communications Library). Implements tensor parallelism for splitting large models across multiple GPUs.

**Requirements:**
- `cuda` feature (automatically enabled)
- Multiple NVIDIA GPUs
- NCCL library
- World size must be a power of 2 (1, 2, 4, 8, etc.)

**Usage:**
```bash
cargo build --release --features "cuda nccl"

# Run with specific GPU count
MISTRALRS_MN_LOCAL_WORLD_SIZE=2 mistralrs serve -m Qwen/Qwen3-30B-A3B-Instruct
```

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Number of GPUs to use (defaults to all) |
| `MISTRALRS_NO_NCCL=1` | Disable NCCL and use device mapping instead |

**Multi-node setup** requires additional environment variables. See [NCCL documentation](DISTRIBUTED/NCCL.md) for details.

> Note: When NCCL is enabled, automatic device mapping is disabled.

---

### `ring`

Enables distributed tensor-parallel inference using a TCP-based ring topology. Works across multiple machines without requiring NCCL.

**Requirements:**
- World size must be a power of 2 (2, 4, 8, etc.)
- TCP ports must be open between nodes

**Usage:**
```bash
cargo build --release --features ring

# Configure via JSON file
export RING_CONFIG=path/to/ring_config.json
mistralrs serve -m model-id
```

**Configuration:**

Create a JSON configuration file for each process:

```json
{
  "master_ip": "0.0.0.0",
  "master_port": 1234,
  "port": 12345,
  "right_port": 12346,
  "rank": 0,
  "world_size": 2
}
```

| Field | Description |
|-------|-------------|
| `master_ip` | IP address for master node |
| `master_port` | Port for master node |
| `port` | Local port for incoming connections |
| `right_port` | Port of right neighbor in ring |
| `right_ip` | IP of right neighbor (optional, defaults to localhost) |
| `rank` | Process rank (0 to world_size-1) |
| `world_size` | Total number of processes (must be power of 2) |

See [Ring documentation](DISTRIBUTED/RING.md) for detailed setup instructions.

---

## Feature Combinations

### Recommended Combinations by Hardware

| Hardware | Recommended Features |
|----------|---------------------|
| NVIDIA Ampere+ (RTX 30/40, A100) | `cuda cudnn flash-attn` |
| NVIDIA Hopper (H100) | `cuda cudnn flash-attn-v3` |
| NVIDIA older GPUs | `cuda cudnn` |
| Apple Silicon | `metal accelerate` |
| Intel CPU | `mkl` |
| Generic CPU | (no features needed) |
| Multi-GPU NVIDIA | `cuda cudnn flash-attn nccl` |
| Multi-node/cross-platform | `ring` (plus GPU features) |

### Installation Examples

```bash
# NVIDIA GPU with all optimizations
cargo install mistralrs-cli --features "cuda cudnn flash-attn"

# Apple Silicon
cargo install mistralrs-cli --features "metal accelerate"

# Intel CPU
cargo install mistralrs-cli --features "mkl"

# Multi-GPU NVIDIA setup
cargo install mistralrs-cli --features "cuda cudnn flash-attn nccl"

# Build from source with CUDA
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo build --release --features "cuda cudnn flash-attn"
```

---

## Internal Features

These features are primarily for library development and are not typically used directly:

| Feature | Description |
|---------|-------------|
| `pyo3_macros` | Python bindings support (used by mistralrs-pyo3) |
| `utoipa` | OpenAPI documentation generation |

---

## Python Package Features

The Python SDK is distributed as separate packages with features pre-configured:

| Package | Equivalent Features |
|---------|---------------------|
| `mistralrs-cuda` | `cuda cudnn flash-attn` |
| `mistralrs-metal` | `metal accelerate` |
| `mistralrs-mkl` | `mkl` |
| `mistralrs` | CPU only |

```bash
pip install mistralrs-cuda    # NVIDIA GPUs
pip install mistralrs-metal   # Apple Silicon
pip install mistralrs-mkl     # Intel CPUs
pip install mistralrs         # Generic CPU
```

---

## Troubleshooting

### Diagnosing Issues

Use `mistralrs doctor` to diagnose your system configuration and verify features are working correctly:

```bash
mistralrs doctor
```

This command checks:
- Detected hardware (GPUs, CPU features)
- Installed libraries (CUDA, cuDNN, etc.)
- Feature compatibility
- Common configuration issues

### Feature not working

1. Run `mistralrs doctor` to check system configuration

2. Verify the feature is enabled in your build:
   ```bash
   cargo build --release --features "your-features" -v
   ```

3. Check hardware compatibility (especially for flash-attn)

4. Ensure required libraries are installed (CUDA, cuDNN, MKL, etc.)

### Conflicting features

- `flash-attn` and `flash-attn-v3` are mutually exclusive
- `metal` is macOS-only; don't use with `cuda`
- `nccl` requires `cuda`

### Build errors

- **CUDA not found**: Ensure CUDA toolkit is installed and `nvcc` is in PATH
- **MKL not found**: Install Intel oneAPI or standalone MKL
- **Metal errors on Linux**: Remove `metal` feature (macOS only)

See [Troubleshooting](TROUBLESHOOTING.md) for more solutions.
