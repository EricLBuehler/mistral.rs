# Python SDK Installation

## Quick Install from PyPI (Recommended)

Pre-built wheels are available for common platforms. Choose the package that matches your hardware:

| Hardware | Install Command |
|----------|----------------|
| **Recommended (auto-optimized)** | `pip install mistralrs` |
| NVIDIA GPUs (CUDA) | `pip install mistralrs-cuda` |
| Apple Silicon (Metal) | `pip install mistralrs-metal` |
| Apple Accelerate | `pip install mistralrs-accelerate` |
| Intel CPUs (MKL) | `pip install mistralrs-mkl` |

### Platform-Specific Optimizations

The `mistralrs` base package includes platform-specific optimizations:
- **macOS Apple Silicon**: Metal GPU support built-in
- **Linux/Windows x86_64**: Intel MKL optimizations built-in
- **Linux aarch64**: CPU-only (use `mistralrs-cuda` for GPU support)

All packages install the `mistralrs` Python module. The package suffix controls which accelerator features are enabled.

### Supported Platforms

| Package | Linux x86_64 | Linux aarch64 | Windows x86_64 | macOS aarch64 |
|---------|:------------:|:-------------:|:--------------:|:-------------:|
| mistralrs | MKL | CPU | MKL | Metal |
| mistralrs-cuda | CUDA | CUDA | CUDA | - |
| mistralrs-metal | - | - | - | Metal |
| mistralrs-accelerate | - | - | - | Accelerate |
| mistralrs-mkl | MKL | - | MKL | - |

**Python version**: 3.10+ (wheels use abi3 for forward compatibility)

```bash
# Example: Install with CUDA support
pip install mistralrs-cuda -v
```

## Build from Source

Building from source gives you access to the latest features and allows customization of build options.

### Prerequisites

1. **Install system packages:**

   Ubuntu/Debian:
   ```bash
   sudo apt install libssl-dev pkg-config
   ```

   macOS:
   ```bash
   brew install openssl pkg-config
   ```

2. **Install Rust** from https://rustup.rs/:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. **(Optional) Set up HuggingFace authentication** for gated models:
   ```bash
   mkdir -p ~/.cache/huggingface
   echo "YOUR_HF_TOKEN" > ~/.cache/huggingface/token
   ```
   Or use `huggingface-cli login`.

### Build Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EricLBuehler/mistral.rs.git
   cd mistral.rs/mistralrs-pyo3
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install maturin** (Rust + Python build tool):
   ```bash
   pip install maturin[patchelf]
   ```

4. **Build and install:**
   ```bash
   maturin develop -r --features <your-features>
   ```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA GPU support |
| `flash-attn` | Flash Attention (CUDA, Ampere+) |
| `flash-attn-v3` | Flash Attention v3 (CUDA, Hopper) |
| `cudnn` | cuDNN optimizations |
| `metal` | Apple Silicon GPU (macOS only) |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL |

Example with CUDA and Flash Attention:
```bash
maturin develop -r --features "cuda flash-attn cudnn"
```

## Verify Installation

```python
import mistralrs
print(mistralrs.__version__)
```

Quick test:
```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-0.6B"),
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50,
    )
)
print(response.choices[0].message.content)
```

## Next Steps

- [SDK Documentation](PYTHON_SDK.md) - Full SDK reference
- [Examples](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python) - Python examples
- [Cookbook](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/cookbook.ipynb) - Interactive tutorial

---

## Building Wheels (For Maintainers)

### Prerequisites

1. **Maturin**: `pip install maturin[patchelf]`
2. **Docker** (Linux only, for manylinux builds): Install from docker.com
3. **Twine** (for uploads): `pip install twine`

### Build Script Usage

The `scripts/build_wheels.py` script auto-detects the platform and builds appropriate wheels.

```bash
# Navigate to repository root
cd mistral.rs

# List what can be built on this machine
python scripts/build_wheels.py --list

# Build all supported packages
python scripts/build_wheels.py --all

# Build specific packages
python scripts/build_wheels.py -p mistralrs mistralrs-cuda

# Specify output directory
python scripts/build_wheels.py --all -o ./dist
```

### Upload Script Usage

```bash
# Dry run to verify
python scripts/upload_wheels.py ./wheels --dry-run

# Upload to TestPyPI first
python scripts/upload_wheels.py ./wheels --test --token $TESTPYPI_TOKEN

# Upload to PyPI
python scripts/upload_wheels.py ./wheels --token $PYPI_TOKEN

# Upload specific packages only
python scripts/upload_wheels.py ./wheels -p mistralrs-cuda
```

### Build Workflow

**Box 1 (Linux aarch64 + CUDA):**
```bash
python scripts/build_wheels.py -p mistralrs mistralrs-cuda
```

**Box 2 (Linux/Windows x86_64 + CUDA + MKL):**
```bash
# Linux
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl
# Windows
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl
```

**Box 3 (macOS aarch64):**
```bash
python scripts/build_wheels.py --all
```

**Central Upload:**
```bash
# Collect wheels from all boxes to a single directory
python scripts/upload_wheels.py ./all_wheels --token $PYPI_TOKEN
```
