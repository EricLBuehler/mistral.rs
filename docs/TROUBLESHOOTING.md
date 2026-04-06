# Troubleshooting

Common issues and solutions for mistral.rs.

## Debug Mode

Enable debug mode for more information:

```bash
MISTRALRS_DEBUG=1 mistralrs run -m <model>
```

Debug mode causes:
- If loading a GGUF or GGML model, outputs a file containing the names, shapes, and types of each tensor:
  - `mistralrs_gguf_tensors.txt` or `mistralrs_ggml_tensors.txt`
- Increased logging verbosity

## System Diagnostics

Run the built-in diagnostics tool:

```bash
mistralrs doctor
```

This checks your system configuration and reports any issues.

## Common Issues

### CUDA Issues

**Setting the CUDA compiler path:**
- Set the `NVCC_CCBIN` environment variable during build

**Error: `recompile with -fPIE`:**
- Some Linux distributions require compiling with `-fPIE`
- Set during build: `CUDA_NVCC_FLAGS=-fPIE cargo build --release --features cuda`

**Error: `CUDA_ERROR_NOT_FOUND` or symbol not found:**
- For non-quantized models, specify the data type to load and run in
- Use one of `f32`, `f16`, `bf16` or `auto` (auto chooses based on device)
- Example: `mistralrs run -m <model> -d auto`

**Minimum CUDA compute capability:**
- The minimum supported CUDA compute cap is **5.3**
- Set a specific compute cap with: `CUDA_COMPUTE_CAP=80 cargo build --release --features cuda`

### Metal Issues (macOS)

**Metal not found (error: unable to find utility "metal"):**

1. Install Xcode:
   ```bash
   xcode-select --install
   ```

2. Set the active developer directory:
   ```bash
   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
   ```

**error: cannot execute tool 'metal' due to missing Metal toolchain**

1. Install Metal Toolchain:
   ```bash
   xcodebuild -downloadComponent MetalToolchain
   ```

**Disabling Metal kernel precompilation:**
- By default, Metal kernels are precompiled during build time for better performance
- To skip precompilation (useful for CI or when Metal is not needed):
  ```bash
  MISTRALRS_METAL_PRECOMPILE=0 cargo build --release --features metal
  ```

### Memory Issues

**Disabling mmap loading:**
- Set `MISTRALRS_NO_MMAP=1` to disable memory-mapped file loading
- Forces all tensor data into memory
- Useful if you're seeing mmap-related errors

**Out of memory errors:**
- Try using quantization: `--isq q4k` or `--isq q8_0`
- Use device mapping to offload layers: `-n 0:16;cpu:16`
- Reduce context length with PagedAttention: `--pa-context-len 4096`

### Model Loading Issues

**Model type not auto-detected:**
- If auto-detection fails, please [raise an issue](https://github.com/EricLBuehler/mistral.rs/issues)
- You can manually specify the architecture if needed

**Chat template issues:**
- Templates are usually auto-detected
- Override with: `-c /path/to/template.jinja`
- See [Chat Templates](CHAT_TOK.md) for details

### Python SDK Installation Issues

**maturin build failures (missing system libraries):**

Building the Python SDK from source requires system libraries. If the build fails with errors about missing headers or libraries, install the prerequisites:

Ubuntu/Debian:
```bash
sudo apt install libssl-dev pkg-config
```

macOS:
```bash
brew install openssl pkg-config
```

You also need a working Rust toolchain. Install it from https://rustup.rs/ if you have not already.

**Wheel not found for platform:**

Pre-built wheels are only available for supported platform combinations. Check the [Python Installation](PYTHON_INSTALLATION.md) page for the full support matrix. If no wheel exists for your platform, build from source:

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs/mistralrs-pyo3
python -m venv .venv && source .venv/bin/activate
pip install maturin[patchelf]
maturin develop -r --features <your-features>
```

**CUDA version mismatch between pip package and system:**

The `mistralrs-cuda` pip package is compiled against a specific CUDA toolkit version. If your system has a different CUDA version, you may see errors like `libcudart.so.XX: cannot open shared object file` or symbol-not-found crashes at runtime. Solutions:

1. Install the matching CUDA toolkit version, or
2. Build from source against your local CUDA installation:
   ```bash
   maturin develop -r --features "cuda flash-attn cudnn"
   ```

### Out of Memory (OOM)

**OOM during model loading:**

The model itself may not fit into GPU memory at full precision. Reduce memory usage by quantizing on the fly with In-Situ Quantization (ISQ):

```bash
mistralrs run -m <model> --isq q4k
```

Alternatively, load a pre-quantized model in GGUF or UQFF format, which are already compressed on disk:

```bash
# GGUF
mistralrs run --format gguf -m <repo> -f <file.gguf>

# UQFF (generate first with `mistralrs quantize`, then load)
mistralrs run --format uqff -m <model> -f <file.uqff>
```

You can also offload some layers to CPU with device mapping:

```bash
mistralrs run -m <model> -n "0:16;cpu:16"
```

**OOM during inference:**

If the model loads successfully but you hit OOM while generating, the KV cache is likely consuming too much memory. When PagedAttention is enabled (the default on CUDA), it pre-allocates GPU memory for the KV cache at startup. Reduce this allocation:

```bash
# Limit KV cache to a specific context length
mistralrs serve -m <model> --pa-context-len 4096

# Or set a fixed memory budget in MB
mistralrs serve -m <model> --pa-memory-mb 2048

# Or reduce the fraction of free GPU memory used (default is 0.90)
mistralrs serve -m <model> --pa-memory-fraction 0.5
```

Reducing the maximum number of concurrent sequences also lowers peak memory usage:

```bash
mistralrs serve -m <model> --max-seqs 8
```

If none of these help, disable PagedAttention entirely to use a dynamically allocated cache:

```bash
mistralrs serve -m <model> --no-paged-attn
```

**OOM with multiple models:**

When running multiple models via the [multi-model](multi_model/overview.md) configuration, each loaded model consumes GPU memory. Unload models you are not actively using, or configure auto-reload so that only one model is resident at a time.

### Model Output Issues

**Garbage or repetitive output:**

This is usually a chat template mismatch. The model expects a specific prompt format and produces nonsense when it receives the wrong one. Try providing an explicit Jinja template:

```bash
mistralrs run -m <model> --jinja-explicit chat_templates/<template>.jinja
```

Browse the `chat_templates/` directory for templates matching your model family. See [Chat Templates](CHAT_TOK.md) for details.

**Model produces tool calls when not expected:**

Some models are fine-tuned for tool use and may emit tool-call JSON even when no tools are provided. To suppress this, set `tool_choice` to `"none"` in your request:

```json
{
  "tool_choice": "none",
  "messages": [...]
}
```

**Thinking tags appearing in output:**

Models that support thinking mode (such as Qwen3 and DeepSeek) emit `<think>...</think>` blocks containing chain-of-thought reasoning. This is expected behavior. To disable thinking entirely, pass `--thinking false`:

```bash
mistralrs run -m Qwen/Qwen3-4B --thinking false
```

If you omit the `--thinking` flag, the chat template's default behavior applies. See [CLI Reference](CLI.md#thinking-mode) for details.

### Docker Issues

**GPU not available inside container:**

Docker does not expose GPUs by default. You must pass the `--gpus all` flag (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --gpus all -p 1234:1234 ghcr.io/ericlbuehler/mistral.rs:latest \
  serve -m Qwen/Qwen3-4B
```

If you see `could not select device driver "" with capabilities: [[gpu]]`, install the NVIDIA Container Toolkit and restart Docker.

**Permission denied on Hugging Face cache:**

The container needs a writable directory for downloading model files. Mount a host volume for the HF cache:

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 1234:1234 \
  ghcr.io/ericlbuehler/mistral.rs:latest \
  serve -m Qwen/Qwen3-4B
```

This also avoids re-downloading models every time you start a new container.

### Authentication Issues

**"Gated model" or "access denied" errors:**

Some models on Hugging Face (such as Llama and Mistral) require you to accept a license agreement and authenticate. First, accept the license on the model's Hugging Face page, then authenticate:

```bash
# Interactive login (saves token to ~/.cache/huggingface/token)
mistralrs login

# Or provide the token directly
mistralrs login --token hf_xxxxxxxxxxxxx

# Or pass a token via environment variable at runtime
mistralrs run -m <model> --token-source env:HF_TOKEN
```

**"Model not found" errors:**

1. Double-check the model ID for typos (e.g., `mistralai/Mistral-7B-Instruct-v0.1`, not `mistral/Mistral-7B`)
2. Verify Hugging Face connectivity by running `mistralrs doctor` -- it tests the connection and token validity
3. If you are behind a corporate proxy or firewall, ensure `https://huggingface.co` is reachable
4. Check whether the model has been renamed or moved on Hugging Face

## Getting Help

If you're still stuck:

- [Discord](https://discord.gg/SZrecqK8qw) - Community support
- [Matrix](https://matrix.to/#/#mistral.rs:matrix.org) - Alternative chat
- [GitHub Issues](https://github.com/EricLBuehler/mistral.rs/issues) - Bug reports and feature requests

When reporting issues, please include:
1. Output of `mistralrs doctor`
2. Full error message
3. Command you ran
4. Hardware (GPU model, OS)
