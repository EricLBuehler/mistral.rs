# Frequently Asked Questions

## General

### How much VRAM do I need?

It depends on the model, quantization level, context length, and batch size. The best way to find out is to let mistral.rs analyze your hardware:

```bash
mistralrs tune -m <model>
```

This shows a table of quantization options with estimated memory usage, context headroom, and quality trade-offs — specific to your GPU. Quantization (`--isq 4`) typically reduces memory by 3-4x compared to FP16.

### Which quantization should I use?

- **Just want it to work?** Use `--isq 4`. mistral.rs picks the best format for your hardware.
- **Have plenty of VRAM?** Use `--isq 8` for near-lossless quality.
- **Want the fastest loading?** Use a pre-quantized [UQFF](UQFF.md) or [GGUF](QUANTS.md#using-a-gguf-quantized-model) model — no quantization at load time.
- **Need per-layer control?** Use a [topology file](TOPOLOGY.md) to set different quantization per layer.
- **On Apple Silicon?** `--isq 4` uses AFQ4, which is optimized for Metal.

See the [Quantization Overview](QUANTS.md) for all options.

### What's the difference between ISQ, GGUF, and UQFF?

- **ISQ** quantizes at load time. Any HF model works, but loading is slower because quantization happens on the fly.
- **GGUF** is a pre-quantized format from the llama.cpp ecosystem. Download a GGUF file and load it directly.
- **UQFF** is mistral.rs's own pre-quantized format. Supports more quantization types than GGUF (HQQ, FP8, AFQ, etc.) and loads instantly.

### Can I use models from local files?

Yes. Pass a local path instead of a Hugging Face model ID:

```bash
mistralrs run -m /path/to/model
```

### How do I use gated models (Llama, etc.)?

1. Accept the model license on Hugging Face
2. Authenticate:
   ```bash
   mistralrs login
   ```
3. Run normally:
   ```bash
   mistralrs run -m meta-llama/Llama-3.2-3B-Instruct
   ```

Alternatively, use `--token-source env:HF_TOKEN` with a Hugging Face token in your environment.

### How do I update mistral.rs?

Re-run the install script — it will build the latest version:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

Or if installed via cargo:
```bash
cargo install mistralrs-cli --features "<your-features>" --force
```

### Does mistral.rs work on WSL?

Yes. WSL2 with CUDA support works well. Install the [NVIDIA CUDA toolkit for WSL](https://developer.nvidia.com/cuda-downloads) and build with `--features cuda`.

### Is Windows supported?

Yes, but **WSL2 is recommended** for the best experience. Native Windows builds work but PagedAttention is only available on Unix-like platforms.

## Performance

### Why is the first request slow?

mistral.rs performs a warmup run when loading a model — it sends a short dummy request to initialize CUDA kernels and caches. You'll see "Beginning dummy run..." and "Dummy run completed" in the logs. After this, subsequent requests are much faster.

### How do I increase throughput?

1. Enable PagedAttention (on by default for CUDA)
2. Increase `--max-seqs` for more concurrent batching (default: 32)
3. Use FlashAttention (compile with `--features flash-attn`)
4. Use quantization to free VRAM for larger batches

See the [Performance Guide](PERFORMANCE.md) for detailed optimization recipes.

### PagedAttention is using too much VRAM

By default, PagedAttention allocates 90% of free VRAM for the KV cache. Reduce it:

```bash
# Allocate for a specific context length
mistralrs serve -m <model> --pa-context-len 4096

# Or set a memory cap
mistralrs serve -m <model> --pa-memory-mb 2048

# Or use a fraction
mistralrs serve -m <model> --pa-memory-fraction 0.5
```

### How do I reduce Time to First Token (TTFT)?

- Compile with FlashAttention (`--features flash-attn`) to accelerate prefill
- Enable prefix caching (`--prefix-cache-n 16`, on by default) to reuse KV cache across turns
- Use quantization to reduce memory bandwidth pressure

## Compatibility

### Can I use OpenAI client libraries?

Yes. mistral.rs provides an OpenAI-compatible HTTP API. Use the standard `openai` Python package:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1/", api_key="foobar")
```

The `api_key` can be any non-empty string — mistral.rs doesn't validate it.

### Can I use LangChain or LlamaIndex?

Yes. Point them at your mistral.rs server as an OpenAI-compatible endpoint:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:1234/v1/", api_key="foobar")
```

### Which models are supported?

See [Supported Models](SUPPORTED_MODELS.md) for the complete list. Architecture is auto-detected — just point mistral.rs at the model and it works.

### Can I run multiple models at once?

Yes. Use a TOML config file with multiple `[[models]]` entries:

```bash
mistralrs from-config --file config.toml
```

See [Multi-Model Support](multi_model/overview.md) for details.
