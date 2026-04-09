# Getting Started with mistral.rs

mistral.rs is a fast, flexible LLM inference engine. It supports 40+ model families (text, vision, audio, speech, image generation, and embeddings) with an OpenAI-compatible API, Python SDK, and Rust SDK.

This tutorial takes you from zero to running models in about 5 minutes.

## Step 1: Install

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

The install script detects your hardware (CUDA, Metal, MKL) and builds with the right features automatically.

> For manual installation, Docker, or building from source, see the [full Installation guide](INSTALLATION.md).

## Step 2: Run Your First Model

```bash
mistralrs run -m Qwen/Qwen3-4B
```

That's it. mistral.rs auto-detects the model architecture, downloads weights from Hugging Face, and drops you into an interactive chat. Type a message and press Enter:

```
> What is Rust's ownership model?
Rust uses an ownership system where each value has a single owner...
> /exit
```

Use `/exit` to quit, `/clear` to reset the conversation, or `/help` for all commands.

## Step 3: Reduce Memory with Quantization

Most models load in BF16 by default. If you're short on VRAM, quantize at load time with `--isq`:

```bash
mistralrs run --isq 4 -m Qwen/Qwen3-4B
```

The `--isq 4` flag quantizes weights to 4-bit as they load, so the full model never needs to fit in memory. mistral.rs picks the best quantization format for your hardware automatically (AFQ on Metal, Q4K on CUDA/CPU).

> Not sure what settings to use? Run `mistralrs tune -m <model>` and it will analyze your hardware and recommend the best configuration.

## Step 4: Start an HTTP Server

Serve any model as an OpenAI-compatible API:

```bash
mistralrs serve --isq 4 -m Qwen/Qwen3-4B
```

The server starts on `http://localhost:1234`. Test it with curl:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or use the OpenAI Python client directly, no changes needed:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1/", api_key="foobar")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Add `--ui` to get a built-in web chat interface:

```bash
mistralrs serve --ui --isq 4 -m Qwen/Qwen3-4B
# Visit http://localhost:1234/ui
```

To override the built-in UI with your own files, create a `ui/` directory next to the `mistralrs` executable. Files placed there are served in place of the embedded UI.

### Auto-Discovery and Lazy Loading

Instead of specifying a single model with `-m`, you can point `--models-dir` at a directory of models. Each subdirectory is treated as a separate model (GGUF, GGML, or Safetensors/Plain format is auto-detected). Models are loaded lazily on first request and the directory is polled for changes, so adding or removing a model subdirectory is picked up automatically.

Combined with `--idle-timeout-secs`, models are automatically unloaded after a period of inactivity and reloaded on demand. This is useful for managing GPU memory when serving many models.

```bash
mistralrs serve --models-dir ./models --ui --idle-timeout-secs 1800
```

## Step 5: Use the Python SDK

```bash
pip install mistralrs        # CPU
pip install mistralrs-cuda   # NVIDIA GPU
pip install mistralrs-metal  # Apple Silicon
```

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

> For the Rust SDK, see the [Rust SDK guide](RUST_SDK.md).

## Step 6: Try a Multimodal Model

mistral.rs handles images, video, and audio out of the box:

```bash
# Describe an image
mistralrs run -m google/gemma-4-E4B-it --image photo.jpg -i "Describe this image"

# Analyze a video
mistralrs run -m google/gemma-4-E4B-it --video clip.mp4 -i "What happens here?"

# Process audio
mistralrs run -m google/gemma-4-E4B-it --audio recording.wav -i "Transcribe this"
```

In interactive mode, just paste file paths directly into your prompt. mistral.rs detects image, video, and audio files automatically.

## Step 7: Auto-Tune Your Setup

Let mistral.rs recommend the best configuration for your hardware:

```bash
mistralrs tune -m Qwen/Qwen3-4B
```

This shows a table of quantization options with estimated VRAM usage, context length headroom, and quality trade-offs. Add `--emit-config config.toml` to save the recommended settings:

```bash
mistralrs tune -m Qwen/Qwen3-4B --emit-config config.toml
mistralrs from-config -f config.toml
```

## Next Steps

| I want to... | Read... |
|---|---|
| Learn all CLI commands and options | [CLI Reference](CLI.md) |
| Optimize performance and memory | [Performance Guide](PERFORMANCE.md) |
| Build agents with tools and web search | [Agentic Features Guide](AGENTS.md) |
| Use the Python SDK in depth | [Python SDK](PYTHON_SDK.md) |
| Use the Rust SDK in depth | [Rust SDK](RUST_SDK.md) |
| Serve models over HTTP | [HTTP Server](HTTP.md) |
| Connect to MCP tool servers | [MCP Client](MCP/client.md) |
| See all supported models | [Supported Models](SUPPORTED_MODELS.md) |
| Choose a quantization method | [Quantization Overview](QUANTS.md) |
| Run multiple models at once | [Multi-Model Support](multi_model/overview.md) |
