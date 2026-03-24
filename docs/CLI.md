# mistralrs CLI Reference

This is the comprehensive CLI reference for `mistralrs`. The CLI provides commands for interactive mode, HTTP server, builtin UI, quantization, and system diagnostics.

## Table of Contents

- [Commands](#commands)
  - [run](#run---interactive-mode): run model in interactive mode
  - [serve](#serve---http-server): start HTTP/MCP server and (optionally) the UI
  - [from-config](#from-config---toml-configuration): run from a [TOML configuration file](CLI_CONFIG.md)
  - [quantize](#quantize---uqff-generation): generate UQFF quantized model file
  - [tune](#tune---recommendations): recommend quantization + device mapping for a model
  - [doctor](#doctor---system-diagnostics): run system diagnostics and environment checks
  - [login](#login---huggingface-authentication): authenticate with HuggingFace Hub
  - [cache](#cache---model-management): manage the HuggingFace model cache
  - [bench](#bench---performance-benchmarking): run performance benchmarks
  - [completions](#completions---shell-completions): generate shell completions
- [Model Types](#model-types)
  - [auto](#auto)
  - [text](#text)
  - [vision](#vision)
  - [diffusion](#diffusion)
  - [speech](#speech)
  - [embedding](#embedding)
- [Features](#features)
  - [ISQ Quantization](#isq-quantization)
  - [UQFF Files](#uqff-files)
  - [PagedAttention](#pagedattention)
  - [Device Mapping](#device-mapping)
  - [LoRA and X-LoRA](#lora-and-x-lora)
  - [Chat Templates](#chat-templates)
  - [Web Search](#web-search)
  - [Thinking Mode](#thinking-mode)
- [Global Options](#global-options)
- [Interactive Commands](#interactive-commands)

---

## Commands

### run - Interactive Mode

Start a model in interactive mode for conversational use.

```bash
mistralrs run [MODEL_TYPE] -m <MODEL_ID> [OPTIONS]
```

Note: `MODEL_TYPE` is optional and defaults to `auto` if not specified. This allows a shorter syntax.

**Examples:**

```bash
# Run a text model interactively (shorthand - auto type is implied)
mistralrs run -m Qwen/Qwen3-4B

# Explicit auto type (equivalent to above)
mistralrs run -m Qwen/Qwen3-4B

# Run with thinking mode enabled
mistralrs run -m Qwen/Qwen3-4B --enable-thinking

# Run a vision model
mistralrs run -m google/gemma-3-4b-it
```

**Options:**

| Option | Description |
|--------|-------------|
| `--enable-thinking` | Enable thinking mode for models that support it |

The `run` command also accepts all [runtime options](#runtime-options).

---

### serve - HTTP Server

Start an HTTP server with OpenAI-compatible API endpoints.

```bash
mistralrs serve [MODEL_TYPE] -m <MODEL_ID> [OPTIONS]
```

Note: `MODEL_TYPE` is optional and defaults to `auto` if not specified.

**Examples:**

```bash
# Start server on default port 1234 (shorthand)
mistralrs serve -m Qwen/Qwen3-4B

# Explicit auto type (equivalent to above)
mistralrs serve -m Qwen/Qwen3-4B

# Start server with web UI
mistralrs serve -m Qwen/Qwen3-4B --ui

# Start server on custom port
mistralrs serve -m Qwen/Qwen3-4B -p 3000

# Start server with MCP support
mistralrs serve -m Qwen/Qwen3-4B --mcp-port 8081
```

**Server Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --port <PORT>` | `1234` | HTTP server port |
| `--host <HOST>` | `0.0.0.0` | Bind address |
| `--ui` | disabled | Serve built-in web UI at `/ui` |
| `--mcp-port <PORT>` | none | MCP protocol server port |
| `--mcp-config <PATH>` | none | MCP client configuration file |

The `serve` command also accepts all [runtime options](#runtime-options).

---

### quantize - UQFF Generation

Generate UQFF (Unified Quantized File Format) files from a model. Supports multiple quantization types in a single command.

```bash
mistralrs quantize <MODEL_TYPE> -m <MODEL_ID> --isq <LEVEL>[,<LEVEL>...] -o <OUTPUT>
```

**Examples:**

```bash
# Quantize to a single type (file output)
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k -o qwen3-4b-uqff/qwen3-4b-q4k.uqff

# Quantize to a single type (directory output, auto-named)
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k -o qwen3-4b-uqff/

# Quantize to multiple types at once (directory output)
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k,q8_0 -o qwen3-4b-uqff/

# Equivalent: repeated --isq flags
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k --isq q8_0 -o qwen3-4b-uqff/

# Quantize a vision model
mistralrs quantize -m google/gemma-3-4b-it --isq 4 -o gemma3-4b-uqff/

# Quantize with imatrix for better quality
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k --imatrix imatrix.dat -o qwen3-4b-uqff/qwen3-4b-q4k.uqff
```

When using directory output mode, the `quantize` command automatically:
- Generates a `README.md` model card with Hugging Face frontmatter and example commands
- Prints the `huggingface-cli upload` command to upload your UQFF to Hugging Face

**Quantize Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `-m, --model-id <ID>` | Yes | Model ID or local path |
| `--isq <LEVEL>` | Yes | Quantization level(s), comma-separated or repeated (see [ISQ Quantization](#isq-quantization)) |
| `-o, --output <PATH>` | Yes | Output path: `.uqff` file (single ISQ) or directory (auto-named per ISQ type) |
| `--isq-organization <TYPE>` | No | ISQ organization strategy: `default` or `moqe` |
| `--imatrix <PATH>` | No | imatrix file for enhanced quantization |
| `--calibration-file <PATH>` | No | Calibration file for imatrix generation |
| `--no-readme` | No | Skip automatic README.md model card generation |

---

### tune - Recommendations

Get quantization and device mapping recommendations for a model. The tune command analyzes your hardware and shows all quantization options with their estimated memory usage, context room, and quality trade-offs.

```bash
mistralrs tune [MODEL_TYPE] -m <MODEL_ID> [OPTIONS]
```

Note: `MODEL_TYPE` is optional and defaults to `auto` if not specified, which supports all model types. See [details](#auto).

**Examples:**

```bash
# Get balanced recommendations (shorthand)
mistralrs tune -m Qwen/Qwen3-4B

# Get quality-focused recommendations
mistralrs tune -m Qwen/Qwen3-4B --profile quality

# Get fast inference recommendations
mistralrs tune -m Qwen/Qwen3-4B --profile fast

# Output as JSON
mistralrs tune -m Qwen/Qwen3-4B --json

# Generate a TOML config file with recommendations
mistralrs tune -m Qwen/Qwen3-4B --emit-config config.toml
```

**Example Output (CUDA):**

```
Tuning Analysis
===============

Model: Qwen/Qwen3-4B
Profile: Balanced
Backend: cuda
Total VRAM: 24.0 GB

Quantization Options
--------------------
┌─────────────┬───────────┬────────┬──────────────┬───────────────┬──────────────────┐
│ Quant       │ Est. Size │ VRAM % │ Context Room │ Quality       │ Status           │
├─────────────┼───────────┼────────┼──────────────┼───────────────┼──────────────────┤
│ None (FP16) │ 8.50 GB   │ 35%    │ 48k          │ Baseline      │ ✅ Fits          │
│ Q8_0        │ 4.50 GB   │ 19%    │ 96k          │ Near-lossless │ 🚀 Recommended   │
│ Q6K         │ 3.70 GB   │ 15%    │ 128k (max)   │ Good          │ ✅ Fits          │
│ Q5K         │ 3.20 GB   │ 13%    │ 128k (max)   │ Good          │ ✅ Fits          │
│ Q4K         │ 2.60 GB   │ 11%    │ 128k (max)   │ Acceptable    │ ✅ Fits          │
│ Q3K         │ 2.00 GB   │ 8%     │ 128k (max)   │ Degraded      │ ✅ Fits          │
│ Q2K         │ 1.50 GB   │ 6%     │ 128k (max)   │ Degraded      │ ✅ Fits          │
└─────────────┴───────────┴────────┴──────────────┴───────────────┴──────────────────┘

Recommended Command
-------------------
  mistralrs serve -m Qwen/Qwen3-4B --isq q8_0

[INFO] PagedAttention is available (mode: auto)
```

**Example Output (Metal):**

On macOS with Metal, the command recommends Apple Format Quantization (AFQ) types:

```
Quantization Options
--------------------
┌─────────────┬───────────┬────────┬──────────────┬───────────────┬──────────────────┐
│ Quant       │ Est. Size │ VRAM % │ Context Room │ Quality       │ Status           │
├─────────────┼───────────┼────────┼──────────────┼───────────────┼──────────────────┤
│ None (FP16) │ 8.50 GB   │ 53%    │ 24k          │ Baseline      │ ✅ Fits          │
│ AFQ8        │ 4.50 GB   │ 28%    │ 56k          │ Near-lossless │ 🚀 Recommended   │
│ AFQ6        │ 3.70 GB   │ 23%    │ 64k          │ Good          │ ✅ Fits          │
│ AFQ4        │ 2.60 GB   │ 16%    │ 128k (max)   │ Acceptable    │ ✅ Fits          │
│ AFQ3        │ 2.00 GB   │ 13%    │ 128k (max)   │ Degraded      │ ✅ Fits          │
│ AFQ2        │ 1.50 GB   │ 9%     │ 128k (max)   │ Degraded      │ ✅ Fits          │
└─────────────┴───────────┴────────┴──────────────┴───────────────┴──────────────────┘
```

**Status Legend:**
- 🚀 **Recommended**: Best option for your profile and hardware
- ✅ **Fits**: Model fits entirely in GPU memory
- ⚠️ **Hybrid**: Model requires CPU offloading (slower due to PCIe bottleneck)
- ❌ **Too Large**: Model doesn't fit even with CPU offload

**Tune Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--profile <PROFILE>` | `balanced` | Tuning profile: `quality`, `balanced`, or `fast` |
| `--json` | disabled | Output JSON instead of human-readable text |
| `--emit-config <PATH>` | none | Emit a TOML config file with recommended settings |

---

### doctor - System Diagnostics

Run comprehensive system diagnostics and environment checks. The doctor command helps identify configuration issues and validates your system is ready for inference.

```bash
mistralrs doctor [OPTIONS]
```

**Examples:**

```bash
# Run diagnostics
mistralrs doctor

# Output as JSON
mistralrs doctor --json
```

**Checks Performed:**
- **CPU Extensions**: AVX, AVX2, AVX-512, FMA support (x86 only; ARM shows NEON)
- **Binary/Hardware Match**: Validates CUDA/Metal features match detected hardware
- **GPU Compute Capability**: Reports compute version and Flash Attention v2/v3 compatibility
- **Flash Attention Features**: Warns if hardware supports FA but binary doesn't have it enabled
- **Hugging Face Connectivity**: Tests connection and token validity using a gated model
- **HF Cache**: Verifies cache directory is writable
- **Disk Space**: Checks available storage

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output JSON instead of human-readable text |

---

### login - HuggingFace Authentication

Authenticate with HuggingFace Hub by saving your token to the local cache.

```bash
mistralrs login [OPTIONS]
```

**Examples:**

```bash
# Interactive login (prompts for token)
mistralrs login

# Provide token directly
mistralrs login --token hf_xxxxxxxxxxxxx
```

The token is saved to the standard HuggingFace cache location:
- Linux/macOS: `~/.cache/huggingface/token`
- Windows: `C:\Users\<user>\.cache\huggingface\token`

If the `HF_HOME` environment variable is set, the token is saved to `$HF_HOME/token`.

**Options:**

| Option | Description |
|--------|-------------|
| `--token <TOKEN>` | Provide token directly (non-interactive) |

---

### cache - Model Management

Manage the HuggingFace model cache. List cached models or delete specific models.

```bash
mistralrs cache <SUBCOMMAND>
```

**Subcommands:**

#### cache list

List all cached models with their sizes and last used times.

```bash
mistralrs cache list
```

**Example output:**

```
HuggingFace Model Cache
-----------------------

┌──────────────────────────┬──────────┬─────────────┐
│ Model                    │ Size     │ Last Used   │
├──────────────────────────┼──────────┼─────────────┤
│ Qwen/Qwen3-4B            │ 8.5 GB   │ today       │
│ google/gemma-3-4b-it     │ 6.2 GB   │ 2 days ago  │
│ meta-llama/Llama-3.2-3B  │ 5.8 GB   │ 1 week ago  │
└──────────────────────────┴──────────┴─────────────┘

Total: 3 models, 20.5 GB
Cache directory: /home/user/.cache/huggingface/hub
```

#### cache delete

Delete a specific model from the cache.

```bash
mistralrs cache delete -m <MODEL_ID>
```

**Examples:**

```bash
# Delete a specific model
mistralrs cache delete -m Qwen/Qwen3-4B

# Delete a model with organization
mistralrs cache delete -m meta-llama/Llama-3.2-3B
```

---

### bench - Performance Benchmarking

Run performance benchmarks to measure prefill and decode speeds.

```bash
mistralrs bench [MODEL_TYPE] -m <MODEL_ID> [OPTIONS]
```

Note: `MODEL_TYPE` is optional and defaults to `auto` if not specified.

**Examples:**

```bash
# Run default benchmark (512 prompt tokens, 128 generated tokens, 3 iterations)
mistralrs bench -m Qwen/Qwen3-4B

# Custom prompt and generation lengths
mistralrs bench -m Qwen/Qwen3-4B --prompt-len 1024 --gen-len 256

# More iterations for better statistics
mistralrs bench -m Qwen/Qwen3-4B --iterations 10

# With ISQ quantization
mistralrs bench -m Qwen/Qwen3-4B --isq q4k
```

**Example output:**

```
Benchmark Results
=================

Model: Qwen/Qwen3-4B
Iterations: 3

┌────────────────────────┬─────────────────┬─────────────────┐
│ Test                   │ T/s             │ Latency         │
├────────────────────────┼─────────────────┼─────────────────┤
│ Prefill (512 tokens)   │ 2847.3 ± 45.2   │ 179.82 ms (TTFT)│
│ Decode (128 tokens)    │ 87.4 ± 2.1      │ 11.44 ms/T      │
└────────────────────────┴─────────────────┴─────────────────┘
```

- **T/s**: Tokens per second (throughput)
- **Latency**: For prefill, shows TTFT (Time To First Token) in milliseconds. For decode, shows ms per token.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt-len <N>` | `512` | Number of tokens in prompt (prefill test) |
| `--gen-len <N>` | `128` | Number of tokens to generate (decode test) |
| `--iterations <N>` | `3` | Number of benchmark iterations |
| `--warmup <N>` | `1` | Number of warmup runs (discarded) |

The `bench` command also accepts all model loading options (ISQ, device mapping, etc.).

---

### from-config - TOML Configuration

Run the CLI from a TOML configuration file. This is the recommended way to run multiple models simultaneously, including models of different types (e.g., text + vision + embedding).

See [CLI_CONFIG.md](CLI_CONFIG.md) for full TOML configuration format details.

```bash
mistralrs from-config --file <PATH>
```

**Example:**

```bash
mistralrs from-config --file config.toml
```

**Multi-model example** (config.toml):

```toml
command = "serve"

[server]
port = 1234
ui = true

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-4B"

[[models]]
kind = "vision"
model_id = "google/gemma-3-4b-it"

[[models]]
kind = "embedding"
model_id = "google/embeddinggemma-300m"
```

---

### completions - Shell Completions

Generate shell completions for your shell.

```bash
mistralrs completions <SHELL>
```

**Examples:**

```bash
# Generate bash completions
mistralrs completions bash > ~/.local/share/bash-completion/completions/mistralrs

# Generate zsh completions
mistralrs completions zsh > ~/.zfunc/_mistralrs

# Generate fish completions
mistralrs completions fish > ~/.config/fish/completions/mistralrs.fish
```

**Supported Shells:** `bash`, `zsh`, `fish`, `elvish`, `powershell`

---

## Model Types

### auto

Auto-detect model type. This is the recommended option for most models and is on by default simply by leaving out the explicit model type.

```bash
mistralrs run -m Qwen/Qwen3-4B
mistralrs serve -m Qwen/Qwen3-4B
```

The `auto` type supports text, vision, and other model types through automatic detection.

### text

Explicit text generation model configuration.

```bash
mistralrs run text -m Qwen/Qwen3-4B
mistralrs serve text -m Qwen/Qwen3-4B
```

### vision

Vision-language models that can process images and text.

```bash
mistralrs run vision -m google/gemma-3-4b-it
mistralrs serve vision -m google/gemma-3-4b-it
```

**Vision Options:**

| Option | Description |
|--------|-------------|
| `--max-edge <SIZE>` | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <N>` | Maximum number of images per request |
| `--max-image-length <SIZE>` | Maximum image dimension for device mapping |

### diffusion

Image generation models using diffusion.

```bash
mistralrs run diffusion -m black-forest-labs/FLUX.1-schnell
mistralrs serve diffusion -m black-forest-labs/FLUX.1-schnell
```

### speech

Speech synthesis models.

```bash
mistralrs run speech -m nari-labs/Dia-1.6B
mistralrs serve speech -m nari-labs/Dia-1.6B
```

### embedding

Text embedding models. These do not support interactive mode but can be used with the HTTP server.

```bash
mistralrs serve embedding -m google/embeddinggemma-300m
```

---

## Features

### ISQ Quantization

In-situ quantization (ISQ) reduces model memory usage by quantizing weights at load time. See [details about ISQ here](ISQ.md).

**Usage:**

```bash
# Simple bit-width quantization
mistralrs run -m Qwen/Qwen3-4B --isq 4
mistralrs run -m Qwen/Qwen3-4B --isq 8

# GGML-style quantization
mistralrs run -m Qwen/Qwen3-4B --isq q4_0
mistralrs run -m Qwen/Qwen3-4B --isq q4_1
mistralrs run -m Qwen/Qwen3-4B --isq q4k
mistralrs run -m Qwen/Qwen3-4B --isq q5k
mistralrs run -m Qwen/Qwen3-4B --isq q6k
```

**ISQ Organization:**

```bash
# Use MOQE organization for potentially better quality
mistralrs run -m Qwen/Qwen3-4B --isq q4k --isq-organization moqe
```

---

### UQFF Files

UQFF (Unified Quantized File Format) provides pre-quantized model files for faster loading.

**Generate UQFF files:**

```bash
mistralrs quantize -m Qwen/Qwen3-4B --isq q4k -o qwen3-4b-uqff/
```

**Load from UQFF:**

```bash
# Specify just the first shard -- remaining shards are auto-discovered
mistralrs run -m Qwen/Qwen3-4B --from-uqff q4k-0.uqff
```

**Multiple UQFF files (semicolon-separated, for different quantizations in one load):**

```bash
mistralrs run -m Qwen/Qwen3-4B --from-uqff "q4k-0.uqff;q8_0-0.uqff"
```

> Note: Shard auto-discovery means you no longer need to list every shard file. Specifying `q4k-0.uqff` will automatically find `q4k-1.uqff`, `q4k-2.uqff`, etc.

---

### PagedAttention

PagedAttention enables efficient memory management for the KV cache. It is automatically enabled on CUDA and disabled on Metal/CPU by default.

**Control PagedAttention:**

```bash
# Auto mode (default): enabled on CUDA, disabled on Metal/CPU
mistralrs serve -m Qwen/Qwen3-4B --paged-attn auto

# Force enable
mistralrs serve -m Qwen/Qwen3-4B --paged-attn on

# Force disable
mistralrs serve -m Qwen/Qwen3-4B --paged-attn off
```

**Memory allocation options (mutually exclusive):**

```bash
# Allocate for specific context length (recommended)
mistralrs serve -m Qwen/Qwen3-4B --pa-context-len 8192

# Allocate specific GPU memory in MB
mistralrs serve -m Qwen/Qwen3-4B --pa-memory-mb 4096

# Allocate fraction of GPU memory (0.0-1.0)
mistralrs serve -m Qwen/Qwen3-4B --pa-memory-fraction 0.8
```

**Additional options:**

| Option | Description |
|--------|-------------|
| `--pa-block-size <SIZE>` | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <TYPE>` | KV cache quantization type (default: auto) |

---

### Device Mapping

Control how model layers are distributed across devices.

**Automatic mapping:**

```bash
# Use defaults (automatic)
mistralrs run -m Qwen/Qwen3-4B
```

**Manual layer assignment:**

```bash
# Assign 10 layers to GPU 0, 20 layers to GPU 1
mistralrs run -m Qwen/Qwen3-4B -n "0:10;1:20"

# Equivalent long form
mistralrs run -m Qwen/Qwen3-4B --device-layers "0:10;1:20"
```

**CPU-only execution:**

```bash
mistralrs run -m Qwen/Qwen3-4B --cpu
```

**Topology file:**

```bash
mistralrs run -m Qwen/Qwen3-4B --topology topology.yaml
```

**Custom HuggingFace cache:**

```bash
mistralrs run -m Qwen/Qwen3-4B --hf-cache /path/to/cache
```

**Device mapping options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --device-layers <MAPPING>` | auto | Device layer mapping (format: `ORD:NUM;...`) |
| `--topology <PATH>` | none | Topology YAML file for device mapping |
| `--hf-cache <PATH>` | none | Custom HuggingFace cache directory |
| `--cpu` | disabled | Force CPU-only execution |
| `--max-seq-len <LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <SIZE>` | `1` | Max batch size for automatic device mapping |

---

### LoRA and X-LoRA

Apply LoRA or X-LoRA adapters to models.

**LoRA:**

```bash
# Single LoRA adapter
mistralrs run -m Qwen/Qwen3-4B --lora my-lora-adapter

# Multiple LoRA adapters (semicolon-separated)
mistralrs run -m Qwen/Qwen3-4B --lora "adapter1;adapter2"
```

**X-LoRA:**

```bash
# X-LoRA adapter with ordering file
mistralrs run -m Qwen/Qwen3-4B --xlora my-xlora-adapter --xlora-order ordering.json

# With target non-granular index
mistralrs run -m Qwen/Qwen3-4B --xlora my-xlora-adapter --xlora-order ordering.json --tgt-non-granular-index 2
```

---

### Chat Templates

Override the model's default chat template.

**Use a template file:**

```bash
# JSON template file
mistralrs run -m Qwen/Qwen3-4B --chat-template template.json

# Jinja template file
mistralrs run -m Qwen/Qwen3-4B --chat-template template.jinja
```

**Explicit Jinja override:**

```bash
mistralrs run -m Qwen/Qwen3-4B --jinja-explicit custom.jinja
```

---

### Web Search

Enable web search capabilities (requires an embedding model).

```bash
# Enable search with default embedding model
mistralrs run -m Qwen/Qwen3-4B --enable-search

# Specify embedding model
mistralrs run -m Qwen/Qwen3-4B --enable-search --search-embedding-model embedding-gemma
```

---

### Thinking Mode

Enable thinking/reasoning mode for models that support it (like DeepSeek, Qwen3).

```bash
mistralrs run -m Qwen/Qwen3-4B --enable-thinking
```

In interactive mode, thinking content is displayed in gray text before the final response.

---

## Global Options

These options apply to all commands.

| Option | Default | Description |
|--------|---------|-------------|
| `--seed <SEED>` | none | Random seed for reproducibility |
| `-l, --log <PATH>` | none | Log all requests and responses to file |
| `--token-source <SOURCE>` | `cache` | HuggingFace authentication token source |
| `-V, --version` | N/A | Print version information and exit |
| `-h, --help` | N/A | Print help message (use with any subcommand) |

**Token source formats:**

- `cache` - Use cached HuggingFace token (default)
- `literal:<token>` - Use literal token value
- `env:<var>` - Read token from environment variable
- `path:<file>` - Read token from file
- `none` - No authentication

**Examples:**

```bash
# Set random seed
mistralrs run -m Qwen/Qwen3-4B --seed 42

# Enable logging
mistralrs run -m Qwen/Qwen3-4B --log requests.log

# Use token from environment variable
mistralrs run -m meta-llama/Llama-3.2-3B-Instruct --token-source env:HF_TOKEN
```

---

## Runtime Options

These options are available for both `run` and `serve` commands.

| Option | Default | Description |
|--------|---------|-------------|
| `--max-seqs <N>` | `32` | Maximum concurrent sequences |
| `--no-kv-cache` | disabled | Disable KV cache entirely |
| `--prefix-cache-n <N>` | `16` | Number of prefix caches to hold (0 to disable) |
| `-c, --chat-template <PATH>` | none | Custom chat template file (.json or .jinja) |
| `-j, --jinja-explicit <PATH>` | none | Explicit JINJA template override |
| `--enable-search` | disabled | Enable web search |
| `--search-embedding-model <MODEL>` | none | Embedding model for search |

---

## Model Source Options

These options are common across model types.

| Option | Description |
|--------|-------------|
| `-m, --model-id <ID>` | HuggingFace model ID or local path (required) |
| `-t, --tokenizer <PATH>` | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` | Model architecture (auto-detected if not specified) |
| `--dtype <TYPE>` | Model data type (default: `auto`) |

---

## Format Options

For loading quantized models.

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Model format: `plain`, `gguf`, or `ggml` (auto-detected) |
| `-f, --quantized-file <FILE>` | Quantized model filename(s) for GGUF/GGML (semicolon-separated) |
| `--tok-model-id <ID>` | Model ID for tokenizer when using quantized format |
| `--gqa <VALUE>` | GQA value for GGML models (default: 1) |

**Examples:**

```bash
# Load a GGUF model
mistralrs run -m Qwen/Qwen3-4B --format gguf -f model.gguf

# Multiple GGUF files
mistralrs run -m Qwen/Qwen3-4B --format gguf -f "model-part1.gguf;model-part2.gguf"
```

---

## Interactive Commands

When running in interactive mode (`mistralrs run`), the following commands are available:

| Command | Description |
|---------|-------------|
| `\help` | Display help message |
| `\exit` | Quit interactive mode |
| `\system <message>` | Add a system message without running the model |
| `\clear` | Clear the chat history |
| `\temperature <float>` | Set sampling temperature (0.0 to 2.0) |
| `\topk <int>` | Set top-k sampling value (>0) |
| `\topp <float>` | Set top-p sampling value (0.0 to 1.0) |

**Examples:**

```
> \system Always respond as a pirate.
> \temperature 0.7
> \topk 50
> Hello!
Ahoy there, matey! What brings ye to these waters?
> \clear
> \exit
```

**Vision Model Interactive Mode:**

For vision models, you can include images in your prompts by specifying file paths or URLs:

```
> Describe this image: /path/to/image.jpg
> Compare these images: image1.png image2.png
> Describe the image and transcribe the audio: photo.jpg recording.mp3
```

**Note**: The CLI automatically detects paths to supported image and audio files within your prompt. You do not need special syntax; simply paste the absolute or relative path to the file.

Supported image formats: PNG, JPEG, BMP, GIF, WebP
Supported audio formats: WAV, MP3, FLAC, OGG
