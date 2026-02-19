# mistralrs-cli TOML Config

`mistralrs-cli` can run entirely from a single TOML configuration file. This config supports multiple models and mirrors the CLI options.

## Usage

```bash
mistralrs from-config --file path/to/config.toml
```

## Quick Example

```toml
command = "serve"

[server]
port = 1234
ui = true

[runtime]
max_seqs = 32

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "q4k"
```

## Complete Reference

### Top-Level Options

| Option | Commands | Description |
|--------|----------|-------------|
| `command` | all | Required. Either `"serve"` or `"run"` |
| `enable_thinking` | run | Enable thinking mode (default: false) |
| `default_model_id` | serve | Default model ID for API requests (must match a model_id in [[models]]) |

### [global] Section

Global options that apply to the entire run.

| Option | Default | Description |
|--------|---------|-------------|
| `seed` | none | Random seed for reproducibility |
| `log` | none | Log all requests/responses to this file path |
| `token_source` | `"cache"` | HuggingFace auth: `"cache"`, `"none"`, `"literal:<token>"`, `"env:<var>"`, `"path:<file>"` |

### [server] Section (serve only)

HTTP server configuration.

| Option | Default | Description |
|--------|---------|-------------|
| `port` | `1234` | HTTP server port |
| `host` | `"0.0.0.0"` | Bind address |
| `ui` | `false` | Serve built-in web UI at `/ui` |
| `mcp_port` | none | MCP protocol server port (enables MCP if set) |
| `mcp_config` | none | MCP client configuration file path |

### [runtime] Section

Runtime inference options.

| Option | Default | Description |
|--------|---------|-------------|
| `max_seqs` | `32` | Maximum concurrent sequences |
| `no_kv_cache` | `false` | Disable KV cache entirely |
| `prefix_cache_n` | `16` | Number of prefix caches to hold (0 to disable) |
| `chat_template` | none | Custom chat template file (.json or .jinja) |
| `jinja_explicit` | none | Explicit JINJA template override |
| `enable_search` | `false` | Enable web search |
| `search_embedding_model` | none | Embedding model for search (e.g., `"embedding-gemma"`) |

### [paged_attn] Section

PagedAttention configuration.

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `"auto"` | `"auto"` (CUDA on, Metal off), `"on"`, or `"off"` |
| `context_len` | none | Allocate KV cache for this context length |
| `memory_mb` | none | GPU memory to allocate in MB (conflicts with context_len) |
| `memory_fraction` | none | GPU memory utilization 0.0-1.0 (conflicts with above) |
| `block_size` | `32` | Tokens per block |
| `cache_type` | `"auto"` | KV cache type |

**Note:** If none of `context_len`, `memory_mb`, or `memory_fraction` are specified, defaults to 90% of available VRAM. Each are mutually exclusive.

### [[models]] Section

Define one or more models. Each `[[models]]` entry creates a new model.

#### Top-Level Model Options

| Option | Required | Description |
|--------|----------|-------------|
| `kind` | yes | Model type: `"auto"`, `"text"`, `"vision"`, `"diffusion"`, `"speech"`, `"embedding"` |
| `model_id` | yes | HuggingFace model ID or local path |
| `tokenizer` | no | Path to local tokenizer.json |
| `arch` | no | Model architecture (auto-detected if not specified) |
| `dtype` | `"auto"` | Data type: `"auto"`, `"f16"`, `"bf16"`, `"f32"` |
| `chat_template` | no | Per-model chat template override |
| `jinja_explicit` | no | Per-model JINJA template override |

#### [models.format] - Model Format

| Option | Default | Description |
|--------|---------|-------------|
| `format` | auto | `"plain"` (safetensors), `"gguf"`, or `"ggml"` |
| `quantized_file` | none | Quantized filename(s) for GGUF/GGML (semicolon-separated) |
| `tok_model_id` | none | Model ID for tokenizer when using quantized format |
| `gqa` | `1` | GQA value for GGML models |

#### [models.adapter] - LoRA/X-LoRA

| Option | Description |
|--------|-------------|
| `lora` | LoRA adapter ID(s), semicolon-separated |
| `xlora` | X-LoRA adapter ID (conflicts with lora) |
| `xlora_order` | X-LoRA ordering JSON file (requires xlora) |
| `tgt_non_granular_index` | Target non-granular index for X-LoRA |

#### [models.quantization] - ISQ/UQFF

| Option | Description |
|--------|-------------|
| `in_situ_quant` | ISQ level: `"4"`, `"8"`, `"q4_0"`, `"q4k"`, `"q6k"`, etc. |
| `from_uqff` | UQFF file(s) to load (semicolon-separated). Shards are auto-discovered: specifying the first shard (e.g., `q4k-0.uqff`) automatically finds `q4k-1.uqff`, etc. |
| `isq_organization` | ISQ strategy: `"default"` or `"moqe"` |
| `imatrix` | imatrix file for enhanced quantization |
| `calibration_file` | Calibration file for imatrix generation |

#### [models.device] - Device Mapping

| Option | Default | Description |
|--------|---------|-------------|
| `cpu` | `false` | Force CPU-only (must be consistent across all models) |
| `device_layers` | auto | Layer mapping, e.g., `["0:10", "1:20"]` format: `ORD:NUM;...` |
| `topology` | none | Topology YAML file |
| `hf_cache` | none | Custom HuggingFace cache directory |
| `max_seq_len` | `4096` | Max sequence length for auto device mapping |
| `max_batch_size` | `1` | Max batch size for auto device mapping |

#### [models.vision] - Vision Options

| Option | Description |
|--------|-------------|
| `max_edge` | Maximum edge length for image resizing |
| `max_num_images` | Maximum images per request |
| `max_image_length` | Maximum image dimension for device mapping |

## Full Examples

### Multi-Model Server with UI

```toml
command = "serve"

[global]
seed = 42

[server]
host = "0.0.0.0"
port = 1234
ui = true

[runtime]
max_seqs = 32
enable_search = true
search_embedding_model = "embedding-gemma"

[paged_attn]
mode = "auto"

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
dtype = "auto"

[models.quantization]
in_situ_quant = "q4k"

[[models]]
kind = "vision"
model_id = "Qwen/Qwen2-VL-2B-Instruct"

[models.vision]
max_num_images = 4

[[models]]
kind = "embedding"
model_id = "google/embeddinggemma-300m"
```

### Interactive Mode with Thinking

```toml
command = "run"
enable_thinking = true

[runtime]
max_seqs = 16

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-4B"
```

### GGUF Model

```toml
command = "serve"

[server]
port = 1234

[[models]]
kind = "text"
model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

[models.format]
format = "gguf"
quantized_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
tok_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
```

### Device Layer Mapping

```toml
command = "serve"

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-70B-Instruct"

[models.device]
device_layers = ["0:40", "1:40"]

[models.quantization]
in_situ_quant = "q4k"
```

## Notes

- `cpu` must be consistent across all models if specified
- `default_model_id` (serve only) must match a `model_id` in `[[models]]`
- `search_embedding_model` requires `enable_search = true`
