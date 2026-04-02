# TOML Configuration Reference

Complete catalogue of all `config.toml` fields for the `mistralrs from-config` command.

**Source of truth**: `mistralrs-core/src/toml_selector.rs`, `mistralrs-cli/src/args/`

---

## Top-Level Fields

| Field | Type | Default | Commands | Description |
|-------|------|---------|----------|-------------|
| `command` | `"serve"` \| `"run"` | required | all | Which subcommand to run |
| `enable_thinking` | bool | `false` | run | Enable thinking/reasoning mode |
| `default_model_id` | string | none | serve | Default model for API (must match a `model_id` in `[[models]]`) |

---

## [global]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | integer | none | Random seed for reproducibility |
| `log` | string (path) | none | Log all requests/responses to file |
| `token_source` | string | `"cache"` | HF auth: `"cache"`, `"none"`, `"literal:<token>"`, `"env:<var>"`, `"path:<file>"` |

---

## [server] (serve only)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | integer | `1234` | HTTP server port |
| `host` | string | `"0.0.0.0"` | Bind address |
| `ui` | bool | `false` | Enable built-in web UI at `/ui` |
| `mcp_port` | integer | none | MCP server port (enables MCP if set) |
| `mcp_config` | string (path) | none | MCP client configuration file |

---

## [runtime]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seqs` | integer | `32` | Maximum concurrent sequences |
| `no_kv_cache` | bool | `false` | Disable KV cache entirely |
| `prefix_cache_n` | integer | `16` | Prefix caches to hold (0 = disable) |
| `chat_template` | string (path) | none | Custom chat template (.json or .jinja) |
| `jinja_explicit` | string (path) | none | Explicit Jinja template override |
| `enable_search` | bool | `false` | Enable web search |
| `search_embedding_model` | string | none | Embedding model for search |

---

## [paged_attn]

**Note**: At most one of `context_len`, `memory_mb`, `memory_fraction` may be set.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"auto"` \| `"on"` \| `"off"` | `"auto"` | `auto` = enabled on CUDA, off on Metal/CPU |
| `context_len` | integer | none | Allocate KV cache for this context length |
| `memory_mb` | integer | none | GPU memory for KV blocks in MB |
| `memory_fraction` | float (0.0–1.0) | `0.9` | GPU fraction for KV blocks |
| `block_size` | integer | `32` | Tokens per block |
| `cache_type` | string | `"auto"` | KV cache element type |

---

## [[models]]

### Top-Level Model Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | string | yes | `"auto"`, `"text"`, `"vision"`, `"diffusion"`, `"speech"`, `"embedding"` |
| `model_id` | string | yes | HuggingFace ID or local path |
| `tokenizer` | string | no | Path to local tokenizer.json |
| `arch` | string | no | Model architecture (auto-detected) |
| `dtype` | string | `"auto"` | `"auto"`, `"f16"`, `"bf16"`, `"f32"` |
| `chat_template` | string | no | Per-model chat template override |
| `jinja_explicit` | string | no | Per-model Jinja override |

---

### [models.format]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | auto | `"plain"`, `"gguf"`, `"ggml"` |
| `quantized_file` | string | none | GGUF/GGML filename(s), semicolon-separated |
| `tok_model_id` | string | none | Tokenizer source for quantized models |
| `gqa` | integer | `1` | GQA value for GGML models |

---

### [models.adapter]

| Field | Type | Description |
|-------|------|-------------|
| `lora` | string | LoRA adapter ID(s), semicolon-separated |
| `xlora` | string | X-LoRA adapter ID (mutually exclusive with `lora`) |
| `xlora_order` | string (path) | X-LoRA ordering JSON file (required with `xlora`) |
| `tgt_non_granular_index` | integer | Target non-granular index for X-LoRA |

---

### [models.quantization]

| Field | Type | Description |
|-------|------|-------------|
| `in_situ_quant` | string | ISQ level: `"q4k"`, `"q8"`, `"q4_0"`, `"q6k"`, etc. |
| `from_uqff` | string | UQFF file(s) to load, semicolon-separated |
| `write_uqff` | string (path) | Write quantized UQFF output |
| `isq_organization` | string | `"default"` or `"moqe"` |
| `imatrix` | string (path) | imatrix file for enhanced quantization |
| `calibration_file` | string (path) | Calibration data for imatrix |

---

### [models.device]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cpu` | bool | `false` | Force CPU-only (must be consistent across all models) |
| `device_layers` | string[] | auto | Layer mapping, e.g., `["0:40", "1:40"]` |
| `topology` | string (path) | none | Topology YAML file |
| `hf_cache` | string (path) | none | Custom HuggingFace cache directory |
| `max_seq_len` | integer | `4096` | Max sequence length for auto device mapping |
| `max_batch_size` | integer | `1` | Max batch size for auto device mapping |

---

### [models.vision] (vision models only)

| Field | Type | Description |
|-------|------|-------------|
| `max_edge` | integer | Maximum edge length for image resizing |
| `max_num_images` | integer | Maximum images per request |
| `max_image_length` | integer | Maximum image dimension for device mapping |

---

### [models.cache] (requires `kvcache-compression` feature)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kv_compression_bits` | integer (2\|3\|4) | none (disabled) | TurboQuant bits per coordinate. Enables compression when set. |
| `kv_compression_threshold` | integer | `128` | Tokens before compression begins. Only active with `kv_compression_bits`. |

**Env var equivalents**: `MISTRALRS_KV_CACHE_BITS`, `MISTRALRS_KV_CACHE_THRESHOLD`

```toml
[models.cache]
# Requires: cargo build --features kvcache-compression
kv_compression_bits = 3          # 2=max compression, 4=best quality, 3=recommended
kv_compression_threshold = 4096  # compress after 4096 tokens
```
