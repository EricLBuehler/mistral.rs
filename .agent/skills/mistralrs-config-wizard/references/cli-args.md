# CLI Arguments Reference

All `mistralrs` command-line flags with their TOML equivalents and environment variable fallbacks.

## Global Flags (all commands)

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--seed <N>` | `[global] seed` | — | none | Random seed for reproducibility |
| `-l, --log <path>` | `[global] log` | — | none | Log all requests/responses to file |
| `--token-source <s>` | `[global] token_source` | — | `cache` | HF auth: `cache`, `none`, `literal:<t>`, `env:<v>`, `path:<f>` |

## Runtime Flags (run + serve)

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--max-seqs <N>` | `[runtime] max_seqs` | — | `32` | Max concurrent sequences |
| `--no-kv-cache` | `[runtime] no_kv_cache` | — | false | Disable KV cache entirely |
| `--prefix-cache-n <N>` | `[runtime] prefix_cache_n` | — | `16` | Prefix caches to hold |
| `-c, --chat-template <p>` | `[runtime] chat_template` | — | none | Custom chat template file |
| `-j, --jinja-explicit <p>` | `[runtime] jinja_explicit` | — | none | Explicit Jinja override |
| `--enable-search` | `[runtime] enable_search` | — | false | Enable web search |
| `--search-embedding-model <m>` | `[runtime] search_embedding_model` | — | none | Embedding model for search |

## KV-Cache Compression Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--kv-cache-bits <2\|3\|4>` | `[models.cache] kv_compression_bits` | `MISTRALRS_KV_CACHE_BITS` | disabled | TurboQuant bits per coordinate. Requires `kvcache-compression` feature. |
| `--kv-cache-threshold <N>` | `[models.cache] kv_compression_threshold` | `MISTRALRS_KV_CACHE_THRESHOLD` | `128` | Tokens before compression starts. Requires `--kv-cache-bits`. |

## PagedAttention Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--paged-attn-mode <m>` | `[paged_attn] mode` | — | `auto` | `auto`, `on`, or `off` |
| `--paged-attn-context-len <N>` | `[paged_attn] context_len` | — | none | KV cache context length (exclusive with memory options) |
| `--paged-attn-memory-mb <N>` | `[paged_attn] memory_mb` | — | none | GPU MB for KV blocks |
| `--paged-attn-memory-fraction <f>` | `[paged_attn] memory_fraction` | — | `0.9` | GPU fraction for KV blocks |
| `--paged-attn-block-size <N>` | `[paged_attn] block_size` | — | `32` | Tokens per block |

## Serve-Only Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `-p, --port <N>` | `[server] port` | — | `1234` | HTTP server port |
| `--serve-ip <addr>` | `[server] host` | — | `0.0.0.0` | Bind address |
| `--ui` | `[server] ui` | — | false | Enable built-in web UI |
| `--mcp-port <N>` | `[server] mcp_port` | — | none | MCP server port |
| `--mcp-config <p>` | `[server] mcp_config` | `MCP_CONFIG_PATH` | none | MCP client config path |

## Model Source Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `-m, --model-id <id>` | `[[models]] model_id` | — | required | HF ID or local path |
| `-t, --tokenizer <p>` | `[[models]] tokenizer` | — | none | Local tokenizer.json |
| `-a, --arch <arch>` | `[[models]] arch` | — | auto | Model architecture |
| `--dtype <type>` | `[[models]] dtype` | — | `auto` | Data type |

## Format Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--format <fmt>` | `[models.format] format` | — | auto | `plain`, `gguf`, or `ggml` |
| `-f, --quantized-file <f>` | `[models.format] quantized_file` | — | none | GGUF/GGML filename(s) |
| `--tok-model-id <id>` | `[models.format] tok_model_id` | — | none | Tokenizer source for GGUF |

## Quantization Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--isq <level>` | `[models.quantization] in_situ_quant` | — | none | ISQ level: `q4k`, `q8`, `q4_0`, etc. |
| `--from-uqff <p>` | `[models.quantization] from_uqff` | — | none | UQFF file(s) to load |
| `--write-uqff <p>` | `[models.quantization] write_uqff` | — | none | Write UQFF output |

## Device Mapping Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--cpu` | `[models.device] cpu` | — | false | Force CPU-only |
| `--topology <p>` | `[models.device] topology` | — | none | Topology YAML file |
| `--hf-cache <p>` | `[models.device] hf_cache` | — | none | Custom HF cache dir |

## Run-Only Flags

| Flag | TOML | Env Var | Default | Description |
|------|------|---------|---------|-------------|
| `--enable-thinking` | `enable_thinking` (top-level) | — | false | Enable thinking/reasoning mode |
