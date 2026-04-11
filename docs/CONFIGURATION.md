# Configuration Reference

This document covers environment variables and server configuration for mistral.rs.

## Runtime Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRALRS_DEBUG=1` | Enable debug mode: outputs tensor info files for GGUF/GGML models, increases logging verbosity |
| `MISTRALRS_NO_MMAP=1` | Disable memory-mapped file loading, forcing all tensor data into memory |
| `MISTRALRS_NO_MLA=1` | Disable [MLA](MLA.md) (Multi-head Latent Attention) optimization for DeepSeek V2/V3 and GLM-4.7-Flash |
| `MISTRALRS_ISQ_SINGLETHREAD=1` | Force ISQ (In-Situ Quantization) to run single-threaded |
| `MISTRALRS_IGPU_MEMORY_FRACTION` | Memory fraction for integrated/unified-memory CUDA GPUs (e.g. NVIDIA Grace Blackwell, Jetson). Float between 0.0 and 1.0, default: `0.75` |
| `MCP_CONFIG_PATH` | Fallback path for MCP client configuration (used if `--mcp-config` not provided) |
| `KEEP_ALIVE_INTERVAL` | SSE keep-alive interval in milliseconds (default: 10000) |
| `HF_HUB_CACHE` | Override Hugging Face Hub cache directory |

### KV-Cache Compression (TurboQuant)

These variables enable TurboQuant KV-cache compression (requires the `kvcache-compression` Cargo feature). They are the environment-variable fallback for `--kv-cache-bits` and `--kv-cache-threshold`; explicit CLI flags always take precedence.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MISTRALRS_KV_CACHE_BITS` | `2` \| `3` \| `4` | unset (disabled) | Bits per coordinate for TurboQuant compression. Omit or unset to disable compression entirely. `3` is the recommended starting point (≈7× compression, <0.1% quality impact). |
| `MISTRALRS_KV_CACHE_THRESHOLD` | integer ≥ 0 | `128` | Minimum number of tokens to accumulate before compression begins. Only active when `MISTRALRS_KV_CACHE_BITS` is set. Higher values preserve full-precision quality on short turns. |

**Quick guidance:**

| VRAM headroom | Recommended `BITS` | Recommended `THRESHOLD` |
|---------------|--------------------|--------------------------|
| > 30 % free | (disabled) | — |
| 15–30 % free | `4` | `4096` |
| 5–15 % free | `3` | `4096` |
| < 5 % free | `3` | `128` (compress early) |
| Critical (< 2 %) | `2` | `0` |

**Example:**

```bash
export MISTRALRS_KV_CACHE_BITS=3
export MISTRALRS_KV_CACHE_THRESHOLD=4096
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct
```

See [KV-Cache Compression Guide](prometheus-enhancements/KVCACHE-COMPRESSION.md) for full details.

## Build-Time Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRALRS_METAL_PRECOMPILE=0` | Skip Metal kernel precompilation (useful for CI) |
| `NVCC_CCBIN` | Set CUDA compiler path |
| `CUDA_NVCC_FLAGS=-fPIE` | Required on some Linux distributions |
| `CUDA_COMPUTE_CAP` | Override CUDA compute capability (e.g., "80" for RTX 3090) |

## Server Defaults

When running the HTTP server with `mistralrs serve`, these defaults apply:

| Setting | Default Value |
|---------|---------------|
| Server IP | `0.0.0.0` (all interfaces) |
| Max request body | 50 MB |
| Max running sequences | 32 |
| Prefix cache count | 16 |
| SSE keep-alive | 10 seconds |
| PagedAttention (CUDA) | Enabled |
| PagedAttention (Metal) | Disabled |
| PA GPU memory usage | 90% of free memory |
| PA block size | 32 tokens |

## Multi-Node Distributed Configuration

For multi-node setups, configure the head node and workers using environment variables.

### Head Node

| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total number of devices across all nodes |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Number of worker nodes |
| `MISTRALRS_MN_HEAD_PORT` | Port for head node communication |

### Worker Nodes

| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Address of head server to connect to |
| `MISTRALRS_MN_WORKER_ID` | This worker's ID |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Number of GPUs on this node |
| `MISTRALRS_NO_NCCL=1` | Disable NCCL (use alternative backend) |

## See Also

- [CLI Reference](CLI.md): Command-line options
- [CLI TOML Configuration](CLI_CONFIG.md): File-based configuration
- [Distributed Inference](DISTRIBUTED/DISTRIBUTED.md): Multi-node setup guide
- [PagedAttention](PAGED_ATTENTION.md): Memory management options
