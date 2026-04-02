# Environment Variables Reference

All `MISTRALRS_*` environment variables. CLI flags always take precedence over env vars.

**Source of truth**: `docs/CONFIGURATION.md`, `mistralrs-cli/src/args/paged_attn.rs`

---

## KV-Cache Compression (TurboQuant)

Requires the `kvcache-compression` Cargo feature.

| Variable | Type | Default | CLI Equivalent | TOML Equivalent |
|----------|------|---------|----------------|-----------------|
| `MISTRALRS_KV_CACHE_BITS` | `2` \| `3` \| `4` | unset (disabled) | `--kv-cache-bits` | `[models.cache] kv_compression_bits` |
| `MISTRALRS_KV_CACHE_THRESHOLD` | integer ≥ 0 | `128` | `--kv-cache-threshold` | `[models.cache] kv_compression_threshold` |

```bash
export MISTRALRS_KV_CACHE_BITS=3
export MISTRALRS_KV_CACHE_THRESHOLD=4096
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct
```

---

## Runtime Behavior

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MISTRALRS_DEBUG` | `1` to enable | off | Debug mode: outputs tensor info, increases log verbosity |
| `MISTRALRS_NO_MMAP` | `1` to enable | off | Disable memory-mapped file loading |
| `MISTRALRS_NO_MLA` | `1` to enable | off | Disable MLA optimization (DeepSeek V2/V3, GLM-4.7-Flash) |
| `MISTRALRS_ISQ_SINGLETHREAD` | `1` to enable | off | Force ISQ to run single-threaded |
| `MISTRALRS_IGPU_MEMORY_FRACTION` | float (0.0–1.0) | `0.75` | Memory fraction for integrated/unified-memory CUDA GPUs |
| `KEEP_ALIVE_INTERVAL` | integer (ms) | `10000` | SSE keep-alive interval |

---

## Server / Networking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_CONFIG_PATH` | string (path) | none | Fallback MCP client config path (if `--mcp-config` not provided) |
| `HF_HUB_CACHE` | string (path) | `~/.cache/huggingface` | Override HuggingFace Hub cache directory |

---

## Multi-Node Distributed

| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total devices across all nodes |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Number of worker nodes (head node) |
| `MISTRALRS_MN_HEAD_PORT` | Head node communication port |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Head server address (worker nodes) |
| `MISTRALRS_MN_WORKER_ID` | This worker's ID |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | GPUs on this node |
| `MISTRALRS_NO_NCCL` | `1` to disable NCCL |

---

## Build-Time

| Variable | Description |
|----------|-------------|
| `MISTRALRS_METAL_PRECOMPILE` | Set to `0` to skip Metal kernel precompilation (CI) |
| `NVCC_CCBIN` | CUDA compiler path override |
| `CUDA_NVCC_FLAGS` | Extra NVCC flags (e.g., `-fPIE` on some Linux distros) |
| `CUDA_COMPUTE_CAP` | Override CUDA compute capability (e.g., `"80"` for RTX 3090) |

---

## Usage Patterns

### Docker / Compose

```yaml
# docker-compose.yml
environment:
  - MISTRALRS_KV_CACHE_BITS=3
  - MISTRALRS_KV_CACHE_THRESHOLD=4096
  - MISTRALRS_DEBUG=0
```

### systemd Service

```ini
[Service]
Environment="MISTRALRS_KV_CACHE_BITS=3"
Environment="MISTRALRS_KV_CACHE_THRESHOLD=4096"
ExecStart=/usr/local/bin/mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct
```

### .env File

```bash
# .env — source before running
MISTRALRS_KV_CACHE_BITS=3
MISTRALRS_KV_CACHE_THRESHOLD=4096
HF_HUB_CACHE=/data/models/cache
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mistralrs-env
stringData:
  MISTRALRS_KV_CACHE_BITS: "3"
  MISTRALRS_KV_CACHE_THRESHOLD: "4096"
```
