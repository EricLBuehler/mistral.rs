# Deployment Patterns

Canonical deployment configurations for mistral.rs with TurboQuant KV-cache compression.

---

## Pattern 1: local-dev

Minimal configuration for local experimentation. No auth, single GPU, UI enabled.

```toml
command = "serve"

[server]
port = 1234
ui = true

[runtime]
max_seqs = 4

[paged_attn]
mode = "auto"

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-4B"
dtype = "auto"

[models.quantization]
in_situ_quant = "q4k"

[models.cache]
# Requires --features kvcache-compression
kv_compression_bits = 3
kv_compression_threshold = 4096
```

```bash
mistralrs from-config --file config.toml
```

---

## Pattern 2: production-server (single GPU, 16–24 GB)

High-throughput server for API consumers.

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234
ui = false

[runtime]
max_seqs = 16
prefix_cache_n = 32

[paged_attn]
mode = "auto"
memory_fraction = 0.88

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
dtype = "auto"

[models.quantization]
in_situ_quant = "q4k"

[models.device]
max_seq_len = 131072
max_batch_size = 4

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096
```

---

## Pattern 3: docker-compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  mistralrs:
    image: ghcr.io/ericllbuehler/mistralrs:latest
    restart: unless-stopped
    ports:
      - "1234:1234"
    env_file: .env
    volumes:
      - ./config.toml:/app/config.toml:ro
      - hf-cache:/root/.cache/huggingface
    command: from-config --file /app/config.toml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
volumes:
  hf-cache:
```

```bash
# .env
MISTRALRS_KV_CACHE_BITS=3
MISTRALRS_KV_CACHE_THRESHOLD=4096
```

```bash
docker-compose up -d
```

---

## Pattern 4: apple-silicon (M-series Mac)

Metal backend, higher compression threshold due to fast unified memory.

```toml
command = "serve"

[server]
port = 1234
ui = true

[runtime]
max_seqs = 8

[paged_attn]
mode = "off"  # PagedAttention disabled on Metal

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-8B"
dtype = "bf16"

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 8192  # higher threshold for Apple Silicon
```

```bash
cargo build --release --features kvcache-compression,metal
./target/release/mistralrs from-config --file config.toml
```

---

## Pattern 5: multi-gpu (2× A100 80GB)

Device layer mapping across two GPUs.

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234

[runtime]
max_seqs = 32

[paged_attn]
mode = "auto"
memory_fraction = 0.90

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-70B-Instruct"
dtype = "bf16"

[models.device]
device_layers = ["0:40", "1:40"]
max_seq_len = 131072

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096
```

---

## Pattern 6: cpu-only

For hardware without a GPU. Higher threshold recommended to minimize latency impact.

```toml
command = "serve"

[server]
port = 1234

[runtime]
max_seqs = 1

[paged_attn]
mode = "off"

[[models]]
kind = "auto"
model_id = "Qwen/Qwen3-4B"
dtype = "f32"

[models.device]
cpu = true

[models.quantization]
in_situ_quant = "q4_0"

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 16384  # compress only for very long contexts
```

---

## Pattern 7: multi-model server

Text + vision + embedding in a single server.

```toml
command = "serve"
default_model_id = "meta-llama/Llama-3.1-8B-Instruct"

[server]
port = 1234
ui = true

[runtime]
max_seqs = 16
enable_search = true
search_embedding_model = "google/embeddinggemma-300m"

[paged_attn]
mode = "auto"
memory_fraction = 0.85

# Text model with compression
[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
dtype = "auto"

[models.quantization]
in_situ_quant = "q4k"

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096

# Vision model — no compression (shorter contexts)
[[models]]
kind = "vision"
model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Embedding model
[[models]]
kind = "embedding"
model_id = "google/embeddinggemma-300m"
```

---

## KV-Cache Compression Summary Across Patterns

| Pattern | GPU | Bits | Threshold | Context gain |
|---------|-----|------|-----------|--------------|
| local-dev | Any 8–16 GB | 3 | 4096 | ~7× |
| production | 16–24 GB CUDA | 3 | 4096 | ~7× |
| docker-compose | Via env var | 3 | 4096 | ~7× |
| apple-silicon | M-series | 3 | 8192 | ~7× (less benefit) |
| multi-gpu | 2× 80 GB | 3 | 4096 | ~7× |
| cpu-only | CPU RAM | 3 | 16384 | ~7× (RAM) |
