# KV-Cache Compression (TurboQuant)

TurboQuant (Google Research, ICLR 2026) compresses the transformer attention KV-cache to as few as 2–4 bits per coordinate, achieving 4–16× memory reduction with virtually zero accuracy loss. It is implemented in the `mistralrs-kvcache-compression` crate and gated behind the `kvcache-compression` Cargo feature.

**Related docs:** [CLI Reference → KV-Cache Compression](../CLI.md#kv-cache-compression) · [TOML Config](../CLI_CONFIG.md) · [Configuration / Env Vars](../CONFIGURATION.md)

---

## Table of Contents

- [What Problem Does It Solve?](#what-problem-does-it-solve)
- [How TurboQuant Works](#how-turbo-quant-works)
- [Building with Compression Support](#building-with-compression-support)
- [Configuration Reference](#configuration-reference)
  - [CLI Flags](#cli-flags)
  - [Environment Variables](#environment-variables)
  - [TOML File (`from-config`)](#toml-file-from-config)
- [Choosing Bits and Threshold](#choosing-bits-and-threshold)
- [Memory Math](#memory-math)
- [Platform Notes](#platform-notes)
- [Complete Examples](#complete-examples)
- [Quality Trade-offs](#quality-trade-offs)
- [Troubleshooting](#troubleshooting)

---

## What Problem Does It Solve?

Every token processed by a transformer is stored in a KV-cache in GPU memory. Without compression:

```
KV memory per token ≈ 2 × num_layers × d_model × 2 bytes  (FP16)
```

For an 8B model with 32 layers and d_model=4096 on a 16 GB GPU:
- Model weights: ~16 GB (FP16) or ~8 GB (Q4)
- Remaining VRAM for KV cache: ~8 GB
- Max context before OOM: ~32K tokens

With 3-bit TurboQuant: same 8 GB headroom fits **≈220K tokens** — a 7× increase.

---

## How TurboQuant Works

Two complementary algorithms:

**1. PolarQuant** (AISTATS 2026): Applies a randomized Hadamard rotation to the KV vectors, then performs Lloyd-Max scalar quantization per coordinate. MSE-optimal under the rotation.

**2. QJL** (AAAI 2025): 1-bit residual correction using a Quantized Johnson-Lindenstrauss transform for unbiased inner-product estimation.

**Combined result:** 3-bit representation with ≈9.8× compression ratio and <0.1% quality degradation on standard benchmarks.

The implementation adds <2–3% latency on modern GPUs — negligible compared to the compute cost of the forward pass.

---

## Building with Compression Support

```bash
# Debug build
cargo build --features kvcache-compression

# Release build
cargo build --release --features kvcache-compression

# macOS / Metal (required on Apple Silicon)
cargo build --release --features kvcache-compression,metal

# CUDA
cargo build --release --features kvcache-compression,cuda

# Combined with other features
cargo build --release --features kvcache-compression,cuda,flash-attn
```

Without `--features kvcache-compression`, any `--kv-cache-bits` flag or `kv_compression_bits` TOML field is silently ignored.

---

## Configuration Reference

KV-cache compression can be configured via three mechanisms. CLI flags take highest priority, then environment variables, then TOML config.

### CLI Flags

```
--kv-cache-bits <2|3|4>
    Bits per coordinate for TurboQuant compression.
    2 = highest compression (~16×, ~1% quality loss)
    3 = recommended (~7×, <0.1% quality loss)
    4 = conservative (~4×, <0.05% quality loss)
    Omit to disable compression entirely.

--kv-cache-threshold <N>      (default: 128, requires --kv-cache-bits)
    Minimum tokens to accumulate before compression begins.
    Higher values preserve full-precision quality on short turns.
    Set to 0 to compress every token from the start.
```

**Examples:**

```bash
# Recommended: 3-bit, compress after 4096 tokens
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --kv-cache-threshold 4096

# Conservative: 4-bit for high-quality long-context
mistralrs serve -m Qwen/Qwen3-8B \
  --kv-cache-bits 4 \
  --kv-cache-threshold 8192

# Aggressive: 3-bit, compress immediately (very tight VRAM)
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --kv-cache-threshold 0

# Interactive mode
mistralrs run -m Qwen/Qwen3-4B --kv-cache-bits 3

# Quantize command (for UQFF generation)
mistralrs quantize auto -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --output-uqff model.uqff
```

---

### Environment Variables

Set these when you want compression to apply without changing launch scripts:

```bash
# Enable 3-bit compression (fallback when --kv-cache-bits is not passed)
export MISTRALRS_KV_CACHE_BITS=3

# Start compressing after 4096 tokens (fallback when --kv-cache-threshold is not passed)
export MISTRALRS_KV_CACHE_THRESHOLD=4096

# Then run normally — compression activates automatically
mistralrs serve -m meta-llama/Llama-3.1-8B-Instruct
```

CLI flags always override the env vars:

```bash
# CLI flag wins — uses 4 bits despite env var saying 3
MISTRALRS_KV_CACHE_BITS=3 mistralrs serve -m ... --kv-cache-bits 4
```

Useful patterns:

```bash
# Docker / container: set in Dockerfile or docker-compose.yml
ENV MISTRALRS_KV_CACHE_BITS=3
ENV MISTRALRS_KV_CACHE_THRESHOLD=4096

# systemd service: add to [Service] section
Environment="MISTRALRS_KV_CACHE_BITS=3"
Environment="MISTRALRS_KV_CACHE_THRESHOLD=4096"

# .env file (sourced before launch)
MISTRALRS_KV_CACHE_BITS=3
MISTRALRS_KV_CACHE_THRESHOLD=4096
```

---

### TOML File (`from-config`)

The `[models.cache]` section in a TOML config file:

```toml
[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

[models.cache]
kv_compression_bits = 3          # 2, 3, or 4 — omit to disable
kv_compression_threshold = 4096  # tokens before compression starts (default: 128)
```

Full server example with TOML:

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234
ui = true

[runtime]
max_seqs = 16

[paged_attn]
mode = "auto"
memory_fraction = 0.85

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
dtype = "auto"

[models.quantization]
in_situ_quant = "q4k"

[models.device]
max_seq_len = 131072

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096
```

Run with:

```bash
mistralrs from-config --file config.toml
```

---

## Choosing Bits and Threshold

### Decision Table

| VRAM headroom | `bits` | `threshold` | Compression | Quality impact |
|---------------|--------|-------------|-------------|----------------|
| > 30 % free | disabled | — | 1× (none) | 0% |
| 15–30 % free | `4` | `4096` | ~4× | <0.05% |
| 5–15 % free | **`3`** | **`4096`** | **~7×** | **<0.1%** |
| < 5 % free | `3` | `128` | ~7× | <0.1% |
| Critical < 2 % | `2` | `0` | ~16× | ~0.5–1% |

### Threshold guidance

The threshold is the number of tokens in the KV-cache before compression begins. A higher threshold means short conversations run at full precision; only long contexts get compressed.

```
threshold = 0       → compress every token immediately (maximum memory savings)
threshold = 128     → default; compress after only 128 tokens
threshold = 4096    → preserve quality for conversations up to 4K tokens
threshold = 8192    → for Apple Silicon (fast unified memory, higher quality bar)
threshold = 32768   → compress only for very long documents / retrieval tasks
```

---

## Memory Math

### How much context can you fit?

```
# KV memory per token (rough estimate)
kv_bytes_per_token = 2 × num_layers × d_model × 2   # FP16

# With 3-bit TurboQuant
kv_bytes_compressed = kv_bytes_per_token / 7

# Available KV memory
available_kv_mb = (total_vram_gb × 1024) - (model_vram_gb × 1024) - os_overhead_mb

# Max context without compression
max_tokens = available_kv_mb × 1024 × 1024 / kv_bytes_per_token

# Max context with 3-bit
max_tokens_compressed = max_tokens × 7
```

### Practical examples

| Model | GPU | Q4 weight size | KV budget | Max context | With 3-bit |
|-------|-----|---------------|-----------|-------------|------------|
| Llama-3.1-8B | RTX 4090 (24 GB) | ~5 GB | ~8 GB | ~32K | ~200K |
| Llama-3.1-70B | A100 80 GB | ~40 GB | ~20 GB | ~16K | ~100K |
| Qwen3-8B | M3 Max 64 GB | ~5 GB | ~45 GB | ~80K | 128K+ native |
| Phi-4 14B | A100 40 GB | ~9 GB | ~22 GB | ~24K | ~150K |

---

## Platform Notes

### CUDA

Full support. Hardware-accelerated Hadamard transform where available. Recommended threshold: `4096`.

### Metal (Apple Silicon)

Supported. Apple Silicon has fast unified memory with high bandwidth, so compression delivers smaller but still meaningful wins. Recommended threshold: `8192` (higher than CUDA to preserve more short-turn quality given the faster memory).

```bash
# Apple Silicon — higher threshold recommended
mistralrs serve --features kvcache-compression,metal \
  -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --kv-cache-threshold 8192
```

### CPU-only

Compression is functional on CPU but adds measurable latency. Recommended only if RAM pressure is extreme:

```bash
mistralrs serve --cpu -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --kv-cache-threshold 16384
```

---

## Complete Examples

### Example 1: 16 GB GPU, long-document Q&A

```bash
cargo build --release --features kvcache-compression,cuda

./target/release/mistralrs serve \
  -m meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-bits 3 \
  --kv-cache-threshold 4096 \
  --isq q4k \
  --max-seqs 8 \
  --port 1234
```

### Example 2: Environment variable approach (Docker)

```dockerfile
FROM ubuntu:22.04
# ... build steps ...
ENV MISTRALRS_KV_CACHE_BITS=3
ENV MISTRALRS_KV_CACHE_THRESHOLD=4096
CMD ["mistralrs", "serve", "-m", "meta-llama/Llama-3.1-8B-Instruct"]
```

### Example 3: TOML config for production server

```toml
# production.toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234
ui = false

[runtime]
max_seqs = 32
no_kv_cache = false
prefix_cache_n = 16

[paged_attn]
mode = "auto"
memory_fraction = 0.90

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

```bash
mistralrs from-config --file production.toml
```

### Example 4: Apple Silicon (M3 Max, 64 GB)

```bash
cargo build --release --features kvcache-compression,metal

./target/release/mistralrs serve \
  -m Qwen/Qwen3-8B \
  --kv-cache-bits 3 \
  --kv-cache-threshold 8192 \
  --port 1234
```

### Example 5: Multi-model server with mixed compression

```toml
command = "serve"

[server]
port = 1234

# Large text model — compress aggressively
[[models]]
kind = "text"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096

# Vision model — no compression (shorter contexts typical)
[[models]]
kind = "vision"
model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Embedding model — no KV cache needed
[[models]]
kind = "embedding"
model_id = "google/embeddinggemma-300m"
```

---

## Quality Trade-offs

| Bits | Compression | Quality loss | Recommendation |
|------|-------------|--------------|----------------|
| `4` | ~4× | <0.05% | Tight but not critical VRAM |
| `3` | ~7–8× | <0.1% | **Recommended for most cases** |
| `2` | ~16× | ~0.5–1% | Extreme memory pressure only |

Quality loss is measured on standard LLM benchmarks (MMLU, HumanEval, GSM8K). For most practical use cases — chat, Q&A, summarization, coding — 3-bit compression is imperceptible.

---

## Troubleshooting

**Compression appears to be ignored (no memory savings)**

- Verify the binary was built with `--features kvcache-compression`
- Check that `MISTRALRS_KV_CACHE_BITS` is set or `--kv-cache-bits` is passed
- PagedAttention must be enabled (`--paged-attn-mode auto` or `on`) for compression to apply

**OOM despite compression enabled**

- Lower `--kv-cache-threshold` to begin compression earlier
- Try `--kv-cache-bits 2` for maximum compression
- Reduce `--max-seqs` to limit concurrent KV footprints
- Reduce `--paged-attn-memory-fraction` to leave more headroom

**Noticeable quality degradation**

- Increase `--kv-cache-bits` from `2` → `3` or `3` → `4`
- Raise `--kv-cache-threshold` to preserve more short-context quality
- For reasoning models (DeepSeek, Qwen3 thinking), use `--kv-cache-threshold 8192` or higher

**Build errors**

```
error: the package does not contain this feature: kvcache-compression
```
Check that `mistralrs-kvcache-compression` is listed in `[workspace.members]` in the root `Cargo.toml`.

---

## See Also

- [CLI Reference](../CLI.md#kv-cache-compression)
- [TOML Configuration](../CLI_CONFIG.md)
- [Configuration / Env Vars](../CONFIGURATION.md)
- [PagedAttention](../PAGED_ATTENTION.md) — KV memory management (works alongside compression)
- [ISQ Quantization](../ISQ.md) — Weight quantization (orthogonal to KV compression)
- [TurboQuant source](https://github.com/Prometheus-AGS/turboquant-rs)
