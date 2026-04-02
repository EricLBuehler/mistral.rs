# TurboQuant KV-Cache Compression Guide

TurboQuant (Google Research, ICLR 2026) compresses the transformer KV-cache to 2–4 bits per coordinate, achieving 4–16× memory reduction with <0.1% quality loss at 3 bits.

**Source of truth**: `docs/prometheus-enhancements/KVCACHE-COMPRESSION.md`

---

## What It Does

Every token processed produces a KV-cache entry:
```
KV bytes/token = 2 × num_layers × d_model × 2  (FP16)
```

A 7B model (32 layers, d_model=4096) uses ~2 MB per token. At 32K context → 64 GB KV alone — more than most GPUs have. TurboQuant compresses each entry to 3 bits, reducing that to ~9 MB/token.

---

## Algorithms

**PolarQuant** — Randomized Hadamard rotation + Lloyd-Max scalar quantization. Minimizes MSE under the rotation. Handles the bulk of compression.

**QJL** — 1-bit residual correction via Quantized Johnson-Lindenstrauss transform. Provides unbiased inner-product estimation for the residual.

Together: 3-bit KV with ≈9.8× compression and <0.1% benchmark degradation.

---

## Three Configuration Mechanisms

### 1. CLI flags (highest priority)
```bash
mistralrs serve -m <model> --kv-cache-bits 3 --kv-cache-threshold 4096
```

### 2. Environment variables (middle priority)
```bash
export MISTRALRS_KV_CACHE_BITS=3
export MISTRALRS_KV_CACHE_THRESHOLD=4096
mistralrs serve -m <model>
```

### 3. TOML config (lowest priority)
```toml
[models.cache]
# Requires --features kvcache-compression
kv_compression_bits = 3
kv_compression_threshold = 4096
```

---

## Bits Decision Table

| VRAM headroom after model load | Bits | Threshold (CUDA) | Threshold (Metal) | Compression | Quality impact |
|-------------------------------|------|-----------------|-------------------|-------------|----------------|
| > 30% free | disabled | — | — | 1× | 0% |
| 15–30% free | `4` | `4096` | `8192` | ~4× | <0.05% |
| 5–15% free | **`3`** | **`4096`** | **`8192`** | **~7×** | **<0.1%** |
| < 5% free | `3` | `128` | `4096` | ~7× | <0.1% |
| Critical < 2% | `2` | `0` | `0` | ~16× | ~0.5–1% |

**Apple Silicon**: Use higher thresholds because fast unified memory access makes compression less urgent on short contexts.

---

## Headroom Calculation

```python
# Rough model VRAM estimate
model_vram_gb = model_params_b * 2.0    # FP16
model_vram_gb = model_params_b * 0.5    # Q4 (ISQ)

# Available for KV + other
available_vram_gb = total_vram_gb - model_vram_gb - 1.0  # ~1GB OS overhead

# Headroom fraction
headroom_pct = available_vram_gb / total_vram_gb
```

---

## Context Capacity Math

```python
# KV bytes per token (FP16)
kv_bytes = 2 * num_layers * d_model * 2

# Max context without compression (given available_kv_mb)
max_tokens = available_kv_mb * 1024 * 1024 / kv_bytes

# Max context with N-bit compression
ratio = {2: 16, 3: 7, 4: 4}[bits]
max_tokens_compressed = max_tokens * ratio
```

### Practical Examples

| Model | GPU | ISQ | Normal max ctx | 3-bit max ctx |
|-------|-----|-----|---------------|---------------|
| Llama-3.1-8B | RTX 4090 24GB | q4k | ~32K | ~224K |
| Llama-3.1-70B | A100 80GB | q4k | ~16K | ~112K |
| Qwen3-8B | M3 Max 64GB | none | ~80K | 128K+ native |
| Phi-4 14B | A100 40GB | q4k | ~24K | ~168K |

---

## Threshold Guidance

```
threshold = 0       → compress every token (maximum savings, always-on)
threshold = 128     → default; compress after only 128 tokens (early)
threshold = 4096    → preserve quality for short conversations (RECOMMENDED for CUDA)
threshold = 8192    → recommended for Apple Silicon
threshold = 32768   → only for very long document/retrieval tasks
```

Higher threshold = better quality on short turns, less total savings.
Lower threshold = more aggressive compression, lower quality on short turns.

---

## Build Requirement

```bash
# Must include kvcache-compression feature
cargo build --release --features kvcache-compression
cargo build --release --features kvcache-compression,cuda   # CUDA
cargo build --release --features kvcache-compression,metal  # macOS

# Without feature, kv_compression_bits is silently ignored
```

---

## Latency Impact

- 3-bit: adds ~2–3% latency on modern CUDA GPUs (negligible)
- 3-bit on Apple Silicon: ~3–5% (fast unified memory reduces impact)
- 2-bit: adds ~4–6% latency

The latency overhead is constant per forward pass regardless of compression ratio — compression/decompression happens in a single fused kernel.
