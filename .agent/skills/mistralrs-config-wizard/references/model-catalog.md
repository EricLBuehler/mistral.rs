# Model Catalog

Supported model architectures in mistral.rs with hardware fit guidance and TurboQuant compression recommendations.

**Source of truth**: `docs/SUPPORTED_MODELS.md`, `candle-transformers/src/models/`

---

## Text Models

| Model Family | Example IDs | Params | VRAM (Q4) | Context | Notes |
|-------------|-------------|--------|-----------|---------|-------|
| Llama 3.x | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~5 GB | 128K | Recommended all-rounder |
| Llama 3.x | `meta-llama/Llama-3.1-70B-Instruct` | 70B | ~40 GB | 128K | Needs multi-GPU or 80GB |
| Qwen 3 | `Qwen/Qwen3-4B` | 4B | ~2.5 GB | 128K | Fast, low VRAM |
| Qwen 3 | `Qwen/Qwen3-8B` | 8B | ~5 GB | 128K | Best 8B for coding/math |
| Qwen 3 | `Qwen/Qwen3-14B` | 14B | ~8 GB | 128K | Strong reasoning |
| Qwen 3 | `Qwen/Qwen3-30B-A3B` | 30B/3B active | ~18 GB | 128K | MoE — efficient |
| Phi-4 | `microsoft/Phi-4` | 14B | ~8 GB | 16K | Strong at STEM |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~4.5 GB | 32K | Classic baseline |
| Gemma 3 | `google/gemma-3-4b-it` | 4B | ~2.5 GB | 128K | Multimodal-ready |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` | 671B | ~400 GB | 128K | Multi-node only |
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | ~5 GB | 128K | Strong reasoning distill |
| GLM-4 | `THUDM/glm-4-9b-chat` | 9B | ~5.5 GB | 128K | Chinese/English |

---

## Vision Models

| Model Family | Example IDs | Params | VRAM (Q4) | Notes |
|-------------|-------------|--------|-----------|-------|
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | 2B | ~2 GB | Efficient vision |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | 7B | ~5 GB | Strong vision |
| Qwen3-VL | `Qwen/Qwen3-VL-7B-Instruct` | 7B | ~5 GB | Latest vision |
| Gemma 3 | `google/gemma-3-4b-it` | 4B | ~2.5 GB | Auto-detected as vision |
| Gemma 3 | `google/gemma-3-27b-it` | 27B | ~16 GB | High quality vision |
| LLaVA | `llava-hf/llava-1.5-7b-hf` | 7B | ~5 GB | Classic multimodal |
| Idefics 3 | `HuggingFaceM4/Idefics3-8B-Llama3` | 8B | ~5 GB | Document understanding |
| Phi-4 MM | `microsoft/Phi-4-multimodal-instruct` | 14B | ~8 GB | Strong STEM vision |

---

## Embedding Models

| Model | VRAM | Notes |
|-------|------|-------|
| `google/embeddinggemma-300m` | ~0.5 GB | Recommended for semantic search |
| `Qwen/Qwen3-Embedding-0.6B` | ~0.4 GB | Multilingual |

---

## Hardware Fit Guide

### 8 GB GPU (RTX 3060, etc.)

| Model | ISQ | KV Bits | Context |
|-------|-----|---------|---------|
| Qwen3-4B | q4k | 3 | ~70K |
| Llama-3.1-8B | q4k | 3 | ~40K |
| Phi-4 14B | q4k | 3 | ~20K |

### 16 GB GPU (RTX 4080, RTX 3090, etc.)

| Model | ISQ | KV Bits | Context |
|-------|-----|---------|---------|
| Llama-3.1-8B | q4k | 3 | ~200K |
| Qwen3-14B | q4k | 3 | ~60K |
| Llama-3.1-70B | q4k | 3 | ~5K (tight) |

### 24 GB GPU (RTX 4090, A5000, etc.)

| Model | ISQ | KV Bits | Context |
|-------|-----|---------|---------|
| Llama-3.1-8B | q4k | 3 | ~224K |
| Qwen3-14B | q4k | 3 | ~100K |
| Qwen3-30B-A3B | q4k | 3 | ~40K |

### 40 GB GPU (A100 40GB)

| Model | ISQ | KV Bits | Context |
|-------|-----|---------|---------|
| Llama-3.1-70B | q4k | 3 | ~60K |
| Qwen3-30B-A3B | bf16 | 3 | ~80K |

### 80 GB GPU (A100 80GB, H100)

| Model | ISQ | KV Bits | Context |
|-------|-----|---------|---------|
| Llama-3.1-70B | bf16 | 3 | ~100K |
| Qwen3-30B-A3B | bf16 | disabled | ~100K |

### Apple Silicon (M3 Max 64 GB)

| Model | dtype | KV Bits | Threshold | Context |
|-------|-------|---------|-----------|---------|
| Qwen3-8B | bf16 | 3 | 8192 | 128K+ |
| Llama-3.1-8B | bf16 | 3 | 8192 | 128K+ |
| Qwen3-14B | bf16 | 4 | 8192 | ~80K |

---

## KV Compression Per Architecture

All supported text/vision architectures benefit equally from TurboQuant. Compression is applied at the cache level after attention projection, independent of model architecture. The only consideration is context length: models with very short native contexts (e.g., ≤8K) benefit less.

---

## GGUF / GGML Models

For GGUF quantized models from TheBloke / other providers, KV compression still applies:

```toml
[[models]]
kind = "text"
model_id = "TheBloke/Llama-2-7B-Chat-GGUF"

[models.format]
format = "gguf"
quantized_file = "llama-2-7b-chat.Q4_K_M.gguf"
tok_model_id = "meta-llama/Llama-2-7b-chat-hf"

[models.cache]
kv_compression_bits = 3
kv_compression_threshold = 4096
```
