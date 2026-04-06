# Performance Guide

This guide helps you choose the right performance settings for your hardware and workload. Each section gives you the key decision, then links to the detailed reference page.

> Don't want to read all this? Run `mistralrs tune -m <model>` and it will recommend the best configuration for your hardware automatically.

## Quick Recommendations

| Hardware | Suggested command |
|---|---|
| NVIDIA GPU (16+ GB VRAM) | `mistralrs serve --isq 4 -m <model>` |
| NVIDIA GPU (8 GB VRAM) | `mistralrs serve --isq 4 -m <7B-model>` |
| Apple Silicon (16+ GB RAM) | `mistralrs serve --isq 4 -m <model>` |
| CPU only | `mistralrs serve --isq 4 --cpu -m <small-model>` |
| Multi-GPU NVIDIA | `mistralrs serve -m <model>` (auto tensor parallelism) |

For precise recommendations, use `mistralrs tune -m <model>` which analyzes your available memory and model size.

## The Performance Stack

When a request arrives, it passes through several optimization layers:

```
Request
  |
  v
[Quantization] ---- Reduces model weight memory (ISQ, GGUF, UQFF)
  |
  v
[Prefill] --------- Processes prompt tokens (FlashAttention accelerates this)
  |
  v
[Decode] ---------- Generates tokens one at a time (PagedAttention manages KV cache)
  |                  (MLA compresses KV cache for DeepSeek/GLM models)
  |
  v
[Speculative] ----- Draft model proposes multiple tokens at once (optional)
  |
  v
Response
```

Each layer is independent — you can use any combination.

## Quantization: Choosing a Method

Quantization reduces model size by using lower-precision weights. mistral.rs offers several approaches:

| Method | When to use | Details |
|---|---|---|
| **ISQ** (`--isq 4`) | Default choice. Quantize any model at load time | [ISQ docs](ISQ.md) |
| **GGUF** (`--format gguf`) | Load pre-quantized GGUF files from the community | [GGUF section](QUANTS.md#using-a-gguf-quantized-model) |
| **UQFF** (`--from-uqff`) | Load pre-quantized UQFF files (faster startup than ISQ) | [UQFF docs](UQFF.md) |
| **Topology** (`--topology`) | Per-layer control: different quantization per layer | [Topology docs](TOPOLOGY.md) |
| **GPTQ/AWQ** | Use GPTQ or AWQ models from HF (auto-detected, CUDA only) | [GPTQ/AWQ section](QUANTS.md#using-a-gptq-quantized-model) |

**ISQ levels at a glance** (approximate, for a 7B parameter model):

| Level | VRAM | Quality | Best for |
|---|---|---|---|
| `--isq 8` (Q8_0) | ~7 GB | Near-lossless | When you have the VRAM |
| `--isq 6` (Q6K) | ~5.5 GB | Good | Balanced quality/size |
| `--isq 4` (Q4K) | ~4 GB | Acceptable | Most common choice |
| `--isq 3` (Q3K) | ~3 GB | Degraded | Tight VRAM budget |
| `--isq 2` (Q2K) | ~2.5 GB | Significantly degraded | Extreme constraint |

On **Metal**, `--isq 4` uses AFQ4 (optimized for Apple Silicon). On **CUDA/CPU**, it uses Q4K.

To improve ISQ accuracy, use an importance matrix: `--calibration-file calibration_data/calibration_datav3_small.txt`. See [Importance Matrix](IMATRIX.md).

For MoE models (DeepSeek, Qwen3 MoE, GLM-4.7), use `--isq-organization moqe` to only quantize expert layers. See [MoQE](ISQ.md#isq-quantization-types).

> Full quantization reference: [Quantization Overview](QUANTS.md)

## Memory Management: PagedAttention

PagedAttention manages KV cache memory efficiently, enabling longer contexts and higher throughput through continuous batching.

**Default behavior:**
- **CUDA**: Enabled automatically, allocates 90% of free VRAM for KV cache
- **Metal**: Disabled by default (enable with `--paged-attn on`)
- **CPU**: Not supported

**Tuning KV cache allocation:**

```bash
# Allocate for a specific context length (recommended)
mistralrs serve -m <model> --pa-context-len 8192

# Or set a fixed memory budget
mistralrs serve -m <model> --pa-memory-mb 4096

# Or use a fraction of available VRAM
mistralrs serve -m <model> --pa-memory-fraction 0.8
```

**FP8 KV cache quantization** halves KV cache memory, allowing ~2x longer contexts:

```bash
mistralrs serve -m <model> --pa-cache-type f8e4m3
```

> Full reference: [PagedAttention](PAGED_ATTENTION.md)

## Attention Acceleration: FlashAttention

FlashAttention accelerates the prefill phase (processing the prompt). It requires CUDA and a compatible GPU.

| GPU generation | Feature flag | Notes |
|---|---|---|
| Ampere (RTX 30xx, A100) | `--features flash-attn` | FlashAttention V2 |
| Ada Lovelace (RTX 40xx) | `--features flash-attn` | FlashAttention V2 |
| Hopper (H100) | `--features flash-attn-v3` | FlashAttention V3 |
| Blackwell (RTX 50xx) | `--features flash-attn` | FlashAttention V2 |

FlashAttention V2 and V3 are mutually exclusive. If compiled with FlashAttention and PagedAttention is enabled, both work together automatically.

> Full reference: [FlashAttention](FLASH_ATTENTION.md)

## Multi-head Latent Attention (MLA)

MLA compresses the KV cache for DeepSeek V2/V3 and GLM-4.7-Flash models, reducing memory usage and enabling longer contexts. It activates automatically when:
- The model supports it (DeepSeek V2, V3, GLM-4.7-Flash)
- PagedAttention is enabled
- Running on CUDA

Disable with `MISTRALRS_NO_MLA=1` if needed.

> Full reference: [MLA](MLA.md)

## Multi-GPU and Distributed Inference

**Single machine, multiple GPUs:**

mistral.rs auto-detects multiple CUDA GPUs and uses tensor parallelism via NCCL. No configuration needed:

```bash
# Uses all available GPUs automatically
mistralrs serve -m <large-model>

# Or specify GPU count
MISTRALRS_MN_LOCAL_WORLD_SIZE=2 mistralrs serve -m <large-model>
```

If the model doesn't fit on GPUs even with parallelism, disable NCCL to use automatic device mapping (GPU + CPU offloading):

```bash
MISTRALRS_NO_NCCL=1 mistralrs serve --isq 4 -m <large-model>
```

**Multiple machines:** Use the [Ring backend](DISTRIBUTED/RING.md) for cross-machine inference over TCP.

> Full reference: [Device Mapping](DEVICE_MAPPING.md) | [NCCL](DISTRIBUTED/NCCL.md) | [Ring](DISTRIBUTED/RING.md)

## Speculative Decoding

Speculative decoding uses a small "draft" model to propose tokens, which the large "target" model validates in parallel. This can significantly speed up generation when the draft model is accurate.

**When it helps:**
- Long generation tasks (100+ tokens)
- Draft model from the same family (e.g., Llama 3.2-1B drafting for Llama 3.1-8B)
- Low batch sizes

**When to skip it:**
- High-throughput batched serving (standard decoding is more efficient)
- No good draft model available

```bash
# Configure via TOML
mistralrs run --from-toml speculative.toml
```

> Full reference: [Speculative Decoding](SPECULATIVE_DECODING.md)

## Benchmarking Your Setup

**Quick benchmark:**

```bash
mistralrs bench -m <model> --isq 4
```

This measures prefill speed (tokens/sec processing the prompt) and decode speed (tokens/sec generating output). Key metrics:

- **Prefill T/s**: Higher is better. Affected by FlashAttention, GPU compute.
- **Decode T/s**: Higher is better. Affected by memory bandwidth, quantization, PagedAttention.
- **TTFT (Time to First Token)**: The prefill latency in ms. Critical for interactive use.

**Automated recommendations:**

```bash
mistralrs tune -m <model>
```

Shows a table of quantization options ranked by fit, quality, and performance for your hardware. Use `--profile quality` or `--profile fast` to shift the ranking.

## Optimization Recipes

### Maximum throughput server (CUDA)

```bash
mistralrs serve -m <model> \
  --isq 4 \
  --pa-memory-fraction 0.9 \
  --max-seqs 32
```

Key: maximize PagedAttention memory, allow many concurrent sequences.

### Lowest VRAM for a 70B model

```bash
MISTRALRS_NO_NCCL=1 mistralrs serve -m <70B-model> \
  --isq 3 \
  --pa-context-len 4096 \
  --max-seq-len 4096
```

Key: aggressive quantization, limit context length, auto device mapping offloads to CPU.

### Fastest TTFT for interactive chat

```bash
mistralrs serve -m <model> \
  --isq 4 \
  --prefix-cache-n 16
```

Key: prefix caching reuses KV cache across turns. FlashAttention (compile-time flag) accelerates prefill.

### Apple Silicon optimization

```bash
mistralrs serve -m <model> --isq 4
```

On Metal, `--isq 4` automatically uses AFQ4 which is optimized for Apple Silicon. PagedAttention is off by default on Metal — enable with `--paged-attn on` if you need higher throughput. The auto device mapper respects unified memory limits.
