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

mistral.rs has several independent optimization layers. You can use any combination:

- **Quantization**: Reduces model weight memory (ISQ, GGUF, UQFF)
- **FlashAttention**: Accelerates prompt processing (prefill phase)
- **PagedAttention**: Efficiently manages KV cache memory during generation
- **MLA**: Compresses KV cache for DeepSeek/GLM models
- **Speculative Decoding**: Uses a draft model to propose multiple tokens at once

## Quantization: Choosing a Method

Quantization reduces model size by using lower-precision weights. mistral.rs offers several approaches:

| Method | When to use | Details |
|---|---|---|
| **ISQ** (`--isq <level>`) | Default choice. Quantize any model at load time. Levels: 2-8 | [ISQ docs](ISQ.md) |
| **GGUF** (`--format gguf`) | Load pre-quantized GGUF files from the community | [GGUF section](QUANTS.md#using-a-gguf-quantized-model) |
| **UQFF** (`--from-uqff`) | Load pre-quantized UQFF files (faster startup than ISQ) | [UQFF docs](UQFF.md) |
| **Topology** (`--topology`) | Per-layer control: different quantization per layer | [Topology docs](TOPOLOGY.md) |
| **GPTQ/AWQ** | Use GPTQ or AWQ models from HF (auto-detected, CUDA only) | [GPTQ/AWQ section](QUANTS.md#using-a-gptq-quantized-model) |

**ISQ levels at a glance:**

| Level | Quality | Notes |
|---|---|---|
| `--isq 8` (Q8_0/AFQ8) | Near-lossless | Use when you have VRAM headroom |
| `--isq 6` (Q6K/AFQ6) | Good | Balanced quality and size |
| `--isq 5` (Q5K) | Good | Slightly better quality than Q4K |
| `--isq 4` (Q4K/AFQ4) | Acceptable | Most common choice |
| `--isq 3` (Q3K/AFQ3) | Degraded | Tight VRAM budget |
| `--isq 2` (Q2K/AFQ2) | Significantly degraded | Extreme constraint |

On **Metal**, ISQ will prefer using AFQ (optimized for Apple Silicon). On **CUDA/CPU**, it uses Q/K quantization.

To improve ISQ accuracy, use an importance matrix: `--calibration-file calibration_data/calibration_datav3_small.txt`. See [Importance Matrix](IMATRIX.md).

### MoQE: Smarter Quantization for MoE Models

For Mixture of Experts models (DeepSeek V2/V3, Qwen3 MoE, Qwen3-VL MoE, GLM-4.7), [MoQE](https://arxiv.org/abs/2310.02410) quantizes only the expert layers while keeping attention and other layers at higher precision. This preserves more quality than uniform quantization since expert layers make up the bulk of parameters in MoE models.

```bash
mistralrs run --isq 4 --isq-organization moqe -m deepseek-ai/DeepSeek-R1
```

MoQE is available in the CLI (`--isq-organization moqe`), Python SDK (`organization="MoQE"`), and Rust SDK. See [ISQ docs](ISQ.md#isq-quantization-types) for the full list of supported models.

> Full quantization reference: [Quantization Overview](QUANTS.md)

## Memory Management: PagedAttention

PagedAttention manages KV cache memory efficiently, enabling longer contexts and higher throughput through continuous batching.

**Default behavior:**
- **CUDA**: Enabled automatically
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
- **TTFT (Time to First Token)**: The prefill latency in ms.

**Automated recommendations:**

```bash
mistralrs tune -m <model>
```

Shows a table of quantization options ranked by fit, quality, and performance for your hardware. Use `--profile quality` or `--profile fast` to shift the ranking.
