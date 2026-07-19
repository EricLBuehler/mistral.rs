---
title: Quantization types
description: Every runtime ISQ (in-situ quantization) type mistralrs supports, what hardware it works on, and how it compares.
---

ISQ (in-situ quantization) types supported by mistral.rs. For format selection guidance and underlying tradeoffs, see the [quantization guide](/mistral.rs/guides/quantization/quantize-a-model/).

Flag choice for normal CLI usage:

- `--quant N` - normal usage.
- `--isq N` - force runtime ISQ and skip the [UQFF (Universal Quantized File Format)](/mistral.rs/reference/uqff-format/) lookup.

## Numeric shorthands

mistral.rs resolves N to a format based on the detected backend (see table). This happens when `--quant` falls back to runtime ISQ, or when you pass `--isq N` directly.

| Shorthand | Metal resolves to | CUDA / CPU resolves to |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

## Format-specific types

### AFQ family

Affine quantization, optimized for Apple Silicon. Runs on Metal (native kernels), CUDA
(dedicated backend), and CPU (fallback).

| Type | Bits |
|---|---|
| `afq2` | 2 |
| `afq3` | 3 |
| `afq4` | 4 |
| `afq6` | 6 |
| `afq8` | 8 |

### Q*K family

GGML K-quant formats. Supported on all backends.

| Type | Bits |
|---|---|
| `q2k` | 2 |
| `q3k` | 3 |
| `q4k` | 4 |
| `q5k` | 5 |
| `q6k` | 6 |

### Legacy GGML types

Supported for GGUF compatibility:

| Type | Bits |
|---|---|
| `q4_0`, `q4_1` | 4 |
| `q5_0`, `q5_1` | 5 |
| `q8_0` | 8 |

### FP8

E4M3 FP8. Native acceleration on NVIDIA Ada/Hopper (compute 8.9+); runs emulated elsewhere.

| Type | Bits | Layout |
|---|---|---|
| `fp8` | 8 | E4M3 (4-bit exponent, 3-bit mantissa) |
| `f8q8` | 8 | FP8 weights, INT8 activations |

### MXFP4

4-bit microscaling format. Native on Blackwell; emulated elsewhere.

| Type | Bits |
|---|---|
| `mxfp4` | 4 |

### HQQ

Half-quadratic quantization.

| Type | Bits |
|---|---|
| `hqq4` | 4 |
| `hqq8` | 8 |

## GPTQ and AWQ

Not ISQ types, pre-quantized formats. Load directly when a Hugging Face model is available as GPTQ or AWQ:

```bash
mistralrs run --format plain -m <gptq-or-awq-repo>
```

mistral.rs detects the quantization from the model's config. No `--quant` or `--isq` required.

See the [quantization guide](/mistral.rs/guides/quantization/quantize-a-model/) for format selection.
