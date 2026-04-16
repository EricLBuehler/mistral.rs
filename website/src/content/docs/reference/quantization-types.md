---
title: Quantization types
description: Every ISQ type mistralrs supports, what hardware it works on, and how it compares.
sidebar:
  order: 13
---

This page documents the specific ISQ types mistralrs understands. For guidance on picking one, see the [quantization decision guide](/mistral.rs/guides/perf/pick-a-quantization/). For the concept behind quantization tradeoffs, see [the explanation page](/mistral.rs/explanation/quantization-tradeoffs/).

## Numeric shorthands

These are the easiest inputs. Pass `--isq N` where N is a number and mistralrs picks a format appropriate for your hardware.

| Shorthand | Metal resolves to | CUDA / CPU resolves to |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

The shorthand is always the right choice unless you have a specific format in mind.

## Format-specific types

### AFQ family (Metal-optimized)

AFQ is our native quantization designed specifically for Apple Silicon GPUs. It uses adaptive float quantization that maps well to Metal's math operations.

| Type | Bits | Notes |
|---|---|---|
| `afq2` | 2 | Aggressive. Quality loss noticeable. |
| `afq3` | 3 | Good balance for tight memory. |
| `afq4` | 4 | The sweet spot for most workloads. |
| `afq6` | 6 | Near-lossless; slightly larger than Q4. |
| `afq8` | 8 | Effectively lossless. |

AFQ types only work on Metal. Loading an AFQ file on CUDA or CPU returns an error.

### Q*K family (CUDA and CPU)

The Q*K formats come from the GGML ecosystem. They work on everything and are a reasonable default for non-Metal.

| Type | Bits | Notes |
|---|---|---|
| `q2k` | 2 | Aggressive. Only use when nothing else fits. |
| `q3k` | 3 | Decent for memory-tight cases. |
| `q4k` | 4 | The most common choice. Works well across hardware. |
| `q5k` | 5 | A step up from Q4K with modest memory cost. |
| `q6k` | 6 | Near full precision. |

### Legacy GGML types

Supported for GGUF compatibility:

| Type | Bits | Notes |
|---|---|---|
| `q4_0`, `q4_1` | 4 | Older 4-bit formats. Prefer Q4K. |
| `q5_0`, `q5_1` | 5 | Older 5-bit formats. Prefer Q5K. |
| `q8_0` | 8 | 8-bit; a common choice for high-quality quantization. |

If you have a GGUF file using one of these types, it loads correctly. For new UQFF conversions, prefer the Q*K or AFQ types.

### FP8 formats (NVIDIA Hopper and newer)

Native FP8 on GPUs that support it:

| Type | Bits | Notes |
|---|---|---|
| `fp8_e4m3` | 8 | 4-bit exponent, 3-bit mantissa. Better range. |
| `fp8_e5m2` | 8 | 5-bit exponent, 2-bit mantissa. Better precision for small values. |

Require compute capability 8.9 or higher. On older hardware, falling back to Q8_0 is typically what you want.

### MXFP4 (Blackwell)

4-bit microscaling format:

| Type | Bits | Notes |
|---|---|---|
| `mxfp4` | 4 | Native on Blackwell; emulated elsewhere with worse performance. |

Very new. Model support is still expanding.

### HQQ (experimental)

Half-quadratic quantization. Aggressive and quality-sensitive:

| Type | Bits | Notes |
|---|---|---|
| `hqq4` | 4 | Alternative 4-bit scheme. Quality varies by model. |
| `hqq8` | 8 | Alternative 8-bit scheme. |

HQQ is research-adjacent; we ship it but recommend Q4K or AFQ4 by default.

## GPTQ and AWQ

These are not ISQ types but pre-quantized formats. If a model on Hugging Face is available as GPTQ or AWQ, load it directly:

```bash
mistralrs run --format plain -m <gptq-or-awq-repo>
```

mistralrs detects the quantization from the model's config. No `--isq` needed.

## How to pick

The decision tree, in order:

1. If you are on Metal, use an AFQ type (the shorthand handles this automatically).
2. If you are on CUDA or CPU, use a Q*K type (again, shorthand handles it).
3. For bit width, start at 4 (the sweet spot) and move up to 8 if you have memory, or down to 3 if you do not.
4. Only reach for FP8, MXFP4, or HQQ if you have a specific reason. They are for users who know what they want.

The [pick-a-quantization guide](/mistral.rs/guides/perf/pick-a-quantization/) has prose explanations for each of these choices.
