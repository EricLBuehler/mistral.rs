---
title: Quantization types
description: Every ISQ type mistralrs supports, what hardware it works on, and how it compares.
sidebar:
  order: 13
---

ISQ types supported by mistral.rs. For format selection guidance, see the [quantization decision guide](/mistral.rs/guides/perf/pick-a-quantization/). For underlying tradeoffs, see [the explanation page](/mistral.rs/explanation/quantization-tradeoffs/).

## Numeric shorthands

Pass `--isq N` where N is a number; mistral.rs picks a format appropriate for the hardware.

| Shorthand | Metal resolves to | CUDA / CPU resolves to |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

Use the shorthand unless a specific format is required.

## Format-specific types

### AFQ family (Metal-optimized)

AFQ is the native quantization designed for Apple Silicon GPUs, using adaptive float quantization that maps well to Metal's math operations.

| Type | Bits | Notes |
|---|---|---|
| `afq2` | 2 | Aggressive. Quality loss noticeable. |
| `afq3` | 3 | Good balance under tight memory. |
| `afq4` | 4 | Sweet spot for most workloads. |
| `afq6` | 6 | Near-lossless; slightly larger than Q4. |
| `afq8` | 8 | Effectively lossless. |

AFQ is Metal-only. Loading AFQ on CUDA or CPU returns an error.

### Q*K family (CUDA and CPU)

Q*K formats come from the GGML ecosystem. They work on all backends and are the default for non-Metal.

| Type | Bits | Notes |
|---|---|---|
| `q2k` | 2 | Aggressive. Use only when nothing else fits. |
| `q3k` | 3 | Decent for memory-constrained cases. |
| `q4k` | 4 | Most common choice. Works well across hardware. |
| `q5k` | 5 | One step up from Q4K with modest memory cost. |
| `q6k` | 6 | Near full precision. |

### Legacy GGML types

Supported for GGUF compatibility:

| Type | Bits | Notes |
|---|---|---|
| `q4_0`, `q4_1` | 4 | Older 4-bit formats. Prefer Q4K. |
| `q5_0`, `q5_1` | 5 | Older 5-bit formats. Prefer Q5K. |
| `q8_0` | 8 | 8-bit; a common choice for high-quality quantization. |

GGUF files using these types load correctly. For new UQFF conversions, prefer Q*K or AFQ.

### FP8 formats (NVIDIA Hopper and newer)

Native FP8 on GPUs that support it:

| Type | Bits | Notes |
|---|---|---|
| `fp8_e4m3` | 8 | 4-bit exponent, 3-bit mantissa. Better range. |
| `fp8_e5m2` | 8 | 5-bit exponent, 2-bit mantissa. Better precision for small values. |

Require compute capability 8.9+. Older hardware should fall back to Q8_0.

### MXFP4 (Blackwell)

4-bit microscaling format:

| Type | Bits | Notes |
|---|---|---|
| `mxfp4` | 4 | Native on Blackwell; emulated elsewhere with worse performance. |

New. Model support is expanding.

### HQQ (experimental)

Half-quadratic quantization. Aggressive and quality-sensitive:

| Type | Bits | Notes |
|---|---|---|
| `hqq4` | 4 | Alternative 4-bit scheme. Quality varies by model. |
| `hqq8` | 8 | Alternative 8-bit scheme. |

HQQ is research-adjacent. Q4K or AFQ4 are the default recommendations.

## GPTQ and AWQ

Not ISQ types — pre-quantized formats. Load directly when a Hugging Face model is available as GPTQ or AWQ:

```bash
mistralrs run --format plain -m <gptq-or-awq-repo>
```

mistral.rs detects the quantization from the model's config. No `--isq` required.

## How to pick

Decision tree:

1. On Metal, use an AFQ type (shorthand handles automatically).
2. On CUDA or CPU, use a Q*K type (shorthand handles automatically).
3. Start at bit width 4. Move to 8 if memory permits or 3 if it does not.
4. Use FP8, MXFP4, or HQQ only with a specific reason.

For prose, see the [pick-a-quantization guide](/mistral.rs/guides/perf/pick-a-quantization/).
