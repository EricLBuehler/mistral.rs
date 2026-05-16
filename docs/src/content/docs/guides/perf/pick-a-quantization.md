---
title: Pick a quantization method
description: Choosing a quantization format for the hardware and workload.
sidebar:
  order: 1
---

mistral.rs supports multiple quantization formats. `--isq 4` is a common starting point.

## Numeric shorthand

`--isq N` resolves to a hardware-appropriate format:

| Shorthand | Metal | CUDA / CPU |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

Explicit format names (e.g., `q4k`, `afq8`) bypass the device check.

## Pre-quantized formats

| Format | When to use |
|---|---|
| UQFF | Native pre-quantized format. Loads via `--from-uqff`. See [UQFF guide](/mistral.rs/guides/perf/use-uqff/). |
| GGUF | Loaded via `--format gguf -f <file>`. |
| GPTQ, AWQ | Loaded directly with `--format plain` when the source repo is pre-quantized. |

## Bit width

Independent of format. Fewer bits produces a smaller model.

Supported widths: 2, 3, 4, 5, 6, 8. Full bit-width by format support: [quantization reference](/mistral.rs/reference/quantization-types/).

## Automated selection

`mistralrs tune -m <model>` recommends per-host quantization. See the [auto-tune guide](/mistral.rs/guides/perf/auto-tune/).
