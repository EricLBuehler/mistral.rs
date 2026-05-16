---
title: Quantization tradeoffs
description: How mistralrs applies in-situ quantization and chooses formats per device.
sidebar:
  order: 4
---

## In-situ quantization

`--isq` quantizes weights as the model loads. The full-precision weights are never resident in memory at the same time; the engine reads each weight, produces its quantized form, and discards the source before moving to the next.

First-run load is slower than pre-quantized formats (GGUF, UQFF), which have no conversion work to do.

## Numeric shorthand resolution

`--isq N` maps to a specific format based on the active device:

| N | Metal | CUDA / CPU |
|---|---|---|
| 2 | AFQ2 | Q2K |
| 3 | AFQ3 | Q3K |
| 4 | AFQ4 | Q4K |
| 5 | Q5K | Q5K |
| 6 | AFQ6 | Q6K |
| 8 | AFQ8 | Q8_0 |

Explicit format names (`q4k`, `afq8`, etc.) bypass the device check. Incompatible combinations (e.g., FP8 on pre-8.9 GPUs) are rejected at load time.

## Format families

- **Q\*K** (`Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`): GGML-compatible block quantization. Broadly applicable.
- **AFQ** (`AFQ2`, `AFQ3`, `AFQ4`, `AFQ6`, `AFQ8`): optimized for Apple Silicon. Also usable on CUDA.
- **Legacy GGML** (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`): supported for GGUF compatibility.
- **FP8** (`F8E4M3`): native FP8 matmul on compute capability 8.9+.
- **MXFP4**: 4-bit microscaling; native fast path on Blackwell.
- **HQQ** (`HQQ4`, `HQQ8`): alternative 4- and 8-bit schemes.

The numeric shorthand picks a format the active device supports; the explicit names override that.

## Organization: `default` vs `moqe`

`--isq-organization` selects which layers get quantized:

- `default`: every linear layer the pipeline exposes for quantization.
- `moqe`: only MoE expert layers; the shared (non-expert) trunk stays at native precision.

`moqe` is useful on mixture-of-experts models where the experts dominate parameter count. Non-MoE models do nothing meaningful with it.

## imatrix

An importance matrix is a per-weight scaling factor derived from running the unquantized model on calibration data and measuring each weight's contribution to output activations. The quantizer uses it to allocate precision to higher-impact weights.

Two flags:

- `--imatrix <path>`: load an existing imatrix file.
- `--calibration-file <path>`: generate an imatrix from calibration text at load time.

The two conflict. `--imatrix` is reused across runs; `--calibration-file` re-generates on every load. imatrix affects the Q*K and HQQ formats; AFQ and legacy GGML formats are unaffected.

## Interaction with paged attention and flash attention

ISQ applies to weights. The KV cache is a separate budget, paged attention manages its memory independently, and `--pa-cache-type` quantizes the cache itself.

Flash attention operates on activations, not weights, and composes with any ISQ format.

## UQFF

UQFF files are a serialized form of an ISQ-quantized model. `mistralrs quantize` runs ISQ and writes the result; `--from-uqff` loads that file without re-running the quantization step. Quality is identical at the same ISQ type; only load time differs.

## See also

- Guide: [pick a quantization](/mistral.rs/guides/perf/pick-a-quantization/).
- Reference: [quantization types](/mistral.rs/reference/quantization-types/), [UQFF format](/mistral.rs/reference/uqff-format/).
