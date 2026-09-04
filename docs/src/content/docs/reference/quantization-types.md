---
title: Quantization types
description: Supported runtime ISQ formats, numeric shorthands, and backend constraints.
---

ISQ (in-situ quantization) types supported by mistral.rs. For format selection guidance and underlying tradeoffs, see the [quantization guide](/guides/quantization/quantize-a-model/).

For `run`, `serve`, and `bench`:

- `--quant N` selects a matching pre-quantized artifact. For safetensors sources, a missing UQFF
  falls back to runtime ISQ.
- `--isq N` forces runtime ISQ and skips the
  [UQFF (Universal Quantized File Format)](/reference/uqff-format/) lookup.

For a GGUF repository, `--quant` selects a matching published file. To requantize GGUF weights
instead, select an exact file with `-f` and pass `--isq`. See
[GGUF support](/reference/gguf-support/) for the accepted file formats. GGUF selection
requires an explicit bit width or format name; `--quant auto` is not supported.

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

## Sensitive tensor precision

Token embeddings and output heads use a higher-precision default than the rest of an aggressively
quantized model:

| Default model type | Effective embedding and output-head type |
|---|---|
| AFQ2, AFQ3, AFQ4 | AFQ6 |
| AFQ6, AFQ8 | AFQ8 |
| Q2K, Q3K, Q4K, Q4_0, Q4_1 | Q6K |
| Q5K, Q6K, Q8K, Q5_0, Q5_1, Q8_0, Q8_1 | Q8_0 |

Q8_0 is the common high-precision Q target because quantized embedding kernels support it across
CPU, CUDA, and Metal. This policy applies to token embeddings, quantized per-layer token embeddings,
`lm_head`, and the top-level `output` head. Gemma 4 applies it to the PLE token-embedding table while
keeping PLE projections at the model default and norms dense. Gemma 3n applies it to the PLE
token-embedding table in the default full configuration; explicit MatFormer slices keep that table
dense.

Each supported model loader declares the exact language embedding and output-head paths that receive
this policy. A similarly named tensor in a vision, audio, or auxiliary subtree is not promoted merely
because its name ends in `embed_tokens`, `word_embeddings`, or `lm_head`.

A tied output head reuses the effective embedding instead of storing a second copy. An explicit
per-tensor ISQ type in a [topology](/guides/perf/topology/) takes precedence over these
defaults.

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

GGML K-quant formats. Q2K through Q6K are supported on all backends; Q8K is available where the
backend supports it.

| Type | Bits |
|---|---|
| `q2k` | 2 |
| `q3k` | 3 |
| `q4k` | 4 |
| `q5k` | 5 |
| `q6k` | 6 |
| `q8k` | 8 |

### Legacy GGML types

Supported for GGUF compatibility:

| Type | Bits |
|---|---|
| `q4_0`, `q4_1` | 4 |
| `q5_0`, `q5_1` | 5 |
| `q8_0`, `q8_1` | 8 |

### FP8

E4M3 FP8 can be produced with ISQ or loaded directly from native FP8, compressed-tensors, and
NVIDIA ModelOpt checkpoints. The checkpoint adapters normalize tensor names, scale shapes, target
rules, exclusions, and tensor-parallel shards into the same linear runtime schemes. The
compressed-tensors metadata may use either `quantization_config` or the legacy
`compression_config` key.

| Checkpoint format | Supported dense linear schemes |
|---|---|
| Native `quant_method: "fp8"` | Tensor-scaled W8A16, static or dynamic tensor-scaled W8A8, and 128x128 weight with dynamic 1x128 activation scaling |
| compressed-tensors | Symmetric E4M3 W8A16 with tensor, channel, or block weight scales; static tensor W8A8; dynamic per-token tensor/channel W8A8; and dynamic block W8A8 |
| ModelOpt | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO`, and `MIXED_PRECISION` configurations composed of these schemes and unquantized layers |

Weight scales may be stored as `weight_scale` or `weight_scale_inv`; both names contain the
dequantization multiplier. Static activation scales may be stored as `input_scale` or
`activation_scale`. Scalar, channel `[N]`/`[N, 1]`, block `[N/128, K/128]`, and ModelOpt block
`[N/128, 1, K/128, 1]` layouts are normalized automatically. Older ModelOpt repositories that
place their configuration in `hf_quant_config.json` are also detected.

On cuTile builds, W8A16 keeps E4M3 weights resident and converts each weight tile to BF16 or F16
inside the GEMM. Tensor and channel W8A8 use dedicated CUDA static-tensor or dynamic per-token
activation quantizers followed by a cuTile FP8 GEMM. The existing 128x128 block W8A8 providers
remain available, including CUTLASS and cuTile paths. Tensor and channel schemes use a cached
dequantized-weight A16 matmul when their accelerated CUDA path is unavailable; this fallback does
not emulate activation quantize/dequantize rounding.

The checkpoint adapters currently cover dense projections and recognized `gate_up_proj`/`qkv_proj`
fusions. Partitioned tensor scales require equal-size fused chunks that the model loader exposes as
separate output shards. Direct fused linears with a vector of scales, MXFP8, E5M2/FNUZ, asymmetric
FP8, output-activation quantization, FP8 KV cache checkpoint metadata, and checkpoint-specific MoE
scale layouts are separate formats.

| Type | Bits | Layout |
|---|---|---|
| `fp8` | 8 | E4M3 (4-bit exponent, 3-bit mantissa) |
| `f8q8` | 8 | CPU-only F8Q8 weights |

### MXFP4

4-bit microscaling format for CUDA and Metal. CPU is not supported; CUDA kernel availability
depends on the build and GPU.

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

See the [quantization guide](/guides/quantization/quantize-a-model/) for format selection.
