---
title: Quantization types
description: Supported runtime ISQ and pretrained checkpoint formats, numeric shorthands, and backend constraints.
---

Quantization types supported by mistral.rs, including ISQ (in-situ quantization) and pretrained
checkpoints. For format selection guidance and underlying tradeoffs, see the
[quantization guide](/guides/quantization/quantize-a-model/).

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

These defaults apply to language token embeddings and output heads. Explicit [topology](/guides/perf/topology/) settings take precedence. Tied heads share the embedding storage. Auxiliary vision and audio tensors keep their model-specific settings.

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
| ModelOpt | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO`, [MXFP8](#mxfp8), and `MIXED_PRECISION` configurations combining supported schemes and unquantized layers |

Weight scales may be stored as `weight_scale` or `weight_scale_inv`; both names contain the
dequantization multiplier. Static activation scales may be stored as `input_scale` or
`activation_scale`. Scalar, channel `[N]`/`[N, 1]`, block `[N/128, K/128]`, and ModelOpt block
`[N/128, 1, K/128, 1]` layouts are normalized automatically. Older ModelOpt repositories that
place their configuration in `hf_quant_config.json` are also detected.

Acceleration depends on the GPU and build. Tensor/channel FP8 falls back to a cached dequantized-weight A16 matmul when its accelerated path is unavailable; that fallback does not reproduce activation quantization rounding. See [quantized execution](/developer/quantized-execution/) for backend details.

The checkpoint adapters currently cover dense projections and recognized `gate_up_proj`/`qkv_proj`
fusions. Partitioned tensor scales require equal-size fused chunks that the model loader exposes as
separate output shards. Direct fused linears with a vector of scales, E5M2/FNUZ, asymmetric
FP8, output-activation quantization, FP8 KV cache checkpoint metadata, and checkpoint-specific MoE
scale layouts are separate formats.

| Type | Bits | Layout |
|---|---|---|
| `fp8` | 8 | E4M3 (4-bit exponent, 3-bit mantissa) |
| `f8q8` | 8 | CPU-only F8Q8 weights |

### MXFP8

ModelOpt `MXFP8` checkpoints use block-scaled E4M3 weights. Dense projections and expert stacks are supported on CUDA, with a CPU reference path. Blackwell GPUs with CUDA 13.3 or newer and `cuda,cutile` use native block-scaled tensor cores for prefill.

Input-dimension shards must align to 32 values. MXFP8 creation through ISQ and UQFF serialization are not supported.

### NVFP4

Load pretrained NVFP4 safetensors checkpoints directly. NVFP4 stores two E2M1 values per byte,
with an E4M3 scale per 16 weights and a separate FP32 global scale.

| Checkpoint format | Supported schemes |
|---|---|
| NVIDIA ModelOpt | `NVFP4` (W4A4), `W4A16_NVFP4`, and `MIXED_PRECISION` combining supported FP8/MXFP8/NVFP4 schemes and unquantized layers |
| compressed-tensors `nvfp4-pack-quantized` | Symmetric `tensor_group` weights with `group_size: 16`; W4A16 or calibrated W4A4 with `dynamic: "local"` input activations |

W4A16 preserves BF16/F16 activations. W4A4 quantizes activation blocks to E2M1 using the
checkpoint's calibrated global scale. The loader preserves this choice, including mixed checkpoints,
and reads older ModelOpt metadata from `hf_quant_config.json`. ModelOpt uses `weight_scale_2` and
`input_scale`; compressed-tensors uses reciprocal `weight_global_scale` and `input_global_scale`.
These conventions are normalized during loading.

ModelOpt `weight_scale_encoding: "csf"` is supported for dense and expert projections. NVFP4 Qwen n-gram embedding tables are also supported.

Accelerated inference requires an NVIDIA Blackwell GPU, CUDA 13.3 or newer, a compatible `tileiras`,
and a binary built with `cuda,cutile`. Use BF16 or F16 model dtype. With CUDA 13.3 or newer selected in the
build and runtime environment, load [NVIDIA's Qwen3-14B-NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3-14B-NVFP4):

```bash
cargo install --path mistralrs-cli --features cuda,cutile
mistralrs run -m nvidia/Qwen3-14B-NVFP4 --dtype bf16
```

The model config selects NVFP4 automatically; omit `--quant` and `--isq`. See
[cuTile setup](/developer/moe-backends/) for toolkit selection and installation.

The Rust API and Python source builds also expose the `cutile` feature. Run the Rust example with:

```bash
cargo run --release -p mistralrs --example nvfp4 --features cuda,cutile
```

For a [Python source build](/developer/from-source/#python-wheels), run
`maturin develop --release --features cuda,cutile,pyo3/extension-module` from `mistralrs-pyo3`.

Dense and MoE projections are supported. Experts within each projection must share a quantization
scheme, and input-dimension shards must align to 16 weights. NVFP4 creation through ISQ, NVFP4 UQFF
serialization, and NVFP4 GGUF loading are not supported.

Kernel selection, precision behavior, and benchmarking are described in [quantized execution](/developer/quantized-execution/#nvfp4).

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
