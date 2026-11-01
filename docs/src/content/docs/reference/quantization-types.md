---
title: Quantization types
description: Supported runtime ISQ and pretrained checkpoint formats, numeric shorthands, and backend constraints.
---

The quantized weight types mistral.rs can run. You can create most of them yourself with ISQ
(in-situ quantization), where mistral.rs quantizes full-precision weights as it loads them. Others
only load from checkpoints that were published already quantized. For help choosing, see
[Quantize a model](/guides/quantization/quantize-a-model/).

Pass a bit width or one of the ISQ type names below to `--quant` or `--isq` on `run`, `serve`, and
`bench`:

```bash
mistralrs run -m Qwen/Qwen3-4B --quant 4      # bit width; the type depends on your backend
mistralrs run -m Qwen/Qwen3-4B --quant q6k    # an exact type
mistralrs run -m Qwen/Qwen3-4B --isq afq8     # always quantize while loading
```

`--quant` prefers a matching prebuilt file.
[Quantize a model](/guides/quantization/quantize-a-model/#how-mistralrs-chooses-what-to-load)
explains how it chooses and when it quantizes while loading instead. For GGUF models, see
[GGUF support](/reference/gguf-support/).

## Numeric shorthands

When mistral.rs quantizes while loading, a bit width maps to a type based on your backend:

| Shorthand | Metal | CUDA and CPU |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

## Sensitive tensor precision

Token embeddings and the output head are sensitive to quantization, so mistral.rs stores them at a
higher precision than the rest of the model:

| Model quantized as | Embeddings and output head stored as |
|---|---|
| AFQ2, AFQ3, AFQ4 | AFQ6 |
| AFQ6, AFQ8 | AFQ8 |
| Q2K, Q3K, Q4K, Q4_0, Q4_1 | Q6K |
| Q5K, Q6K, Q8K, Q5_0, Q5_1, Q8_0, Q8_1 | Q8_0 |

A type you set for a tensor in a [topology](/guides/perf/topology/) file overrides this. Vision and
audio tensors keep their own settings.

## Format-specific types

Some formats below describe their schemes as WxAy: weights stored in x bits and activations
computed in y bits. For example, W8A16 has 8-bit weights and 16-bit activations.

### AFQ family

Affine quantization, designed for Apple Silicon. Metal has native kernels, and AFQ also runs on
CUDA and CPU.

| Type | Bits |
|---|---|
| `afq2` | 2 |
| `afq3` | 3 |
| `afq4` | 4 |
| `afq6` | 6 |
| `afq8` | 8 |

### Q*K family

GGML K-quant formats. `q2k` through `q6k` work on all backends. As an ISQ type, `q8k` isn't
available on CUDA.

| Type | Bits |
|---|---|
| `q2k` | 2 |
| `q3k` | 3 |
| `q4k` | 4 |
| `q5k` | 5 |
| `q6k` | 6 |
| `q8k` | 8 |

### Legacy GGML types

Older GGML block formats, mainly for GGUF compatibility. As an ISQ type, `q8_1` isn't available on
CUDA.

| Type | Bits |
|---|---|
| `q4_0`, `q4_1` | 4 |
| `q5_0`, `q5_1` | 5 |
| `q8_0`, `q8_1` | 8 |

### FP8

FP8 weights use the E4M3 layout. Create them with `--isq fp8`, or load an FP8 checkpoint in one of
these formats:

| Checkpoint format | Supported dense linear schemes |
|---|---|
| Native `quant_method: "fp8"` | Tensor-scaled W8A16, static or dynamic tensor-scaled W8A8, and 128x128 weight with dynamic 1x128 activation scaling |
| compressed-tensors | Symmetric E4M3 W8A16 with tensor, channel, or block weight scales; static tensor W8A8; dynamic per-token tensor/channel W8A8; and dynamic block W8A8 |
| ModelOpt | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO`, [MXFP8](#mxfp8), and `MIXED_PRECISION` configurations combining supported schemes and unquantized layers |

The scheme comes from the checkpoint's config, including older layouts: compressed-tensors
`compression_config` and ModelOpt `hf_quant_config.json`. Accepted scale tensors:

- Weight scales: `weight_scale` or `weight_scale_inv`. Both hold the dequantization multiplier.
- Static activation scales: `input_scale` or `activation_scale`.
- Scale shapes: scalar, per-channel `[N]` or `[N, 1]`, block `[N/128, K/128]`, or ModelOpt block
  `[N/128, 1, K/128, 1]`.

Speed depends on the GPU and build. When there's no fast FP8 kernel for tensor or channel scales,
mistral.rs dequantizes the weights once and runs a 16-bit matmul. W8A8 outputs can then differ
slightly, because activations aren't rounded to FP8. See
[quantized execution](/developer/quantized-execution/) for backend details.

A fused `gate_up_proj` or `qkv_proj` projection with one scale per chunk needs equal-size chunks.
Other fused projections with per-chunk scales fail to load, as do these variants:

- E5M2 and FNUZ FP8.
- Asymmetric FP8.
- Output-activation quantization.
- MoE expert scales in a checkpoint-specific layout.

A checkpoint's own KV-cache quantization settings aren't applied. To store the KV cache in FP8, use
[`--pa-cache-type f8e4m3`](/guides/perf/paged-attention/#fp8-kv-cache).

| Type | Bits | Layout |
|---|---|---|
| `fp8` | 8 | E4M3 (4-bit exponent, 3-bit mantissa) |
| `f8q8` | 8 | CPU-only F8Q8 weights |

### MXFP8

Loads ModelOpt `MXFP8` checkpoints, which store E4M3 weights with one scale per 32 values. Dense
layers and MoE experts run on CUDA, and a CPU path is also available. On Blackwell GPUs, a build
with `cuda,cutile` and CUDA 13.3 or newer processes prompts on block-scaled tensor cores.

Limits:

- With tensor parallelism, each GPU's slice of a layer's input dimension must be a multiple of 32.
- You can't create MXFP8 with ISQ or save it to UQFF.

### NVFP4

Loads pretrained NVFP4 safetensors checkpoints. Each weight is a 4-bit E2M1 float, with one E4M3
scale per 16 weights and an FP32 global scale.

| Checkpoint format | Supported schemes |
|---|---|
| NVIDIA ModelOpt | `NVFP4` (W4A4), `W4A16_NVFP4`, and `MIXED_PRECISION` combining supported FP8/MXFP8/NVFP4 schemes and unquantized layers |
| compressed-tensors `nvfp4-pack-quantized` | Symmetric `tensor_group` weights with `group_size: 16`; W4A16 or calibrated W4A4 with `dynamic: "local"` input activations |

W4A16 keeps activations in BF16 or F16. W4A4 also quantizes activations to 4 bits, using the scale
calibrated in the checkpoint. Mixed checkpoints keep each layer's scheme. Older ModelOpt checkpoints
that keep their config in `hf_quant_config.json` also load.

ModelOpt `weight_scale_encoding: "csf"` is supported for dense and expert projections.
Qwen3.8-Flash-Next's n-gram embedding tables can also be NVFP4.

Fast NVFP4 inference needs:

- An NVIDIA Blackwell GPU.
- CUDA 13.3 or newer, in both the build and runtime environment, with a compatible `tileiras`.
- A binary built with `cuda,cutile`.
- BF16 or F16 model dtype.

For example, to run [NVIDIA's Qwen3-14B-NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3-14B-NVFP4):

```bash
cargo install --path mistralrs-cli --features cuda,cutile
mistralrs run -m nvidia/Qwen3-14B-NVFP4 --dtype bf16
```

The model config selects NVFP4 automatically, so don't pass `--quant` or `--isq`. See
[cuTile setup](/developer/moe-backends/) for toolkit selection and installation.

The Rust API and Python source builds also expose the `cutile` feature. Run the Rust example with:

```bash
cargo run --release -p mistralrs --example nvfp4 --features cuda,cutile
```

For a [Python source build](/developer/from-source/#python-wheels), run
`maturin develop --release --features cuda,cutile,pyo3/extension-module` from `mistralrs-pyo3`.

Limits:

- Dense and MoE projections are supported. All experts in a projection must use the same scheme.
- With tensor parallelism, each GPU's slice of a layer's input dimension must be a multiple of 16.
- You can't create NVFP4 with ISQ, save it to UQFF, or load it from GGUF.

Kernel selection, precision behavior, and benchmarking are described in [quantized execution](/developer/quantized-execution/#nvfp4).

### MXFP4

4-bit microscaling format for CUDA and Metal. CPU isn't supported, and CUDA kernel availability
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

GPTQ and AWQ checkpoints load as published; you can't create them with ISQ. mistral.rs reads the
quantization settings from the model's config, so don't pass `--quant` or `--isq`:

```bash
mistralrs run --format plain -m <gptq-or-awq-repo>
```
