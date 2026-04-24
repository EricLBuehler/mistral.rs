---
title: Quantize a model
description: Shrink a language model at load time so it fits on the GPU you have. Measure the savings and compare the quality. About ten minutes.
sidebar:
  order: 6
---

In-situ quantization (ISQ) stores weights in fewer bits at load time, without a pre-converted file. A 14B model in BF16 needs about 28 GB for weights; at 4 bits the same model is about 7 GB. The model used here is Gemma 4.

## Basic usage

Pass `--isq` with the bit width:

```bash
mistralrs run --isq 4 -m google/gemma-4-E4B-it
```

`--isq 4` quantizes every weight to 4 bits as the model loads, choosing a format per backend: AFQ4 on Metal, Q4K on CUDA or CPU. Weights are quantized as they arrive; the full unquantized model is never resident in memory.

Memory footprint scales roughly linearly with bits per weight: a model in BF16 (2 bytes/param) uses about half the memory at `--isq 8` and a quarter at `--isq 4`. KV cache memory depends on context length and is independent of quantization. Use `nvidia-smi` (or equivalent) and `mistralrs tune` to measure on your hardware.

## Bit widths

Supported widths: 2, 3, 4, 5, 6, 8. Fewer bits means less memory and more quality degradation.

## Picking a specific format

`--isq` also accepts format names:

```bash
mistralrs run --isq q4k -m google/gemma-4-E4B-it     # Q4K, CUDA/CPU friendly
mistralrs run --isq afq4 -m google/gemma-4-E4B-it    # AFQ4, Metal-optimized
mistralrs run --isq q8_0 -m google/gemma-4-E4B-it    # Q8_0, the GGUF standard
```

The full list is in the [quantization reference](/mistral.rs/reference/quantization-types/).

## Letting the tune command decide

`mistralrs tune` measures the memory/quality tradeoff for a specific model and host:

```bash
mistralrs tune -m google/gemma-4-E4B-it
```

The command loads the model at several quantization levels, runs a short benchmark on each, and prints memory usage, context length headroom, and a quality proxy. To emit the recommendation as config, pass `--emit-config recommended.toml` and run `mistralrs from-config -f recommended.toml`.

## Quantization from Python and Rust

The same option exists in both SDKs. From Python, pass `in_situ_quant`:

```python
runner = Runner(
    which=Which.Plain(model_id="google/gemma-4-E4B-it"),
    in_situ_quant="4",
)
```

From Rust, use `with_auto_isq`:

```rust
let model = ModelBuilder::new("google/gemma-4-E4B-it")
    .with_auto_isq(IsqBits::Four)
    .build()
    .await?;
```

Both accept the same values as the CLI flag.

## Notes

ISQ runs at model load time. The engine quantizes weights in parallel and on-the-fly as they arrive into the target format, so loading can take longer than loading an unquantized model. To avoid the conversion on repeated loads, save the result in UQFF format. See the [UQFF guide](/mistral.rs/guides/perf/use-uqff/).

Not every ISQ format works on every accelerator. Q*K works on all backends; AFQ formats require Metal; FP8 formats require an NVIDIA GPU with compute capability 8.9+. Loading an incompatible format returns an error. The numeric shorthand picks a compatible format for the detected backend.

Pre-quantized GGUF files are a separate path from ISQ. They load directly without conversion. See the [GGUF guide](/mistral.rs/guides/perf/pick-a-quantization/).

## See also

- [Guides](/mistral.rs/guides/) for specific tasks.
- [Reference](/mistral.rs/reference/) for flags and APIs.
- [Troubleshooting](/mistral.rs/reference/troubleshooting/).
