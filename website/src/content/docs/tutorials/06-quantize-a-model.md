---
title: Quantize a model
description: Shrink a language model at load time so it fits on the GPU you have. Measure the savings and compare the quality. About ten minutes.
sidebar:
  order: 6
---

Earlier tutorials used models small enough to fit on a typical GPU at native precision. Larger models require quantization. A 14B model in BF16 needs about 28 GB for weights alone; 30B and 70B models exceed the memory of most consumer cards.

Quantization stores each weight in fewer bits than the native format. Four bits per weight in place of sixteen yields a quarter-sized model with a small, usually tolerable drop in output quality. This tutorial covers in-situ quantization (ISQ) — quantization at load time, with no pre-converted file.

The model is Gemma 4.

## The lazy path

Pass `--isq` with the bit width:

```bash
mistralrs run --isq 4 -m google/gemma-4-E4B-it
```

`--isq 4` quantizes every weight to 4 bits as the model loads, choosing a format appropriate for the accelerator: AFQ4 on Metal, Q4K on CUDA or CPU. The full unquantized model is never resident in memory — this is what allows 70B-class models to fit on a 24 GB card.

Memory footprint scales roughly linearly with bits per weight: a model in BF16 (2 bytes/param) uses about half the memory at `--isq 8` and a quarter at `--isq 4`. KV cache memory depends on context length and is independent of quantization, so a running server uses more than the weight footprint alone. Use `nvidia-smi` (or equivalent) and `mistralrs tune` to see the per-model numbers on your hardware.

## What changes with each bit width

Fewer bits mean less memory and usually faster inference, with some quality degradation.

8 bits: largely indistinguishable from full precision on tracked benchmarks. The safe choice when memory is not tight.

4 bits: the practical sweet spot. Output is noticeably quantized on hard reasoning problems in side-by-side comparison; for chat, code generation, and summarization the difference is often imperceptible.

2 and 3 bits: for fitting very large models on very little memory. Quality drops more sharply. Worth trying when the alternative is not running the model at all.

## Picking a specific format

`--isq` also accepts format names:

```bash
mistralrs run --isq q4k -m google/gemma-4-E4B-it     # Q4K, CUDA/CPU friendly
mistralrs run --isq afq4 -m google/gemma-4-E4B-it    # AFQ4, Metal-optimized
mistralrs run --isq q8_0 -m google/gemma-4-E4B-it    # Q8_0, the GGUF standard
```

The full list is in the [quantization reference](/mistral.rs/reference/quantization-types/). The numeric shorthand picks well in most cases.

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

ISQ runs at model load time, so loading a quantized model from a fresh cache is slightly slower than loading the unquantized version because the engine dequantizes weights as they arrive and re-quantizes them into the target format. To avoid the conversion on repeated loads, save the result in UQFF format. See the [UQFF guide](/mistral.rs/guides/perf/use-uqff/).

Not every ISQ format works on every accelerator. Q4K works everywhere; AFQ formats require Metal; FP8 formats require an NVIDIA GPU with FP8 tensor cores. Loading a specific incompatible format fails with an explanatory message. The numeric shorthand picks a compatible format automatically.

Pre-quantized GGUF files on the Hugging Face hub are a separate path from ISQ. They load directly without conversion. When a model is available in GGUF, that is usually the fastest way to run it. See the [GGUF guide](/mistral.rs/guides/perf/pick-a-quantization/).

## What's next

This is the last tutorial. The [Guides](/mistral.rs/guides/) cover specific tasks; the [Reference](/mistral.rs/reference/) lists every flag and API method; the [Explanation](/mistral.rs/explanation/) pages cover design rationale. For known issues, see the [troubleshooting reference](/mistral.rs/reference/troubleshooting/).
