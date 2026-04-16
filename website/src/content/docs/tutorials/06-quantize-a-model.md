---
title: Quantize a model
description: Shrink a language model at load time so it fits on the GPU you have. Measure the savings and compare the quality. About ten minutes.
sidebar:
  order: 6
---

Every model so far in these tutorials has loaded small enough that we could be casual about memory: Qwen3-4B and Gemma 4 both fit comfortably on a modestly specced GPU at their native precision. That stops being true as soon as you reach for something larger. A 14-billion-parameter model in BF16 needs around 28 GB just for weights, and any 30B-class or 70B-class model is out of reach for almost any consumer card.

Quantization is how you get those models into the memory you actually have. The idea is straightforward: store each weight with fewer bits than the native format uses. Four bits per weight instead of sixteen gives you a model that is a quarter of the size, with a small and usually tolerable loss in output quality. This tutorial walks through in-situ quantization with mistral.rs, which is the flavor that applies at load time without needing a pre-converted file.

We will use Gemma 4 again. Small enough that the numbers are easy to reason about, and large enough that the memory savings are noticeable.

## The lazy path

The shortest way to quantize is to pass `--isq` with the bit width you want:

```bash
mistralrs run --isq 4 -m google/gemma-4-E4B-it
```

That is all. `--isq 4` tells the engine to quantize every weight to 4 bits as it loads, picking a format that is appropriate for whatever accelerator you are on. On Metal you get AFQ4, on CUDA or CPU you get Q4K. Either way, the full unquantized model never has to exist in memory at any point during loading, which matters because it is what makes 70B-class models fit on a 24 GB card.

Watch the memory footprint go down as you change the bit width. Without quantization, Gemma 4 E4B takes around 10 GB on a CUDA device:

```bash
mistralrs run -m google/gemma-4-E4B-it
# nvidia-smi: ~10 GB used
```

With `--isq 8`, halved to 5 GB:

```bash
mistralrs run --isq 8 -m google/gemma-4-E4B-it
# nvidia-smi: ~5.1 GB used
```

With `--isq 4`, down to around 2.8 GB:

```bash
mistralrs run --isq 4 -m google/gemma-4-E4B-it
# nvidia-smi: ~2.8 GB used
```

Those numbers are just the model weights. You also need headroom for the KV cache, which depends on context length and is independent of how you quantized, so a running server will always use a bit more than the weight footprint alone suggests.

## What changes with each bit width

The bit widths follow the intuition you would hope for. Fewer bits means less memory and usually faster inference, at the cost of some degradation in output quality. Where on that curve you want to sit depends on what you are doing with the model.

Eight bits is mostly indistinguishable from the full-precision model on the benchmarks we track. If you only care about memory and you are not pushing against a tight VRAM budget, this is the safe choice.

Four bits is the sweet spot for most practical use. Model output is noticeably quantized if you compare side-by-side on tricky reasoning problems, but for normal chat, code generation, and summarization tasks the difference is often not something you would notice without looking for it.

Two and three bits exist for extreme cases where you are trying to fit something very large onto very little memory. Quality drops more sharply here. Worth trying if the alternative is not being able to run the model at all; otherwise stick with four.

## Picking a specific format

The numeric shorthand picks a reasonable default for your platform, but if you want a particular format directly, `--isq` accepts format names as well:

```bash
mistralrs run --isq q4k -m google/gemma-4-E4B-it     # Q4K, CUDA/CPU friendly
mistralrs run --isq afq4 -m google/gemma-4-E4B-it    # AFQ4, Metal-optimized
mistralrs run --isq q8_0 -m google/gemma-4-E4B-it    # Q8_0, the GGUF standard
```

The full list of accepted names and what each one does is in the [quantization reference](/mistral.rs/reference/quantization-types/). Most people never need to care about the specific names; the numeric shorthand picks well.

## Letting the tune command decide

If you are not sure which bit width will give you the best balance of memory and quality for a specific model and your hardware, `mistralrs tune` will measure it for you:

```bash
mistralrs tune -m google/gemma-4-E4B-it
```

The command loads the model at several quantization levels, runs a short benchmark on each one, and prints a table showing memory usage, context length headroom, and a quality proxy so you can see the tradeoff spelled out. If you want the recommendation in a form you can feed back into the CLI, pass `--emit-config recommended.toml` and then run `mistralrs from-config -f recommended.toml`.

## Quantization from Python and Rust

The same knob exists in both SDKs. From Python, pass `in_situ_quant` to the `Runner`:

```python
runner = Runner(
    which=Which.Plain(model_id="google/gemma-4-E4B-it"),
    in_situ_quant="4",
)
```

From Rust, use `with_auto_isq` on the `ModelBuilder`:

```rust
let model = ModelBuilder::new("google/gemma-4-E4B-it")
    .with_auto_isq(IsqBits::Four)
    .build()
    .await?;
```

Both accept the same set of values as the CLI flag.

## Before you leave

A few things that come up once you start quantizing in anger.

In-situ quantization does its work at model load time. That means loading a quantized model from a fresh cache takes slightly longer than loading the unquantized version, because the engine is dequantizing weights as they arrive and then re-quantizing into the target format. If you are going to load the same model repeatedly, it is worth converting it once and saving the result in UQFF format, which loads directly without the extra conversion step. The [UQFF guide](/mistral.rs/guides/perf/use-uqff/) covers that workflow.

Not every combination of ISQ format and hardware is supported. Q4K works everywhere, AFQ formats work on Metal, FP8 formats need an NVIDIA GPU with fp8 tensor cores. When you pick a specific format that does not work on your machine, the CLI will refuse to load and print a message about it. When you use the numeric shorthand, this is not something you have to think about because the engine chooses a compatible format for you.

Pre-quantized models from the Hugging Face hub (GGUF files) are a separate path from in-situ quantization. Those already have the quantized weights on disk, so mistral.rs loads them directly without doing any conversion. If a model you want is available in GGUF form, that is almost always the fastest way to run it; see the [GGUF guide](/mistral.rs/guides/perf/pick-a-quantization/) for details.

## What's next

This is the last tutorial. From here, the [Guides](/mistral.rs/guides/) cover specific tasks in more detail, the [Reference](/mistral.rs/reference/) has every flag and API method, and the [Explanation](/mistral.rs/explanation/) pages go into the reasoning behind the design choices that showed up across these six tutorials. If you run into trouble, the [troubleshooting reference](/mistral.rs/reference/troubleshooting/) is the fastest way to find a known fix.
