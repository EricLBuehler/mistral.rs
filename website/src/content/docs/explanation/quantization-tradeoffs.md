---
title: Quantization tradeoffs
description: The speed, quality, and memory triangle. Why different quantization methods exist and when each one is the right answer.
sidebar:
  order: 4
---

Quantization replaces high-precision weights (usually 16-bit floating point) with lower-precision approximations. Less precision means smaller models, which means either that the same hardware can run a bigger model, or that the same model runs faster. The cost is some loss of output quality. This page is about how to reason about that cost.

## What quantization actually does

A language model's weights are mostly matrix multiplication operands. During inference, you multiply the activation vector (what the model is "thinking" right now) by each layer's weight matrix to get the next hidden state. If you can do that multiplication faster or with less memory, everything downstream is faster or uses less memory too.

Standard fp16 or bf16 weights are 2 bytes each. A 7B parameter model in bf16 is 14 GB of weight data. Quantization to 4 bits per weight brings that down to 3.5 GB. At 2 bits per weight, 1.75 GB.

The trick is that the weights are not random numbers. They are the result of training, which clusters them into distributions that are predictable enough that you can represent them with fewer bits and still get approximately the same outputs. The better a quantization format matches the statistics of real model weights, the less quality you lose for a given bit width.

## The triangle

Three things trade off:

- **Memory**: more bits means more bytes per weight means bigger models need more VRAM.
- **Speed**: more bits means more memory bandwidth per multiply, which is usually the bottleneck on modern GPUs. Less memory is faster.
- **Quality**: fewer bits means less precision means more quantization noise means worse outputs.

The interesting cases are where two of these are in tension with the third.

Want maximum speed? Use the fewest bits you can tolerate. 4 bits is the usual choice; 2 is possible but degrades quality sharply.

Want maximum quality? Stay at native precision. bf16 on modern hardware, fp16 on older.

Want to fit a large model on small hardware? Quantize aggressively. A 70B model at 4 bits is feasible on a 48 GB card; at native precision it needs much more.

The triangle is not free. Most formats sit near the Pareto frontier, meaning you cannot generally get "more speed for free" without giving up quality. You are picking a point on a curve.

## Why several formats exist

If quantization is just "represent weights with fewer bits," why are there half a dozen methods?

Because the statistics of weights are not uniform. Some layers tolerate aggressive quantization; others do not. Some GPU architectures have fast kernels for specific bit widths but not others. Different training regimes produce weights that cluster differently in ways that affect which quantization scheme works best.

The Q*K family (Q4K, Q5K, Q6K, Q8_0) is the most broadly applicable. It uses a mixed scheme where attention-relevant weights get more precision than MLP-relevant ones, which matches empirical sensitivity.

The AFQ family (AFQ4, AFQ6, AFQ8) is designed specifically for Apple Silicon's math pipeline. On Metal it is meaningfully faster than Q*K at the same quality level. On other hardware it does not apply.

FP8 formats use the modern GPU fp8 tensor cores for native low-precision math, which can be faster than integer quantization plus dequantization. They only work on hardware that has fp8 tensor cores (Hopper and newer).

MXFP4 is a newer microscaling 4-bit format that does something similar to the Q*K block structure but with different math. On Blackwell it has a fast native path; elsewhere it is emulated and offers no speed benefit.

The choice between formats is about matching the format to the hardware. Within a hardware class, the choice of bit width is about the speed-quality tradeoff.

## The quality cost is not linear

Going from 16 bits to 8 bits loses almost no quality on most benchmarks. Going from 8 to 4 loses a little more. Going from 4 to 3 is a noticeable step. Going from 3 to 2 is a large step, often including systematic errors on hard tasks.

The specific numbers depend on the model and the benchmark. Perplexity measures tend to show smooth degradation; task-specific benchmarks (reasoning, code, math) sometimes show cliffs where a particular bit width is noticeably worse. The [MMLU paper and many follow-ups](https://arxiv.org/abs/2212.10560) have detailed per-task numbers for common quantization schemes.

In practice, the rule of thumb is: use 4 bits as default, use 8 if you have the memory, use 2 or 3 only when the alternative is not running the model at all.

## Why some workloads are more sensitive

Not every task uses the model the same way. A task that needs precise long chains of reasoning (a hard math problem) can fail entirely if any step is off. A task that is generative and open-ended (creative writing) has many correct answers and is more forgiving.

Code generation sits somewhere in the middle. Small errors are often caught by syntax failures, but subtle logic bugs can pass unnoticed.

Multimodal generation is more sensitive than text, because the vision encoder's outputs are high-dimensional and small perturbations propagate through the cross-attention.

Diffusion models are more sensitive than language models, because generation is iterative and errors compound across denoising steps.

These sensitivities argue for different quantization defaults per modality. mistralrs does not currently pick different defaults automatically, but the [topology feature](/mistral.rs/guides/perf/topology/) lets you quantize different layers at different levels if you have a strong opinion.

## The imatrix technique

Importance matrices (imatrix) are a way to improve quantization quality without changing the bit width. The technique: run the full-precision model on a small calibration dataset, record how much each weight contributes to the model's output, and then quantize with more precision allocated to the weights that matter more.

This works. Models quantized with imatrix tend to score 0.5 to 2 perplexity points better than the same models quantized without, at the same bit width. For 4-bit or 3-bit quantization where quality is already on the edge, that is a meaningful improvement.

The cost is the calibration data and the time to run the model on it. For models you quantize once and serve many times, imatrix is worth the effort. For one-off quantization, it is usually not.

## When quantization is the wrong knob

Not every performance problem is memory-bound. If your model fits comfortably and you want more throughput, quantization helps by reducing memory bandwidth per operation. If your model is latency-bound on a single request, quantization helps less because the bottleneck is the sequential nature of generation, not the per-step cost.

Levers that help with sequential latency:

- Flash attention
- Speculative decoding
- Smaller model (different tradeoff)
- Faster GPU (also a different tradeoff)

Levers that help with throughput under concurrent load:

- Paged attention
- Quantization (indirectly, by freeing memory for more concurrent sequences)
- Multi-GPU tensor parallelism

The rule of thumb: if `nvidia-smi` shows your GPU memory is full, quantization helps. If your GPU is not memory-full but is still slow, something else is going on.

## The practical answer

For most users, most of the time: use `--isq 4`. It quantizes weights to 4 bits at load time, picks a format appropriate for your hardware, and gives you a model that is almost always indistinguishable from full precision on normal tasks.

When that is not what you want, the [pick-a-quantization guide](/mistral.rs/guides/perf/pick-a-quantization/) has a decision tree. For the underlying hardware compatibility, the [quantization types reference](/mistral.rs/reference/quantization-types/) has the exhaustive table.
