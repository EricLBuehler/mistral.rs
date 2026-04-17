---
title: Quantization tradeoffs
description: The speed, quality, and memory triangle. Why different quantization methods exist and when each one is the right answer.
sidebar:
  order: 4
---

Quantization replaces high-precision weights (usually 16-bit floating point) with lower-precision approximations. Less precision means smaller models, allowing the same hardware to run larger models or the same model to run faster. The cost is some output quality loss.

## What quantization actually does

Model weights are mostly matrix multiplication operands. During inference, the activation vector is multiplied by each layer's weight matrix to produce the next hidden state. Faster or smaller multiplication makes everything downstream faster or smaller.

Standard fp16 or bf16 weights are 2 bytes each. A 7B model in bf16 is 14 GB. At 4 bits per weight: 3.5 GB. At 2 bits: 1.75 GB.

Weights are not random. Training clusters them into distributions predictable enough to represent with fewer bits while preserving approximate outputs. Better matching of the format to weight statistics means less quality loss for a given bit width.

## The triangle

Three things trade off:

- **Memory** — more bits, more bytes per weight, more VRAM required.
- **Speed** — more bits, more memory bandwidth per multiply, the modern GPU bottleneck. Less memory is faster.
- **Quality** — fewer bits, less precision, more quantization noise, worse outputs.

Cases where two are in tension with the third:

For maximum speed: use the fewest bits tolerable. 4 is typical; 2 is possible but degrades sharply.

For maximum quality: stay at native precision. bf16 on modern hardware, fp16 on older.

For fitting a large model on small hardware: quantize aggressively. A 70B model at 4 bits fits on a 48 GB card; at native precision it does not.

The triangle is not free. Most formats sit near the Pareto frontier — more speed costs quality. The choice is a point on a curve.

## Why several formats exist

Weight statistics are not uniform. Some layers tolerate aggressive quantization; others do not. Some GPU architectures have fast kernels for specific bit widths but not others. Different training regimes produce weights that cluster differently.

The Q*K family (Q4K, Q5K, Q6K, Q8_0) is the most broadly applicable. It uses a mixed scheme where attention weights get more precision than MLP weights, matching empirical sensitivity.

The AFQ family (AFQ4, AFQ6, AFQ8) is designed for Apple Silicon's math pipeline. On Metal it is meaningfully faster than Q*K at the same quality; elsewhere it does not apply.

FP8 formats use modern GPU FP8 tensor cores for native low-precision math, which can outperform integer quantization plus dequantization. Hopper and newer only.

MXFP4 is a newer microscaling 4-bit format with Q*K-like block structure but different math. Native fast path on Blackwell; emulated elsewhere with no speed benefit.

Format choice matches the hardware. Within a hardware class, bit width choice handles the speed-quality tradeoff.

## The quality cost is not linear

16 → 8 bits loses almost no quality on most benchmarks. 8 → 4 loses a little more. 4 → 3 is a noticeable step. 3 → 2 is large, often introducing systematic errors on hard tasks.

Specific numbers depend on model and benchmark. Perplexity tends to degrade smoothly; task-specific benchmarks (reasoning, code, math) sometimes show cliffs at particular bit widths. The [MMLU paper and follow-ups](https://arxiv.org/abs/2212.10560) have detailed per-task numbers.

Rule of thumb: 4 bits as default, 8 with available memory, 2 or 3 only when nothing else fits.

## Why some workloads are more sensitive

Tasks vary in their use of the model. Long chains of precise reasoning (hard math) can fail entirely on a single error. Generative open-ended tasks (creative writing) have many correct answers and tolerate more.

Code generation sits between. Syntax failures catch many small errors, but subtle logic bugs pass unnoticed.

Multimodal generation is more sensitive than text. The vision encoder's high-dimensional outputs propagate small perturbations through cross-attention.

Diffusion models are more sensitive than language models. Iterative generation compounds errors across denoising steps.

These sensitivities argue for different per-modality quantization defaults. mistral.rs does not currently differentiate automatically; the [topology feature](/mistral.rs/guides/perf/topology/) supports per-layer quantization for opinionated configurations.

## The imatrix technique

Importance matrices (imatrix) improve quantization quality without changing the bit width. The technique: run the full-precision model on a small calibration dataset, record per-weight contribution to output, and allocate more precision to higher-impact weights.

Quantization with imatrix typically scores 0.5 to 2 perplexity points better than without at the same bit width. Meaningful improvement at 4-bit or 3-bit where quality is already on the edge.

The cost is calibration data and runtime. Worth it for models quantized once and served many times. Usually not worth it for one-off quantization.

## When quantization is the wrong knob

Not every performance problem is memory-bound. If the model fits comfortably and more throughput is the goal, quantization helps by reducing memory bandwidth per operation. If a single request is latency-bound, quantization helps less because the bottleneck is sequential generation, not per-step cost.

Latency levers:

- Flash attention
- Speculative decoding
- Smaller model (different tradeoff)
- Faster GPU (different tradeoff)

Concurrent throughput levers:

- Paged attention
- Quantization (indirectly, by freeing memory for more concurrent sequences)
- Multi-GPU tensor parallelism

Rule of thumb: full GPU memory in `nvidia-smi` means quantization helps. Slow inference with non-full memory means something else is the issue.

## The practical answer

For most users: use `--isq 4`. Quantizes to 4 bits at load time, picks a hardware-appropriate format, produces a model usually indistinguishable from full precision on normal tasks.

For other cases, see the [pick-a-quantization guide](/mistral.rs/guides/perf/pick-a-quantization/) for a decision tree, and the [quantization types reference](/mistral.rs/reference/quantization-types/) for the hardware compatibility table.
