---
title: Pick a quantization method
description: Which quantization format to use, and when. A decision guide for ISQ, GGUF, UQFF, AWQ, and the rest.
sidebar:
  order: 1
---

mistral.rs supports a lot of quantization formats. This page is a guide to picking the right one. The short answer, for most people, is "use `--isq 4` and move on." The long answer covers when that is wrong.

## The three questions that matter

Before looking at format names, answer three things about your workload:

**What GPU are you running on?** Metal-based hardware benefits from AFQ formats. NVIDIA hardware benefits from Q*K formats. CPUs can use either but are slower than GPUs regardless. Fp8 formats require compute capability 8.9 or higher (L40, 40-series, H-series).

**Do you already have a GGUF file, or are you downloading from a full-precision checkpoint?** If you have a GGUF, use it directly; there is nothing to decide. If you are starting from an unquantized checkpoint, you have options.

**How much loading speed matters.** In-situ quantization (ISQ) does its work at load time, which adds seconds or minutes to startup depending on the model. Pre-quantized formats (GGUF, UQFF) load faster because the conversion was done once, in advance.

## Format quick reference

| Format | Source | Best for | Notes |
|---|---|---|---|
| ISQ (`--isq 4`) | Unquantized checkpoint | First-time use, experimentation | Engine picks AFQ or Q*K for you. Loads slower than GGUF on first run. |
| UQFF | Pre-converted on disk | Production, fast boot | Our native pre-quantized format. Loads directly, no conversion. |
| GGUF | llama.cpp ecosystem | When someone else already converted it | Broadly compatible, often community-maintained. |
| GPTQ, AWQ | HuggingFace, offline quantized | Running a specific community quant | Loaded natively; no conversion step. |
| HQQ | Research workloads | Very aggressive quantization | Experimental; quality varies. |
| FP8 | NVIDIA fp8-capable GPUs | Squeezing more out of modern cards | Native FP8 when the hardware supports it. |
| MXFP4 | NVIDIA Blackwell | Absolute lowest-bit on the newest cards | Very new; check model support. |

## The decision

Run this top to bottom and stop at the first match:

1. **Is there a UQFF already on Hugging Face for this model?** Use it. Load is fast, quality matches ISQ, no conversion needed. See the [UQFF guide](/mistral.rs/guides/perf/use-uqff/).
2. **Is there a GGUF already on Hugging Face?** Use it. Load is fast, quality is usually comparable to ISQ at the same bit width, and you get format compatibility with llama.cpp.
3. **Do you need to run on Apple Silicon at maximum speed?** Use `--isq 4` (or your bit width of choice). The AFQ formats it selects are meaningfully faster than GGUF Q4 on Metal.
4. **Do you need a specific community quant (GPTQ, AWQ)?** Use it directly, no conversion required.
5. **Default for everything else.** Use `--isq 4` on the unquantized checkpoint. On CUDA and CPU you get Q4K, which is a well-tested format.

## Bit width selection

Orthogonal to the format, you also choose a bit width. Fewer bits is smaller and usually faster but tradeoffs grow.

- **8 bits (`--isq 8`)**: Near-lossless on benchmarks. Good default when memory is not scarce.
- **4 bits (`--isq 4`)**: The most common production choice. Clearly distinguishable from full precision on hard reasoning tasks, but usually indistinguishable on everyday chat, code, and summarization.
- **3 bits (`--isq 3`)**: Meaningful quality loss. Reach for it when 4 bits does not fit and the alternative is not running the model.
- **2 bits (`--isq 2`)**: Very aggressive. The model starts to make systematic errors. Use only when you have no better option.

The [quantization reference](/mistral.rs/reference/quantization-types/) has the exhaustive list of bit-width and format combinations, including which ones work on which hardware.

## When to offload work to `mistralrs tune`

If you do not want to make these decisions yourself, `mistralrs tune -m <model>` runs a benchmark at several quantization levels and prints a table of the tradeoffs for your specific hardware. It is the closest thing to an automated answer we have. Details in the [auto-tune guide](/mistral.rs/guides/perf/auto-tune/).

## When quantization is the wrong lever

Quantization helps when memory is the bottleneck. When speed is the bottleneck on a model that already fits, the levers are different:

- **Flash attention** for CUDA, especially on Hopper. See the [flash attention guide](/mistral.rs/guides/perf/use-flash-attention/).
- **Paged attention** for high-concurrency serving. See the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/).
- **Speculative decoding** if you have a compatible draft model available. See the [speculative decoding guide](/mistral.rs/guides/perf/speculative-decoding/).

If those do not help either, you are usually memory-bandwidth-bound, and at that point a different GPU is the next lever.
