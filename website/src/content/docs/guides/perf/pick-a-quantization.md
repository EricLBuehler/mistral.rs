---
title: Pick a quantization method
description: Which quantization format to use, and when. A decision guide for ISQ, GGUF, UQFF, AWQ, and the rest.
sidebar:
  order: 1
---

mistral.rs supports many quantization formats. The default for most cases is `--isq 4`. This page covers when to choose differently.

## The three questions that matter

**What GPU?** Metal hardware benefits from AFQ formats. NVIDIA hardware benefits from Q*K formats. CPUs run either, slower. FP8 formats require compute capability 8.9+ (L40, 40-series, H-series).

**GGUF or unquantized checkpoint?** GGUF: use it directly. Unquantized: choose a format below.

**Loading speed.** ISQ runs at load time, adding seconds or minutes per startup. Pre-quantized formats (GGUF, UQFF) load directly.

## Format quick reference

| Format | Source | Best for | Notes |
|---|---|---|---|
| ISQ (`--isq 4`) | Unquantized checkpoint | First-time use, experimentation | Engine selects AFQ or Q*K. Slower first load than GGUF. |
| UQFF | Pre-converted on disk | Production, fast boot | Native pre-quantized format. Loads directly. |
| GGUF | llama.cpp ecosystem | Pre-converted by community | Broadly compatible. |
| GPTQ, AWQ | HuggingFace, offline quantized | Specific community quants | Loaded natively, no conversion. |
| HQQ | Research workloads | Aggressive quantization | Experimental; quality varies. |
| FP8 | NVIDIA FP8-capable GPUs | Modern cards | Native FP8 when supported. |
| MXFP4 | NVIDIA Blackwell | Lowest-bit on newest cards | Very new; check model support. |

## The decision

Stop at the first match:

1. **UQFF available on Hugging Face?** Use it. Fast load, ISQ-equivalent quality, no conversion. See the [UQFF guide](/mistral.rs/guides/perf/use-uqff/).
2. **GGUF available on Hugging Face?** Use it. Fast load, comparable quality at the same bit width, llama.cpp interop.
3. **Apple Silicon, max speed?** Use `--isq 4`. AFQ is meaningfully faster than GGUF Q4 on Metal.
4. **Specific community quant (GPTQ, AWQ)?** Use it directly.
5. **Default.** `--isq 4` on the unquantized checkpoint. CUDA and CPU get Q4K.

## Bit width selection

Independent of format. Fewer bits is smaller and usually faster, with growing tradeoffs.

- **8 bits (`--isq 8`)** — near-lossless on benchmarks. Default when memory is not scarce.
- **4 bits (`--isq 4`)** — common production choice. Distinguishable from full precision on hard reasoning; usually indistinguishable on chat, code, summarization.
- **3 bits (`--isq 3`)** — meaningful quality loss. Use when 4 bits does not fit.
- **2 bits (`--isq 2`)** — very aggressive. The model makes systematic errors. Last resort.

Full bit-width × format support: [quantization reference](/mistral.rs/reference/quantization-types/).

## When to defer to `mistralrs tune`

`mistralrs tune -m <model>` benchmarks several quantization levels and prints per-hardware tradeoffs. Details: [auto-tune guide](/mistral.rs/guides/perf/auto-tune/).

## When quantization is the wrong lever

Quantization addresses memory bottlenecks. For speed bottlenecks on a model that already fits:

- **Flash attention** on CUDA, especially Hopper. See the [flash attention guide](/mistral.rs/guides/perf/use-flash-attention/).
- **Paged attention** for high-concurrency serving. See the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/).
- **Speculative decoding** with a compatible draft model. See the [speculative decoding guide](/mistral.rs/guides/perf/speculative-decoding/).

If none of these help, the workload is memory-bandwidth-bound, and a different GPU is the next lever.
