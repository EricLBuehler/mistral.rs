---
title: "mistral.rs v0.9.5: faster Qwen3.8-Flash-Next on GB10"
author: Eric Buehler
date: 2026-10-28
slug: v0-9-5-qwen-flash-next
tags: [benchmarks, cuda, gguf, fp4]
---

# mistral.rs v0.9.5: faster Qwen3.8-Flash-Next on GB10

Qwen3.8-Flash-Next brings a demanding mix of routed experts, hyper-connections, attention, and per-layer embeddings. We optimized those paths in mistral.rs and measured warmed text generation on one NVIDIA GB10 at concurrency one.

With the same GGUF weights, mistral.rs delivers **60.5-69.1% higher effective prefill rates and 11.4-17.2% higher decode rates than the tested llama.cpp controls**. With the official FP4 checkpoint, it delivers **4.7-13.5% higher effective prefill rates and 2.3-3.7% higher decode rates than the tested optimized vLLM configuration**.

These are separate checkpoint comparisons, each with 512, 2,048, and 8,192 prompt tokens, 128 generated tokens, and no speculative decoding.

## GGUF: a lead in both phases

![GGUF effective prefill and decode throughput](figures/gguf_throughput.png)

Both engines load the same four `UD-Q4_K_XL` GGUF shards. Each llama.cpp bar is its strongest recorded median for that metric and prompt length across four controls, including both 512- and 8,192-token chunk settings. This gives the comparator the benefit of its best recorded setting in every cell.

| Prompt tokens | mistral.rs prefill | llama.cpp prefill | Gain | mistral.rs decode | llama.cpp decode | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,239.52 | 772.23 | +60.5% | 31.48 | 27.66 | +13.8% |
| 2,048 | 1,849.43 | 1,093.76 | +69.1% | 30.58 | 27.44 | +11.4% |
| 8,192 | 1,949.88 | 1,211.21 | +61.0% | 30.44 | 25.98 | +17.2% |

Rates are tokens/s. Results use medians of three measured requests after warmup. Both engines use BF16 KV caches; mistral.rs uses BF16 activations and casts F32 dense weights to BF16, while llama.cpp retains F32 dense storage. Generated tokens and expert routing can differ. Native `--quant 4` is a different quantization and is not used for this comparison.

## Official FP4: ahead of optimized vLLM at C1

![Official FP4 effective prefill and decode throughput](figures/fp4_throughput.png)

Both engines load the same official `nvidia/Qwen3.8-Flash-Next-FP4-Local` revision and keep its original BF16 output head. The vLLM comparator is a pinned GB10-optimized B12X image with resident device PLE, not a generic default configuration.

The activation policies differ: both use A4 expert prefill, while mistral.rs follows the checkpoint's A4 expert decode recipe and vLLM uses A16 expert decode. This is a comparison of the same checkpoint weights with those policies, not equal activation precision.

| Prompt tokens | mistral.rs TTFT (ms) | vLLM TTFT (ms) | Prefill rate gain | mistral.rs decode (tok/s) | vLLM decode (tok/s) | Decode gain |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 234.83 | 255.16 | +8.7% | 26.45 | 25.50 | +3.7% |
| 2,048 | 684.93 | 777.11 | +13.5% | 26.11 | 25.52 | +2.3% |
| 8,192 | 2,763.16 | 2,893.87 | +4.7% | 26.00 | 25.41 | +2.3% |

Each length has 18 measured requests across three rounds. mistral.rs leads in both metrics in all three rounds at every tested length. All 54 measured outputs repeat the conditioning outputs within each engine; additional repeated requests produce identical text and logprobs. Four short reference answers also match across engines. These checks establish repeatability on the fixture, not broad quality equivalence.

## Shared improvements

The work includes shared GGUF projection and expert kernels, mixed NVFP4/MXFP8 checkpoint loading, native packed FP4 decode, and cuTile prefill kernels. A shared hyper-connection layer fuses the up projection with gated residual mixing. File-mapped embedding storage avoids another full-table disk allocation, and kernel warmup now reports progress.

These paths dispatch on quantization format, dimensions, routing, and hardware capabilities. The serving results measure their combined effect; they do not isolate each kernel's contribution.

## Evidence and reproduction

Effective prefill is prompt tokens divided by client-observed TTFT, so it includes scheduling and first-token work. Decode is the rate after the first token, excluding prompt processing. FP4 uses reciprocal mean latency across 18 requests per length; GGUF uses medians of three. The GGUF snapshot and final FP4 implementation have separately recorded source identities.

The package includes curated raw responses, exact prompts, full configuration and precision disclosures, normalized JSONL/CSV, and scripts that regenerate both figures without running inference. Serving runners print the recorded launch settings for a fresh reproduction.

Read the [full report](report.md) for methodology, environment measurements, source provenance, and commands. These results cover this model family on this GB10 at C1; higher concurrency remains a separate measurement.
