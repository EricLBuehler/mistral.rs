---
title: "mistral.rs v0.9.5: Qwen3.8-Flash-Next on GB10"
author: Eric Buehler
date: 2026-09-30
slug: v0-9-5-flash-next
tags: [benchmarks, cuda, gguf, moe]
draft: true
---

# mistral.rs v0.9.5: Qwen3.8-Flash-Next on GB10

mistral.rs v0.9.5 adds Qwen3.8-Flash-Next text and vision support from safetensors
and GGUF, plus built-in multi-token prediction from safetensors/UQFF. It also
fixes long-prompt cuTile GDN prefill and improves memory budgeting on unified-memory
CUDA devices such as DGX Spark and Jetson.

We measured the implementation on one NVIDIA GB10 with 128 GB of unified memory.
With Q4K in-situ quantization and built-in MTP, the final mixed-prompt workload
reaches **54.7 output tok/s at concurrency one and 131.7 aggregate tok/s at
concurrency eight**.

## Run Flash-Next

Load safetensors with Q4K quantization and the checkpoint's built-in MTP head:

```bash
mistralrs run -m Qwen/Qwen3.8-Flash-Next --isq q4k --mtp \
  --max-model-len 16384 --max-seqs 8
```

Or select a GGUF variant and its vision projector together:

```bash
mistralrs run -m unsloth/Qwen3.8-Flash-Next-GGUF --quant 4 \
  --max-model-len 16384 --max-seqs 8
```

These commands use the context and sequence limits tested on GB10. The GGUF
loader reconstructs the model configuration from metadata, so it does
not need a separate `config.json`. Built-in MTP is supported through safetensors
and UQFF; it is not available with GGUF loading.

## Same GGUF, compared with llama.cpp

![Flash-Next UD-Q4_K_XL serving: prefill, individual prompts, and eight-request bursts](figures/flash_next_gguf.png)

Using the same UD-Q4_K_XL files and BF16 KV cache, mistral.rs delivers:

- **55-58% higher prefill throughput** across 512-, 2,048-, and 8,192-token inputs.
- **12.7% higher eight-request burst throughput:** 74.7 versus 66.2 output tok/s.
- Essentially equal ordinary single-request throughput: **26.3 versus 26.5 tok/s**,
  averaged across eight prompts.

The llama.cpp serving reference includes a documented allocation-padding patch
needed to complete these requests. These are shared HTTP-harness results from
the recorded same-GGUF baseline, separate from native `llama-bench` timings and
the later ISQ/MTP runs. The [report](report.md#same-gguf-serving-comparison)
includes the patch, exact versions, and uncertainty.

## Built-in MTP and concurrent serving

![Flash-Next Q4K ISQ with built-in MTP at concurrency one, six, and eight](figures/flash_next_mtp.png)

| Concurrency | Aggregate output tok/s | Output tok/s per active request |
| ---: | ---: | ---: |
| 1 | 54.70 | 54.71 |
| 6 | 119.85 | 20.80 |
| 8 | 131.74 | 17.32 |

These are means of five measured closed-loop trials after two warmups. Each
request generates 128 tokens, and completed requests are replaced until the
trial finishes. ISQ and the GGUF above use different quantization choices.

We also checked the BF16 cuTile MoE path against vLLM using the same
Qwen3.5-35B-A3B checkpoint, with no quantization or MTP:

| Concurrency | mistral.rs output tok/s | vLLM output tok/s |
| ---: | ---: | ---: |
| 1 | 31.84 | 30.92 |
| 6 | 84.33 | 72.51 |
| 8 | 93.85 | 95.47 |

That is a different model and execution path. Both engines scale about 3x from
C1 to C8 in this control; it does not predict vLLM's Flash-Next performance.

## Reliability and loading improvements

The cuTile GDN prefill fixes address a block-solve compiler issue and BF16
rounding drift that could produce NaNs on long repetitive inputs. They apply to
the Qwen3.5/3.8 GDN path, beyond the new model.

Unified-memory CUDA devices now budget available RAM minus 1 GiB by default and
cap the KV cache to the context length. The existing memory-fraction override
remains available. GGUF device mapping uses exact per-layer weight sizes, and
raw completion responses preserve leading whitespace, including Python indentation.

The measurements use prerelease builds reporting version 0.9.4, with exact source
and binary identities recorded. Read the [full report](report.md) for methods,
limitations, raw-data links, and commands to regenerate every chart.
