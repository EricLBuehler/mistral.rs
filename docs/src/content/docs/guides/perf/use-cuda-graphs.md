---
title: Use CUDA graphs
description: Reduce CUDA decode launch overhead with graph replay.
sidebar:
  order: 5
---

CUDA graphs capture a fixed decode step once and replay it with new token and metadata inputs. This reduces CPU launch overhead during autoregressive decoding. It does not change model math, sampling, or output quality.

## Default behavior

CUDA graphs are enabled by default for supported CUDA decode paths. To disable them for comparison or debugging:

```bash
MISTRALRS_CUDA_GRAPHS=0 mistralrs serve --paged-attn on -m <model>
```

They require a CUDA build and a CUDA device. They currently apply to decode, not prompt prefill.

## Requirements

Graph replay is attempted only when all of these are true:

- The model implementation declares CUDA decode graph support.
- PagedAttention is active.
- The step is single-token decode (`q_len == 1`), not the initial prompt chunk.
- The request is not using a speculative proposer path.
- The graph key matches the input shape, dtype, cache metadata shapes, and context bucket.

If any condition is not met, mistral.rs runs the normal CUDA path.

## Capture and replay

The first time a decode shape is seen, mistral.rs runs a normal warmup forward, captures a graph for that shape, uploads it, and caches it. Later matching decode steps copy the current input ids and PagedAttention metadata into graph-owned buffers and replay the graph.

The decode graph cache holds a small number of recent graph entries. New batch sizes, tensor shapes, or metadata layouts can trigger another capture.

If capture or replay fails, CUDA graphs are disabled for that loaded pipeline and a warning is logged. In that case generation continues through the normal CUDA path.

## Interaction with PagedAttention and FlashInfer

CUDA graphs are most useful with PagedAttention because the paged metadata gives the graph stable tensor shapes while the values inside those tensors change from step to step.

On CUDA, PagedAttention uses FlashInfer-backed paged kernels for supported decode paths by default. CUDA graphs can replay those kernels as part of the decode graph. To compare against the non-FlashInfer paged path:

```bash
MISTRALRS_FLASHINFER_DECODE=0 mistralrs serve --paged-attn on -m <model>
```

## When it helps

CUDA graphs help most when decode is limited by CPU launch overhead or many small kernels. They usually do little for prompt prefill, where larger matrix and attention kernels dominate.

For apples-to-apples benchmarking, keep the same prompt length, generation length, batch size, PagedAttention mode, and FlashInfer setting across runs.
