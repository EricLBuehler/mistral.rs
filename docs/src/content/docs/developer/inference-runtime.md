---
title: Inference runtime internals
description: Memory admission, paged attention kernels, and CUDA graph replay.
---

For serving configuration, use [paged attention](/guides/perf/paged-attention/) and [throughput tuning](/guides/perf/throughput-tuning/). These notes describe constraints relevant when changing the runtime.

## Memory planning and admission

The planner distinguishes persistent allocations from temporary execution workspace. Weights, KV pools, retained encoder caches, recurrent prefix snapshots, and CUDA graphs consume persistent capacity. Activation workspace and one recurrent staging snapshot use the execution budget.

`Auto` leaves the larger of 5% or 1 GiB as headroom on discrete GPUs. Unified-memory devices leave 25%, capped at 16 GiB, and Metal also observes its recommended working-set limit. Temporary work can use automatic headroom; an explicit utilization target remains a hard limit. Moving weights to host RAM does not free capacity on unified-memory devices.

Startup estimates use configured concurrency and prompt limits. Forward admission uses scheduled tokens and the selected attention path, reserving encoder workspace for multimodal work. Unknown paths keep conservative estimates. Allocation is checked again before serving: automatic capacities can shrink, while required capacities fail if they cannot fit.

Multi-model groups share persistent and execution budgets according to maximum serving concurrency. Reserve the other models' workspace before admitting retained graphs. Rust callers must set memory goals before `plan_paged_kv`; changed goals require re-planning.

## Attention dispatch

Metal's default HND cache supports native paged prefill and decode. Structured masks travel as lengths, positions, windows, and ranges. Arbitrary masks are read directly without materializing an attention-score matrix. MLA uses a separate layout and accounting path.

CUDA uses FlashInfer for compatible decode shapes, direct paged FlashAttention for eligible prefill chunks, and a gather-based fallback otherwise. Scratch allocations may remain reserved for reuse; reserved device memory is not all live tensors.

FP8 K/V scales remain fixed for the cache lifetime so existing entries are never reinterpreted. The default scale is 1.0. Cache storage type is independent of model-weight quantization.

## CUDA graphs

Replay requires model-declared support, active paging, single-token decode, no speculative proposer, and a matching graph key. The key includes shape, dtype, cache metadata, and context bucket. Inputs and metadata are copied into graph-owned buffers before replay; scheduler admission still happens normally.

Supported pipelines warm up common batch sizes at startup. New shapes can trigger additional capture; retained entries are bounded. Capture waits for paged group members to load and skips groups containing unplanned, nonpaged models. Memory pressure defers capture to ordinary execution. Other capture or replay failures disable graphs for that loaded pipeline and log a warning.
