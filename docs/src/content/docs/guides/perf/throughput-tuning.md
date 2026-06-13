---
title: Throughput tuning
description: Concurrency, prefix caching, and scheduling knobs for server workloads.
---

A handful of flags control how many requests run at once and how much repeated work the engine can skip. The defaults are reasonable for a single-user server; tune them when traffic is concurrent or shares long prompts.

```bash
mistralrs serve -m Qwen/Qwen3-4B \
  --max-seqs 64 \
  --pa-memory-fraction 0.9
```

Measure with `mistralrs bench -m <model>` before and after changing anything; flags that help one workload can hurt another.

## Concurrency: --max-seqs

`--max-seqs` caps the number of sequences running at once (default 32). Waiting requests queue until a slot frees. Raise it for high-concurrency serving; lower it to bound per-request latency or memory.

The SDK equivalents are `Runner(max_seqs=...)` in Python (default 16) and `.with_max_num_seqs(...)` on the Rust builders (default 32).

More running sequences need more KV cache. With paged attention, size the block pool to match: one of `--pa-context-len`, `--pa-memory-mb`, or `--pa-memory-fraction` (default: 90% of available VRAM). If the pool runs out of blocks mid-generation, the scheduler preempts a running sequence, frees its blocks, and requeues it, which costs recomputation.

## When paged attention matters

Concurrency beyond a few sequences is the main reason to care about [paged attention](/mistral.rs/guides/perf/paged-attention/). It is the difference between per-sequence contiguous caches sized for the maximum context and a shared block pool with a fixed budget:

- With paged attention (default on CUDA): continuous batching against the block pool, block-level prefix sharing, and the CUDA decode-graph and FlashInfer fast paths.
- Without it (Metal, CPU, or `--paged-attn off`): the default scheduler is first-come-first-served and batches running sequences by equal length; sequences at other lengths wait their turn and accumulate priority.

## Prefix caching

Prefix caching skips prefill for tokens the engine has already processed, which is most valuable for multi-turn chat and shared system prompts.

`--prefix-cache-n` sets how many cached prefixes stay on device (default 16, `0` disables). `--no-kv-cache` disables the KV cache entirely and prefix caching with it.

The mechanism depends on the attention mode:

- Paged attention: sequences sharing a token prefix reuse the same reference-counted cache blocks; matching blocks of a new request are not recomputed.
- Non-paged: completed sequences' caches are kept whole; a new request whose tokens extend a cached sequence resumes from it. Beyond `--prefix-cache-n` device entries, caches are evicted to CPU memory.

## Memory planning for automatic device mapping

Automatic device mapping reserves activation and KV memory for a worst case you declare: `--max-seq-len` (default 4096) and `--max-batch-size` (default 1). If you expect long prompts or large batches, raise these so layers are not placed too greedily; if memory is tight, the defaults already keep reservations small. See [distributed inference](/mistral.rs/guides/perf/distributed-inference/) for manual placement.

## CUDA defaults worth knowing

On CUDA, two decode fast paths are on by default and need no tuning: FlashInfer paged kernels and CUDA decode graphs. They exist as env-var switches (`MISTRALRS_FLASHINFER_DECODE`, `MISTRALRS_CUDA_GRAPHS`) only for debugging and benchmarking comparisons; details on the [paged attention page](/mistral.rs/guides/perf/paged-attention/).

Quantization is the other big throughput lever, traded against quality: see [quantize a model](/mistral.rs/guides/quantization/quantize-a-model/).
