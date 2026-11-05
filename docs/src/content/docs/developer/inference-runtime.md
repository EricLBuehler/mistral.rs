---
title: Inference runtime internals
description: Memory planning, attention kernel selection, and CUDA graph replay.
---

These notes are for contributors changing how mistral.rs budgets device memory, picks attention kernels, or replays CUDA graphs. To configure these features, see [paged attention](/guides/perf/paged-attention/) and [throughput tuning](/guides/perf/throughput-tuning/).

## Memory planning

Memory planning is spread across a few modules in `mistralrs-core/src/`:

| File | Role |
|---|---|
| `memory_plan.rs` | Turns a `MemoryGoal` and a snapshot of device memory into a byte budget. |
| `resource_plan.rs` | `plan_paged_kv` splits the budget between the models of a multi-model server. |
| `memory_budget.rs` | Tracks each model's share, and the runtime reservations models publish so they don't take each other's memory. |
| `pipeline/loaders/activation_memory.rs` | Estimates activation memory at load time. |
| `paged_attention/cache_allocation.rs` | Allocates the KV cache, measures real usage, and shrinks the cache if needed. |
| `engine/memory.rs` | Checks each scheduled batch against the budget before its forward pass. |

### Persistent memory and workspace

The budget code tracks two kinds of memory:

- **Persistent memory** stays allocated while the model is loaded: weights, the KV cache, retained encoder caches, recurrent-state snapshots kept for prefix caching, and captured CUDA graphs.
- **Execution workspace** is held only during a forward pass: activations, plus one staging slot for a recurrent-state snapshot.

### Budget limits

`resolve_device_budget` turns the memory goal into a limit on what mistral.rs may allocate:

| Goal | Limit |
|---|---|
| `Auto` (default), `KvBytes`, or `KvTokens` | Total memory minus headroom. |
| `Utilization(f)` | `f` times total memory. |

Headroom is 5% of total memory (at least 1 GiB) on discrete GPUs, and 25% (at most 16 GiB) on unified-memory devices. On Metal, the limit is also capped at the device's recommended working-set size.

Forward passes get a looser limit from `resolve_forward_budget`. Without a utilization goal, temporary work may also use the headroom, up to the Metal working-set limit on unified memory. A utilization goal stays a hard cap for forward passes too.

On unified-memory devices the CPU and GPU share RAM, so automatic device mapping (`pipeline/loaders/auto_device_map.rs`) never adds the CPU as a fallback device. Moving layers to the CPU wouldn't free anything there.

### Sizing the KV cache at load

1. Activation estimates assume the configured limits: `max_num_seqs`, `max_num_batched_tokens`, and the prefill chunk size.
2. The KV cache gets the persistent budget that's left after weights and reservations.
3. `allocate_cache` allocates the cache, measures the memory actually in use, and checks that the runtime reservations still fit.
4. If they don't, an automatic goal (`Auto` or a utilization fraction) shrinks the cache and tries again. A fixed size (`KvBytes` or `KvTokens`) fails with an error instead.

### Admitting each forward pass

Before each forward pass, `forward_workspace_bytes` estimates the workspace the scheduled batch needs:

- The estimate scales with the tokens actually scheduled.
- The attention part comes from the selected attention kernel. If the kernel can't report a complete figure, or the model uses an arbitrary custom mask, the load-time estimate is used as a floor.
- A batch with multimodal input also reserves encoder workspace on the primary device.

`enforce_memory_goal` then compares the estimate with the forward budget on each device. When a prompt batch doesn't fit:

1. On CUDA, cached graphs are freed first.
2. The engine keeps the first half of the batch and defers the rest.
3. This repeats until one request is left. If that one still doesn't fit, it fails with an error.

### Multiple models

`plan_paged_kv` gives each model a share of the budget in proportion to its `max_num_seqs`. If any model sets a utilization goal, all of them must use the same value.

Each model publishes a runtime reservation (its workspace plus pending recurrent snapshots) through `memory_budget.rs`. Before a model captures or keeps a CUDA graph, it checks that the other models' reservations still fit (`other_workspace_bytes`). If they don't, it runs that step without a graph.

`plan_paged_kv` reads the memory goal once. Rust callers must set the goal before calling it, and call it again if the goal changes.

## Attention kernels

### Metal

- The default paged KV cache layout is HND: `[blocks, kv_heads, block_size, head_dim]` per layer (`paged_attention/cache_engine.rs`). Set `MISTRALRS_METAL_PAGED_LAYOUT=legacy` for the older layout.
- Native paged kernels in `mistralrs-paged-attn/src/metal/` handle both prefill and decode on the HND layout.
- Common mask shapes reach the prefill kernel as parameters instead of a mask tensor: sequence lengths and positions, causal and sliding windows, chunked attention, and multimodal token ranges.
- Any other mask is passed as a tensor that the kernel reads directly. The full attention-score matrix is never built.
- MLA models such as DeepSeek V3 use their own cache layout and memory accounting.

### CUDA

In the tables below, "gather" means copying a sequence's cached K/V blocks into one contiguous buffer so a regular attention kernel can read them.

Decode picks a kernel in `DecodePlan::choose` (`paged_attention/plan.rs`): FlashInfer, the standard paged-attention kernel, or gather plus SDPA. Custom masks always use gather plus SDPA.

Prefill picks a kernel in `PrefixPrefillPlan::choose`, in the same file:

| Plan | Used when |
|---|---|
| `Fa3Fp8Paged` | BF16 queries over an FP8 cache with head size 256, and no ALiBi, sinks, softcap, sliding window, or custom mask. |
| `FlashAttentionPaged` | Built with `flash-attn`, F16 or BF16 with a cache of the same dtype, and no ALiBi, sinks, or custom mask. |
| `GatherSdpa` | Everything else. |

The first two read the paged cache directly. Both have further shape checks, listed in the function.

Device memory in use is higher than the memory held by live tensors. While graphs or the engine hold the CUDA memory pool, its release threshold is raised, so freed scratch buffers stay in the pool for reuse.

### FP8 KV cache scales

FP8 cache scales never change while the cache exists, so values already stored keep their meaning. Each attention layer reads `q_scale`, `k_scale`, and `v_scale` from the checkpoint. If they're missing, it uses 1.0 and logs a warning (`paged_attention/scales.rs`). The cache dtype is independent of weight quantization.

## CUDA graphs

A CUDA graph is a recorded sequence of kernel launches that can be replayed with one call, which removes most of the CPU cost of launching each kernel. The graph code is in `pipeline/cuda_graph.rs`. `pipeline/normal.rs` and `pipeline/multimodal.rs` decide when to capture and replay.

Target model graphs come in two kinds, and both live in the same per-pipeline cache:

- **Decode graphs** run a step where each sequence adds one token.
- **Verify graphs** run a speculative verify step, where the target model checks each sequence's last token plus `depth` drafted tokens in one forward of width `1 + depth`. See [speculative decoding](#speculative-decoding).

The cache key (`CudaDecodeGraphKey`) covers the device, the input shape (batch bucket and width) and dtype, the batch kind (`RecurrentBatchKind::Decode` or `RecurrentBatchKind::SpeculativeDecode`), the bucketed context length, and the shapes and dtypes of the attention metadata tensors. A verify graph never matches a decode graph.

A step is replayed from a graph only when all of these hold:

- `MISTRALRS_CUDA_GRAPHS` isn't set to off.
- The model supports graphs (`supports_cuda_decode_graphs`, or `supports_cuda_decode_graphs_for_args` in the multimodal pipeline), and no attention layer needs context lengths on the host.
- Paged attention is active, and the step isn't a prompt chunk.
- The step is a one-token decode, or a verify step that meets the conditions under [speculative decoding](#speculative-decoding).
- The batch size has a graph bucket. Decode graphs use exact sizes 1 to 16, verify graphs use exact sizes 1 to 8, and both then use multiples of 8 up to 128.
- A graph with a matching key exists, or one can be captured.

Before a replay, the inputs and metadata are copied into buffers the graph owns. Scheduling and memory admission still run as usual. Only the model forward is replayed.

Capture works like this:

- At startup, supported pipelines capture the common batch buckets up to the configured maximum batch size, for width 1 and for every planned verify width. Small batches of every width go first. Buckets that don't fit in memory are left for later.
- New shapes are captured the first time they appear. The cache keeps 64 to 96 graphs and evicts the least recently used.
- In a multi-model group, capture waits until every paged model has loaded and published its reservation. If any model in the group runs without paged attention, capture is off for the whole group, because that model publishes no reservation for graphs to respect.

When capture fails:

- If memory is short, the step runs without a graph.
- Any other capture or replay error logs a warning and turns graphs off for that pipeline until it's reloaded.

### Speculative decoding

Verify graphs are only captured in the multimodal pipeline (`pipeline/multimodal.rs`). That's where the models with built-in proposers load: Gemma 4 (MTP), Qwen3.5 (MTP or DFlash), and Qwen4-Exp (MTP). The text pipeline (`pipeline/normal.rs`) skips all graphs, including startup capture, when a proposer is attached. No text-pipeline model attaches one today.

With a proposer attached, a step uses a verify graph only when both of these hold:

- **The width and batch match a plan.** Each model lists its verify graphs through `speculative_graph_plans`, one `SpeculativeGraphPlan` per depth (`speculative/policy.rs`). A plan can cap the batch bucket. Extra depths added by depth autotuning are capped at batch 8 (`speculative/autotuner.rs`), and Qwen4-Exp sets a cap for each depth.
- **The model returns proposer state.** The proposer reads outputs of the target forward, such as hidden states. The model hands these over through `take_speculative_graph_state`. After a replay, the pipeline installs the graph's copy back into the model with `install_speculative_graph_state`. If the model returns nothing, the step runs without a graph.

Proposers keep their own graphs separate from the target cache:

- The DFlash drafter has its own cache in `speculative/dflash.rs`, keyed by batch bucket, draft block size, and token selection mode. It holds 40 graphs by default and is precaptured at load through `precapture_speculative_cuda_graphs`. It needs a Linux build with `cuda` and `flash-attn`. When no drafter graph fits a step, the draft runs without one.
- MTP proposers have no graph cache of their own.

When memory runs short, the least recently used target graphs are evicted first, then proposer graphs (`reclaim_cuda_graph_entries` in `pipeline/cuda_graph.rs`).
