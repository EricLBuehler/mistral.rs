---
title: Prometheus metrics
description: Metric names, types, labels, and interpretation for mistralrs serve.
---

`GET /metrics` exposes these metrics in Prometheus text format. For configuration and dashboard queries, see [observability](/guides/deploy/observability/).

HTTP and streaming metrics carry [request labels](#request-labels). Engine metrics have no `model` label: they add up every model in the process. For per-model engine dashboards, run each model in its own process.

## HTTP metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `http_requests_total` | counter | `method`, `path`, `model`, `status` | Completed request count. |
| `http_request_duration_seconds` | histogram | `method`, `path`, `model`, `status` | End-to-end request latency. |
| `http_requests_in_flight` | gauge | `method`, `path`, `model` | Requests currently running. |
| `http_request_body_bytes` | histogram | `method`, `path`, `model` | Request body bytes when the body size is known. |
| `mistralrs_request_outcomes_total` | counter | `outcome` | Finished requests by result: `completed`, `error` (a 4xx or 5xx status, or an error mid-stream), or `client_disconnected` (the client closed a stream before it ended). |

## Streaming latency

For streaming (`text/event-stream`) responses, the middleware also records per-stream latency:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_time_to_first_token_seconds` | histogram | `method`, `path`, `model`, `status` | Time from request start to the first token. Includes queueing and prefill. |
| `mistralrs_inter_token_latency_seconds` | histogram | `method`, `path`, `model`, `status` | Gaps between consecutive token steps. One sample per step after the first. |

SSE keep-alives and other protocol events don't add samples. A high quantile of inter-token latency shows long decode stalls.

## Engine metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_tokens_processed_total` | counter | - | Tokens processed (prefill plus decode). Always equals the sum of the two phase counters below. |
| `mistralrs_prefill_tokens_processed_total` | counter | - | Prompt (prefill) tokens processed. |
| `mistralrs_decode_tokens_processed_total` | counter | - | Generated (decode) tokens processed. With speculative decoding only verified tokens count. |
| `mistralrs_prefix_cache_lookups_total` | counter | - | New requests that checked the prefix cache. |
| `mistralrs_prefix_cache_hits_total` | counter | - | New requests that reused a cached prefix. Divide by lookups for the hit rate. |
| `mistralrs_prefix_cache_evictions_total` | counter | - | Cached prefix data dropped to make room. With paged attention this counts KV blocks, plus saved recurrent states on hybrid models; without it, whole cached requests. |
| `mistralrs_sequences_running` | gauge | - | Sequences currently running. |
| `mistralrs_sequences_waiting` | gauge | - | Sequences in the waiting queue. |
| `mistralrs_sequences_capacity` | gauge | - | Maximum sequences the engine schedules concurrently (set at engine startup). |
| `mistralrs_sequences_completed_total` | counter | `reason` | Sequences that reached a terminal state. `reason` is one of `stop`, `length`, `canceled`, `generated_image`, `generated_speech`, `tool_calls`, `error`. |
| `mistralrs_paged_preemptions_total` | counter | - | Running requests paused and sent back to the queue because the KV cache ran out. They're recomputed later. If this keeps climbing, lower `--max-seqs` or give the cache more memory. |

## Multimodal encoder cache

Multimodal models cache the encoder output for each image, video, or audio input, so a repeated input isn't encoded again.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_encoder_cache_hits_total` | counter | - | Inputs whose encoder output came from the cache. |
| `mistralrs_encoder_cache_misses_total` | counter | - | Inputs that had to be encoded. |
| `mistralrs_encoder_cache_intra_batch_deduplications_total` | counter | - | Repeated inputs within one batch that were encoded only once. |
| `mistralrs_encoder_cache_evictions_total` | counter | `reason` | Cached outputs dropped. `reason` is `entry_capacity` (the entry limit was reached), `logical_byte_capacity` (the `--encoder-cache-memory-mb` limit was reached), or `incompatible_shape` (a cached output no longer matched). |
| `mistralrs_encoder_cache_insert_rejections_total` | counter | `reason` | Outputs that weren't cached. `reason` is `entry_exceeds_logical_byte_capacity` (one output is larger than the whole `--encoder-cache-memory-mb` budget) or `storage_compaction_failed`. |
| `mistralrs_encoder_cache_resident_entries` | gauge | - | Encoder outputs currently cached, across all models in the process. |
| `mistralrs_encoder_cache_resident_logical_bytes` | gauge | - | Size of the cached encoder outputs in bytes, across all models in the process. |

## KV cache blocks

With [paged attention](/guides/perf/paged-attention/), the KV cache is a pool of fixed-size blocks (32 tokens each by default, set by `--pa-block-size`).

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_kv_cache_blocks_total` | gauge | - | Blocks in the pool. |
| `mistralrs_kv_cache_blocks_active` | gauge | - | Blocks holding the cache of requests in flight. |
| `mistralrs_kv_cache_blocks_used` | gauge | - | Same value as `mistralrs_kv_cache_blocks_active`. |
| `mistralrs_kv_cache_blocks_reserved` | gauge | - | Blocks set aside for prompts that have started prefill but aren't finished. They're promised to those requests but not filled yet. |
| `mistralrs_kv_cache_blocks_available` | gauge | - | Blocks new work can use: total minus active minus reserved. Cached prefixes count as available, since they're dropped when space is needed. Near zero, new requests wait and running ones may be preempted. |
| `mistralrs_kv_cache_blocks_prefix_cached` | gauge | - | Blocks holding a cached prefix that a later request with the same start can reuse. Includes blocks that running requests are also using. |

## Hybrid model prefix cache

Hybrid models mix attention with recurrent layers, such as Qwen3.5 or Granite 4.0. To reuse a cached prefix, they need a saved copy of the recurrent state at the end of that prefix, and the KV blocks for the prefix must stay in the cache. These metrics track those saved states and the blocks kept for them.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_paged_recurrent_prefix_owners_used` | gauge | - | Cached prefixes that have a saved recurrent state. |
| `mistralrs_paged_recurrent_prefix_owners_capacity` | gauge | - | Most cached prefixes that can have one, from `--prefix-cache-n` or its automatic value. |
| `mistralrs_paged_recurrent_prefix_checkpoints_used` | gauge | - | Saved recurrent states. Lower than `owners_used` when several prefixes share one state. |
| `mistralrs_paged_recurrent_prefix_bytes_used` | gauge | - | Memory held by saved recurrent states, in bytes. |
| `mistralrs_paged_recurrent_prefix_validation_total` | counter | `outcome`, `reason` | Checks of a cached prefix before reuse. `outcome` is `hit` (the prefix is reused) or `miss` (the prompt is recomputed). `reason` gives the cause, such as `checkpoint_unavailable` when no matching saved state exists. |
| `mistralrs_paged_recurrent_prefix_owner_evictions_total` | counter | `reason` | Cached prefixes that lost their saved state. `capacity`: the `owners_capacity` limit was reached. `block_pressure`: the KV cache needed the prefix's blocks. If `capacity` climbs while many conversations are active, raise `--prefix-cache-n`. |
| `mistralrs_kv_cache_blocks_prefix_retained` | gauge | - | KV blocks kept for saved recurrent states, so those prefixes stay reusable. |
| `mistralrs_paged_prefix_retention_free_blocks` | gauge | - | Of those, blocks no running request is using. The cache takes them back when it runs short. |
| `mistralrs_paged_prefix_retention_entries` | gauge | - | Saved recurrent states whose KV blocks are being kept. Drops as soon as the cache takes blocks back. |
| `mistralrs_paged_prefix_retention_hashes` | gauge | - | Distinct cached blocks, by content, kept for saved recurrent states. |
| `mistralrs_paged_prefix_retention_pressure_evictions_total` | counter | `reason` | Times the cache stopped keeping a saved state's blocks. `allocation_pressure`: running requests needed the space. `reset`: the prefix cache was cleared. A steady climb in `allocation_pressure` means the KV cache is too small to hold both running requests and cached prefixes. |

## Speculative decoding (MTP)

These metrics are only populated when a model with a speculative proposer (MTP) is attached. `position` is the 0-based draft position within a proposal.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_speculative_drafts_total` | counter | - | Verified sequences: one per sequence per step that used speculative verification. |
| `mistralrs_speculative_draft_tokens_proposed_total` | counter | - | Draft tokens proposed for verification. |
| `mistralrs_speculative_draft_tokens_accepted_total` | counter | - | Draft tokens the target model accepted. |
| `mistralrs_speculative_draft_tokens_accepted_per_pos_total` | counter | `position` | Draft tokens accepted at each proposal position. The drop-off across positions shows where proposals start failing. |
| `mistralrs_speculative_staged_drops_total` | counter | - | Staged proposals discarded without verification (preemption, batch-shape mismatch, or a step that cannot verify). |

## Request labels

The `path` label is the matched route pattern, such as `/v1/responses/{response_id}`, not the raw URI. Requests that match no route get `<unmatched>`.

The `model` label depends on the route:

| Request | `model` label |
|---|---|
| Inference: chat completions, completions, responses, messages, embeddings, image generation, speech | The `model` field. If it's omitted or `default`, the server's default model. |
| LoRA adapter load, unload, and list | The `model` field or query parameter, with the same default. |
| Model management: `/v1/models/unload`, `reload`, `status`, `tune` | The `model_id` field. |
| A body that isn't valid JSON, or names a model the server doesn't have | `unknown` |
| Any other route | `none` |

HTTP metrics skip housekeeping traffic, so scrapes and probes don't show up in request dashboards: `OPTIONS` requests, `/`, `/health`, `/metrics`, `/docs`, the OpenAPI JSON, and `/ui`.
