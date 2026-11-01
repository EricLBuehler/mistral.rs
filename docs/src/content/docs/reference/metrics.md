---
title: Prometheus metrics
description: Metric names, types, labels, and interpretation for mistralrs serve.
---

`GET /metrics` exposes these metrics in Prometheus text format. For configuration and dashboard queries, see [observability](/guides/deploy/observability/).

HTTP and streaming metrics carry request labels. Engine metrics are process-wide: a `model` label on HTTP counters does not make engine counters per-model. Use separate processes for per-model engine dashboards.

## HTTP metrics

The HTTP middleware records:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `http_requests_total` | counter | `method`, `path`, `model`, `status` | Completed request count. |
| `http_request_duration_seconds` | histogram | `method`, `path`, `model`, `status` | End-to-end request latency. |
| `http_requests_in_flight` | gauge | `method`, `path`, `model` | Requests currently running. |
| `http_request_body_bytes` | histogram | `method`, `path`, `model` | Request body bytes when the body size is known. |

## Streaming latency

For streaming (`text/event-stream`) responses, the middleware also records per-stream latency:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_time_to_first_token_seconds` | histogram | `method`, `path`, `model`, `status` | Time from request start to the first engine token step. Includes queueing and prefill. |
| `mistralrs_inter_token_latency_seconds` | histogram | `method`, `path`, `model`, `status` | Gaps between consecutive engine token steps. One sample per step after the first. |

Timing measures engine token steps, including queueing and prefill for the first token. SSE keep-alives and protocol bookkeeping events do not contribute samples. Use histogram quantiles to find long decode stalls.


## Engine metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mistralrs_tokens_processed_total` | counter | - | Tokens processed (prefill plus decode). Always equals the sum of the two phase counters below. |
| `mistralrs_prefill_tokens_processed_total` | counter | - | Prompt (prefill) tokens processed. |
| `mistralrs_decode_tokens_processed_total` | counter | - | Generated (decode) tokens processed. With speculative decoding only verified tokens count. |
| `mistralrs_prefix_cache_lookups_total` | counter | - | New sequences that requested a prefix cache lookup. |
| `mistralrs_prefix_cache_hits_total` | counter | - | New sequences that found a cached KV prefix. |
| `mistralrs_prefix_cache_evictions_total` | counter | - | Prefix cache entries, logical recurrent owners, or paged KV hash keys evicted from cache. |
| `mistralrs_encoder_cache_hits_total` | counter | - | Multimodal encoder cache hits (multimodal models only). |
| `mistralrs_encoder_cache_misses_total` | counter | - | Multimodal encoder cache misses (multimodal models only). |
| `mistralrs_encoder_cache_intra_batch_deduplications_total` | counter | - | Repeated media items coalesced within one encoder batch. |
| `mistralrs_encoder_cache_evictions_total` | counter | `reason` | Encoder outputs removed because of entry, configured logical-byte, or shape constraints. |
| `mistralrs_encoder_cache_insert_rejections_total` | counter | `reason` | Encoder outputs rejected because one entry exceeds the configured byte budget or compact storage allocation failed. |
| `mistralrs_encoder_cache_resident_entries` | gauge | - | Encoder outputs resident across all encoder caches in the process. |
| `mistralrs_encoder_cache_resident_logical_bytes` | gauge | - | Logical tensor payload bytes resident across all encoder caches in the process. Byte-budgeted entries use compact storage. |
| `mistralrs_sequences_running` | gauge | - | Sequences currently running. |
| `mistralrs_sequences_waiting` | gauge | - | Sequences in the waiting queue. |
| `mistralrs_sequences_capacity` | gauge | - | Maximum sequences the engine schedules concurrently (set at engine startup). |
| `mistralrs_sequences_completed_total` | counter | `reason` | Sequences that reached a terminal state. `reason` is one of `stop`, `length`, `canceled`, `generated_image`, `generated_speech`, `tool_calls`, `error`. |
| `mistralrs_kv_cache_blocks_used` | gauge | - | KV cache blocks currently in use (paged attention). |
| `mistralrs_kv_cache_blocks_total` | gauge | - | Total KV cache blocks (paged attention). |
| `mistralrs_kv_cache_blocks_active` | gauge | - | Physical KV cache blocks referenced by running sequences. |
| `mistralrs_kv_cache_blocks_reserved` | gauge | - | Prompt-tail blocks logically reserved but not yet materialized. |
| `mistralrs_kv_cache_blocks_available` | gauge | - | Physical KV blocks available after subtracting prompt-tail reservations. |
| `mistralrs_kv_cache_blocks_prefix_cached` | gauge | - | Physical KV blocks that retain at least one prefix-cache key. |
| `mistralrs_kv_cache_blocks_prefix_retained` | gauge | - | Physical KV blocks protected by a paired recurrent prefix checkpoint. |
| `mistralrs_paged_prefix_retention_free_blocks` | gauge | - | Protected physical KV blocks not referenced by a running sequence. |
| `mistralrs_paged_prefix_retention_entries` | gauge | - | Active paired recurrent checkpoint leases. This drops immediately when allocator pressure revokes a lease. |
| `mistralrs_paged_prefix_retention_hashes` | gauge | - | Distinct block hashes protected by active paired checkpoint leases. |
| `mistralrs_paged_prefix_retention_pressure_evictions_total` | counter | `reason` | Paired checkpoint leases revoked to preserve allocation progress. `reason` is `allocation_pressure` or `reset`. |
| `mistralrs_paged_recurrent_prefix_owners_used` | gauge | - | Logical recurrent prefix owners currently retained. |
| `mistralrs_paged_recurrent_prefix_owners_capacity` | gauge | - | Configured logical recurrent prefix owner capacity. |
| `mistralrs_paged_recurrent_prefix_checkpoints_used` | gauge | - | Recurrent prefix checkpoints currently retained. |
| `mistralrs_paged_recurrent_prefix_bytes_used` | gauge | - | Logical tensor payload bytes retained by recurrent prefix checkpoints. |
| `mistralrs_paged_recurrent_prefix_validation_total` | counter | `outcome`, `reason` | Prefix validation outcomes, including attention-only hits and recurrent restore misses. |
| `mistralrs_paged_recurrent_prefix_owner_evictions_total` | counter | `reason` | Logical recurrent owners evicted by `capacity` or after `block_pressure` revokes their paired blocks. |
| `mistralrs_paged_preemptions_total` | counter | - | Running sequences preempted and returned to the waiting queue (paged attention). |

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

The `path` label is the matched route pattern, such as `/v1/responses/{response_id}`, not the raw URI. The `model` label is the resolved model id for inference requests, defaults to the server default model when the request omits `model`, uses explicit `model_id` values for model-management requests, uses `unknown` when the request body cannot be read or parsed as JSON, and uses `none` for routes that do not target a model. Unmatched requests are labeled `<unmatched>`. Housekeeping endpoints are excluded from HTTP metrics to keep scrape and probe traffic out of request dashboards.

