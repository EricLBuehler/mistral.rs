#!/usr/bin/env python3
"""Production correctness, scheduler, and endurance testing for an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import shlex
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "default"
DEFAULT_SEED = 20260841
DEFAULT_TIMEOUT_SECONDS = 1_200.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_CONCURRENCIES = (1, 8, 16)
DEFAULT_CONTEXT_LENGTHS = (1_024, 8_192, 32_768, 100_000)
DEFAULT_LONG_CORRECTNESS_LENGTHS = (60_000, 65_535, 65_536, 65_537, 100_000)
DEFAULT_LONG_CORRECTNESS_CONCURRENCIES = (3,)
DEFAULT_RESIDENT_CONTEXT_LENGTHS = (60_000, 65_536, 100_000)
DEFAULT_RESIDENT_CONCURRENCIES = (1, 3, 8, 16)
DEFAULT_RESIDENT_REQUESTS = 16
ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS = 100_000
ADVERSARIAL_LONG_RESIDENT_REQUESTS = 3
DEFAULT_ADVERSARIAL_LONG_RESIDENT_MAX_TOKENS = 64
DEFAULT_ADVERSARIAL_LONG_RESIDENT_MIN_DECODE_TOK_S = {1: 190.0, 3: 300.0}
DEFAULT_ADVERSARIAL_CHURN_ROUNDS = 3
DEFAULT_CONTEXT_MIX = ((1_024, 45), (8_192, 30), (32_768, 20), (100_000, 5))
DEFAULT_PRODUCTION_DURATION_SECONDS = 14_400.0
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 5.0
DEFAULT_PROBE_INTERVAL_SECONDS = 30.0
DEFAULT_PRODUCTION_DIAGNOSTIC_CONCURRENCY = len(DEFAULT_CONTEXT_LENGTHS)
DEFAULT_COMPARISON_WINDOW_SECONDS = 3_600.0
DEFAULT_MIN_COMPARISON_WINDOW_SAMPLES = 32
DEFAULT_PRODUCTION_MIN_OUTPUT_TOK_S_BY_CONCURRENCY = {8: 450.0, 16: 500.0}
DEFAULT_OUTPUT_ROOT = Path("artifacts/production_soak")
OUTPUT_PREVIEW_CHARS = 256
DEFAULT_MAX_REPEATED_NGRAM_RATIO = 0.20
REPETITION_NGRAM_WIDTH = 4
REPETITION_ALLOWED_OCCURRENCES = 2
REPETITION_TAIL_TOKENS = 64
MAX_PERIODIC_PATTERN_TOKENS = 32
MIN_PERIODIC_REPETITIONS = 3.0
MIN_PERIODIC_LOOP_TOKENS = 16
DEFAULT_FAIRNESS_MAX_SLOWDOWN = 3.0
DEFAULT_FAIRNESS_STAGGER_SECONDS = 0.05
DEFAULT_FAIRNESS_MAX_SHORT_TTFT_SECONDS = 1.0
DEFAULT_FAIRNESS_MAX_SHORT_TPOT_SECONDS = 0.05
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 30.0
DEFAULT_CLEANUP_POLL_SECONDS = 0.25
DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION = 0.05
DEFAULT_MIN_PREFIX_REUSE_FRACTION = 0.98
DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO = 0.98
DEFAULT_ALLOWED_CUDA_GRAPH_SKIP_REASONS = ("prefill",)
DEFAULT_MIN_MTP_ACCEPTANCE_RATE = 0.05
DEFAULT_MIN_MTP_MEAN_ADVANCE = 1.10
DEFAULT_MIN_MTP_PROPOSAL_DEPTH = 2.0
DEFAULT_MAX_SPARSE_VERIFIER_FALLBACK_RATIO = 0.01
DEFAULT_MIN_SPARSE_VERIFIER_ACCOUNTING_COVERAGE = 0.98
DEFAULT_PREFIX_PRESSURE_CAPACITY_FRACTION = 1.10
DEFAULT_PREFIX_PRESSURE_KV_HEADROOM_FRACTION = 0.10
DEFAULT_KV_BLOCK_SIZE_TOKENS = 32
DEFAULT_PREFIX_PRESSURE_MAX_ENTRIES = 128
DEFAULT_MIN_OUTPUT_EVENT_COVERAGE = 0.98
DEFAULT_MAX_LATENCY_DEGRADATION_FRACTION = 0.20
DEFAULT_MIN_TELEMETRY_COVERAGE = 0.95
DEFAULT_MIN_TELEMETRY_SAMPLES = 2
DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB = 512.0
DEFAULT_MAX_PROCESS_RSS_DRIFT_FRACTION = 0.10
DEFAULT_MAX_PROCESS_RSS_HIGH_WATER_MIB = 2_048.0
DEFAULT_MAX_GPU_MEMORY_DRIFT_MIB = 512.0
DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB = 2_048.0
DEFAULT_MAX_KV_BLOCK_UTILIZATION = 0.95
DEFAULT_MAX_RECURRENT_SLOT_UTILIZATION = 1.0
MAX_WINDOWED_KV_SLOT_UTILIZATION = 1.0
MIN_PRODUCTION_PROBES_PER_PHASE = 2
MIN_COLD_LONG_CONTEXT_GRAPH_CAPTURES = 2
DEFAULT_OVERLAP_BASELINE_SECONDS = 2.0
DEFAULT_OVERLAP_QUEUE_POLL_SECONDS = 0.01
DEFAULT_MIN_OVERLAP_BASELINE_COMPLETIONS = 2
DEFAULT_MIN_OVERLAP_BASELINE_EVENTS = 256
DEFAULT_MAX_OVERLAP_DECODE_GAP_SECONDS = 0.25
DEFAULT_MAX_OVERLAP_PREFILL_TTFT_SECONDS = 30.0
DEFAULT_SPECULATIVE_PREFIX_REPLAY_TOKENS = 0
DEFAULT_MAX_PROBE_LATENCY_SLOWDOWN = 3.0
DEFAULT_MAX_FINAL_C1_LATENCY_SLOWDOWN = 1.20
DEFAULT_MIN_FINAL_C1_DECODE_RATIO = 0.95
DEFAULT_MAX_SCHEDULE_LATENESS_SECONDS = 1.0
DEFAULT_MULTIMODAL_PHASE_DURATION_SECONDS = 300.0
DEFAULT_MULTIMODAL_CONCURRENCY = 8
DEFAULT_MULTIMODAL_IMAGE_INTERVAL_SECONDS = 30.0
DEFAULT_MULTIMODAL_MIN_TEXT_REQUESTS_PER_PHASE = 32
DEFAULT_MULTIMODAL_MIN_IMAGE_REQUESTS = 10
DEFAULT_MULTIMODAL_MIN_MIXED_THROUGHPUT_RATIO = 0.90
DEFAULT_MULTIMODAL_MAX_MIXED_TPOT_RATIO = 1.20
DEFAULT_MULTIMODAL_MAX_MIXED_TTFT_P99_SECONDS = 2.0
DEFAULT_MULTIMODAL_MIN_RECOVERY_THROUGHPUT_RATIO = 0.95
DEFAULT_MULTIMODAL_IMAGE = "website/public/og.png"
DEFAULT_MULTIMODAL_IMAGE_PROMPT = (
    "Read the prominent title and subtitle exactly, then identify the background "
    "color and the colors used for the title lettering."
)
DEFAULT_MULTIMODAL_REQUIRED_PHRASES = (
    "mistral.rs",
    "fast, flexible llm inference",
)
DEFAULT_MULTIMODAL_EXPECTED_ATTRIBUTES = (
    ("black background", "dark background"),
    ("white lettering", "white text", "white letters"),
    ("orange lettering", "orange text", "orange letters"),
)
ADVERSARIAL_FAIRNESS_SHORT_CONTEXT_TOKENS = 1_024
ADVERSARIAL_FAIRNESS_LONG_CONTEXT_TOKENS = 100_000
ADVERSARIAL_FAIRNESS_MIN_OVERLAP_OUTPUT_EVENTS = 2
DEFAULT_QUALITY_REPLAY_CONCURRENCY = 16
DEFAULT_QUALITY_REPLAY_PRESSURE_WAVES = 8
DEFAULT_QUALITY_REPLAY_PRESSURE_ENTRIES = 16
DEFAULT_QUALITY_REPLAY_PRESSURE_CONTEXT_TOKENS = 8_192
DEFAULT_QUALITY_REPLAY_PRESSURE_MAX_TOKENS = 8
DEFAULT_QUALITY_REPLAY_MAX_STABILITY_PASSES = 4
QUALITY_REPLAY_CONTROL_SEED_OFFSET = 80_000_000
QUALITY_REPLAY_PRESSURE_SEED_OFFSET = 81_000_000
PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_GAUGE = (
    "mistralrs_paged_recurrent_prefix_owners_capacity"
)
PAGED_RECURRENT_PREFIX_OWNERS_USED_GAUGE = (
    "mistralrs_paged_recurrent_prefix_owners_used"
)
PAGED_PREFIX_RETENTION_PRESSURE_EVICTIONS_COUNTER = (
    "mistralrs_paged_prefix_retention_pressure_evictions_total"
)
PAGED_PREFIX_RETAINED_BLOCKS_GAUGE = "mistralrs_kv_cache_blocks_prefix_retained"
EXACT_TEXT_SOURCE_MARGIN_TOKENS = 8_192
EXACT_TEXT_LENGTH_REFINEMENT_STEPS = 16
EXACT_TEXT_LOCAL_SEARCH_RADIUS = 16
EXACT_TEXT_PADDING_REFINEMENT_STEPS = 16
EXACT_TEXT_PADDING_STABILITY_REPEATS = 32
SCHEDULE_TIME_EPSILON_SECONDS = 1e-6
PROCESS_CPU_CLOCK_TICKS_PER_SECOND = int(os.sysconf("SC_CLK_TCK"))
PART1_PRODUCTION_SAMPLING_POLICY = "production"
PART1_CUSTOM_SAMPLING_POLICY = "custom"
PART1_SAMPLING_POLICIES = (
    PART1_PRODUCTION_SAMPLING_POLICY,
    PART1_CUSTOM_SAMPLING_POLICY,
)
CHURN_NEAR_CAPACITY_HEADROOM = 1
MIN_CHURN_NEAR_CAPACITY_SAMPLES = 3
MIN_CHURN_NEAR_CAPACITY_SAMPLE_FRACTION = 0.10
PRODUCTION_SENTINEL_STAGES = (("early", 0.10), ("middle", 0.50), ("late", 0.90))
RETRIEVAL_PADDING_SLACK_TOKENS = 64
RETRIEVAL_SOURCE_MARGIN_TOKENS = 256
RETRIEVAL_INITIAL_SAMPLE_ROWS = 512
RETRIEVAL_PADDING_ADJUSTMENT_STEPS = 64
RETRIEVAL_PADDING_CANDIDATES = (
    " x",
    " z",
    " q",
    " a",
    " .",
    " X",
    " Z",
    " Q",
    "!",
    ";",
    ",",
    ":",
    "|",
)
RETRIEVAL_LENGTH_REFINEMENT_STEPS = 16
PROMPT_CALIBRATION_RAW_LENGTHS = (1_024, 2_048)
CONTEXT_PROMPT_PROFILE = "context"
RETRIEVAL_PROMPT_PROFILE = "retrieval_no_thinking"
EXACT_CONTEXT_SUFFIX = (
    "\nEnd of deterministic production-soak context.\n"
    "Respond with varied original prose without quoting or repeating the context.\n"
)
PROMETHEUS_METRIC_PREFIXES = ("mistralrs_", "http_requests_in_flight")
KV_CACHE_ACTIVE_GAUGE = "mistralrs_kv_cache_blocks_active"
KV_CACHE_PREFIX_CACHED_GAUGE = "mistralrs_kv_cache_blocks_prefix_cached"
REQUEST_OUTCOMES_COUNTER = "mistralrs_request_outcomes_total"
SPARSE_VERIFIER_GPU_COUNTER = "mistralrs_speculative_sparse_gpu_verify_total"
SPARSE_VERIFIER_FALLBACK_COUNTER = "mistralrs_speculative_sparse_gpu_fallback_total"
CUDA_GRAPH_DISPATCH_COUNTER = "mistralrs_cuda_graph_dispatch_total"
CUDA_GRAPH_EVENTS_COUNTER = "mistralrs_cuda_graph_events_total"
CUDA_GRAPH_EVICTIONS_COUNTER = "mistralrs_cuda_graph_evictions_total"
CUDA_GRAPH_CACHE_POPULATION_REASON = "cache_population"
CUDA_MEMORY_PENDING_GAUGE = "mistralrs_cuda_memory_maintenance_pending"
CUDA_MEMORY_MAINTENANCE_COUNTER = "mistralrs_cuda_memory_maintenance_total"
CUDA_MEMORY_PRESSURE_COUNTER = "mistralrs_cuda_memory_pressure_total"
CUDA_MEMORY_RECLAIMED_BYTES_COUNTER = "mistralrs_cuda_memory_reclaimed_bytes_total"
CUDA_PROMPT_BATCH_REDUCTIONS_COUNTER = "mistralrs_cuda_prompt_batch_reductions_total"
CUDA_PROMPT_SEQUENCES_DEFERRED_COUNTER = "mistralrs_cuda_prompt_sequences_deferred_total"
CUDA_PROMPT_MEMORY_REJECTIONS_COUNTER = "mistralrs_cuda_prompt_memory_rejections_total"
WINDOWED_KV_SLOTS_USED_GAUGE = "mistralrs_windowed_kv_slots_used"
WINDOWED_KV_SLOTS_TOTAL_GAUGE = "mistralrs_windowed_kv_slots_total"
DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE = (
    f'{WINDOWED_KV_SLOTS_USED_GAUGE}{{component="dflash",pool="live"}}'
)
DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE = (
    f'{WINDOWED_KV_SLOTS_TOTAL_GAUGE}{{component="dflash",pool="live"}}'
)
DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE = (
    f'{WINDOWED_KV_SLOTS_USED_GAUGE}{{component="dflash",pool="checkpoint"}}'
)
DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE = (
    f'{WINDOWED_KV_SLOTS_TOTAL_GAUGE}{{component="dflash",pool="checkpoint"}}'
)
REQUIRED_PRODUCTION_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_sequences_capacity",
    "mistralrs_requests_pending_admission",
    KV_CACHE_ACTIVE_GAUGE,
    KV_CACHE_PREFIX_CACHED_GAUGE,
    "mistralrs_kv_cache_blocks_total",
    "mistralrs_recurrent_state_slots_used",
    "mistralrs_recurrent_state_slots_total",
)
MULTIMODAL_FRESH_COUNTERS = (
    REQUEST_OUTCOMES_COUNTER,
    "mistralrs_sequences_completed_total",
    "mistralrs_sequences_rejected_total",
    "mistralrs_tokens_processed_total",
    "mistralrs_request_queue_duration_seconds_count",
    "mistralrs_prefix_cache_lookups_total",
    "mistralrs_prefix_cache_hits_total",
    "mistralrs_prefix_cache_tokens_matched_total",
    "mistralrs_prefix_cache_tokens_reused_total",
    "mistralrs_prefix_cache_evictions_total",
    "mistralrs_speculative_prefix_cache_hits_total",
    "mistralrs_speculative_prefix_cache_misses_total",
    "mistralrs_speculative_prefix_cache_captures_total",
    "mistralrs_speculative_prefix_cache_restore_copies_total",
    "mistralrs_speculative_prefix_replay_tokens_avoided_total",
    "mistralrs_encoder_cache_hits_total",
    "mistralrs_encoder_cache_misses_total",
)
MULTIMODAL_FRESH_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
    KV_CACHE_ACTIVE_GAUGE,
    KV_CACHE_PREFIX_CACHED_GAUGE,
    "mistralrs_kv_cache_blocks_used",
    "mistralrs_recurrent_state_slots_used",
    DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
    DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
)
MULTIMODAL_TRANSIENT_CLEANUP_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
    KV_CACHE_ACTIVE_GAUGE,
    "http_requests_in_flight",
    DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
    CUDA_MEMORY_PENDING_GAUGE,
)
DFLASH_REQUIRED_PRODUCTION_GAUGES = (
    DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
    DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE,
    DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
    DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
)
EOS_TOKEN_CANDIDATES = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "</s>",
)
CONTEXT_PARAGRAPH = (
    "This is deterministic production-soak context. It describes scheduling, memory, "
    "cache ownership, request isolation, and reproducible inference. Each sentence exists "
    "only to provide varied natural-language tokens for a long prompt. The service must "
    "preserve correctness while requests arrive, finish, retry, and disconnect.\n"
)
CANARY_PROMPTS = (
    "Explain why request isolation matters in a concurrent inference server.",
    "Give three practical ways to detect a stale cache entry.",
    "Describe a fair policy for mixing long prefills with short decode requests.",
    "Write a compact argument for deterministic seeded sampling in test suites.",
    "Compare throughput and latency without assuming that they are interchangeable.",
    "Summarize how cancellation should release accelerator resources.",
    "Explain why a common wall clock is required for aggregate throughput.",
    "Describe one useful invariant for a paged key-value cache.",
    "Give a short example of a production retry policy.",
    "Explain why variable batch shapes can expose kernel dispatch gaps.",
    "Describe how prefix-cache correctness can be tested after eviction pressure.",
    "Explain the difference between time to first token and time per output token.",
    "List two risks of mixing multimodal encoder work with decode traffic.",
    "Describe what a long-running serving soak should measure.",
    "Explain why fixed seeds should survive request reordering.",
    "Give a concise definition of starvation-free scheduling.",
)

SERVER_COMMAND_SECRET_FLAGS = frozenset(
    {
        "--api-key",
        "--hf-token",
        "--token",
        "--token-source",
    }
)
SERVE_CONFIGURATION_FLAGS = (
    "--paged-attn",
    "--pa-context-len",
    "--pa-memory-mb",
    "--pa-memory-fraction",
    "--pa-block-size",
    "--pa-cache-type",
    "--max-seqs",
    "--prefix-cache-n",
    "--max-num-batched-tokens",
    "--max-prefill-chunk-tokens",
    "--max-decode-steps-before-prefill",
    "--mtp-model",
    "--mtp-n-predict",
    "--mtp-draft-sampling",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_graph_components(value: str) -> tuple[str, ...]:
    components = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"target", "dflash"}
    if not components or any(component not in allowed for component in components):
        raise argparse.ArgumentTypeError("expected target, dflash, or target,dflash")
    if len(set(components)) != len(components):
        raise argparse.ArgumentTypeError("graph components must be unique")
    return components


def parse_context_mix(value: str) -> tuple[tuple[int, int], ...]:
    items: list[tuple[int, int]] = []
    for part in value.split(","):
        try:
            length, weight = part.split(":", maxsplit=1)
            parsed = (int(length), int(weight))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected context mix such as 1024:45,8192:30"
            ) from exc
        if parsed[0] <= 0 or parsed[1] <= 0:
            raise argparse.ArgumentTypeError("context lengths and weights must be positive")
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError("context mix cannot be empty")
    return tuple(items)


def parse_nonempty_phrase(value: str) -> str:
    phrase = value.strip()
    if not phrase:
        raise argparse.ArgumentTypeError("phrase cannot be empty")
    return phrase


def parse_phrase_alternatives(value: str) -> tuple[str, ...]:
    alternatives = tuple(part.strip() for part in value.split("|") if part.strip())
    if not alternatives:
        raise argparse.ArgumentTypeError("expected one or more phrases separated by |")
    return alternatives


def parse_concurrency_thresholds(value: str) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for part in value.split(","):
        try:
            raw_concurrency, raw_threshold = part.split(":", maxsplit=1)
            concurrency = int(raw_concurrency)
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected concurrency thresholds such as 1:150,8:800,16:1200"
            ) from exc
        if concurrency <= 0 or not math.isfinite(threshold) or threshold <= 0:
            raise argparse.ArgumentTypeError(
                "concurrency and throughput thresholds must be positive"
            )
        if concurrency in thresholds:
            raise argparse.ArgumentTypeError(
                f"duplicate concurrency threshold for {concurrency}"
            )
        thresholds[concurrency] = threshold
    if not thresholds:
        raise argparse.ArgumentTypeError("throughput thresholds cannot be empty")
    return thresholds


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "p50": percentile(data, 0.50),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_base_url(base_url: str) -> tuple[str, str]:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        return value, value[:-3]
    return f"{value}/v1", value


def sanitize_record(record: dict[str, Any], keep_output: bool) -> dict[str, Any]:
    result = dict(record)
    text = result.pop("output_text", None)
    transcript = result.pop("output_transcript", None)
    reasoning = result.pop("reasoning_text", None)
    tool_calls = result.pop("tool_calls", None)
    if text is not None:
        result["output_sha256"] = stable_hash(text)
        result["output_chars"] = len(text)
        result["output_preview"] = text[:OUTPUT_PREVIEW_CHARS]
        if keep_output:
            result["output_text"] = text
    if transcript is not None:
        result["output_transcript_sha256"] = stable_hash(transcript)
        if keep_output:
            result["output_transcript"] = transcript
            result["reasoning_text"] = reasoning
            result["tool_calls"] = tool_calls
    return result


def merge_tool_call_delta(
    accumulated: dict[int, dict[str, Any]], delta_calls: Sequence[dict[str, Any]]
) -> None:
    for fallback_index, delta in enumerate(delta_calls):
        index = int(delta.get("index", fallback_index))
        call = accumulated.setdefault(
            index,
            {"index": index, "id": "", "type": "", "function": {"name": "", "arguments": ""}},
        )
        for key in ("id", "type"):
            value = delta.get(key)
            if isinstance(value, str):
                call[key] += value
        function = delta.get("function") or {}
        for key in ("name", "arguments"):
            value = function.get(key)
            if isinstance(value, str):
                call["function"][key] += value


def stream_delta_token_count(
    content: Any,
    reasoning: Any,
    tool_calls: Any,
    tokenizer: TokenizerAdapter | None,
) -> int:
    def strings(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)

    text = "".join(
        part
        for value in (content, reasoning, tool_calls)
        for part in strings(value)
    )
    if not text:
        return 0
    if tokenizer is None:
        return 1
    return max(1, tokenizer.count(text))


def channel_transcript(
    reasoning_text: str, content_text: str, tool_calls: Sequence[dict[str, Any]]
) -> str:
    return json.dumps(
        {
            "reasoning_content": reasoning_text,
            "content": content_text,
            "tool_calls": list(tool_calls),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        try:
            self._file = path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError(f"refusing to append to existing evidence file: {path}") from exc
        self._lock = asyncio.Lock()

    async def emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, "timestamp": utc_now(), **fields}
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        async with self._lock:
            self._file.write(f"{line}\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


class TokenizerAdapter:
    def __init__(self, source: str) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "context-shaped modes require the optional `tokenizers` package"
            ) from exc
        path = Path(source)
        self.source = source
        self.tokenizer = Tokenizer.from_file(str(path)) if path.is_file() else Tokenizer.from_pretrained(source)
        self._exact_text_source_cache: dict[int, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def eos_token_id(self) -> int | None:
        vocabulary = self.tokenizer.get_vocab(with_added_tokens=True)
        for token in EOS_TOKEN_CANDIDATES:
            if token in vocabulary:
                return vocabulary[token]
        return None

    def exact_text(self, target_tokens: int, label: str) -> str:
        if target_tokens <= 0:
            raise ValueError("target token count must be positive")
        prefix = f"Production soak case {label}.\n"
        fixed_tokens = self.count(prefix + EXACT_CONTEXT_SUFFIX)
        if fixed_tokens >= target_tokens:
            raise ValueError(f"target length {target_tokens} is too short for context scaffold")
        body_tokens = self.encode(CONTEXT_PARAGRAPH)
        if not body_tokens:
            raise RuntimeError("tokenizer produced no tokens for the context corpus")
        required = target_tokens + EXACT_TEXT_SOURCE_MARGIN_TOKENS
        repeats = max(1, math.ceil(required / len(body_tokens)))
        source_cache = getattr(self, "_exact_text_source_cache", None)
        if source_cache is None:
            source_cache = {}
            self._exact_text_source_cache = source_cache
        while True:
            source_ids = source_cache.get(repeats)
            if source_ids is None:
                source_ids = self.encode(CONTEXT_PARAGRAPH * repeats)
                source_cache[repeats] = source_ids
            if len(source_ids) >= required:
                break
            repeats = max(
                repeats + 1,
                math.ceil(repeats * required / max(1, len(source_ids))),
            )

        def build(candidate_length: int, padding: str = "") -> str:
            return (
                prefix
                + self.decode(source_ids[:candidate_length])
                + padding
                + EXACT_CONTEXT_SUFFIX
            )

        evaluated: dict[int, tuple[str, int]] = {}

        def evaluate(candidate_length: int) -> tuple[str, int]:
            cached = evaluated.get(candidate_length)
            if cached is not None:
                return cached
            text = build(candidate_length)
            result = (text, self.count(text))
            evaluated[candidate_length] = result
            return result

        candidate_length = min(
            len(source_ids),
            max(0, target_tokens - fixed_tokens),
        )
        for _ in range(EXACT_TEXT_LENGTH_REFINEMENT_STEPS):
            text, actual = evaluate(candidate_length)
            if actual == target_tokens:
                return text
            corrected = min(
                len(source_ids),
                max(0, candidate_length + target_tokens - actual),
            )
            if corrected == candidate_length or corrected in evaluated:
                break
            candidate_length = corrected

        best_candidate, (_, best_actual) = min(
            evaluated.items(),
            key=lambda item: abs(target_tokens - item[1][1]),
        )
        predicted = min(
            len(source_ids),
            max(0, best_candidate + target_tokens - best_actual),
        )
        local_candidates = [predicted]
        for offset in range(1, EXACT_TEXT_LOCAL_SEARCH_RADIUS + 1):
            if predicted - offset >= 0:
                local_candidates.append(predicted - offset)
            if predicted + offset <= len(source_ids):
                local_candidates.append(predicted + offset)
        for candidate_length in local_candidates:
            text, actual = evaluate(candidate_length)
            if actual == target_tokens:
                return text

        below = [
            (candidate, text, actual)
            for candidate, (text, actual) in evaluated.items()
            if actual < target_tokens
        ]
        if below:
            candidate_length, _, actual = max(below, key=lambda item: item[2])
            stable_padding = [
                unit
                for unit in RETRIEVAL_PADDING_CANDIDATES
                if self.count(unit) == 1
                and self.count(unit * EXACT_TEXT_PADDING_STABILITY_REPEATS)
                == EXACT_TEXT_PADDING_STABILITY_REPEATS
            ]
            for unit in stable_padding:
                padding_tokens = target_tokens - actual
                visited_padding: set[int] = set()
                for _ in range(EXACT_TEXT_PADDING_REFINEMENT_STEPS):
                    if padding_tokens <= 0 or padding_tokens in visited_padding:
                        break
                    visited_padding.add(padding_tokens)
                    text = build(candidate_length, unit * padding_tokens)
                    padded_actual = self.count(text)
                    if padded_actual == target_tokens:
                        return text
                    padding_tokens += target_tokens - padded_actual
        raise RuntimeError(
            f"could not construct round-trip-stable text with exactly {target_tokens} tokens"
        )

    def retrieval_text(self, target_tokens: int, label: str) -> tuple[str, str]:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest().upper()
        begin_key = f"BEGIN-{digest[:12]}"
        middle_key = f"MIDDLE-{digest[12:24]}"
        end_key = f"END-{digest[24:36]}"
        expected = f"{begin_key}|{middle_key}|{end_key}"
        prefix = (
            "This is a deterministic memory retrieval test. Keep the three keys and answer "
            "the final question using only the requested pipe-separated value.\n"
            f"The beginning key is {begin_key}.\n"
        )
        middle = f"\nThe middle key is {middle_key}.\n"
        suffix = (
            f"\nThe ending key is {end_key}.\n"
            "Final question: what are the beginning, middle, and ending keys? Reply exactly "
            f"with {expected} and nothing else.\nAnswer:"
        )
        fixed_tokens = self.count(prefix + middle + suffix)
        if fixed_tokens >= target_tokens:
            raise ValueError(
                f"target length {target_tokens} is too short for retrieval scaffold"
            )

        rng_seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(rng_seed)
        rows: list[str] = []

        def append_rows(count: int) -> None:
            first = len(rows)
            rows.extend(
                f"Ledger {index:06d} maps request {rng.getrandbits(48):012x} to lane "
                f"{rng.randrange(257):03d} at epoch {rng.randrange(10_000_000):07d}; "
                f"checksum {rng.getrandbits(64):016x}.\n"
                for index in range(first, first + count)
            )

        required_source_tokens = target_tokens + RETRIEVAL_SOURCE_MARGIN_TOKENS
        append_rows(RETRIEVAL_INITIAL_SAMPLE_ROWS)
        filler_ids = self.encode("".join(rows))
        while len(filler_ids) < required_source_tokens:
            tokens_per_row = max(1.0, len(filler_ids) / len(rows))
            missing_rows = math.ceil(
                (required_source_tokens - len(filler_ids)) / tokens_per_row
            )
            append_rows(max(RETRIEVAL_INITIAL_SAMPLE_ROWS, missing_rows))
            filler_ids = self.encode("".join(rows))

        def build(filler_tokens: int, padding: str = "") -> str:
            midpoint = filler_tokens // 2
            left = self.decode(filler_ids[:midpoint])
            right = self.decode(filler_ids[midpoint:filler_tokens])
            return prefix + left + middle + right + padding + suffix

        candidate_tokens = target_tokens - fixed_tokens - RETRIEVAL_PADDING_SLACK_TOKENS
        if candidate_tokens <= 0:
            raise ValueError(f"target length {target_tokens} leaves no room for retrieval data")
        visited: set[int] = set()
        below_candidates: list[tuple[int, str, int]] = []
        for _ in range(RETRIEVAL_LENGTH_REFINEMENT_STEPS):
            if candidate_tokens in visited:
                break
            visited.add(candidate_tokens)
            text = build(candidate_tokens)
            actual = self.count(text)
            if actual == target_tokens:
                return text, expected
            if actual < target_tokens:
                below_candidates.append((candidate_tokens, text, actual))
            candidate_tokens = max(
                1,
                min(len(filler_ids), candidate_tokens + target_tokens - actual),
            )
        if not below_candidates:
            raise RuntimeError(
                f"could not construct a retrieval prompt below {target_tokens} tokens"
            )
        padding_adjustments = [0]
        for step in range(1, RETRIEVAL_PADDING_ADJUSTMENT_STEPS + 1):
            padding_adjustments.extend((-step, step))
        below_candidates.sort(
            key=lambda candidate: abs(
                target_tokens - candidate[2] - RETRIEVAL_PADDING_SLACK_TOKENS
            )
        )
        for candidate_tokens, _, actual in below_candidates:
            deficit = target_tokens - actual
            for unit in RETRIEVAL_PADDING_CANDIDATES:
                if self.count(unit) != 1:
                    continue
                for adjustment in padding_adjustments:
                    padding_tokens = deficit + adjustment
                    if padding_tokens <= 0:
                        continue
                    text = build(candidate_tokens, unit * padding_tokens)
                    if self.count(text) == target_tokens:
                        return text, expected
        candidate_tokens, _, actual = max(below_candidates, key=lambda candidate: candidate[2])
        raise RuntimeError(
            f"could not fit a {target_tokens}-token retrieval prompt from "
            f"{actual} tokens using {candidate_tokens} filler tokens"
        )


@dataclass(slots=True)
class SamplingPolicy:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = DEFAULT_MIN_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

    def payload(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }


def production_sampling_policy_evidence(
    name: str | None,
    policy: SamplingPolicy | dict[str, Any],
) -> dict[str, Any]:
    actual = asdict(policy) if isinstance(policy, SamplingPolicy) else dict(policy)
    expected = asdict(SamplingPolicy())
    matches = {
        field: actual.get(field) == expected_value
        for field, expected_value in expected.items()
    }
    named = name == PART1_PRODUCTION_SAMPLING_POLICY
    return {
        "passed": named and all(matches.values()),
        "name": name,
        "required_name": PART1_PRODUCTION_SAMPLING_POLICY,
        "named": named,
        "actual": {field: actual.get(field) for field in expected},
        "expected": expected,
        "matches": matches,
    }


@dataclass(frozen=True, slots=True)
class ProductionMemoryLimits:
    min_coverage: float = DEFAULT_MIN_TELEMETRY_COVERAGE
    max_process_rss_drift_mib: float = DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB
    max_process_rss_drift_fraction: float = DEFAULT_MAX_PROCESS_RSS_DRIFT_FRACTION
    max_process_rss_high_water_mib: float = DEFAULT_MAX_PROCESS_RSS_HIGH_WATER_MIB
    max_gpu_memory_drift_mib: float = DEFAULT_MAX_GPU_MEMORY_DRIFT_MIB
    max_gpu_memory_high_water_mib: float = DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB
    max_kv_block_utilization: float = DEFAULT_MAX_KV_BLOCK_UTILIZATION
    max_recurrent_slot_utilization: float = DEFAULT_MAX_RECURRENT_SLOT_UTILIZATION
    require_dflash_windowed_kv: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ProductionMemoryLimits:
        return cls(
            min_coverage=args.min_telemetry_coverage,
            max_process_rss_drift_mib=args.max_process_rss_drift_mib,
            max_process_rss_drift_fraction=args.max_process_rss_drift_fraction,
            max_process_rss_high_water_mib=args.max_process_rss_high_water_mib,
            max_gpu_memory_drift_mib=args.max_gpu_memory_drift_mib,
            max_gpu_memory_high_water_mib=args.max_gpu_memory_high_water_mib,
            max_kv_block_utilization=args.max_kv_block_utilization,
            max_recurrent_slot_utilization=args.max_recurrent_slot_utilization,
            require_dflash_windowed_kv="dflash" in args.expected_graph_components,
        )


@dataclass(frozen=True, slots=True)
class PrefixPressureConfig:
    entries: int
    max_sequences: int
    context_tokens: int
    max_completion_tokens: int
    block_size_tokens: int
    kv_headroom_fraction: float


@dataclass(slots=True)
class RequestSpec:
    case_id: str
    seed: int
    max_tokens: int
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    context_tokens: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QualityReplayCase:
    case_id: str
    seed: int
    concurrency: int
    worker_index: int
    request_index: int
    context_tokens: int
    prompt_index: int
    prompt_label: str
    source_output_transcript_sha256: str
    source_quality_failure: bool


@dataclass(frozen=True, slots=True)
class QualityReplayPressureConfig:
    waves: int
    entries: int
    context_tokens: int
    max_tokens: int
    block_size_tokens: int
    headroom_fraction: float


@dataclass(slots=True)
class RequestResult:
    case_id: str
    seed: int
    ok: bool
    status_code: int | None
    started: float
    ended: float
    ttft_seconds: float | None
    tpot_seconds: float | None
    client_queue_seconds: float
    completion_tokens: int
    prompt_tokens: int | None
    finish_reason: str | None
    output_text: str
    reasoning_text: str
    tool_calls: list[dict[str, Any]]
    output_transcript: str
    output_chunks: int
    stream_done: bool
    usage_received: bool
    request_id: str | None
    error: str | None
    error_kind: str | None
    context_tokens: int | None
    tags: dict[str, Any]
    streamed_output_tokens: int = 0
    output_event_times: list[float] = field(default_factory=list)
    output_event_token_counts: list[int] = field(default_factory=list)
    output_event_window_counts: list[int] = field(default_factory=list)
    output_token_window_counts: list[int] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return self.ended - self.started

    def record(self, keep_output: bool) -> dict[str, Any]:
        value = asdict(self)
        output_event_times = value.pop("output_event_times")
        output_event_token_counts = value.pop("output_event_token_counts")
        value["retained_output_event_timestamps"] = len(output_event_times)
        value["retained_output_event_token_counts"] = len(
            output_event_token_counts
        )
        value["elapsed_seconds"] = self.elapsed_seconds
        return sanitize_record(value, keep_output)


class SoakClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        policy: SamplingPolicy,
        tokenizer: TokenizerAdapter | None,
    ) -> None:
        self.api_base, self.root_base = normalized_base_url(base_url)
        self.model = model
        self.policy = policy
        self.tokenizer = tokenizer
        self.timeout_seconds = timeout_seconds
        self.prompt_overhead_tokens: dict[str, int] = {}
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=128)
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds),
            limits=limits,
        )

    async def close(self) -> None:
        await self.http.aclose()

    def calibrated_content_tokens(self, profile: str, prompt_tokens: int) -> int:
        if profile not in self.prompt_overhead_tokens:
            raise RuntimeError(f"prompt profile {profile!r} has not been calibrated")
        content_tokens = prompt_tokens - self.prompt_overhead_tokens[profile]
        if content_tokens <= 0:
            raise ValueError(
                f"requested prompt length {prompt_tokens} does not exceed the calibrated "
                f"{self.prompt_overhead_tokens[profile]}-token chat template"
            )
        return content_tokens

    def payload(self, spec: RequestSpec, stream: bool) -> dict[str, Any]:
        messages: Any = spec.messages
        if messages is None:
            messages = [{"role": "user", "content": spec.prompt or ""}]
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": spec.max_tokens,
            "seed": spec.seed,
            "stream": stream,
            **self.policy.payload(),
            **spec.extra,
        }

    async def stream_request(
        self,
        spec: RequestSpec,
        scheduled_at: float | None = None,
        timeout_seconds: float | None = None,
        retain_output_event_windows: Sequence[tuple[float, float]] | None = None,
    ) -> RequestResult:
        started = time.perf_counter()
        queued = max(0.0, started - scheduled_at) if scheduled_at is not None else 0.0
        first_token: float | None = None
        token_times: list[float] = []
        token_counts: list[int] = []
        output_event_window_counts = (
            [0] * len(retain_output_event_windows)
            if retain_output_event_windows is not None
            else []
        )
        output_token_window_counts = (
            [0] * len(retain_output_event_windows)
            if retain_output_event_windows is not None
            else []
        )
        streamed_output_tokens = 0
        content: list[str] = []
        reasoning: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        saw_done = False
        status_code: int | None = None
        request_id: str | None = None
        error: str | None = None
        error_kind: str | None = None
        chunks = 0
        timeout = (
            httpx.Timeout(self.timeout_seconds, read=timeout_seconds)
            if timeout_seconds is not None
            else None
        )
        try:
            async with self.http.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json=self.payload(spec, stream=True),
                timeout=timeout,
            ) as response:
                status_code = response.status_code
                request_id = response.headers.get("x-request-id")
                if response.is_error:
                    body = await response.aread()
                    raise RuntimeError(f"HTTP {status_code}: {body.decode(errors='replace')}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        continue
                    if not data:
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    tool_piece = delta.get("tool_calls")
                    reasoning_piece = delta.get("reasoning_content")
                    if piece is not None or tool_piece or reasoning_piece:
                        event_tokens = stream_delta_token_count(
                            piece,
                            reasoning_piece,
                            tool_piece,
                            self.tokenizer,
                        )
                        if event_tokens > 0:
                            now = time.perf_counter()
                            streamed_output_tokens += event_tokens
                            if first_token is None:
                                first_token = now
                            if retain_output_event_windows is None:
                                token_times.append(now)
                                token_counts.append(event_tokens)
                            else:
                                for index, (start, end) in enumerate(
                                    retain_output_event_windows
                                ):
                                    if start <= now < end:
                                        output_event_window_counts[index] += 1
                                        output_token_window_counts[index] += event_tokens
                            chunks += 1
                    if piece:
                        content.append(piece)
                    if isinstance(reasoning_piece, str):
                        reasoning.append(reasoning_piece)
                    if isinstance(tool_piece, list):
                        merge_tool_call_delta(tool_call_parts, tool_piece)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        except httpx.ReadTimeout as exc:
            error_kind = type(exc).__name__
            error = f"{error_kind}: {exc}"
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
            error_kind = type(exc).__name__
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            missing = []
            if not saw_done:
                missing.append("[DONE]")
            if finish_reason is None:
                missing.append("terminal finish_reason")
            if not usage:
                missing.append("final usage")
            if missing:
                error_kind = "IncompleteStream"
                error = f"IncompleteStream: missing {', '.join(missing)}"
        ended = time.perf_counter()
        output = "".join(content)
        reasoning_output = "".join(reasoning)
        tool_calls = [tool_call_parts[index] for index in sorted(tool_call_parts)]
        transcript = channel_transcript(reasoning_output, output, tool_calls)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if completion_tokens == 0 and output:
            completion_tokens = self.tokenizer.count(output) if self.tokenizer else chunks
        prompt_tokens = usage.get("prompt_tokens")
        parsed_prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        if (
            error is None
            and spec.context_tokens is not None
            and parsed_prompt_tokens != spec.context_tokens
        ):
            error_kind = "ContextLengthMismatch"
            error = (
                "ContextLengthMismatch: server reported "
                f"{parsed_prompt_tokens!r} prompt tokens for requested "
                f"{spec.context_tokens}"
            )
        ttft = first_token - started if first_token is not None else None
        tpot = None
        if completion_tokens > 1 and first_token is not None:
            tpot = (ended - first_token) / (completion_tokens - 1)
        return RequestResult(
            case_id=spec.case_id,
            seed=spec.seed,
            ok=error is None and status_code is not None and status_code < 400,
            status_code=status_code,
            started=started,
            ended=ended,
            ttft_seconds=ttft,
            tpot_seconds=tpot,
            client_queue_seconds=queued,
            completion_tokens=completion_tokens,
            prompt_tokens=parsed_prompt_tokens,
            finish_reason=finish_reason,
            output_text=output,
            reasoning_text=reasoning_output,
            tool_calls=tool_calls,
            output_transcript=transcript,
            output_chunks=chunks,
            stream_done=saw_done,
            usage_received=bool(usage),
            request_id=request_id,
            error=error,
            error_kind=error_kind,
            context_tokens=spec.context_tokens,
            tags=dict(spec.tags),
            streamed_output_tokens=streamed_output_tokens,
            output_event_times=token_times,
            output_event_token_counts=token_counts,
            output_event_window_counts=output_event_window_counts,
            output_token_window_counts=output_token_window_counts,
        )

    async def request_json(
        self,
        spec: RequestSpec,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.perf_counter()
        timeout = httpx.Timeout(timeout_seconds) if timeout_seconds is not None else None
        status_code: int | None = None
        request_id: str | None = None
        error: str | None = None
        error_kind: str | None = None
        body: dict[str, Any] | None = None
        try:
            response = await self.http.post(
                f"{self.api_base}/chat/completions",
                json=self.payload(spec, stream=False),
                timeout=timeout,
            )
            status_code = response.status_code
            request_id = response.headers.get("x-request-id")
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            error_kind = type(exc).__name__
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
        return body, {
            "case_id": spec.case_id,
            "seed": spec.seed,
            "ok": error is None,
            "status_code": status_code,
            "request_id": request_id,
            "elapsed_seconds": ended - started,
            "error": error,
            "error_kind": error_kind,
            "tags": spec.tags,
        }

    async def disconnect(
        self,
        spec: RequestSpec,
        after_seconds: float,
        admission_release: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        accepted = asyncio.Event()

        async def consume() -> None:
            async with self.http.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json=self.payload(spec, stream=True),
            ) as response:
                response.raise_for_status()
                accepted.set()
                async for _ in response.aiter_bytes():
                    pass

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(accepted.wait(), timeout=min(30.0, self.timeout_seconds))
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return {
                "case_id": spec.case_id,
                "outcome": "acceptance_timeout",
                "elapsed_seconds": time.perf_counter() - started,
                "disconnect_after_seconds": after_seconds,
                "server_stream_accepted": False,
                "error": "server did not accept the streaming response before timeout",
                "context_tokens": spec.context_tokens,
            }
        if admission_release is not None:
            await admission_release.wait()
        await asyncio.sleep(after_seconds)
        task.cancel()
        outcome = "cancelled"
        error = None
        try:
            await task
            outcome = "completed_before_disconnect"
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            outcome = "error"
            error = f"{type(exc).__name__}: {exc}"
        return {
            "case_id": spec.case_id,
            "outcome": outcome,
            "elapsed_seconds": time.perf_counter() - started,
            "disconnect_after_seconds": after_seconds,
            "server_stream_accepted": True,
            "error": error,
            "context_tokens": spec.context_tokens,
        }

    async def metrics(self) -> dict[str, float]:
        response = await self.http.get(f"{self.root_base}/metrics", timeout=30.0)
        response.raise_for_status()
        return parse_prometheus(response.text)

    async def system_info(self) -> dict[str, Any]:
        response = await self.http.get(f"{self.api_base}/system/info", timeout=30.0)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("system info endpoint returned a non-object response")
        return value

    async def ui_models(self) -> list[dict[str, Any]]:
        response = await self.http.get(f"{self.root_base}/ui/api/list_models", timeout=30.0)
        response.raise_for_status()
        value = response.json()
        models = value.get("models") if isinstance(value, dict) else None
        if not isinstance(models, list) or not all(
            isinstance(model, dict) for model in models
        ):
            raise RuntimeError("UI model endpoint returned an invalid model inventory")
        return models


def parse_prometheus(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        parts = value.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        name, raw_number = parts
        try:
            number = float(raw_number)
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        base_name = name.split("{", maxsplit=1)[0]
        if base_name.startswith(PROMETHEUS_METRIC_PREFIXES):
            result[name] = number
    return result


def metric_total(snapshot: dict[str, float], metric: str) -> float | None:
    base_name = metric.split("{", 1)[0]
    required_labels = dict(re.findall(r'(\w+)="([^"]*)"', metric))
    matches = []
    for key, value in snapshot.items():
        if key.split("{", 1)[0] != base_name:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', key))
        if all(labels.get(name) == expected for name, expected in required_labels.items()):
            matches.append(value)
    return sum(matches) if matches else None


def production_required_gauges(
    expected_graph_components: Sequence[str],
) -> tuple[str, ...]:
    if "dflash" in expected_graph_components:
        return (*REQUIRED_PRODUCTION_GAUGES, *DFLASH_REQUIRED_PRODUCTION_GAUGES)
    return REQUIRED_PRODUCTION_GAUGES


def metric_delta(before: dict[str, float], after: dict[str, float], metric: str) -> float | None:
    start = metric_total(before, metric)
    end = metric_total(after, metric)
    if end is None:
        return None
    return end - (start or 0.0)


def command_option_value(argv: Sequence[str], *names: str) -> str | None:
    value = None
    for index, argument in enumerate(argv):
        for name in names:
            if argument == name and index + 1 < len(argv):
                value = argv[index + 1]
            elif argument.startswith(f"{name}="):
                value = argument.split("=", maxsplit=1)[1]
    return value


def redact_server_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        name = argument.split("=", maxsplit=1)[0]
        if name not in SERVER_COMMAND_SECRET_FLAGS:
            redacted.append(argument)
            continue
        if "=" in argument:
            redacted.append(f"{name}=<redacted>")
        else:
            redacted.append(argument)
            redact_next = True
    return redacted


def parse_serve_configuration(argv: Sequence[str]) -> dict[str, Any]:
    values = {
        flag.removeprefix("--").replace("-", "_"): command_option_value(argv, flag)
        for flag in SERVE_CONFIGURATION_FLAGS
    }
    values["model"] = command_option_value(argv, "-m", "--model")
    values["subcommand"] = "serve" if "serve" in argv else None
    values["paged_attn"] = values["paged_attn"] or "auto"
    values["pa_cache_type"] = values["pa_cache_type"] or "auto"
    return values


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_server_process_provenance(server_pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(server_pid)
    cmdline_bytes = (proc / "cmdline").read_bytes()
    argv = [os.fsdecode(value) for value in cmdline_bytes.split(b"\0") if value]
    executable = Path(os.readlink(proc / "exe"))
    stat = executable.stat()
    redacted_argv = redact_server_argv(argv)
    redacted_cmdline = b"\0".join(os.fsencode(value) for value in redacted_argv) + b"\0"
    return {
        "pid": server_pid,
        "executable": str(executable),
        "executable_sha256": hash_file(executable),
        "executable_size_bytes": stat.st_size,
        "executable_mtime_ns": stat.st_mtime_ns,
        "working_directory": os.readlink(proc / "cwd"),
        "argv_redacted": redacted_argv,
        "command_redacted": shlex.join(redacted_argv),
        "command_sha256": hashlib.sha256(redacted_cmdline).hexdigest(),
        "command_hash_scope": "redacted_nul_delimited_argv",
        "serve_configuration": parse_serve_configuration(argv),
        "process_is_mistralrs": (
            "mistralrs" in executable.name
            or any("mistralrs" in value for value in argv[:1])
        ),
    }


async def gpu_driver_provenance(server_pid: int | None) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi exited with status {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        rows = []
        for fields in csv.reader(stdout.decode().splitlines(), skipinitialspace=True):
            if len(fields) != 5:
                continue
            rows.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1].strip(),
                    "name": fields[2].strip(),
                    "driver_version": fields[3].strip(),
                    "memory_total_mib": float(fields[4]),
                }
            )
        server_process_gpus: list[dict[str, Any]] = []
        server_process_query_error = None
        if server_pid is not None:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            process_stdout, process_stderr = await process.communicate()
            if process.returncode != 0:
                server_process_query_error = (
                    f"nvidia-smi exited with status {process.returncode}: "
                    f"{process_stderr.decode(errors='replace').strip()}"
                )
            else:
                for fields in csv.reader(
                    process_stdout.decode().splitlines(),
                    skipinitialspace=True,
                ):
                    if len(fields) != 2:
                        continue
                    try:
                        pid = int(fields[0])
                    except ValueError:
                        continue
                    if pid == server_pid:
                        server_process_gpus.append(
                            {"pid": pid, "gpu_uuid": fields[1].strip()}
                        )
        return {
            "available": bool(rows),
            "gpus": rows,
            "server_process_gpus": server_process_gpus,
            "server_process_query_error": server_process_query_error,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "gpus": [],
            "server_process_gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def realized_kv_configuration(metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "blocks_total": metric_total(metrics, "mistralrs_kv_cache_blocks_total"),
        "blocks_total_series": labeled_metric_values(
            metrics, "mistralrs_kv_cache_blocks_total"
        ),
        "blocks_active": metric_total(metrics, KV_CACHE_ACTIVE_GAUGE),
        "blocks_active_series": labeled_metric_values(metrics, KV_CACHE_ACTIVE_GAUGE),
        "blocks_prefix_cached": metric_total(metrics, KV_CACHE_PREFIX_CACHED_GAUGE),
        "sequence_capacity": metric_total(metrics, "mistralrs_sequences_capacity"),
        "recurrent_slots_total": metric_total(
            metrics, "mistralrs_recurrent_state_slots_total"
        ),
    }


def server_provenance_evidence(provenance: dict[str, Any]) -> dict[str, Any]:
    build = (provenance.get("system_info") or {}).get("build") or {}
    process = provenance.get("process") or {}
    gpu = provenance.get("gpu_driver") or {}
    serve = process.get("serve_configuration") or {}
    kv = provenance.get("realized_kv_configuration") or {}
    git_revision = str(build.get("git_revision") or "")
    inventory_uuids = {item.get("uuid") for item in gpu.get("gpus") or []}
    process_gpu_uuids = {
        item.get("gpu_uuid") for item in gpu.get("server_process_gpus") or []
    }
    server_gpus_mapped = bool(process_gpu_uuids) and process_gpu_uuids.issubset(
        inventory_uuids
    )
    checks = {
        "git_revision": bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", git_revision)),
        "binary": bool(
            process.get("process_is_mistralrs")
            and process.get("executable")
            and process.get("executable_sha256")
        ),
        "serve_command": bool(
            process.get("command_sha256") and serve.get("subcommand") == "serve"
        ),
        "gpu_driver": bool(
            gpu.get("available")
            and (
                provenance.get("server_pid") is None
                or server_gpus_mapped
            )
        ),
        "kv_configuration": bool(
            serve.get("pa_cache_type") not in (None, "auto")
            and serve.get("paged_attn")
            and kv.get("blocks_total") is not None
            and kv.get("sequence_capacity") is not None
        ),
    }
    return {"complete": all(checks.values()), "checks": checks}


async def collect_server_provenance(
    client: SoakClient, server_pid: int | None
) -> dict[str, Any]:
    provenance: dict[str, Any] = {"server_pid": server_pid, "errors": {}}
    try:
        provenance["system_info"] = await client.system_info()
    except Exception as exc:
        provenance["system_info"] = None
        provenance["errors"]["system_info"] = f"{type(exc).__name__}: {exc}"
    if server_pid is not None:
        try:
            provenance["process"] = await asyncio.to_thread(
                read_server_process_provenance, server_pid
            )
        except (OSError, ValueError) as exc:
            provenance["process"] = None
            provenance["errors"]["process"] = f"{type(exc).__name__}: {exc}"
    else:
        provenance["process"] = None
        provenance["errors"]["process"] = "server PID was not provided"
    provenance["gpu_driver"] = await gpu_driver_provenance(server_pid)
    try:
        metrics = await client.metrics()
        provenance["realized_kv_configuration"] = realized_kv_configuration(metrics)
    except Exception as exc:
        provenance["realized_kv_configuration"] = {}
        provenance["errors"]["metrics"] = f"{type(exc).__name__}: {exc}"
    provenance["evidence"] = server_provenance_evidence(provenance)
    return provenance


def server_provenance_required(args: argparse.Namespace) -> bool:
    return (
        args.require_server_provenance
        or args.mode
        in ("prefix-pressure", "production", "quality-replay", "multimodal")
        or (args.mode == "adversarial" and args.acceptance_grade)
    )


def prefix_pressure_plan(
    snapshot: dict[str, float], config: PrefixPressureConfig
) -> dict[str, int | float]:
    total = metric_total(snapshot, "mistralrs_kv_cache_blocks_total")
    active = metric_total(snapshot, KV_CACHE_ACTIVE_GAUGE)
    prefix_cached = metric_total(snapshot, KV_CACHE_PREFIX_CACHED_GAUGE)
    if total is None or total <= 0:
        raise RuntimeError("prefix pressure requires mistralrs_kv_cache_blocks_total")
    if active is None or active < 0:
        raise RuntimeError(f"prefix pressure requires {KV_CACHE_ACTIVE_GAUGE}")
    if prefix_cached is None or prefix_cached < 0:
        raise RuntimeError(f"prefix pressure requires {KV_CACHE_PREFIX_CACHED_GAUGE}")

    total_blocks = math.floor(total)
    active_blocks = math.ceil(active)
    prefix_cached_blocks = math.ceil(prefix_cached)
    headroom_blocks = math.ceil(total_blocks * config.kv_headroom_fraction)
    blocks_per_request = math.ceil(
        (config.context_tokens + config.max_completion_tokens)
        / config.block_size_tokens
    )
    active_budget_blocks = max(0, total_blocks - active_blocks - headroom_blocks)
    capacity_concurrency = active_budget_blocks // blocks_per_request
    concurrency = min(config.entries, config.max_sequences, capacity_concurrency)
    if concurrency < 1:
        raise RuntimeError(
            "prefix pressure cannot fit one active request with the configured KV "
            "headroom; reduce --prefix-pressure-context-tokens or "
            "--prefix-pressure-kv-headroom-fraction"
        )

    return {
        "concurrency": concurrency,
        "capacity_concurrency": capacity_concurrency,
        "entries": config.entries,
        "max_sequences": config.max_sequences,
        "total_blocks": total_blocks,
        "active_blocks_observed": active_blocks,
        "prefix_cached_blocks_observed_reclaimable": prefix_cached_blocks,
        "headroom_fraction": config.kv_headroom_fraction,
        "headroom_blocks": headroom_blocks,
        "active_budget_blocks": active_budget_blocks,
        "blocks_per_request": blocks_per_request,
        "active_working_set_blocks": concurrency * blocks_per_request,
    }


def interval_union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(
        (start, end)
        for start, end in intervals
        if math.isfinite(start) and math.isfinite(end) and end > start
    )
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def output_interval_evidence(
    results: Sequence[RequestResult],
    start: float,
    end: float,
) -> dict[str, int | float | bool | None]:
    token_weights_complete = all(
        len(result.output_event_times) == len(result.output_event_token_counts)
        for result in results
    )
    output_tokens = (
        sum(
            token_count
            for result in results
            for timestamp, token_count in zip(
                result.output_event_times,
                result.output_event_token_counts,
            )
            if start <= timestamp < end
        )
        if token_weights_complete
        else None
    )
    return {
        "start": start,
        "end": end,
        "seconds": max(0.0, end - start),
        "completed_requests": sum(
            start <= result.ended < end for result in results
        ),
        "output_events": sum(
            start <= timestamp < end
            for result in results
            for timestamp in result.output_event_times
        ),
        "output_tokens": output_tokens,
        "token_weights_complete": token_weights_complete,
    }


def overlap_window_evidence(
    results: Sequence[RequestResult],
    baseline_started: float,
    prefill_started: float,
    prefill_first_token: float | None,
) -> tuple[
    dict[str, int | float | bool | None],
    dict[str, int | float | bool | None] | None,
]:
    baseline = output_interval_evidence(
        results,
        baseline_started,
        prefill_started,
    )
    overlapped = (
        output_interval_evidence(
            results,
            prefill_started,
            prefill_first_token,
        )
        if prefill_first_token is not None
        else None
    )
    return baseline, overlapped


def output_event_gap_evidence(
    results: Sequence[RequestResult],
    started: float,
    ended: float | None,
    maximum_gap_seconds: float,
) -> dict[str, Any]:
    event_times = sorted(
        timestamp
        for result in results
        for timestamp in result.output_event_times
        if ended is not None
        and started <= timestamp <= ended
        and math.isfinite(timestamp)
    )
    if ended is None or ended <= started or not event_times:
        maximum_observed_gap = None
    else:
        maximum_observed_gap = max(
            event_times[0] - started,
            ended - event_times[-1],
            *(right - left for left, right in zip(event_times, event_times[1:])),
        )
    return {
        "passed": (
            maximum_observed_gap is not None
            and maximum_observed_gap <= maximum_gap_seconds
        ),
        "window_started": started,
        "window_ended": ended,
        "observed_output_events": len(event_times),
        "maximum_observed_gap_seconds": maximum_observed_gap,
        "maximum_allowed_gap_seconds": maximum_gap_seconds,
    }


def concurrent_decode_overlap_evidence(
    left: RequestResult,
    right: RequestResult,
) -> dict[str, Any]:
    left_events = sorted(
        timestamp for timestamp in left.output_event_times if math.isfinite(timestamp)
    )
    right_events = sorted(
        timestamp for timestamp in right.output_event_times if math.isfinite(timestamp)
    )
    instrumentation_complete = len(left_events) >= 2 and len(right_events) >= 2
    overlap_started = (
        max(left_events[0], right_events[0]) if instrumentation_complete else None
    )
    overlap_ended = (
        min(left_events[-1], right_events[-1]) if instrumentation_complete else None
    )
    overlap_seconds = (
        max(0.0, overlap_ended - overlap_started)
        if overlap_started is not None and overlap_ended is not None
        else None
    )
    left_overlap_events = (
        sum(overlap_started <= timestamp <= overlap_ended for timestamp in left_events)
        if overlap_seconds is not None and overlap_seconds > 0
        else 0
    )
    right_overlap_events = (
        sum(overlap_started <= timestamp <= overlap_ended for timestamp in right_events)
        if overlap_seconds is not None and overlap_seconds > 0
        else 0
    )
    return {
        "passed": (
            left.ok
            and right.ok
            and instrumentation_complete
            and overlap_seconds is not None
            and overlap_seconds > 0
            and left_overlap_events >= ADVERSARIAL_FAIRNESS_MIN_OVERLAP_OUTPUT_EVENTS
            and right_overlap_events >= ADVERSARIAL_FAIRNESS_MIN_OVERLAP_OUTPUT_EVENTS
        ),
        "instrumentation_complete": instrumentation_complete,
        "left_case_id": left.case_id,
        "right_case_id": right.case_id,
        "left_output_events": len(left_events),
        "right_output_events": len(right_events),
        "left_overlap_events": left_overlap_events,
        "right_overlap_events": right_overlap_events,
        "minimum_overlap_events_per_request": (
            ADVERSARIAL_FAIRNESS_MIN_OVERLAP_OUTPUT_EVENTS
        ),
        "overlap_started": overlap_started,
        "overlap_ended": overlap_ended,
        "overlap_seconds": overlap_seconds,
    }


def summarize_batch(
    results: Sequence[RequestResult],
    wall_seconds: float,
    concurrency: int,
    name: str,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    output_tokens = sum(result.completion_tokens for result in successful)
    decode_candidates = [
        result
        for result in successful
        if result.completion_tokens > 1
    ]
    decode_results = [
        result for result in decode_candidates if result.ttft_seconds is not None
    ]
    decode_intervals_missing = len(decode_candidates) - len(decode_results)
    decode_tokens = sum(result.completion_tokens - 1 for result in decode_candidates)
    decode_active_seconds = interval_union_seconds(
        (result.started + result.ttft_seconds, result.ended)
        for result in decode_results
        if result.ttft_seconds is not None
    )
    common_wall_throughput = (
        output_tokens / wall_seconds if wall_seconds > 0 else None
    )
    decode_active_throughput = (
        decode_tokens / decode_active_seconds
        if decode_active_seconds > 0 and decode_intervals_missing == 0
        else None
    )
    return {
        "name": name,
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "output_tokens": output_tokens,
        "decode_requests": len(decode_candidates),
        "decode_intervals": len(decode_results),
        "decode_intervals_missing": decode_intervals_missing,
        "decode_tokens": decode_tokens,
        "decode_active_seconds": decode_active_seconds,
        "end_to_end_output_tok_s_common_wall": common_wall_throughput,
        "output_tok_s_common_wall": common_wall_throughput,
        "decode_tok_s_active": decode_active_throughput,
        "request_latency_seconds": distribution(result.elapsed_seconds for result in successful),
        "ttft_seconds": distribution(
            result.ttft_seconds for result in successful if result.ttft_seconds is not None
        ),
        "tpot_seconds": distribution(
            result.tpot_seconds for result in successful if result.tpot_seconds is not None
        ),
        "client_queue_seconds": distribution(result.client_queue_seconds for result in successful),
        "completion_tokens": distribution(result.completion_tokens for result in successful),
        "prompt_tokens": distribution(
            result.prompt_tokens for result in successful if result.prompt_tokens is not None
        ),
        "finish_reasons": dict(
            Counter(result.finish_reason or "none" for result in successful)
        ),
    }


def summarize_context_groups(
    results: Sequence[RequestResult],
    wall_seconds: float,
    concurrency: int,
    name: str,
) -> dict[str, dict[str, Any]]:
    lengths = sorted(
        {
            result.context_tokens
            for result in results
            if result.context_tokens is not None
        }
    )
    return {
        str(length): summarize_batch(
            [result for result in results if result.context_tokens == length],
            wall_seconds,
            concurrency,
            f"{name}-{length}",
        )
        for length in lengths
    }


async def run_batch(
    client: SoakClient,
    specs: Sequence[RequestSpec],
    concurrency: int,
    writer: JsonlWriter,
    phase: str,
    keep_output: bool,
) -> tuple[list[RequestResult], dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    scheduled_at = time.perf_counter()

    async def run_one(spec: RequestSpec) -> RequestResult:
        async with semaphore:
            result = await client.stream_request(spec, scheduled_at=scheduled_at)
            await writer.emit(
                "request",
                phase=phase,
                concurrency=concurrency,
                **result.record(keep_output),
            )
            return result

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(run_one(spec) for spec in specs))
    wall_seconds = time.perf_counter() - wall_start
    summary = summarize_batch(results, wall_seconds, concurrency, phase)
    summary["by_context"] = summarize_context_groups(
        results,
        wall_seconds,
        concurrency,
        phase,
    )
    await writer.emit("batch_summary", phase=phase, **summary)
    return results, summary


async def run_staggered_pair(
    client: SoakClient,
    first: RequestSpec,
    second: RequestSpec,
    stagger_seconds: float,
    writer: JsonlWriter,
    phase: str,
) -> tuple[list[RequestResult], dict[str, Any]]:
    wall_start = time.perf_counter()
    first_task = asyncio.create_task(client.stream_request(first))
    await asyncio.sleep(stagger_seconds)
    second_task = asyncio.create_task(client.stream_request(second))
    results = list(await asyncio.gather(first_task, second_task))
    wall_seconds = time.perf_counter() - wall_start
    for result in results:
        await writer.emit(
            "request",
            phase=phase,
            concurrency=2,
            **result.record(True),
        )
    summary = summarize_batch(results, wall_seconds, 2, phase)
    await writer.emit("batch_summary", phase=phase, **summary)
    return results, summary


def make_canary_specs(args: argparse.Namespace) -> list[RequestSpec]:
    count = max(args.requests, max(args.concurrencies))
    return [
        RequestSpec(
            case_id=f"canary-{index:03d}",
            seed=args.seed + index,
            max_tokens=args.max_tokens,
            prompt=CANARY_PROMPTS[index % len(CANARY_PROMPTS)],
            tags={"prompt_index": index % len(CANARY_PROMPTS)},
        )
        for index in range(count)
    ]


def text_units(text: str, tokenizer: TokenizerAdapter | None) -> list[str]:
    if tokenizer:
        return [str(token) for token in tokenizer.encode(text)]
    return re.findall(r"\w+|[^\w\s]", text.lower())


def repeated_ngram_ratio(
    tokens: Sequence[str], width: int = REPETITION_NGRAM_WIDTH
) -> float:
    if len(tokens) < width:
        return 0.0
    ngrams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def excess_repeated_ngram_ratio(
    tokens: Sequence[str],
    width: int = REPETITION_NGRAM_WIDTH,
    allowed_occurrences: int = REPETITION_ALLOWED_OCCURRENCES,
) -> tuple[float, int]:
    if len(tokens) < width:
        return 0.0, 0
    counts = Counter(
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )
    total = len(tokens) - width + 1
    excess = sum(max(count - allowed_occurrences, 0) for count in counts.values())
    return excess / total, max(counts.values())


def periodic_loop_evidence(tokens: Sequence[str]) -> dict[str, float | int | bool]:
    best_span = 0
    best_period = 0
    maximum_period = min(MAX_PERIODIC_PATTERN_TOKENS, len(tokens) // 3)
    for period in range(1, maximum_period + 1):
        matching = 0
        for index in range(period, len(tokens)):
            if tokens[index] == tokens[index - period]:
                matching += 1
                span = matching + period
                if span / period >= MIN_PERIODIC_REPETITIONS and span > best_span:
                    best_span = span
                    best_period = period
            else:
                matching = 0
    repeat_count = best_span / best_period if best_period else 0.0
    detected = (
        best_span >= MIN_PERIODIC_LOOP_TOKENS
        and repeat_count >= MIN_PERIODIC_REPETITIONS
    )
    return {
        "detected": detected,
        "span_tokens": best_span,
        "pattern_tokens": best_period,
        "repeat_count": repeat_count,
        "coverage": best_span / len(tokens) if tokens else 0.0,
    }


def repetition_evidence(
    tokens: Sequence[str], max_excess_ratio: float
) -> dict[str, Any]:
    raw_ratio = repeated_ngram_ratio(tokens)
    excess_ratio, max_occurrences = excess_repeated_ngram_ratio(tokens)
    tail = tokens[-REPETITION_TAIL_TOKENS:]
    tail_raw_ratio = repeated_ngram_ratio(tail)
    tail_excess_ratio, tail_max_occurrences = excess_repeated_ngram_ratio(tail)
    periodic = periodic_loop_evidence(tokens)
    tail_periodic = periodic_loop_evidence(tail)
    degeneration_detected = (
        excess_ratio > max_excess_ratio
        or tail_excess_ratio > max_excess_ratio
        or periodic["detected"]
        or tail_periodic["detected"]
    )
    return {
        "valid": not degeneration_detected,
        "degeneration_detected": degeneration_detected,
        "repeated_ngram_ratio": raw_ratio,
        "excess_repeated_ngram_ratio": excess_ratio,
        "tail_repeated_ngram_ratio": tail_raw_ratio,
        "tail_excess_repeated_ngram_ratio": tail_excess_ratio,
        "max_ngram_occurrences": max_occurrences,
        "tail_max_ngram_occurrences": tail_max_occurrences,
        "max_excess_repeated_ngram_ratio": max_excess_ratio,
        "periodic_loop": periodic,
        "tail_periodic_loop": tail_periodic,
    }


def ks_statistic(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 1.0
    values = sorted(set(left) | set(right))
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    left_index = 0
    right_index = 0
    maximum = 0.0
    for value in values:
        while left_index < len(left_sorted) and left_sorted[left_index] <= value:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= value:
            right_index += 1
        maximum = max(
            maximum,
            abs(left_index / len(left_sorted) - right_index / len(right_sorted)),
        )
    return maximum


def jensen_shannon(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not keys or left_total == 0 or right_total == 0:
        return 1.0
    divergence = 0.0
    for key in keys:
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2.0
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return divergence


def compare_samples(
    candidate: Sequence[RequestResult],
    reference: Sequence[RequestResult],
    tokenizer: TokenizerAdapter | None,
    max_ks: float,
    max_js: float,
) -> dict[str, Any]:
    candidate_tokens = [
        text_units(result.output_transcript, tokenizer) for result in candidate if result.ok
    ]
    reference_tokens = [
        text_units(result.output_transcript, tokenizer) for result in reference if result.ok
    ]
    candidate_lengths = [float(len(tokens)) for tokens in candidate_tokens]
    reference_lengths = [float(len(tokens)) for tokens in reference_tokens]
    candidate_repeats = [repeated_ngram_ratio(tokens) for tokens in candidate_tokens]
    reference_repeats = [repeated_ngram_ratio(tokens) for tokens in reference_tokens]
    length_ks = ks_statistic(candidate_lengths, reference_lengths)
    repetition_ks = ks_statistic(candidate_repeats, reference_repeats)
    unigram_js = jensen_shannon(
        Counter(token for output in candidate_tokens for token in output),
        Counter(token for output in reference_tokens for token in output),
    )
    finish_candidate = Counter(result.finish_reason for result in candidate if result.ok)
    finish_reference = Counter(result.finish_reason for result in reference if result.ok)
    finish_keys = set(finish_candidate) | set(finish_reference)
    candidate_total = sum(finish_candidate.values()) or 1
    reference_total = sum(finish_reference.values()) or 1
    finish_tv = 0.5 * sum(
        abs(finish_candidate[key] / candidate_total - finish_reference[key] / reference_total)
        for key in finish_keys
    )
    passed = (
        length_ks <= max_ks
        and repetition_ks <= max_ks
        and unigram_js <= max_js
        and finish_tv <= max_ks
    )
    return {
        "passed": passed,
        "candidate_samples": len(candidate_tokens),
        "reference_samples": len(reference_tokens),
        "length_ks": length_ks,
        "repetition_ks": repetition_ks,
        "unigram_js": unigram_js,
        "finish_reason_tv": finish_tv,
        "max_ks": max_ks,
        "max_js": max_js,
    }


def exact_output_diagnostics(
    reference: Sequence[RequestResult],
    candidate: Sequence[RequestResult],
    reference_phase: str,
    candidate_phase: str,
    expected_cases: Sequence[RequestSpec] | None = None,
) -> dict[str, Any]:
    reference_successes = [result for result in reference if result.ok]
    candidate_successes = [result for result in candidate if result.ok]
    reference_counts = Counter(
        (result.case_id, result.seed) for result in reference_successes
    )
    candidate_counts = Counter(
        (result.case_id, result.seed) for result in candidate_successes
    )
    expected_counts = (
        Counter((spec.case_id, spec.seed) for spec in expected_cases)
        if expected_cases is not None
        else None
    )
    reference_by_key = {
        (result.case_id, result.seed): result for result in reference_successes
    }
    candidate_by_key = {
        (result.case_id, result.seed): result for result in candidate_successes
    }
    reference_keys = set(reference_by_key)
    candidate_keys = set(candidate_by_key)
    expected_keys = (
        {(spec.case_id, spec.seed) for spec in expected_cases}
        if expected_cases is not None
        else reference_keys | candidate_keys
    )
    reference_missing_expected = sorted(expected_keys - reference_keys)
    candidate_missing_expected = sorted(expected_keys - candidate_keys)
    reference_unexpected = sorted(reference_keys - expected_keys)
    candidate_unexpected = sorted(candidate_keys - expected_keys)
    reference_duplicates = sorted(
        key for key, count in reference_counts.items() if count != 1
    )
    candidate_duplicates = sorted(
        key for key, count in candidate_counts.items() if count != 1
    )
    expected_duplicates = sorted(
        key for key, count in (expected_counts or {}).items() if count != 1
    )
    coverage_complete = (
        bool(expected_keys)
        and not expected_duplicates
        and not reference_missing_expected
        and not candidate_missing_expected
        and not reference_unexpected
        and not candidate_unexpected
        and not reference_duplicates
        and not candidate_duplicates
    )
    shared = sorted(reference_keys & candidate_keys)
    mismatches = []
    transcript_differences = 0
    length_differences = 0
    finish_reason_differences = 0
    for key in shared:
        expected = reference_by_key[key]
        actual = candidate_by_key[key]
        transcript_differs = expected.output_transcript != actual.output_transcript
        if transcript_differs:
            transcript_differences += 1
        if expected.completion_tokens != actual.completion_tokens:
            length_differences += 1
        if expected.finish_reason != actual.finish_reason:
            finish_reason_differences += 1
        if (
            transcript_differs
            or expected.completion_tokens != actual.completion_tokens
            or expected.finish_reason != actual.finish_reason
        ):
            mismatches.append(
                {
                    "case_id": key[0],
                    "seed": key[1],
                    "expected_sha256": stable_hash(expected.output_transcript),
                    "actual_sha256": stable_hash(actual.output_transcript),
                    "expected_completion_tokens": expected.completion_tokens,
                    "actual_completion_tokens": actual.completion_tokens,
                    "expected_finish_reason": expected.finish_reason,
                    "actual_finish_reason": actual.finish_reason,
                }
            )
    missing_candidate = [
        {"case_id": case_id, "seed": seed}
        for case_id, seed in sorted(reference_keys - candidate_keys)
    ]
    missing_reference = [
        {"case_id": case_id, "seed": seed}
        for case_id, seed in sorted(candidate_keys - reference_keys)
    ]
    exact_matches = len(shared) - len(mismatches)
    return {
        "reference_phase": reference_phase,
        "candidate_phase": candidate_phase,
        "passed": coverage_complete and not mismatches,
        "coverage_complete": coverage_complete,
        "expected_cases": len(expected_keys),
        "reference_cases": len(reference_by_key),
        "candidate_cases": len(candidate_by_key),
        "shared_fixed_seed_cases": len(shared),
        "exact_matches": exact_matches,
        "exact_divergences": len(mismatches),
        "exact_match_rate": exact_matches / len(shared) if shared else None,
        "transcript_differences": transcript_differences,
        "completion_length_differences": length_differences,
        "finish_reason_differences": finish_reason_differences,
        "missing_candidate": missing_candidate,
        "missing_reference": missing_reference,
        "reference_missing_expected": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in reference_missing_expected
        ],
        "candidate_missing_expected": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in candidate_missing_expected
        ],
        "reference_unexpected": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in reference_unexpected
        ],
        "candidate_unexpected": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in candidate_unexpected
        ],
        "reference_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in reference_duplicates
        ],
        "candidate_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in candidate_duplicates
        ],
        "expected_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in expected_duplicates
        ],
        "mismatches": mismatches,
    }


def fixed_seed_comparison_evidence(
    exact: dict[str, Any],
    statistical: dict[str, Any],
    semantic_passed: bool,
    exact_gated: bool = True,
) -> dict[str, Any]:
    coverage_complete = exact.get("coverage_complete") is True
    return {
        "exact_diagnostics": exact,
        "exact_diagnostics_gated": exact_gated,
        "case_seed_coverage_complete": coverage_complete,
        "statistical_comparison": statistical,
        "semantic_passed": semantic_passed,
        "passed": (
            coverage_complete
            and statistical["passed"]
            and semantic_passed
            and (not exact_gated or exact["passed"])
        ),
    }


def fixed_seed_invariance_evidence(
    exact_replays: Sequence[dict[str, Any]],
    ordering_comparisons: Sequence[dict[str, Any]],
    cross_phase_comparisons: Sequence[dict[str, Any]],
    phase_count: int,
) -> dict[str, Any]:
    expected_cross_phase_comparisons = max(0, phase_count - 1)
    expected_counts = {
        "exact_replays": phase_count,
        "ordering_comparisons": phase_count,
        "cross_phase_comparisons": expected_cross_phase_comparisons,
    }
    observed_counts = {
        "exact_replays": len(exact_replays),
        "ordering_comparisons": len(ordering_comparisons),
        "cross_phase_comparisons": len(cross_phase_comparisons),
    }
    distribution_comparisons = [*ordering_comparisons, *cross_phase_comparisons]
    comparisons = [*exact_replays, *distribution_comparisons]
    counts_complete = observed_counts == expected_counts
    coverage_complete = bool(comparisons) and all(
        item.get("case_seed_coverage_complete") is True for item in comparisons
    )
    replay_contract_complete = bool(exact_replays) and all(
        item.get("exact_diagnostics_gated") is True
        and (item.get("exact_diagnostics") or {}).get("passed") is True
        and item.get("passed") is True
        for item in exact_replays
    )
    distribution_contract_complete = bool(distribution_comparisons) and all(
        item.get("exact_diagnostics_gated") is False
        and item.get("passed") is True
        for item in distribution_comparisons
    )
    return {
        "passed": (
            counts_complete
            and coverage_complete
            and replay_contract_complete
            and distribution_contract_complete
        ),
        "phase_count": phase_count,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "counts_complete": counts_complete,
        "case_seed_coverage_complete": coverage_complete,
        "same_shape_exact_replays_complete": replay_contract_complete,
        "ordering_and_concurrency_distributions_complete": (
            distribution_contract_complete
        ),
    }


def full_batch_specs(
    specs: Sequence[RequestSpec], concurrency: int
) -> list[RequestSpec]:
    count = len(specs) - len(specs) % concurrency
    if count == 0:
        raise ValueError("measurement requires at least one full concurrent batch")
    return list(specs[:count])


def correctness_order_specs(
    specs: Sequence[RequestSpec], reverse: bool
) -> list[RequestSpec]:
    ordered = list(specs)
    return list(reversed(ordered)) if reverse else ordered


def common_full_batch_specs(
    specs: Sequence[RequestSpec], concurrencies: Sequence[int]
) -> list[RequestSpec]:
    if not concurrencies:
        raise ValueError("common measurement cohort requires a concurrency")
    measured_keys = [
        {(spec.case_id, spec.seed) for spec in full_batch_specs(specs, concurrency)}
        for concurrency in concurrencies
    ]
    common_keys = set.intersection(*measured_keys)
    if not common_keys:
        raise ValueError("measurement concurrencies have no common full-batch cohort")
    return [spec for spec in specs if (spec.case_id, spec.seed) in common_keys]


def common_full_batch_cohort_evidence(
    specs: Sequence[RequestSpec],
    common_specs: Sequence[RequestSpec],
    concurrencies: Sequence[int],
) -> dict[str, Any]:
    requested_counts = Counter((spec.case_id, spec.seed) for spec in specs)
    common_counts = Counter((spec.case_id, spec.seed) for spec in common_specs)
    requested_keys = {(spec.case_id, spec.seed) for spec in specs}
    common_keys = {(spec.case_id, spec.seed) for spec in common_specs}
    measurement_keys = {
        concurrency: {
            (spec.case_id, spec.seed)
            for spec in full_batch_specs(specs, concurrency)
        }
        for concurrency in concurrencies
    }
    expected_common_keys = set.intersection(*measurement_keys.values())
    requested_duplicates = sorted(
        key for key, count in requested_counts.items() if count != 1
    )
    common_duplicates = sorted(
        key for key, count in common_counts.items() if count != 1
    )
    complete = (
        bool(common_keys)
        and common_keys == expected_common_keys
        and not requested_duplicates
        and not common_duplicates
    )
    return {
        "complete": complete,
        "requested_cases": len(requested_keys),
        "common_cases": len(common_keys),
        "measurement_cases_by_concurrency": {
            str(concurrency): len(keys)
            for concurrency, keys in measurement_keys.items()
        },
        "included_cases": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in sorted(common_keys)
        ],
        "excluded_cases": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in sorted(requested_keys - common_keys)
        ],
        "requested_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in requested_duplicates
        ],
        "common_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in common_duplicates
        ],
    }


def fixed_seed_exact_replay_cohort(
    measured_specs: Sequence[RequestSpec], concurrency: int
) -> tuple[list[RequestSpec], dict[str, Any]]:
    if concurrency <= 0 or len(measured_specs) < concurrency:
        raise ValueError("exact replay requires at least one full concurrent batch")
    c1_serial = concurrency == 1
    normal_single_batch = len(measured_specs) == concurrency
    exact_specs = list(measured_specs if c1_serial else measured_specs[:concurrency])
    measured_counts = Counter((spec.case_id, spec.seed) for spec in measured_specs)
    exact_counts = Counter((spec.case_id, spec.seed) for spec in exact_specs)
    measured_duplicates = sorted(
        key for key, count in measured_counts.items() if count != 1
    )
    exact_duplicates = sorted(
        key for key, count in exact_counts.items() if count != 1
    )
    normal_phase_reusable = normal_single_batch or c1_serial
    reuse_reason = (
        "single_full_batch"
        if normal_single_batch
        else (
            "c1_batch_shape_constant"
            if c1_serial
            else "dedicated_single_full_batch_required"
        )
    )
    complete = (
        bool(exact_specs)
        and not measured_duplicates
        and not exact_duplicates
        and (
            (c1_serial and exact_specs == list(measured_specs))
            or (not c1_serial and len(exact_specs) == concurrency)
        )
    )
    exact_keys = set(exact_counts)
    return exact_specs, {
        "complete": complete,
        "concurrency": concurrency,
        "measurement_cases": len(measured_specs),
        "measurement_waves": math.ceil(len(measured_specs) / concurrency),
        "exact_cases": len(exact_specs),
        "cohort_kind": "c1_serial" if c1_serial else "single_full_batch",
        "normal_phase_reusable": normal_phase_reusable,
        "normal_reuse_reason": reuse_reason,
        "included_cases": [
            {"case_id": spec.case_id, "seed": spec.seed} for spec in exact_specs
        ],
        "excluded_cases": [
            {"case_id": spec.case_id, "seed": spec.seed}
            for spec in measured_specs
            if (spec.case_id, spec.seed) not in exact_keys
        ],
        "measurement_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in measured_duplicates
        ],
        "exact_duplicate_keys": [
            {"case_id": case_id, "seed": seed}
            for case_id, seed in exact_duplicates
        ],
    }


def resident_exact_replay_cohort(
    measured_specs: Sequence[RequestSpec], concurrency: int
) -> tuple[list[RequestSpec], dict[str, Any]]:
    if concurrency <= 0:
        raise ValueError("exact replay requires at least one full concurrent batch")
    if len(measured_specs) % concurrency != 0:
        raise ValueError("exact replay measurement cohort must contain full batches")
    return fixed_seed_exact_replay_cohort(measured_specs, concurrency)


def balanced_context_full_batch_specs(
    specs: Sequence[RequestSpec], concurrency: int
) -> list[RequestSpec]:
    contexts = tuple(
        dict.fromkeys(
            spec.context_tokens
            for spec in specs
            if spec.context_tokens is not None
        )
    )
    if not contexts or any(spec.context_tokens is None for spec in specs):
        raise ValueError("balanced measurement requires a context length on every request")
    grouped = {
        context: [spec for spec in specs if spec.context_tokens == context]
        for context in contexts
    }
    requests_per_context = min(len(group) for group in grouped.values())
    while (
        requests_per_context > 0
        and requests_per_context * len(contexts) % concurrency != 0
    ):
        requests_per_context -= 1
    if requests_per_context == 0:
        raise ValueError(
            "measurement requires a context-balanced full concurrent batch"
        )
    selected = {
        (spec.case_id, spec.seed)
        for group in grouped.values()
        for spec in group[:requests_per_context]
    }
    return [spec for spec in specs if (spec.case_id, spec.seed) in selected]


def results_for_specs(
    results: Sequence[RequestResult], specs: Sequence[RequestSpec]
) -> list[RequestResult]:
    keys = {(spec.case_id, spec.seed) for spec in specs}
    return [result for result in results if (result.case_id, result.seed) in keys]


def validate_sampled_output(
    result: RequestResult,
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
) -> tuple[bool, dict[str, Any]]:
    tool_call_text = json.dumps(result.tool_calls, sort_keys=True) if result.tool_calls else ""
    generated = result.reasoning_text + result.output_text + tool_call_text
    units = text_units(generated, tokenizer)
    reasoning_units = text_units(result.reasoning_text, tokenizer)
    content_units = text_units(result.output_text, tokenizer)
    tool_call_units = text_units(tool_call_text, tokenizer)
    channel_repetition = {
        name: repetition_evidence(channel_units, max_repeated_ngram_ratio)
        for name, channel_units in (
            ("reasoning", reasoning_units),
            ("content", content_units),
        )
        if channel_units
    }
    repetition_valid = all(
        evidence["valid"] for evidence in channel_repetition.values()
    )
    combined_repetition = repetition_evidence(units, max_repeated_ngram_ratio)
    valid = (
        result.ok
        and result.completion_tokens > 0
        and result.output_chunks > 0
        and result.finish_reason is not None
        and bool(units)
        and repetition_valid
    )
    return valid, {
        "case_id": result.case_id,
        "seed": result.seed,
        "ok": result.ok,
        "nonempty": bool(units),
        "reasoning_nonempty": bool(reasoning_units),
        "content_nonempty": bool(content_units),
        "tool_calls_nonempty": bool(tool_call_units),
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "repeated_ngram_ratio": combined_repetition["repeated_ngram_ratio"],
        "excess_repeated_ngram_ratio": max(
            (
                evidence["excess_repeated_ngram_ratio"]
                for evidence in channel_repetition.values()
            ),
            default=0.0,
        ),
        "tail_excess_repeated_ngram_ratio": max(
            (
                evidence["tail_excess_repeated_ngram_ratio"]
                for evidence in channel_repetition.values()
            ),
            default=0.0,
        ),
        "max_repeated_ngram_ratio": max_repeated_ngram_ratio,
        "repetition_gate_metric": "excess_after_second_occurrence",
        "repetition_valid": repetition_valid,
        "degeneration_detected": not repetition_valid,
        "channel_repetition": channel_repetition,
        "output_transcript_sha256": stable_hash(result.output_transcript),
    }


def load_canary_artifact(path: Path, phase: str) -> tuple[list[RequestResult], dict[str, Any]]:
    samples: list[RequestResult] = []
    metadata: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if record.get("event") == "run_start":
                metadata["run_start"] = record
            elif record.get("event") == "run_summary":
                metadata["run_summary"] = record
            if record.get("event") != "request" or record.get("phase") != phase:
                continue
            output = record.get("output_text")
            transcript = record.get("output_transcript")
            if output is None or transcript is None:
                raise RuntimeError(
                    f"{path} does not retain full output for {phase}; canary artifacts must keep output"
                )
            samples.append(
                RequestResult(
                    case_id=str(record["case_id"]),
                    seed=int(record["seed"]),
                    ok=bool(record.get("ok")),
                    status_code=record.get("status_code"),
                    started=float(record.get("started") or 0.0),
                    ended=float(record.get("ended") or 0.0),
                    ttft_seconds=record.get("ttft_seconds"),
                    tpot_seconds=record.get("tpot_seconds"),
                    client_queue_seconds=float(record.get("client_queue_seconds") or 0.0),
                    completion_tokens=int(record.get("completion_tokens") or 0),
                    prompt_tokens=record.get("prompt_tokens"),
                    finish_reason=record.get("finish_reason"),
                    output_text=output,
                    reasoning_text=str(record.get("reasoning_text") or ""),
                    tool_calls=record.get("tool_calls") or [],
                    output_transcript=transcript,
                    output_chunks=int(record.get("output_chunks") or 0),
                    stream_done=bool(record.get("stream_done")),
                    usage_received=bool(record.get("usage_received")),
                    request_id=record.get("request_id"),
                    error=record.get("error"),
                    error_kind=record.get("error_kind"),
                    context_tokens=record.get("context_tokens"),
                    tags=record.get("tags") or {},
                )
            )
    if not samples:
        raise RuntimeError(f"no request records for phase {phase!r} in {path}")
    return samples, metadata


def artifact_compatibility(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    candidate_start = candidate.get("run_start") or {}
    reference_start = reference.get("run_start") or {}
    candidate_args = candidate_start.get("arguments") or {}
    reference_args = reference_start.get("arguments") or {}
    fields = (
        "sampling_policy",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "requests",
        "max_tokens",
    )
    differences = {
        field: {"candidate": candidate_args.get(field), "reference": reference_args.get(field)}
        for field in fields
        if candidate_args.get(field) != reference_args.get(field)
    }
    return {"compatible": not differences, "differences": differences}


async def compare_mode(
    args: argparse.Namespace,
    tokenizer: TokenizerAdapter | None,
    writer: JsonlWriter,
) -> dict[str, Any]:
    candidate, candidate_metadata = load_canary_artifact(args.candidate, args.candidate_phase)
    reference, reference_metadata = load_canary_artifact(args.reference, args.reference_phase)
    compatibility = artifact_compatibility(candidate_metadata, reference_metadata)
    statistical = compare_samples(
        candidate,
        reference,
        tokenizer,
        args.stat_max_ks,
        args.stat_max_js,
    )
    statistical_by_concurrency = []
    if args.require_part1_complete:
        for concurrency in DEFAULT_CONCURRENCIES:
            phase = f"canary-c{concurrency}-normal"
            candidate_phase_samples, _ = load_canary_artifact(args.candidate, phase)
            reference_phase_samples, _ = load_canary_artifact(args.reference, phase)
            comparison = compare_samples(
                candidate_phase_samples,
                reference_phase_samples,
                tokenizer,
                args.stat_max_ks,
                args.stat_max_js,
            )
            comparison.update(
                {
                    "concurrency": concurrency,
                    "candidate_phase": phase,
                    "reference_phase": phase,
                    "candidate_errors": sum(
                        not result.ok for result in candidate_phase_samples
                    ),
                    "reference_errors": sum(
                        not result.ok for result in reference_phase_samples
                    ),
                }
            )
            comparison["passed"] = (
                comparison["passed"]
                and comparison["candidate_errors"] == 0
                and comparison["reference_errors"] == 0
                and comparison["candidate_samples"]
                == comparison["reference_samples"]
            )
            statistical_by_concurrency.append(comparison)
            await writer.emit("offline_comparison_by_concurrency", **comparison)
    candidate_by_seed = {
        (item.case_id, item.seed): item.output_transcript for item in candidate if item.ok
    }
    reference_by_seed = {
        (item.case_id, item.seed): item.output_transcript for item in reference if item.ok
    }
    shared = sorted(set(candidate_by_seed) & set(reference_by_seed))
    exact = sum(candidate_by_seed[key] == reference_by_seed[key] for key in shared)
    candidate_canary_passed = (candidate_metadata.get("run_summary") or {}).get("passed")
    reference_canary_passed = (reference_metadata.get("run_summary") or {}).get("passed")
    candidate_summary = candidate_metadata.get("run_summary") or {}
    reference_summary = reference_metadata.get("run_summary") or {}
    candidate_start = candidate_metadata.get("run_start") or {}
    reference_start = reference_metadata.get("run_start") or {}
    candidate_args = candidate_start.get("arguments") or {}
    reference_args = reference_start.get("arguments") or {}
    candidate_concurrencies = set(candidate_args.get("concurrencies") or ())
    reference_concurrencies = set(reference_args.get("concurrencies") or ())
    candidate_mtp = candidate_summary.get("candidate_mtp") or {}
    reference_mtp = reference_summary.get("candidate_mtp") or {}
    candidate_graph = candidate_summary.get("candidate_cuda_graph") or {}
    reference_graph = reference_summary.get("candidate_cuda_graph") or {}
    candidate_edges = candidate_summary.get("edge_cases") or {}
    candidate_sampling = production_sampling_policy_evidence(
        candidate_args.get("sampling_policy"),
        candidate_start.get("policy") or candidate_args,
    )
    reference_sampling = production_sampling_policy_evidence(
        reference_args.get("sampling_policy"),
        reference_start.get("policy") or reference_args,
    )
    candidate_fixed_seed = candidate_summary.get("fixed_seed_invariance") or {}
    reference_fixed_seed = reference_summary.get("fixed_seed_invariance") or {}
    part1_coverage = {
        "candidate_c1_c8_c16": set(DEFAULT_CONCURRENCIES).issubset(
            candidate_concurrencies
        ),
        "reference_c1_c8_c16": set(DEFAULT_CONCURRENCIES).issubset(
            reference_concurrencies
        ),
        "candidate_production_sampling_policy": candidate_sampling["passed"],
        "reference_production_sampling_policy": reference_sampling["passed"],
        "candidate_fixed_seed_invariance": (
            candidate_fixed_seed.get("passed") is True
        ),
        "reference_fixed_seed_invariance": (
            reference_fixed_seed.get("passed") is True
        ),
        "candidate_edge_cases": (
            candidate_edges.get("passed") is True
            and candidate_edges.get("skipped") is not True
        ),
        "candidate_mtp": (
            candidate_mtp.get("passed") is True
            and candidate_mtp.get("active") is True
        ),
        "candidate_target_and_dflash_graphs": (
            candidate_graph.get("passed") is True
            and set((candidate_graph.get("components") or {}).keys())
            == {"target", "dflash"}
        ),
        "reference_target_only": (
            reference_mtp.get("passed") is True
            and reference_mtp.get("active") is False
        ),
        "reference_target_graph": (
            reference_graph.get("passed") is True
            and set((reference_graph.get("components") or {}).keys()) == {"target"}
        ),
        "statistical_comparison": (
            statistical["passed"]
            and len(statistical_by_concurrency) == len(DEFAULT_CONCURRENCIES)
            and all(item["passed"] for item in statistical_by_concurrency)
        ),
    }
    coverage_complete = all(part1_coverage.values())
    comparison_passed = (
        compatibility["compatible"]
        and statistical["passed"]
        and bool(candidate_canary_passed)
        and bool(reference_canary_passed)
    )
    passed = comparison_passed and (
        not args.require_part1_complete or coverage_complete
    )
    summary = {
        "mode": "compare",
        "passed": passed,
        "comparison_passed": comparison_passed,
        "coverage_complete": coverage_complete,
        "part1_coverage": part1_coverage,
        "candidate_sampling_policy": candidate_sampling,
        "reference_sampling_policy": reference_sampling,
        "require_part1_complete": args.require_part1_complete,
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "candidate_phase": args.candidate_phase,
        "reference_phase": args.reference_phase,
        "candidate_canary_passed": candidate_canary_passed,
        "reference_canary_passed": reference_canary_passed,
        "compatibility": compatibility,
        "statistical_comparison": statistical,
        "statistical_comparison_by_concurrency": statistical_by_concurrency,
        "shared_fixed_seed_cases": len(shared),
        "exact_fixed_seed_matches": exact,
        "exact_fixed_seed_match_rate": exact / len(shared) if shared else None,
    }
    if "server_provenance" in candidate_summary:
        summary["server_provenance"] = candidate_summary["server_provenance"]
    await writer.emit("offline_comparison", **summary)
    return summary


async def run_edge_cases(
    client: SoakClient,
    args: argparse.Namespace,
    writer: JsonlWriter,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}

    async def execute(
        name: str,
        spec: RequestSpec,
        validate: Any,
    ) -> None:
        body, record = await client.request_json(spec)
        valid = False
        detail = None
        if body is not None:
            try:
                valid, detail = validate(body)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
        result = {**record, "valid": valid, "detail": detail}
        results[name] = result
        await writer.emit("canary_edge", edge=name, **result)

    base_seed = args.seed + 100_000
    await execute(
        "stop_sequence",
        RequestSpec(
            case_id="edge-stop",
            seed=base_seed,
            max_tokens=32,
            prompt="Emit exactly GREEN STOP_BOUNDARY RED.",
            extra={
                "stop": ["STOP_BOUNDARY"],
                "grammar": {"type": "regex", "value": "GREEN STOP_BOUNDARY RED"},
                "enable_thinking": False,
            },
        ),
        lambda body: validate_stop(body, "GREEN", "STOP_BOUNDARY"),
    )
    await execute(
        "max_length",
        RequestSpec(
            case_id="edge-max-length",
            seed=base_seed + 1,
            max_tokens=args.max_length_tokens,
            prompt="Continue writing varied prose until the token budget is exhausted.",
            extra={"ignore_eos": True},
        ),
        lambda body: validate_max_length(body, args.max_length_tokens),
    )
    await execute(
        "regex",
        RequestSpec(
            case_id="edge-regex",
            seed=base_seed + 2,
            max_tokens=16,
            prompt="Return a soak identifier in the requested format.",
            extra={
                "grammar": {"type": "regex", "value": "SOAK-[0-9]{4}"},
                "enable_thinking": False,
            },
        ),
        lambda body: validate_regex(body, r"SOAK-[0-9]{4}"),
    )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["healthy"]},
            "workers": {"type": "integer", "minimum": 1, "maximum": 16},
        },
        "required": ["status", "workers"],
        "additionalProperties": False,
    }
    await execute(
        "json_schema",
        RequestSpec(
            case_id="edge-json-schema",
            seed=base_seed + 3,
            max_tokens=32,
            prompt="Return a healthy service status and a worker count.",
            extra={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "soak_status", "schema": schema},
                },
                "enable_thinking": False,
            },
        ),
        validate_json_schema,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_soak_status",
                "description": "Record the status of a serving soak.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy"]},
                        "concurrency": {"type": "integer"},
                    },
                    "required": ["status", "concurrency"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]
    await execute(
        "required_tool",
        RequestSpec(
            case_id="edge-tool",
            seed=base_seed + 4,
            max_tokens=64,
            prompt="Record a healthy soak at concurrency 8 using the available tool.",
            extra={
                "tools": tools,
                "tool_choice": "required",
                "enable_thinking": False,
            },
        ),
        validate_tool,
    )
    eos_token_id = args.eos_token_id
    if eos_token_id is None and client.tokenizer:
        eos_token_id = client.tokenizer.eos_token_id()
    if eos_token_id is not None:
        await execute(
            "eos",
            RequestSpec(
                case_id="edge-eos",
                seed=base_seed + 5,
                max_tokens=8,
                prompt="End the response now.",
                extra={"ignore_eos": False, "logit_bias": {str(eos_token_id): 100.0}},
            ),
            validate_eos,
        )
    else:
        result = {
            "ok": True,
            "valid": False,
            "skipped": True,
            "detail": "pass --eos-token-id or --tokenizer; required EOS coverage was not run",
        }
        results["eos"] = result
        await writer.emit("canary_edge", edge="eos", **result)
    return {
        "passed": all(item.get("ok") and item.get("valid") for item in results.values()),
        "edges": results,
    }


def response_choice(body: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    return message.get("content") or "", choice.get("finish_reason"), message


def validate_stop(body: dict[str, Any], expected_prefix: str, stop: str) -> tuple[bool, str]:
    content, finish, _ = response_choice(body)
    valid = content.strip() == expected_prefix and stop not in content and finish == "stop"
    return valid, f"finish={finish!r} content={content!r}"


def validate_max_length(body: dict[str, Any], expected: int) -> tuple[bool, str]:
    _, finish, _ = response_choice(body)
    actual = int((body.get("usage") or {}).get("completion_tokens") or 0)
    valid = actual == expected and finish == "length"
    return valid, f"finish={finish!r} completion_tokens={actual} expected={expected}"


def validate_regex(body: dict[str, Any], pattern: str) -> tuple[bool, str]:
    content, finish, _ = response_choice(body)
    valid = re.fullmatch(pattern, content.strip()) is not None
    return valid, f"finish={finish!r} content={content!r}"


def validate_json_schema(body: dict[str, Any]) -> tuple[bool, str]:
    content, finish, _ = response_choice(body)
    value = json.loads(content)
    valid = (
        isinstance(value, dict)
        and value.get("status") == "healthy"
        and isinstance(value.get("workers"), int)
        and 1 <= value["workers"] <= 16
        and set(value) == {"status", "workers"}
    )
    return valid, f"finish={finish!r} value={value!r}"


def validate_tool(body: dict[str, Any]) -> tuple[bool, str]:
    _, finish, message = response_choice(body)
    calls = message.get("tool_calls") or []
    if not calls:
        return False, f"finish={finish!r} no tool call"
    function = calls[0].get("function") or {}
    arguments = json.loads(function.get("arguments") or "{}")
    valid = (
        function.get("name") == "record_soak_status"
        and arguments.get("status") == "healthy"
        and arguments.get("concurrency") == 8
    )
    return valid, f"finish={finish!r} function={function!r}"


def validate_eos(body: dict[str, Any]) -> tuple[bool, str]:
    _, finish, _ = response_choice(body)
    actual = int((body.get("usage") or {}).get("completion_tokens") or 0)
    valid = finish == "stop" and actual <= 2
    return valid, f"finish={finish!r} completion_tokens={actual}"


async def canary_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    candidate_metrics_before = await safe_metrics(client, writer, "canary-candidate-start")
    specs = make_canary_specs(args)
    baseline: list[RequestResult] | None = None
    baseline_phase = ""
    summaries: list[dict[str, Any]] = []
    all_candidate: list[RequestResult] = []
    candidate_normal_by_concurrency: dict[int, list[RequestResult]] = {}
    cross_phase_results: list[tuple[str, list[RequestResult]]] = []
    exact_replays: list[dict[str, Any]] = []
    exact_replay_cohorts: list[dict[str, Any]] = []
    ordering_comparisons: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    semantic_by_phase: dict[str, bool] = {}
    for concurrency in args.concurrencies:
        exact_specs, exact_cohort = fixed_seed_exact_replay_cohort(
            specs,
            concurrency,
        )
        exact_replay_cohorts.append(exact_cohort)
        await writer.emit("canary_exact_replay_cohort", **exact_cohort)
        stabilization_phase = f"canary-c{concurrency}-stabilize"
        _, stabilization_summary = await run_batch(
            client,
            specs,
            concurrency,
            writer,
            stabilization_phase,
            keep_output=False,
        )
        stabilization_summary["order"] = "stabilize"
        summaries.append(stabilization_summary)
        normal_results: list[RequestResult] | None = None
        normal_phase = ""
        for order in ("normal", "normal-replay", "reverse"):
            ordered = correctness_order_specs(specs, order == "reverse")
            phase = f"canary-c{concurrency}-{order}"
            results, summary = await run_batch(
                client, ordered, concurrency, writer, phase, keep_output=True
            )
            summary["order"] = order
            summaries.append(summary)
            phase_quality = []
            for result in results:
                valid, detail = validate_sampled_output(
                    result, client.tokenizer, args.max_repeated_ngram_ratio
                )
                check = {"phase": phase, "valid": valid, **detail}
                phase_quality.append(check)
                quality_checks.append(check)
            semantic_by_phase[phase] = bool(phase_quality) and all(
                item["valid"] for item in phase_quality
            )
            if order == "normal":
                normal_results = results
                normal_phase = phase
                all_candidate.extend(results)
                candidate_normal_by_concurrency[concurrency] = results
                cross_phase_results.append((phase, results))
                if baseline is None:
                    baseline = results
                    baseline_phase = phase
            elif order == "normal-replay":
                if normal_results is None:
                    raise RuntimeError("normal canary phase must precede its exact replay")
                full_cohort_statistical = compare_samples(
                    results,
                    normal_results,
                    client.tokenizer,
                    args.stat_max_ks,
                    args.stat_max_js,
                )
                full_cohort_semantic = (
                    semantic_by_phase[normal_phase] and semantic_by_phase[phase]
                )
                if exact_cohort["normal_phase_reusable"]:
                    exact_baseline = normal_results
                    exact_results = results
                    exact_baseline_phase = normal_phase
                    exact_replay_phase = phase
                else:
                    exact_baseline_phase = (
                        f"canary-c{concurrency}-exact-baseline"
                    )
                    exact_baseline, exact_baseline_summary = await run_batch(
                        client,
                        exact_specs,
                        concurrency,
                        writer,
                        exact_baseline_phase,
                        keep_output=True,
                    )
                    exact_baseline_summary["order"] = "exact-baseline"
                    summaries.append(exact_baseline_summary)
                    exact_replay_phase = f"canary-c{concurrency}-exact-replay"
                    exact_results, exact_replay_summary = await run_batch(
                        client,
                        exact_specs,
                        concurrency,
                        writer,
                        exact_replay_phase,
                        keep_output=True,
                    )
                    exact_replay_summary["order"] = "exact-replay"
                    summaries.append(exact_replay_summary)
                    for exact_phase, exact_phase_results in (
                        (exact_baseline_phase, exact_baseline),
                        (exact_replay_phase, exact_results),
                    ):
                        exact_quality = []
                        for result in exact_phase_results:
                            valid, detail = validate_sampled_output(
                                result,
                                client.tokenizer,
                                args.max_repeated_ngram_ratio,
                            )
                            check = {"phase": exact_phase, "valid": valid, **detail}
                            exact_quality.append(check)
                            quality_checks.append(check)
                        semantic_by_phase[exact_phase] = bool(exact_quality) and all(
                            item["valid"] for item in exact_quality
                        )
                exact = exact_output_diagnostics(
                    exact_baseline,
                    exact_results,
                    exact_baseline_phase,
                    exact_replay_phase,
                    exact_specs,
                )
                exact_statistical = compare_samples(
                    exact_results,
                    exact_baseline,
                    client.tokenizer,
                    args.stat_max_ks,
                    args.stat_max_js,
                )
                exact_semantic = (
                    semantic_by_phase[exact_baseline_phase]
                    and semantic_by_phase[exact_replay_phase]
                )
                exact_evidence = fixed_seed_comparison_evidence(
                    exact,
                    exact_statistical,
                    exact_semantic,
                )
                replay = {
                    "concurrency": concurrency,
                    "cohort": exact_cohort,
                    "full_cohort_statistical_comparison": full_cohort_statistical,
                    "full_cohort_semantic_passed": full_cohort_semantic,
                    **exact_evidence,
                }
                replay["passed"] = (
                    replay["passed"]
                    and exact_cohort["complete"]
                    and full_cohort_statistical["passed"]
                    and full_cohort_semantic
                )
                exact_replays.append(replay)
                await writer.emit("exact_replay_comparison", **replay)
            else:
                if normal_results is None:
                    raise RuntimeError("normal canary phase must precede reverse ordering")
                exact = exact_output_diagnostics(
                    normal_results,
                    results,
                    normal_phase,
                    phase,
                    specs,
                )
                statistical_ordering = compare_samples(
                    results,
                    normal_results,
                    client.tokenizer,
                    args.stat_max_ks,
                    args.stat_max_js,
                )
                ordering = {
                    "concurrency": concurrency,
                    **fixed_seed_comparison_evidence(
                        exact,
                        statistical_ordering,
                        semantic_by_phase[normal_phase] and semantic_by_phase[phase],
                        exact_gated=False,
                    ),
                }
                ordering_comparisons.append(ordering)
                await writer.emit("exact_ordering_comparison", **ordering)
    if baseline is None:
        raise RuntimeError("canary produced no baseline")

    cross_phase_comparisons = []
    fixed_seed_mismatches = []
    for phase, results in cross_phase_results:
        if phase == baseline_phase:
            continue
        exact = exact_output_diagnostics(
            baseline,
            results,
            baseline_phase,
            phase,
            specs,
        )
        statistical = compare_samples(
            results,
            baseline,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        semantic_passed = semantic_by_phase[baseline_phase] and semantic_by_phase[phase]
        comparison = {
            "phase": phase,
            **fixed_seed_comparison_evidence(
                exact,
                statistical,
                semantic_passed,
                exact_gated=False,
            ),
        }
        cross_phase_comparisons.append(comparison)
        fixed_seed_mismatches.extend(
            {"phase": phase, **mismatch} for mismatch in exact["mismatches"]
        )
        await writer.emit("cross_shape_comparison", **comparison)
    edge_summary = {"passed": True, "skipped": True, "edges": {}}
    if not args.skip_edge_cases:
        edge_summary = await run_edge_cases(client, args, writer)
    statistical = None
    statistical_by_concurrency: list[dict[str, Any]] = []
    reference_summaries: list[dict[str, Any]] = []
    reference_target_only: dict[str, Any] | None = None
    reference_graph: dict[str, Any] | None = None
    if args.reference_url:
        reference = SoakClient(
            args.reference_url,
            args.model,
            args.api_key,
            args.timeout,
            client.policy,
            client.tokenizer,
        )
        all_reference: list[RequestResult] = []
        try:
            reference_metrics_before = await safe_metrics(
                reference, writer, "canary-reference-start"
            )
            for concurrency in args.concurrencies:
                results, summary = await run_batch(
                    reference,
                    specs,
                    concurrency,
                    writer,
                    f"reference-c{concurrency}-normal",
                    keep_output=True,
                )
                all_reference.extend(results)
                reference_summaries.append(summary)
                comparison = compare_samples(
                    candidate_normal_by_concurrency[concurrency],
                    results,
                    client.tokenizer,
                    args.stat_max_ks,
                    args.stat_max_js,
                )
                comparison.update(
                    {
                        "concurrency": concurrency,
                        "candidate_errors": sum(
                            not result.ok
                            for result in candidate_normal_by_concurrency[concurrency]
                        ),
                        "reference_errors": summary["errors"],
                    }
                )
                comparison["passed"] = (
                    comparison["passed"]
                    and comparison["candidate_errors"] == 0
                    and comparison["reference_errors"] == 0
                    and comparison["candidate_samples"]
                    == comparison["reference_samples"]
                )
                statistical_by_concurrency.append(comparison)
                await writer.emit("statistical_comparison_by_concurrency", **comparison)
            reference_metrics_after = await safe_metrics(
                reference, writer, "canary-reference-end"
            )
            reference_target_only = target_only_speculative_evidence(
                reference_metrics_before,
                reference_metrics_after,
            )
            reference_graph = cuda_graph_evidence(
                reference_metrics_before,
                reference_metrics_after,
                reference_metrics_before,
                ("target",),
                args.min_cuda_graph_replay_ratio,
            )
        finally:
            await reference.close()
        statistical = compare_samples(
            all_candidate,
            all_reference,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        await writer.emit("statistical_comparison", **statistical)
    candidate_metrics_after = await safe_metrics(client, writer, "canary-candidate-end")
    candidate_mtp = configured_speculative_evidence(
        candidate_metrics_before,
        candidate_metrics_after,
        args,
        args.require_mtp,
    )
    candidate_graph = cuda_graph_evidence(
        candidate_metrics_before,
        candidate_metrics_after,
        candidate_metrics_before,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    sampling_policy = production_sampling_policy_evidence(
        args.sampling_policy,
        client.policy,
    )
    fixed_seed_invariance = fixed_seed_invariance_evidence(
        exact_replays,
        ordering_comparisons,
        cross_phase_comparisons,
        len(args.concurrencies),
    )
    edge_names = {"stop_sequence", "max_length", "regex", "json_schema", "required_tool", "eos"}
    edge_coverage_complete = (
        not args.skip_edge_cases
        and edge_names.issubset(edge_summary["edges"])
        and edge_summary["passed"]
    )
    part1_coverage = {
        "c1_c8_c16": set(DEFAULT_CONCURRENCIES).issubset(args.concurrencies),
        "production_sampling_policy": sampling_policy["passed"],
        "fixed_seed_invariance": fixed_seed_invariance["passed"],
        "edge_cases": edge_coverage_complete,
        "target_reference": args.reference_url is not None,
        "statistical_comparison": (
            statistical is not None
            and len(statistical_by_concurrency) >= len(DEFAULT_CONCURRENCIES)
        ),
        "candidate_mtp": (
            args.require_mtp and candidate_mtp["passed"] and candidate_mtp["active"]
        ),
        "candidate_target_and_dflash_graphs": candidate_graph["passed"],
        "reference_target_only": (
            reference_target_only is not None and reference_target_only["passed"]
        ),
        "reference_target_graph": (
            reference_graph is not None and reference_graph["passed"]
        ),
    }
    coverage_complete = all(part1_coverage.values())
    request_errors = sum(summary["errors"] for summary in summaries)
    engine_passed = (
        request_errors == 0
        and all(item["passed"] for item in exact_replays)
        and all(item["passed"] for item in ordering_comparisons)
        and all(item["passed"] for item in cross_phase_comparisons)
        and all(item["valid"] for item in quality_checks)
        and edge_summary["passed"]
        and (statistical is None or statistical["passed"])
        and all(item["passed"] for item in statistical_by_concurrency)
        and (not args.require_mtp or (candidate_mtp["passed"] and candidate_graph["passed"]))
    )
    passed = engine_passed and (not args.require_part1_complete or coverage_complete)
    return {
        "mode": "canary",
        "passed": passed,
        "engine_passed": engine_passed,
        "coverage_complete": coverage_complete,
        "part1_coverage": part1_coverage,
        "require_part1_complete": args.require_part1_complete,
        "production_sampling": asdict(client.policy),
        "sampling_policy": sampling_policy,
        "fixed_seed_invariance": fixed_seed_invariance,
        "summaries": summaries,
        "exact_replay_cohorts": exact_replay_cohorts,
        "exact_replays": exact_replays,
        "ordering_comparisons": ordering_comparisons,
        "cross_phase_comparisons": cross_phase_comparisons,
        "fixed_seed_mismatches": fixed_seed_mismatches,
        "quality_checks": quality_checks,
        "edge_cases": edge_summary,
        "reference_summaries": reference_summaries,
        "statistical_comparison": statistical,
        "statistical_comparison_by_concurrency": statistical_by_concurrency,
        "candidate_mtp": candidate_mtp,
        "candidate_cuda_graph": candidate_graph,
        "reference_target_only": reference_target_only,
        "reference_cuda_graph": reference_graph,
    }


async def sweep_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    if args.context_tokens is not None:
        if client.tokenizer is None:
            raise RuntimeError("--context-tokens requires --tokenizer")
        await calibrate_prompt_profiles(client, writer, (CONTEXT_PROMPT_PROFILE,))
        prompt = exact_context(client, args.context_tokens, "sweep")
    else:
        prompt = CANARY_PROMPTS[0]
    count = max(args.requests, max(args.concurrencies))
    specs = [
        RequestSpec(
            case_id=f"sweep-{index:04d}",
            seed=args.seed + index,
            max_tokens=args.max_tokens,
            prompt=prompt,
            context_tokens=args.context_tokens,
        )
        for index in range(count)
    ]
    summaries = []
    for concurrency in args.concurrencies:
        _, summary = await run_batch(
            client,
            specs,
            concurrency,
            writer,
            f"sweep-c{concurrency}",
            keep_output=False,
        )
        summaries.append(summary)
    return {
        "mode": "sweep",
        "passed": all(summary["errors"] == 0 for summary in summaries),
        "summaries": summaries,
    }


def prompt_profile_extra(profile: str) -> dict[str, Any]:
    if profile == CONTEXT_PROMPT_PROFILE:
        return {}
    if profile == RETRIEVAL_PROMPT_PROFILE:
        return {"enable_thinking": False}
    raise ValueError(f"unknown prompt profile {profile!r}")


def calibration_prompt(
    tokenizer: TokenizerAdapter,
    profile: str,
    raw_tokens: int,
    label: str,
) -> str:
    if profile == CONTEXT_PROMPT_PROFILE:
        return tokenizer.exact_text(raw_tokens, label)
    if profile == RETRIEVAL_PROMPT_PROFILE:
        return tokenizer.retrieval_text(raw_tokens, label)[0]
    raise ValueError(f"unknown prompt profile {profile!r}")


async def calibrate_prompt_profiles(
    client: SoakClient,
    writer: JsonlWriter,
    profiles: Sequence[str],
) -> None:
    tokenizer = client.tokenizer
    if tokenizer is None:
        raise RuntimeError("prompt-length calibration requires --tokenizer")
    for profile in profiles:
        if profile in client.prompt_overhead_tokens:
            continue
        observations = []
        for index, raw_tokens in enumerate(PROMPT_CALIBRATION_RAW_LENGTHS):
            prompt = calibration_prompt(
                tokenizer,
                profile,
                raw_tokens,
                f"calibration-{profile}-{index}",
            )
            local_tokens = tokenizer.count(prompt)
            if local_tokens != raw_tokens:
                raise RuntimeError(
                    f"{profile} calibration prompt has {local_tokens} local tokens, "
                    f"expected {raw_tokens}"
                )
            spec = RequestSpec(
                case_id=f"calibration-{profile}-{index}",
                seed=DEFAULT_SEED + 900_000 + index,
                max_tokens=1,
                prompt=prompt,
                tags={"scenario": "prompt_calibration", "profile": profile},
                extra={"ignore_eos": True, **prompt_profile_extra(profile)},
            )
            body, record = await client.request_json(spec)
            usage = body.get("usage") if body is not None else None
            observed_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if not record["ok"] or observed_tokens is None:
                raise RuntimeError(
                    f"prompt calibration failed for {profile}: "
                    f"{record['error'] or 'response did not contain usage.prompt_tokens'}"
                )
            observed_tokens = int(observed_tokens)
            overhead_tokens = observed_tokens - raw_tokens
            if overhead_tokens < 0:
                raise RuntimeError(
                    f"server reported fewer prompt tokens than the tokenizer for {profile}: "
                    f"{observed_tokens} < {raw_tokens}"
                )
            observations.append(overhead_tokens)
            await writer.emit(
                "prompt_calibration_probe",
                profile=profile,
                raw_prompt_tokens=raw_tokens,
                server_prompt_tokens=observed_tokens,
                overhead_tokens=overhead_tokens,
                request=record,
            )
        if len(set(observations)) != 1:
            raise RuntimeError(
                f"chat-template overhead is not stable for {profile}: {observations}"
            )
        client.prompt_overhead_tokens[profile] = observations[0]
        await writer.emit(
            "prompt_calibration",
            profile=profile,
            overhead_tokens=observations[0],
            probe_raw_lengths=list(PROMPT_CALIBRATION_RAW_LENGTHS),
        )


def exact_contexts(
    client: SoakClient,
    lengths: Sequence[int],
    namespace: str,
) -> dict[int, str]:
    if client.tokenizer is None:
        raise RuntimeError("this mode requires --tokenizer for exact context construction")
    return {
        length: exact_context(client, length, f"{namespace}-{length}")
        for length in lengths
    }


def exact_context(
    client: SoakClient,
    length: int,
    label: str,
) -> str:
    if client.tokenizer is None:
        raise RuntimeError("this mode requires --tokenizer for exact context construction")
    content_tokens = client.calibrated_content_tokens(CONTEXT_PROMPT_PROFILE, length)
    return client.tokenizer.exact_text(content_tokens, label)


def retrieval_spec(
    client: SoakClient,
    length: int,
    label: str,
    seed: int,
    max_tokens: int,
    tags: dict[str, Any],
) -> RequestSpec:
    if client.tokenizer is None:
        raise RuntimeError("retrieval cases require --tokenizer")
    content_tokens = client.calibrated_content_tokens(RETRIEVAL_PROMPT_PROFILE, length)
    prompt, expected = client.tokenizer.retrieval_text(content_tokens, label)
    return RequestSpec(
        case_id=label,
        seed=seed,
        max_tokens=max_tokens,
        prompt=prompt,
        context_tokens=length,
        tags={**tags, "expected_answer": expected},
        extra=prompt_profile_extra(RETRIEVAL_PROMPT_PROFILE),
    )


def adversarial_mixed_specs(
    client: SoakClient,
    context_lengths: Sequence[int],
    request_count: int,
    seed: int,
    max_tokens: int,
) -> list[RequestSpec]:
    lengths = (
        tuple(context_lengths)
        * math.ceil(request_count / len(context_lengths))
    )[:request_count]
    return [
        retrieval_spec(
            client,
            length,
            f"mixed-{index:03d}-{length}",
            seed + index,
            max_tokens,
            {"scenario": "mixed", "length": length, "role": "traffic"},
        )
        for index, length in enumerate(lengths)
    ]


def request_cohort_uniqueness_evidence(
    specs: Sequence[RequestSpec],
) -> dict[str, Any]:
    case_ids = [spec.case_id for spec in specs]
    case_seed_keys = [(spec.case_id, spec.seed) for spec in specs]
    prompt_hashes = [stable_hash(spec.prompt or "") for spec in specs]
    return {
        "passed": (
            bool(specs)
            and len(set(case_ids)) == len(specs)
            and len(set(case_seed_keys)) == len(specs)
            and len(set(prompt_hashes)) == len(specs)
        ),
        "requests": len(specs),
        "unique_case_ids": len(set(case_ids)),
        "unique_case_seed_keys": len(set(case_seed_keys)),
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "prompt_sha256": prompt_hashes,
    }


def adversarial_long_resident_cohorts(
    client: SoakClient,
    seed: int,
    max_tokens: int,
    request_extra: dict[str, Any],
) -> tuple[list[RequestSpec], list[RequestSpec]]:
    measured = [
        RequestSpec(
            case_id=f"adversarial-long-resident-{index}",
            seed=seed + index,
            max_tokens=max_tokens,
            prompt=exact_context(
                client,
                ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS,
                f"adversarial-long-resident-{index}",
            ),
            context_tokens=ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS,
            tags={"scenario": "long_resident", "stage": "measure"},
            extra=dict(request_extra),
        )
        for index in range(ADVERSARIAL_LONG_RESIDENT_REQUESTS)
    ]
    warm = [
        RequestSpec(
            case_id=f"{spec.case_id}-warm",
            seed=spec.seed,
            max_tokens=1,
            prompt=spec.prompt,
            context_tokens=spec.context_tokens,
            tags={"scenario": "long_resident", "stage": "warm"},
            extra=dict(spec.extra),
        )
        for spec in measured
    ]
    return measured, warm


def full_length_completion_evidence(
    results: Sequence[RequestResult], requested_max_tokens: int
) -> dict[str, Any]:
    requests = [
        {
            "case_id": result.case_id,
            "seed": result.seed,
            "ok": result.ok,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
            "passed": (
                result.ok
                and result.completion_tokens == requested_max_tokens
                and result.finish_reason == "length"
            ),
        }
        for result in results
    ]
    return {
        "passed": bool(requests) and all(item["passed"] for item in requests),
        "requested_max_tokens": requested_max_tokens,
        "requests": requests,
    }


def validate_retrieval_result(
    result: RequestResult,
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
) -> tuple[bool, dict[str, Any]]:
    expected = str(result.tags["expected_answer"])
    normalized = re.sub(r"\s+", "", result.output_text).upper()
    expected_normalized = re.sub(r"\s+", "", expected).upper()
    repetition = repeated_ngram_ratio(text_units(result.output_text, tokenizer))
    semantic = normalized == expected_normalized
    prompt_tokens_match = result.prompt_tokens == result.context_tokens
    valid = (
        result.ok
        and prompt_tokens_match
        and semantic
        and repetition <= max_repeated_ngram_ratio
    )
    return valid, {
        "case_id": result.case_id,
        "ok": result.ok,
        "requested_prompt_tokens": result.context_tokens,
        "server_prompt_tokens": result.prompt_tokens,
        "prompt_tokens_match": prompt_tokens_match,
        "semantic_match": semantic,
        "repeated_ngram_ratio": repetition,
        "max_repeated_ngram_ratio": max_repeated_ngram_ratio,
        "output_transcript_sha256": stable_hash(result.output_transcript),
    }


async def run_long_context_correctness(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    specs = [
        retrieval_spec(
            client,
            length,
            f"long-correctness-{length}",
            args.seed + 80_000 + index,
            args.long_correctness_max_tokens,
            {"scenario": "long_correctness", "length": length},
        )
        for index, length in enumerate(args.long_correctness_context_lengths)
    ]
    cold_metrics_before = await safe_metrics(
        client,
        writer,
        "long-correctness-c1-cold-start",
    )
    cold_phase = "long-correctness-c1-cold"
    cold, cold_summary = await run_batch(
        client,
        specs,
        1,
        writer,
        cold_phase,
        keep_output=True,
    )
    cold_metrics_after = await safe_metrics(
        client,
        writer,
        "long-correctness-c1-cold-end",
    )
    cold_dispatch = labeled_metric_deltas(
        cold_metrics_before,
        cold_metrics_after,
        "mistralrs_cuda_graph_dispatch_total",
    )
    cold_events = labeled_metric_deltas(
        cold_metrics_before,
        cold_metrics_after,
        "mistralrs_cuda_graph_events_total",
    )
    cold_eager_captures = sum(
        item["delta"]
        for item in cold_dispatch
        if item["labels"].get("component") == "target"
        and item["labels"].get("mode") == "eager"
        and item["labels"].get("reason") == CUDA_GRAPH_CACHE_POPULATION_REASON
    )
    cold_capture_successes = sum(
        item["delta"]
        for item in cold_events
        if item["labels"].get("component") == "target"
        and item["labels"].get("event") == "capture"
        and item["labels"].get("outcome") == "success"
    )
    cold_capture_failures = sum(
        item["delta"]
        for item in cold_events
        if item["labels"].get("component") == "target"
        and item["labels"].get("event") == "capture"
        and item["labels"].get("outcome") == "failure"
    )
    cold_graph_evidence = {
        "passed": (
            cold_eager_captures >= MIN_COLD_LONG_CONTEXT_GRAPH_CAPTURES
            and cold_capture_successes >= MIN_COLD_LONG_CONTEXT_GRAPH_CAPTURES
            and cold_capture_failures == 0
        ),
        "minimum_captures": MIN_COLD_LONG_CONTEXT_GRAPH_CAPTURES,
        "eager_cache_population_dispatches": cold_eager_captures,
        "successful_captures": cold_capture_successes,
        "failed_captures": cold_capture_failures,
        "dispatch": cold_dispatch,
        "events": cold_events,
    }
    await writer.emit("cold_long_context_graph_evidence", **cold_graph_evidence)
    _, stabilization_summary = await run_batch(
        client,
        specs,
        1,
        writer,
        "long-correctness-c1-stabilize",
        keep_output=False,
    )
    baseline_phase = "long-correctness-c1-baseline"
    baseline, baseline_summary = await run_batch(
        client,
        specs,
        1,
        writer,
        baseline_phase,
        keep_output=True,
    )
    summaries = [cold_summary, stabilization_summary, baseline_summary]
    exact_replays: list[dict[str, Any]] = []
    exact_replay_cohorts: list[dict[str, Any]] = []
    ordering_comparisons: list[dict[str, Any]] = []
    cross_phase_results: list[tuple[str, list[RequestResult]]] = []
    semantic_checks: list[dict[str, Any]] = []
    semantic_by_phase: dict[str, bool] = {}
    cold_semantic_checks = []
    for result in cold:
        valid, detail = validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )
        check = {"phase": cold_phase, "valid": valid, **detail}
        cold_semantic_checks.append(check)
        semantic_checks.append(check)
    semantic_by_phase[cold_phase] = bool(cold_semantic_checks) and all(
        item["valid"] for item in cold_semantic_checks
    )
    baseline_semantic_checks = []
    for result in baseline:
        valid, detail = validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )
        check = {"phase": baseline_phase, "valid": valid, **detail}
        baseline_semantic_checks.append(check)
        semantic_checks.append(check)
    semantic_by_phase[baseline_phase] = bool(baseline_semantic_checks) and all(
        item["valid"] for item in baseline_semantic_checks
    )
    c1_exact_specs, c1_exact_cohort = fixed_seed_exact_replay_cohort(specs, 1)
    exact_replay_cohorts.append(c1_exact_cohort)
    await writer.emit("long_correctness_exact_replay_cohort", **c1_exact_cohort)
    cold_exact = exact_output_diagnostics(
        cold,
        baseline,
        cold_phase,
        baseline_phase,
        specs,
    )
    cold_vs_canonical = {
        "passed": (
            cold_exact["passed"]
            and semantic_by_phase[cold_phase]
            and semantic_by_phase[baseline_phase]
        ),
        "exact_diagnostics": cold_exact,
        "exact_diagnostics_gated": True,
        "cold_semantic_passed": semantic_by_phase[cold_phase],
        "canonical_semantic_passed": semantic_by_phase[baseline_phase],
    }
    await writer.emit("cold_long_context_comparison", **cold_vs_canonical)

    replay_phase = "long-correctness-c1-replay"
    replay, replay_summary = await run_batch(
        client,
        specs,
        1,
        writer,
        replay_phase,
        keep_output=True,
    )
    summaries.append(replay_summary)
    replay_diagnostics = exact_output_diagnostics(
        baseline,
        replay,
        baseline_phase,
        replay_phase,
        c1_exact_specs,
    )
    replay_semantic_checks = []
    for result in replay:
        valid, detail = validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )
        check = {"phase": replay_phase, "valid": valid, **detail}
        replay_semantic_checks.append(check)
        semantic_checks.append(check)
    semantic_by_phase[replay_phase] = bool(replay_semantic_checks) and all(
        item["valid"] for item in replay_semantic_checks
    )
    replay_comparison = {
        "concurrency": 1,
        "cohort": c1_exact_cohort,
        **fixed_seed_comparison_evidence(
            replay_diagnostics,
            compare_samples(
                replay,
                baseline,
                client.tokenizer,
                args.long_correctness_stat_max_ks,
                args.long_correctness_stat_max_js,
            ),
            semantic_by_phase[baseline_phase] and semantic_by_phase[replay_phase],
        ),
    }
    replay_comparison["passed"] = (
        replay_comparison["passed"] and c1_exact_cohort["complete"]
    )
    exact_replays.append(replay_comparison)
    await writer.emit("exact_replay_comparison", **replay_comparison)

    for concurrency in args.long_correctness_concurrencies:
        exact_specs, exact_cohort = fixed_seed_exact_replay_cohort(
            specs,
            concurrency,
        )
        exact_replay_cohorts.append(exact_cohort)
        await writer.emit(
            "long_correctness_exact_replay_cohort",
            **exact_cohort,
        )
        _, stabilization_summary = await run_batch(
            client,
            specs,
            concurrency,
            writer,
            f"long-correctness-c{concurrency}-stabilize",
            keep_output=False,
        )
        summaries.append(stabilization_summary)
        normal_results: list[RequestResult] | None = None
        normal_phase = ""
        for order in ("normal", "normal-replay", "reverse"):
            ordered = correctness_order_specs(specs, order == "reverse")
            phase = f"long-correctness-c{concurrency}-{order}"
            results, summary = await run_batch(
                client, ordered, concurrency, writer, phase, keep_output=True
            )
            summaries.append(summary)
            phase_semantic_checks = []
            for result in results:
                valid, detail = validate_retrieval_result(
                    result, client.tokenizer, args.max_repeated_ngram_ratio
                )
                check = {"phase": phase, "valid": valid, **detail}
                phase_semantic_checks.append(check)
                semantic_checks.append(check)
            semantic_by_phase[phase] = bool(phase_semantic_checks) and all(
                item["valid"] for item in phase_semantic_checks
            )
            if order == "normal":
                normal_results = results
                normal_phase = phase
                cross_phase_results.append((phase, results))
            elif order == "normal-replay":
                if normal_results is None:
                    raise RuntimeError("normal long-context phase must precede its exact replay")
                full_cohort_statistical = compare_samples(
                    results,
                    normal_results,
                    client.tokenizer,
                    args.long_correctness_stat_max_ks,
                    args.long_correctness_stat_max_js,
                )
                full_cohort_semantic = (
                    semantic_by_phase[normal_phase] and semantic_by_phase[phase]
                )
                if exact_cohort["normal_phase_reusable"]:
                    exact_baseline = normal_results
                    exact_results = results
                    exact_baseline_phase = normal_phase
                    exact_replay_phase = phase
                else:
                    exact_baseline_phase = (
                        f"long-correctness-c{concurrency}-exact-baseline"
                    )
                    exact_baseline, exact_baseline_summary = await run_batch(
                        client,
                        exact_specs,
                        concurrency,
                        writer,
                        exact_baseline_phase,
                        keep_output=True,
                    )
                    summaries.append(exact_baseline_summary)
                    exact_replay_phase = (
                        f"long-correctness-c{concurrency}-exact-replay"
                    )
                    exact_results, exact_replay_summary = await run_batch(
                        client,
                        exact_specs,
                        concurrency,
                        writer,
                        exact_replay_phase,
                        keep_output=True,
                    )
                    summaries.append(exact_replay_summary)
                    for exact_phase, exact_phase_results in (
                        (exact_baseline_phase, exact_baseline),
                        (exact_replay_phase, exact_results),
                    ):
                        exact_semantic_checks = []
                        for result in exact_phase_results:
                            valid, detail = validate_retrieval_result(
                                result,
                                client.tokenizer,
                                args.max_repeated_ngram_ratio,
                            )
                            check = {
                                "phase": exact_phase,
                                "valid": valid,
                                **detail,
                            }
                            exact_semantic_checks.append(check)
                            semantic_checks.append(check)
                        semantic_by_phase[exact_phase] = bool(
                            exact_semantic_checks
                        ) and all(item["valid"] for item in exact_semantic_checks)
                replay_diagnostics = exact_output_diagnostics(
                    exact_baseline,
                    exact_results,
                    exact_baseline_phase,
                    exact_replay_phase,
                    exact_specs,
                )
                exact_statistical = compare_samples(
                    exact_results,
                    exact_baseline,
                    client.tokenizer,
                    args.long_correctness_stat_max_ks,
                    args.long_correctness_stat_max_js,
                )
                exact_semantic = (
                    semantic_by_phase[exact_baseline_phase]
                    and semantic_by_phase[exact_replay_phase]
                )
                exact_evidence = fixed_seed_comparison_evidence(
                    replay_diagnostics,
                    exact_statistical,
                    exact_semantic,
                )
                replay_comparison = {
                    "concurrency": concurrency,
                    "cohort": exact_cohort,
                    "full_cohort_statistical_comparison": full_cohort_statistical,
                    "full_cohort_semantic_passed": full_cohort_semantic,
                    **exact_evidence,
                }
                replay_comparison["passed"] = (
                    replay_comparison["passed"]
                    and exact_cohort["complete"]
                    and full_cohort_statistical["passed"]
                    and full_cohort_semantic
                )
                exact_replays.append(replay_comparison)
                await writer.emit("exact_replay_comparison", **replay_comparison)
            else:
                if normal_results is None:
                    raise RuntimeError(
                        "normal long-context phase must precede reverse ordering"
                    )
                exact = exact_output_diagnostics(
                    normal_results,
                    results,
                    normal_phase,
                    phase,
                    specs,
                )
                statistical = compare_samples(
                    results,
                    normal_results,
                    client.tokenizer,
                    args.long_correctness_stat_max_ks,
                    args.long_correctness_stat_max_js,
                )
                ordering_diagnostics = {
                    "concurrency": concurrency,
                    **fixed_seed_comparison_evidence(
                        exact,
                        statistical,
                        semantic_by_phase[normal_phase] and semantic_by_phase[phase],
                        exact_gated=False,
                    ),
                }
                ordering_comparisons.append(ordering_diagnostics)
                await writer.emit(
                    "exact_ordering_comparison", **ordering_diagnostics
                )

    cross_phase_comparisons = []
    for phase, results in cross_phase_results:
        exact = exact_output_diagnostics(
            baseline, results, baseline_phase, phase, specs
        )
        statistical = compare_samples(
            results,
            baseline,
            client.tokenizer,
            args.long_correctness_stat_max_ks,
            args.long_correctness_stat_max_js,
        )
        semantic_passed = semantic_by_phase[baseline_phase] and semantic_by_phase[phase]
        comparison = {
            "phase": phase,
            **fixed_seed_comparison_evidence(
                exact,
                statistical,
                semantic_passed,
                exact_gated=False,
            ),
        }
        cross_phase_comparisons.append(comparison)
        await writer.emit("cross_shape_comparison", **comparison)

    passed = (
        all(summary["errors"] == 0 for summary in summaries)
        and cold_graph_evidence["passed"]
        and cold_vs_canonical["passed"]
        and all(item["passed"] for item in exact_replays)
        and all(item["passed"] for item in ordering_comparisons)
        and all(item["passed"] for item in cross_phase_comparisons)
        and all(item["valid"] for item in semantic_checks)
    )
    result = {
        "passed": passed,
        "lengths": list(args.long_correctness_context_lengths),
        "concurrencies": list(args.long_correctness_concurrencies),
        "summaries": summaries,
        "exact_replay_cohorts": exact_replay_cohorts,
        "cold_graph_evidence": cold_graph_evidence,
        "cold_vs_canonical": cold_vs_canonical,
        "exact_replays": exact_replays,
        "ordering_comparisons": ordering_comparisons,
        "cross_phase_comparisons": cross_phase_comparisons,
        "semantic_checks": semantic_checks,
    }
    await writer.emit("long_context_correctness_summary", **result)
    return result


async def safe_metrics(client: SoakClient, writer: JsonlWriter, phase: str) -> dict[str, float]:
    try:
        snapshot = await client.metrics()
        await writer.emit("metrics_snapshot", phase=phase, metrics=snapshot)
        return snapshot
    except Exception as exc:
        await writer.emit(
            "metrics_error", phase=phase, error=f"{type(exc).__name__}: {exc}"
        )
        return {}


CLEANUP_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
    "mistralrs_recurrent_state_slots_used",
    KV_CACHE_ACTIVE_GAUGE,
    "http_requests_in_flight",
)
RESIDENT_TRANSIENT_CLEANUP_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
    "mistralrs_recurrent_state_slots_used",
    KV_CACHE_ACTIVE_GAUGE,
    "http_requests_in_flight",
)
OPTIONAL_CLEANUP_GAUGES = (
    DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
    CUDA_MEMORY_PENDING_GAUGE,
)
DFLASH_ABORT_CLEANUP_GAUGES = (
    DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
    DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
)
CHURN_REQUIRED_GAUGES = (
    "mistralrs_sequences_capacity",
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
)


def cleanup_evidence(
    baseline: dict[str, float],
    observed: dict[str, float],
    baseline_values: dict[str, float | None],
    observed_values: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "baseline": baseline_values,
        "observed": observed_values,
        "kv_cache_blocks": {
            "active": {
                "baseline": metric_total(baseline, KV_CACHE_ACTIVE_GAUGE),
                "observed": metric_total(observed, KV_CACHE_ACTIVE_GAUGE),
                "gated": KV_CACHE_ACTIVE_GAUGE in baseline_values,
            },
            "prefix_cached": {
                "baseline": metric_total(baseline, KV_CACHE_PREFIX_CACHED_GAUGE),
                "observed": metric_total(observed, KV_CACHE_PREFIX_CACHED_GAUGE),
                "gated": False,
            },
        },
        "windowed_kv_slots": {
            pool: {
                "baseline": metric_total(baseline, gauge),
                "observed": metric_total(observed, gauge),
                "gated": gauge in baseline_values,
            }
            for pool, gauge in (
                ("live", DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE),
                (
                    "checkpoint",
                    DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
                ),
            )
        },
    }


def dflash_checkpoint_retention_evidence(
    snapshots: dict[str, dict[str, float]],
    distinct_successful_prefixes: int,
) -> dict[str, Any]:
    required_stages = (
        "before_cold",
        "after_cold",
        "after_hit",
        "after_pressure",
        "after_retry",
        "quiescent",
    )
    stages = {
        stage: {
            "used": metric_total(
                snapshots.get(stage, {}),
                DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
            ),
            "total": metric_total(
                snapshots.get(stage, {}),
                DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
            ),
        }
        for stage in required_stages
    }
    available = any(
        value is not None for stage in stages.values() for value in stage.values()
    )
    instrumentation_complete = all(
        stage["used"] is not None and stage["total"] is not None
        for stage in stages.values()
    )
    baseline_total = stages["before_cold"]["total"]
    stable_total = (
        instrumentation_complete
        and baseline_total is not None
        and baseline_total > 0
        and all(stage["total"] == baseline_total for stage in stages.values())
    )
    within_physical_capacity = instrumentation_complete and all(
        0 <= stage["used"] <= stage["total"] for stage in stages.values()
    )
    retained_capacity = (
        max(0.0, baseline_total - 1.0) if baseline_total is not None else None
    )
    quiescent_used = stages["quiescent"]["used"]
    quiescent_within_retained_capacity = (
        retained_capacity is not None
        and quiescent_used is not None
        and quiescent_used <= retained_capacity
    )
    after_cold_used = stages["after_cold"]["used"]
    after_hit_used = stages["after_hit"]["used"]
    hit_no_growth = (
        after_cold_used is not None
        and after_hit_used is not None
        and after_hit_used <= after_cold_used
    )
    after_pressure_used = stages["after_pressure"]["used"]
    after_retry_used = stages["after_retry"]["used"]
    same_key_retry_no_growth = (
        after_pressure_used is not None
        and after_retry_used is not None
        and after_retry_used <= after_pressure_used
    )
    baseline_used = stages["before_cold"]["used"]
    observed_growth = (
        quiescent_used - baseline_used
        if quiescent_used is not None and baseline_used is not None
        else None
    )
    total_growth_bounded = (
        observed_growth is not None
        and observed_growth <= distinct_successful_prefixes
    )
    checks = {
        "stable_total": {
            "passed": stable_total,
            "expected": baseline_total,
        },
        "within_physical_capacity": {
            "passed": within_physical_capacity,
        },
        "quiescent_within_retained_capacity": {
            "passed": quiescent_within_retained_capacity,
            "used": quiescent_used,
            "maximum": retained_capacity,
        },
        "hit_no_growth": {
            "passed": hit_no_growth,
            "before": after_cold_used,
            "after": after_hit_used,
        },
        "same_key_retry_no_growth": {
            "passed": same_key_retry_no_growth,
            "before": after_pressure_used,
            "after": after_retry_used,
        },
        "total_growth_bounded": {
            "passed": total_growth_bounded,
            "baseline": baseline_used,
            "quiescent": quiescent_used,
            "observed_growth": observed_growth,
            "maximum_growth": distinct_successful_prefixes,
        },
    }
    return {
        "passed": not available
        or (instrumentation_complete and all(check["passed"] for check in checks.values())),
        "available": available,
        "instrumentation_complete": instrumentation_complete,
        "used_gauge": DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
        "total_gauge": DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
        "staging_reserve_slots": 1,
        "retained_capacity": retained_capacity,
        "distinct_successful_prefixes": distinct_successful_prefixes,
        "stages": stages,
        "checks": checks,
    }


def prefix_cached_ownership_evidence(
    cancel_delta: float | None,
    retry_delta: float | None,
    timeout_delta: float | None,
) -> dict[str, Any]:
    stages = {
        "cancellation_cleanup": {
            "delta": cancel_delta,
            "expectation": "non_increasing",
            "passed": cancel_delta is not None and cancel_delta <= 0,
        },
        "successful_retry": {
            "delta": retry_delta,
            "expectation": "increasing",
            "passed": retry_delta is not None and retry_delta > 0,
        },
        "timeout_cleanup": {
            "delta": timeout_delta,
            "expectation": "non_increasing",
            "passed": timeout_delta is not None and timeout_delta <= 0,
        },
    }
    instrumentation_complete = all(item["delta"] is not None for item in stages.values())
    return {
        "passed": instrumentation_complete and all(item["passed"] for item in stages.values()),
        "instrumentation_complete": instrumentation_complete,
        "metric": KV_CACHE_PREFIX_CACHED_GAUGE,
        "stages": stages,
    }


def churn_capacity_evidence(
    queue_samples: Sequence[dict[str, float | None]],
    width: int,
    max_sequences: int,
    server_sequence_capacity: float | None,
    cleanup_ok: bool,
    errors: int,
) -> dict[str, Any]:
    instrumentation_complete = bool(queue_samples) and all(
        sample.get(name) is not None
        for sample in queue_samples
        for name in CHURN_REQUIRED_GAUGES
    )
    running_values = [
        value
        for sample in queue_samples
        if (value := sample.get("mistralrs_sequences_running")) is not None
    ]
    waiting_values = [
        value
        for sample in queue_samples
        if (value := sample.get("mistralrs_sequences_waiting")) is not None
    ]
    pending_values = [
        value
        for sample in queue_samples
        if (value := sample.get("mistralrs_requests_pending_admission")) is not None
    ]
    peak_running = max(running_values, default=None)
    peak_waiting = max(waiting_values, default=None)
    peak_pending = max(pending_values, default=None)
    near_capacity_threshold = min(
        width,
        max_sequences - CHURN_NEAR_CAPACITY_HEADROOM,
    )
    near_capacity_samples = sum(
        value >= near_capacity_threshold for value in running_values
    )
    near_capacity_consecutive_samples = 0
    current_near_capacity_samples = 0
    for value in running_values:
        if value >= near_capacity_threshold:
            current_near_capacity_samples += 1
            near_capacity_consecutive_samples = max(
                near_capacity_consecutive_samples,
                current_near_capacity_samples,
            )
        else:
            current_near_capacity_samples = 0
    near_capacity_sample_fraction = ratio(
        near_capacity_samples,
        len(running_values),
    )
    near_capacity_sustained = (
        near_capacity_threshold > 0
        and near_capacity_consecutive_samples >= MIN_CHURN_NEAR_CAPACITY_SAMPLES
    )
    queue_required = width > max_sequences
    queue_observed = bool(
        (peak_waiting is not None and peak_waiting > 0)
        or (peak_pending is not None and peak_pending > 0)
    )
    capacity_matches = server_sequence_capacity == max_sequences
    capacity_respected = peak_running is not None and peak_running <= max_sequences
    return {
        "passed": (
            instrumentation_complete
            and capacity_matches
            and capacity_respected
            and near_capacity_sustained
            and (not queue_required or queue_observed)
            and cleanup_ok
            and errors == 0
        ),
        "width": width,
        "configured_max_seqs": max_sequences,
        "server_sequence_capacity": server_sequence_capacity,
        "capacity_matches": capacity_matches,
        "instrumentation_complete": instrumentation_complete,
        "queue_required": queue_required,
        "queue_observed": queue_observed,
        "peak_running": peak_running,
        "peak_waiting": peak_waiting,
        "peak_pending_admission": peak_pending,
        "capacity_respected": capacity_respected,
        "near_capacity_threshold": near_capacity_threshold,
        "near_capacity_samples": near_capacity_samples,
        "near_capacity_consecutive_samples": near_capacity_consecutive_samples,
        "near_capacity_total_samples": len(running_values),
        "near_capacity_sample_fraction": near_capacity_sample_fraction,
        "minimum_near_capacity_samples": MIN_CHURN_NEAR_CAPACITY_SAMPLES,
        "minimum_near_capacity_sample_fraction": (
            MIN_CHURN_NEAR_CAPACITY_SAMPLE_FRACTION
        ),
        "near_capacity_sample_fraction_gated": False,
        "near_capacity_sustained": near_capacity_sustained,
        "cleanup_ok": cleanup_ok,
        "errors": errors,
    }


async def poll_for_cleanup(
    client: SoakClient,
    writer: JsonlWriter,
    baseline: dict[str, float],
    timeout_seconds: float,
    poll_seconds: float,
    phase: str,
    gauges: Sequence[str] = CLEANUP_GAUGES,
    optional_gauges: Sequence[str] = OPTIONAL_CLEANUP_GAUGES,
) -> tuple[bool, dict[str, float], dict[str, Any]]:
    gauges = tuple(
        dict.fromkeys(
            (
                *gauges,
                *(
                    gauge
                    for gauge in optional_gauges
                    if metric_total(baseline, gauge) is not None
                ),
            )
        )
    )
    baseline_values = {name: metric_total(baseline, name) for name in gauges}
    missing_baseline = [name for name, value in baseline_values.items() if value is None]
    if missing_baseline:
        detail = cleanup_evidence(baseline, {}, baseline_values, {})
        detail["missing_baseline_metrics"] = missing_baseline
        return False, {}, detail
    deadline = time.perf_counter() + timeout_seconds
    last_snapshot: dict[str, float] = {}
    last_values: dict[str, float | None] = {}
    while time.perf_counter() < deadline:
        last_snapshot = await safe_metrics(client, writer, phase)
        last_values = {name: metric_total(last_snapshot, name) for name in gauges}
        if all(
            value is not None and value <= baseline_values[name]
            for name, value in last_values.items()
        ):
            return True, last_snapshot, cleanup_evidence(
                baseline,
                last_snapshot,
                baseline_values,
                last_values,
            )
        await asyncio.sleep(poll_seconds)
    detail = cleanup_evidence(
        baseline,
        last_snapshot,
        baseline_values,
        last_values,
    )
    detail["timeout_seconds"] = timeout_seconds
    return False, last_snapshot, detail


async def run_prefix_pressure_workflow(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []

    async def record_quality(phase: str, results: Sequence[RequestResult]) -> None:
        for result in results:
            valid, detail = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {"phase": phase, "valid": valid, **detail}
            quality_checks.append(check)
            await writer.emit("prefix_pressure_quality", **check)

    prefix_prompt = exact_context(
        client,
        args.prefix_context_tokens,
        f"{args.prefix_pressure_namespace}-target",
    )
    prefix_seed = args.seed + 60_000
    pre_prefix_metrics = await safe_metrics(client, writer, "prefix-before-cold")
    sequence_capacity = metric_total(
        pre_prefix_metrics,
        "mistralrs_sequences_capacity",
    )
    capacity_matches = sequence_capacity == args.max_seqs
    prefix_spec = RequestSpec(
        case_id="prefix-cold",
        seed=prefix_seed,
        max_tokens=args.max_tokens,
        prompt=prefix_prompt,
        context_tokens=args.prefix_context_tokens,
        tags={"scenario": "prefix", "stage": "cold"},
    )
    cold, cold_summary = await run_batch(
        client,
        [prefix_spec],
        1,
        writer,
        "prefix-cold",
        keep_output=True,
    )
    summaries.append(cold_summary)
    await record_quality("prefix-cold", cold)
    post_cold_metrics = await safe_metrics(client, writer, "prefix-after-cold")
    hit_spec = RequestSpec(
        case_id="prefix-hit",
        seed=prefix_seed,
        max_tokens=args.max_tokens,
        prompt=prefix_prompt,
        context_tokens=args.prefix_context_tokens,
        tags={"scenario": "prefix", "stage": "hit"},
    )
    hit, hit_summary = await run_batch(
        client,
        [hit_spec],
        1,
        writer,
        "prefix-hit",
        keep_output=True,
    )
    summaries.append(hit_summary)
    await record_quality("prefix-hit", hit)
    post_hit_metrics = await safe_metrics(client, writer, "prefix-after-hit")
    hit_evidence = prefix_cache_evidence(
        post_cold_metrics,
        post_hit_metrics,
        [args.prefix_context_tokens],
        args.min_prefix_reuse_fraction,
        args.kv_block_size_tokens,
        args.speculative_prefix_replay_tokens,
    )
    cold_reused_tokens = metric_delta(
        pre_prefix_metrics,
        post_cold_metrics,
        "mistralrs_prefix_cache_tokens_reused_total",
    )
    cold_reuse_fraction = ratio(cold_reused_tokens, args.prefix_context_tokens)
    cold_miss_observed = (
        cold_reuse_fraction is not None
        and cold_reuse_fraction <= 1.0 - args.min_prefix_reuse_fraction
    )

    total_kv_blocks = metric_total(
        post_hit_metrics,
        "mistralrs_kv_cache_blocks_total",
    )
    if total_kv_blocks is None or total_kv_blocks <= 0:
        raise RuntimeError("prefix pressure requires mistralrs_kv_cache_blocks_total")
    pressure_blocks_per_request = math.ceil(
        args.prefix_pressure_context_tokens / args.kv_block_size_tokens
    )
    capacity_pressure_entries = math.ceil(
        total_kv_blocks
        * args.prefix_pressure_capacity_fraction
        / pressure_blocks_per_request
    )
    pressure_entries = max(args.prefix_pressure_entries, capacity_pressure_entries)
    if pressure_entries > args.prefix_pressure_max_entries:
        raise RuntimeError(
            "capacity-derived prefix pressure requires "
            f"{pressure_entries} entries, above --prefix-pressure-max-entries "
            f"{args.prefix_pressure_max_entries}; increase the pressure context length or cap"
        )
    pressure_plan = prefix_pressure_plan(
        post_hit_metrics,
        PrefixPressureConfig(
            entries=pressure_entries,
            max_sequences=args.max_seqs,
            context_tokens=args.prefix_pressure_context_tokens,
            max_completion_tokens=args.prefix_pressure_max_tokens,
            block_size_tokens=args.kv_block_size_tokens,
            kv_headroom_fraction=args.prefix_pressure_kv_headroom_fraction,
        ),
    )
    await writer.emit("prefix_pressure_plan", **pressure_plan)
    eviction_specs = [
        RequestSpec(
            case_id=f"prefix-pressure-{index:04d}",
            seed=args.seed + 61_000 + index,
            max_tokens=args.prefix_pressure_max_tokens,
            prompt=exact_context(
                client,
                args.prefix_pressure_context_tokens,
                f"{args.prefix_pressure_namespace}-pressure-{index}",
            ),
            context_tokens=args.prefix_pressure_context_tokens,
            tags={"scenario": "prefix", "stage": "pressure"},
            extra={"ignore_eos": True},
        )
        for index in range(pressure_entries)
    ]
    pressure_results, pressure_summary = await run_batch(
        client,
        eviction_specs,
        int(pressure_plan["concurrency"]),
        writer,
        "prefix-pressure",
        keep_output=False,
    )
    summaries.append(pressure_summary)
    await record_quality("prefix-pressure", pressure_results)
    post_pressure_metrics = await safe_metrics(
        client,
        writer,
        "prefix-after-pressure-load",
    )
    pressure_evictions = metric_delta(
        post_hit_metrics,
        post_pressure_metrics,
        "mistralrs_prefix_cache_evictions_total",
    )
    pressure_reached_eviction = pressure_evictions is not None and pressure_evictions > 0
    after_spec = RequestSpec(
        case_id="prefix-after-pressure",
        seed=prefix_seed,
        max_tokens=args.max_tokens,
        prompt=prefix_prompt,
        context_tokens=args.prefix_context_tokens,
        tags={"scenario": "prefix", "stage": "after_pressure"},
    )
    after, after_summary = await run_batch(
        client,
        [after_spec],
        1,
        writer,
        "prefix-after-pressure",
        keep_output=True,
    )
    summaries.append(after_summary)
    await record_quality("prefix-after-pressure", after)
    post_after_metrics = await safe_metrics(
        client,
        writer,
        "prefix-after-evicted-retry",
    )
    after_reused_tokens = metric_delta(
        post_pressure_metrics,
        post_after_metrics,
        "mistralrs_prefix_cache_tokens_reused_total",
    )
    after_reuse_fraction = ratio(after_reused_tokens, args.prefix_context_tokens)
    target_was_evicted = (
        after_reuse_fraction is not None
        and after_reuse_fraction <= 1.0 - args.min_prefix_reuse_fraction
    )
    prefix_correct = (
        cold[0].output_transcript
        == hit[0].output_transcript
        == after[0].output_transcript
    )
    cleanup_ok, cleanup_metrics, cleanup = await poll_for_cleanup(
        client,
        writer,
        pre_prefix_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "prefix-pressure-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    distinct_successful_prefixes = sum(
        result.ok for result in (*cold, *pressure_results)
    )
    checkpoint_retention = dflash_checkpoint_retention_evidence(
        {
            "before_cold": pre_prefix_metrics,
            "after_cold": post_cold_metrics,
            "after_hit": post_hit_metrics,
            "after_pressure": post_pressure_metrics,
            "after_retry": post_after_metrics,
            "quiescent": cleanup_metrics,
        },
        distinct_successful_prefixes,
    )
    memory_pressure = cuda_memory_pressure_evidence(
        post_hit_metrics,
        cleanup_metrics,
        require_instrumentation=True,
    )
    await writer.emit("cuda_memory_pressure_summary", **memory_pressure)
    prefix_cache_stages = {
        "cold_miss_observed": cold_miss_observed,
        "cold_reused_tokens": cold_reused_tokens,
        "cold_reuse_fraction": cold_reuse_fraction,
        "hit": hit_evidence,
        "kv_blocks_total": total_kv_blocks,
        "kv_block_size_tokens": args.kv_block_size_tokens,
        "pressure_context_tokens": args.prefix_pressure_context_tokens,
        "pressure_capacity_fraction": args.prefix_pressure_capacity_fraction,
        "pressure_entries_configured_minimum": args.prefix_pressure_entries,
        "pressure_entries_capacity_derived": capacity_pressure_entries,
        "pressure_entries_executed": pressure_entries,
        "pressure_plan": pressure_plan,
        "pressure_evictions": pressure_evictions,
        "pressure_reached_eviction": pressure_reached_eviction,
        "target_retry_reused_tokens": after_reused_tokens,
        "target_retry_reuse_fraction": after_reuse_fraction,
        "target_was_evicted": target_was_evicted,
        "outputs_equal": prefix_correct,
        "dflash_checkpoint_retention": checkpoint_retention,
    }
    passed = (
        capacity_matches
        and all(summary["errors"] == 0 for summary in summaries)
        and all(item["valid"] for item in quality_checks)
        and prefix_correct
        and cold_miss_observed
        and hit_evidence["passed"]
        and pressure_reached_eviction
        and target_was_evicted
        and cleanup_ok
        and checkpoint_retention["passed"]
        and memory_pressure["passed"]
    )
    prefix_cache_stages["passed"] = passed
    result = {
        "mode": "prefix-pressure",
        "passed": passed,
        "summaries": summaries,
        "quality_checks": quality_checks,
        "configured_max_seqs": args.max_seqs,
        "server_sequence_capacity": sequence_capacity,
        "capacity_matches": capacity_matches,
        "prefix_cache": prefix_cache_stages,
        "cleanup_ok": cleanup_ok,
        "cleanup": cleanup,
        "memory_pressure": memory_pressure,
        "metric_deltas": selected_metric_deltas(
            pre_prefix_metrics,
            cleanup_metrics,
        ),
    }
    await writer.emit("prefix_cache_stage_summary", **result)
    return result


async def prefix_pressure_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    await calibrate_prompt_profiles(client, writer, (CONTEXT_PROMPT_PROFILE,))
    return await run_prefix_pressure_workflow(args, client, writer)


async def run_adversarial_long_resident_performance(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
    graph_reference: dict[str, float],
) -> dict[str, Any]:
    production_policy = SamplingPolicy()
    sampling = production_sampling_policy_evidence(
        PART1_PRODUCTION_SAMPLING_POLICY,
        production_policy,
    )
    request_extra = {**production_policy.payload(), "ignore_eos": True}
    specs, warm_specs = adversarial_long_resident_cohorts(
        client,
        args.seed + 91_000,
        args.long_resident_max_tokens,
        request_extra,
    )
    cohort = request_cohort_uniqueness_evidence(specs)
    warm_before = await safe_metrics(client, writer, "long-resident-warm-start")
    warm_results, warm_summary = await run_batch(
        client,
        warm_specs,
        ADVERSARIAL_LONG_RESIDENT_REQUESTS,
        writer,
        "adversarial-long-resident-warm",
        keep_output=False,
    )
    warm_cleanup_ok, warm_after, warm_cleanup = await poll_for_cleanup(
        client,
        writer,
        warm_before,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-long-resident-warm-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    warm_prompt_exact = (
        len(warm_results) == len(warm_specs)
        and all(
            result.prompt_tokens == ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS
            for result in warm_results
        )
    )
    warm_completion = full_length_completion_evidence(warm_results, 1)
    runtime_before = warm_after
    phase_evidence = []
    measured_summaries = []
    all_quality = []
    runtime_after = warm_after
    for concurrency in sorted(args.min_long_resident_decode_tok_s_by_concurrency):
        stabilization_phase = f"adversarial-long-resident-c{concurrency}-stabilize"
        stabilization_before = runtime_after
        stabilization_results, stabilization_summary = await run_batch(
            client,
            specs,
            concurrency,
            writer,
            stabilization_phase,
            keep_output=False,
        )
        stabilization_cleanup_ok, measurement_before, stabilization_cleanup = (
            await poll_for_cleanup(
                client,
                writer,
                stabilization_before,
                args.cleanup_timeout_seconds,
                args.cleanup_poll_seconds,
                f"{stabilization_phase}-cleanup",
                RESIDENT_TRANSIENT_CLEANUP_GAUGES,
            )
        )
        stabilization_completion = full_length_completion_evidence(
            stabilization_results,
            args.long_resident_max_tokens,
        )
        phase = f"adversarial-long-resident-c{concurrency}"
        results, summary = await run_batch(
            client,
            specs,
            concurrency,
            writer,
            phase,
            keep_output=True,
        )
        cleanup_ok, runtime_after, cleanup = await poll_for_cleanup(
            client,
            writer,
            measurement_before,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            f"{phase}-cleanup",
            RESIDENT_TRANSIENT_CLEANUP_GAUGES,
        )
        completion = full_length_completion_evidence(
            results,
            args.long_resident_max_tokens,
        )
        quality = []
        for result, completion_check in zip(results, completion["requests"]):
            valid, detail = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {
                "phase": phase,
                "valid": (
                    valid
                    and result.prompt_tokens
                    == ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS
                    and completion_check["passed"]
                ),
                "prompt_tokens_match": (
                    result.prompt_tokens
                    == ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS
                ),
                "full_length_completion": completion_check["passed"],
                **detail,
            }
            quality.append(check)
            all_quality.append(check)
            await writer.emit("adversarial_long_resident_quality", **check)
        prefix = prefix_cache_evidence(
            measurement_before,
            runtime_after,
            [ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS] * len(specs),
            args.min_prefix_reuse_fraction,
            args.kv_block_size_tokens,
            args.speculative_prefix_replay_tokens,
        )
        detail = {
            "concurrency": concurrency,
            "passed": (
                stabilization_summary["errors"] == 0
                and stabilization_cleanup_ok
                and stabilization_completion["passed"]
                and summary["errors"] == 0
                and cleanup_ok
                and completion["passed"]
                and prefix["passed"]
                and all(item["valid"] for item in quality)
            ),
            "stabilization": stabilization_summary,
            "stabilization_cleanup_ok": stabilization_cleanup_ok,
            "stabilization_cleanup": stabilization_cleanup,
            "stabilization_completion": stabilization_completion,
            "measurement": summary,
            "cleanup_ok": cleanup_ok,
            "cleanup": cleanup,
            "completion": completion,
            "prefix_cache": prefix,
            "quality": quality,
        }
        phase_evidence.append(detail)
        measured_summaries.append(summary)
        await writer.emit("adversarial_long_resident_phase", **detail)
    throughput = exact_throughput_threshold_evidence(
        measured_summaries,
        args.min_long_resident_decode_tok_s_by_concurrency,
        "decode_tok_s_active",
    )
    mtp = configured_speculative_evidence(
        runtime_before,
        runtime_after,
        args,
        args.require_mtp,
    )
    graph = cuda_graph_evidence(
        runtime_before,
        runtime_after,
        graph_reference,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    memory = cuda_memory_pressure_evidence(
        runtime_before,
        runtime_after,
        require_instrumentation=True,
    )
    result = {
        "passed": (
            sampling["passed"]
            and cohort["passed"]
            and warm_summary["errors"] == 0
            and warm_cleanup_ok
            and warm_prompt_exact
            and warm_completion["passed"]
            and all(item["passed"] for item in phase_evidence)
            and all(item["valid"] for item in all_quality)
            and throughput["passed"]
            and mtp["passed"]
            and graph["passed"]
            and memory["passed"]
        ),
        "context_tokens": ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS,
        "requests_per_phase": ADVERSARIAL_LONG_RESIDENT_REQUESTS,
        "max_tokens": args.long_resident_max_tokens,
        "production_sampling": sampling,
        "cohort": cohort,
        "warmup": warm_summary,
        "warmup_prompt_tokens_match": warm_prompt_exact,
        "warmup_completion": warm_completion,
        "warmup_cleanup_ok": warm_cleanup_ok,
        "warmup_cleanup": warm_cleanup,
        "phases": phase_evidence,
        "throughput": throughput,
        "mtp": mtp,
        "cuda_graph": graph,
        "cuda_memory": memory,
    }
    await writer.emit("adversarial_long_resident_performance", **result)
    return result


async def adversarial_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    await calibrate_prompt_profiles(
        client,
        writer,
        (CONTEXT_PROMPT_PROFILE, RETRIEVAL_PROMPT_PROFILE),
    )
    contexts = exact_contexts(client, args.context_lengths, "adversarial")
    shortest = min(args.context_lengths)
    longest = max(args.context_lengths)
    initial_metrics = await safe_metrics(client, writer, "adversarial-start")
    summaries: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    runtime_evidence: list[dict[str, Any]] = []
    runtime_cursor = initial_metrics
    sequence_capacity = metric_total(initial_metrics, "mistralrs_sequences_capacity")
    max_seqs_capacity_ok = sequence_capacity == args.max_seqs == 16

    async def record_quality(phase: str, results: Sequence[RequestResult]) -> None:
        for result in results:
            valid, detail = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {"phase": phase, "valid": valid, **detail}
            quality_checks.append(check)
            await writer.emit("adversarial_quality", **check)

    async def record_retrieval_quality(
        phase: str, results: Sequence[RequestResult]
    ) -> None:
        for result in results:
            valid, detail = validate_retrieval_result(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {"phase": phase, "valid": valid, **detail}
            quality_checks.append(check)
            await writer.emit("adversarial_retrieval_quality", **check)

    async def record_runtime(phase: str, gated: bool) -> None:
        nonlocal runtime_cursor
        after = await safe_metrics(client, writer, f"{phase}-runtime-end")
        evidence = {
            "phase": phase,
            "gated": gated,
            "mtp": configured_speculative_evidence(
                runtime_cursor,
                after,
                args,
                gated and args.require_mtp,
            ),
            "cuda_graph": cuda_graph_evidence(
                runtime_cursor,
                after,
                initial_metrics,
                args.expected_graph_components,
                args.min_cuda_graph_replay_ratio,
            ),
            "cuda_memory": cuda_memory_pressure_evidence(
                runtime_cursor,
                after,
                require_instrumentation=gated,
            ),
            "queue_latency_histograms": queue_histogram_summaries(
                runtime_cursor,
                after,
            ),
            "metric_deltas": selected_metric_deltas(runtime_cursor, after),
        }
        evidence["passed"] = (
            not gated
            or (
                evidence["mtp"]["passed"]
                and evidence["cuda_graph"]["passed"]
                and evidence["cuda_memory"]["passed"]
            )
        )
        runtime_evidence.append(evidence)
        await writer.emit("adversarial_runtime_summary", **evidence)
        runtime_cursor = after

    long_correctness = await run_long_context_correctness(args, client, writer)
    await record_runtime("adversarial-long-correctness", True)
    long_resident_performance = await run_adversarial_long_resident_performance(
        args,
        client,
        writer,
        initial_metrics,
    )
    await record_runtime("adversarial-long-resident-performance", True)

    mixed_templates = {
        length: retrieval_spec(
            client,
            length,
            f"mixed-resident-{length}",
            args.seed + 500 + index,
            args.max_tokens,
            {"scenario": "mixed", "length": length, "role": "template"},
        )
        for index, length in enumerate(args.context_lengths)
    }
    mixed_warmup, mixed_warmup_summary = await run_batch(
        client,
        list(mixed_templates.values()),
        min(args.max_seqs, len(mixed_templates)),
        writer,
        "adversarial-mixed-context-warmup",
        keep_output=True,
    )
    summaries.append(mixed_warmup_summary)
    await record_retrieval_quality(
        "adversarial-mixed-context-warmup", mixed_warmup
    )
    mixed_specs = adversarial_mixed_specs(
        client,
        args.context_lengths,
        args.mixed_requests,
        args.seed,
        args.max_tokens,
    )
    mixed_cohort = request_cohort_uniqueness_evidence(mixed_specs)
    mixed_cohort["stabilization_measurement_specs_reused"] = True
    mixed_cohort["passed"] = (
        mixed_cohort["passed"]
        and mixed_cohort["stabilization_measurement_specs_reused"]
    )
    await writer.emit("adversarial_mixed_context_cohort", **mixed_cohort)
    mixed_throughput_summaries = []
    for concurrency in args.throughput_concurrencies:
        measured_specs = balanced_context_full_batch_specs(
            mixed_specs,
            concurrency,
        )
        stabilization_phase = f"adversarial-mixed-context-c{concurrency}-stabilize"
        stabilization_results, stabilization_summary = await run_batch(
            client,
            measured_specs,
            concurrency,
            writer,
            stabilization_phase,
            keep_output=True,
        )
        summaries.append(stabilization_summary)
        await record_retrieval_quality(
            stabilization_phase,
            stabilization_results,
        )
        await record_runtime(stabilization_phase, False)
        phase = f"adversarial-mixed-context-c{concurrency}"
        mixed_results, summary = await run_batch(
            client,
            measured_specs,
            concurrency,
            writer,
            phase,
            keep_output=True,
        )
        summaries.append(summary)
        mixed_throughput_summaries.append(summary)
        await record_retrieval_quality(phase, mixed_results)
        await record_runtime(phase, True)
    mixed_throughput = serving_throughput_evidence(
        mixed_throughput_summaries,
        args.min_output_tok_s_by_concurrency,
        args.min_scaling_efficiency,
    )
    await writer.emit("adversarial_throughput_evidence", **mixed_throughput)

    short_spec = retrieval_spec(
        client,
        ADVERSARIAL_FAIRNESS_SHORT_CONTEXT_TOKENS,
        "fairness-short-b",
        args.seed + 10_000,
        args.max_tokens,
        {"scenario": "fairness", "role": "short"},
    )
    _, stabilization_summary = await run_batch(
        client,
        [short_spec],
        1,
        writer,
        "fairness-short-stabilize",
        keep_output=False,
    )
    summaries.append(stabilization_summary)
    warmup, warmup_summary = await run_batch(
        client, [short_spec], 1, writer, "fairness-short-warmup", keep_output=True
    )
    summaries.append(warmup_summary)
    isolated, isolated_summary = await run_batch(
        client, [short_spec], 1, writer, "fairness-short-isolated", keep_output=True
    )
    summaries.append(isolated_summary)
    long_specs = {
        order: retrieval_spec(
            client,
            ADVERSARIAL_FAIRNESS_LONG_CONTEXT_TOKENS,
            f"fairness-long-a-{order}",
            args.seed + 10_001 + index,
            args.max_tokens,
            {"scenario": "fairness", "role": "long", "order": order},
        )
        for index, order in enumerate(("a_then_b", "b_then_a"))
    }
    long_warmup, long_warmup_summary = await run_batch(
        client,
        list(long_specs.values()),
        len(long_specs),
        writer,
        "fairness-long-warmup",
        keep_output=True,
    )
    summaries.append(long_warmup_summary)
    await record_retrieval_quality("fairness-long-warmup", long_warmup)
    warmup_short = warmup[0]
    isolated_short = isolated[0]
    fairness_runs: list[dict[str, Any]] = []
    concurrent_short_results: list[RequestResult] = []
    concurrent_long_results: list[RequestResult] = []
    for order in ("a_then_b", "b_then_a"):
        long_spec = long_specs[order]
        first, second = (
            (long_spec, short_spec) if order == "a_then_b" else (short_spec, long_spec)
        )
        phase = f"fairness-{order}"
        pair, pair_summary = await run_staggered_pair(
            client,
            first,
            second,
            args.fairness_stagger_seconds,
            writer,
            phase,
        )
        summaries.append(pair_summary)
        short_result = next(result for result in pair if result.case_id == short_spec.case_id)
        long_result = next(result for result in pair if result.case_id == long_spec.case_id)
        concurrent_short_results.append(short_result)
        concurrent_long_results.append(long_result)
        decode_overlap = concurrent_decode_overlap_evidence(
            long_result,
            short_result,
        )
        fairness_runs.append(
            {
                "order": order,
                "summary": pair_summary,
                "short_ttft_slowdown": ratio(
                    short_result.ttft_seconds, isolated_short.ttft_seconds
                ),
                "short_tpot_slowdown": ratio(
                    short_result.tpot_seconds, isolated_short.tpot_seconds
                ),
                "short_equal_to_isolated": (
                    short_result.output_transcript == isolated_short.output_transcript
                ),
                "decode_overlap": decode_overlap,
            }
        )
    fairness_slowdowns = [
        value
        for run in fairness_runs
        for value in (run["short_ttft_slowdown"], run["short_tpot_slowdown"])
    ]
    relative_slowdown = fairness_relative_slowdown_evidence(
        fairness_slowdowns,
        args.fairness_max_slowdown,
    )
    absolute_short_latency = fairness_short_latency_evidence(
        concurrent_short_results,
        args.fairness_max_short_ttft_seconds,
        args.fairness_max_short_tpot_seconds,
    )
    fairness_semantic_results = [
        warmup_short,
        isolated_short,
        *long_warmup,
        *concurrent_short_results,
        *concurrent_long_results,
    ]
    fairness_semantic = [
        validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )[0]
        for result in fairness_semantic_results
    ]
    fairness_passed = (
        all(result.ok for result in fairness_semantic_results)
        and warmup_short.output_transcript == isolated_short.output_transcript
        and relative_slowdown["passed"]
        and absolute_short_latency["passed"]
        and all(run["decode_overlap"]["passed"] for run in fairness_runs)
        and all(fairness_semantic)
    )
    fairness_result = {
        "passed": fairness_passed,
        "max_slowdown": args.fairness_max_slowdown,
        "relative_slowdown": relative_slowdown,
        "absolute_short_latency": absolute_short_latency,
        "stagger_seconds": args.fairness_stagger_seconds,
        "short_context_tokens": ADVERSARIAL_FAIRNESS_SHORT_CONTEXT_TOKENS,
        "long_context_tokens": ADVERSARIAL_FAIRNESS_LONG_CONTEXT_TOKENS,
        "long_prefix_warmup": long_warmup_summary,
        "warmup_equal_to_isolated": (
            warmup_short.output_transcript == isolated_short.output_transcript
        ),
        "cross_shape_exact_diagnostics_gated": False,
        "isolated_short_ttft_seconds": isolated_short.ttft_seconds,
        "isolated_short_tpot_seconds": isolated_short.tpot_seconds,
        "orders": fairness_runs,
    }
    await writer.emit("fairness_summary", **fairness_result)
    await record_runtime("adversarial-fairness", True)

    overlap_decode_template = retrieval_spec(
        client,
        shortest,
        "overlap-resident-decode",
        args.seed + 20_000,
        args.max_tokens,
        {"scenario": "overlap", "role": "decode"},
    )
    overlap_warmup, overlap_warmup_summary = await run_batch(
        client,
        [overlap_decode_template],
        1,
        writer,
        "adversarial-prefill-overlap-warmup",
        keep_output=True,
    )
    summaries.append(overlap_warmup_summary)
    await record_retrieval_quality(
        "adversarial-prefill-overlap-warmup", overlap_warmup
    )
    overlap_stop = asyncio.Event()
    overlap_decode_results: list[RequestResult] = []

    async def overlap_decode_worker(worker_index: int) -> None:
        request_index = 0
        while not overlap_stop.is_set():
            spec = RequestSpec(
                case_id=f"overlap-decode-w{worker_index}-r{request_index}",
                seed=args.seed + 21_000 + worker_index * 10_000 + request_index,
                max_tokens=overlap_decode_template.max_tokens,
                prompt=overlap_decode_template.prompt,
                context_tokens=overlap_decode_template.context_tokens,
                tags=dict(overlap_decode_template.tags),
                extra=dict(overlap_decode_template.extra),
            )
            result = await client.stream_request(spec)
            overlap_decode_results.append(result)
            await writer.emit(
                "request",
                phase="adversarial-prefill-overlap-decode",
                concurrency=args.overlap_short_requests,
                **result.record(True),
            )
            request_index += 1

    decode_started = time.perf_counter()
    overlap_workers = [
        asyncio.create_task(overlap_decode_worker(index))
        for index in range(args.overlap_short_requests)
    ]
    readiness_deadline = decode_started + args.cleanup_timeout_seconds
    while (
        len(overlap_decode_results) < args.min_overlap_baseline_completions
        and time.perf_counter() < readiness_deadline
    ):
        await asyncio.sleep(args.overlap_queue_poll_seconds)
    baseline_started = time.perf_counter()
    baseline_deadline = baseline_started + args.cleanup_timeout_seconds
    while time.perf_counter() < baseline_deadline:
        baseline_readiness = output_interval_evidence(
            overlap_decode_results,
            baseline_started,
            time.perf_counter(),
        )
        if (
            baseline_readiness["seconds"] >= args.overlap_baseline_seconds
            and baseline_readiness["completed_requests"]
            >= args.min_overlap_baseline_completions
            and baseline_readiness["output_tokens"] is not None
            and baseline_readiness["output_tokens"]
            >= args.min_overlap_baseline_events
        ):
            break
        await asyncio.sleep(args.overlap_queue_poll_seconds)
    prefill_started = time.perf_counter()
    overlap_prefill_spec = retrieval_spec(
        client,
        longest,
        "overlap-large-prefill",
        args.seed + 22_000,
        args.max_tokens,
        {"scenario": "overlap", "role": "large_prefill"},
    )
    overlap_prefill = await client.stream_request(overlap_prefill_spec)
    prefill_completed = time.perf_counter()
    prefill_first_token = (
        overlap_prefill.started + overlap_prefill.ttft_seconds
        if overlap_prefill.ttft_seconds is not None
        else None
    )
    await writer.emit(
        "request",
        phase="adversarial-prefill-overlap-prefill",
        concurrency=args.overlap_short_requests + 1,
        **overlap_prefill.record(True),
    )
    overlap_stop.set()
    await asyncio.gather(*overlap_workers)
    overlap_finished = time.perf_counter()
    baseline_observation, overlapped_observation = overlap_window_evidence(
        overlap_decode_results,
        baseline_started,
        prefill_started,
        prefill_first_token,
    )
    baseline_completions = int(baseline_observation["completed_requests"])
    baseline_seconds = float(baseline_observation["seconds"])
    overlapped_seconds = (
        float(overlapped_observation["seconds"])
        if overlapped_observation is not None
        else None
    )
    baseline_decode_events = int(baseline_observation["output_events"])
    overlapped_decode_events = (
        int(overlapped_observation["output_events"])
        if overlapped_observation is not None
        else 0
    )
    baseline_decode_tokens = baseline_observation["output_tokens"]
    overlapped_decode_tokens = (
        overlapped_observation["output_tokens"]
        if overlapped_observation is not None
        else None
    )
    baseline_decode_event_s = ratio(baseline_decode_events, baseline_seconds)
    overlapped_decode_event_s = ratio(overlapped_decode_events, overlapped_seconds)
    baseline_decode_tok_s = ratio(baseline_decode_tokens, baseline_seconds)
    overlapped_decode_tok_s = ratio(overlapped_decode_tokens, overlapped_seconds)
    overlap_decode_ratio = ratio(overlapped_decode_tok_s, baseline_decode_tok_s)
    overlap_results = [overlap_prefill, *overlap_decode_results]
    overlap_summary = summarize_batch(
        overlap_results,
        overlap_finished - decode_started,
        args.overlap_short_requests + 1,
        "adversarial-prefill-overlap",
    )
    summaries.append(overlap_summary)
    await record_retrieval_quality("adversarial-prefill-overlap", overlap_results)
    overlap_prefill_semantic_ok = validate_retrieval_result(
        overlap_prefill,
        client.tokenizer,
        args.max_repeated_ngram_ratio,
    )[0]
    overlap_decode_semantic_ok = all(
        validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )[0]
        for result in overlap_decode_results
    )
    overlap_output_event_coverage = output_event_coverage_evidence(
        overlap_decode_results,
        args.min_output_event_coverage,
    )
    overlap_decode_gap = output_event_gap_evidence(
        overlap_decode_results,
        prefill_started,
        prefill_first_token,
        args.max_overlap_decode_gap_seconds,
    )
    overlap_evidence = {
        "passed": (
            baseline_completions >= args.min_overlap_baseline_completions
            and baseline_decode_tokens is not None
            and baseline_decode_tokens >= args.min_overlap_baseline_events
            and overlap_prefill.ttft_seconds is not None
            and overlap_prefill.ttft_seconds
            <= args.max_overlap_prefill_ttft_seconds
            and baseline_decode_tokens > 0
            and overlapped_decode_tokens is not None
            and overlapped_decode_tokens > 0
            and overlapped_decode_tok_s is not None
            and overlapped_decode_tok_s
            >= args.min_overlap_decode_events_per_second
            and overlap_decode_ratio is not None
            and overlap_decode_ratio >= args.min_overlap_decode_throughput_ratio
            and overlap_prefill_semantic_ok
            and overlap_decode_semantic_ok
            and overlap_decode_gap["passed"]
            and overlap_output_event_coverage["output_event_coverage_ok"]
        ),
        "decode_workers": args.overlap_short_requests,
        "decode_warmup_seconds": baseline_started - decode_started,
        "baseline_window": baseline_observation,
        "baseline_seconds": baseline_seconds,
        "prefill_overlap_seconds": overlapped_seconds,
        "prefill_completion_seconds": prefill_completed - prefill_started,
        "prefill_ttft_seconds": overlap_prefill.ttft_seconds,
        "maximum_prefill_ttft_seconds": args.max_overlap_prefill_ttft_seconds,
        "baseline_completions_before_prefill": baseline_completions,
        "minimum_baseline_completions": args.min_overlap_baseline_completions,
        "minimum_baseline_output_events": args.min_overlap_baseline_events,
        "minimum_baseline_output_tokens": args.min_overlap_baseline_events,
        "baseline_decode_output_events": baseline_decode_events,
        "overlapped_decode_output_events": overlapped_decode_events,
        "baseline_decode_output_tokens": baseline_decode_tokens,
        "overlapped_decode_output_tokens": overlapped_decode_tokens,
        "baseline_decode_output_events_per_second": baseline_decode_event_s,
        "overlapped_decode_output_events_per_second": overlapped_decode_event_s,
        "baseline_decode_output_tokens_per_second": baseline_decode_tok_s,
        "overlapped_decode_output_tokens_per_second": overlapped_decode_tok_s,
        "minimum_overlapped_decode_events_per_second": (
            args.min_overlap_decode_events_per_second
        ),
        "minimum_overlapped_decode_tokens_per_second": (
            args.min_overlap_decode_events_per_second
        ),
        "overlap_to_baseline_throughput_ratio": overlap_decode_ratio,
        "minimum_overlap_to_baseline_throughput_ratio": (
            args.min_overlap_decode_throughput_ratio
        ),
        "prefill_semantic_ok": overlap_prefill_semantic_ok,
        "all_decode_semantic_ok": overlap_decode_semantic_ok,
        "decode_output_gap": overlap_decode_gap,
        **overlap_output_event_coverage,
        "minimum_output_event_coverage": args.min_output_event_coverage,
    }
    await writer.emit("overlap_decode_evidence", **overlap_evidence)
    await record_runtime("adversarial-prefill-overlap", True)

    pre_cancel_metrics = await safe_metrics(client, writer, "adversarial-pre-cancel")
    cancellation_specs = [
        retrieval_spec(
            client,
            args.cancel_context_tokens,
            f"disconnect-{index:03d}",
            args.seed + 30_000 + index,
            args.max_tokens * 4,
            {"scenario": "disconnect"},
        )
        for index in range(args.cancel_requests)
    ]
    for spec in cancellation_specs:
        spec.extra["ignore_eos"] = True
    cancellation_rng = random.Random(args.seed + 30_000)
    disconnect_delays = [
        args.disconnect_after_seconds * cancellation_rng.uniform(0.5, 2.0)
        for _ in cancellation_specs
    ]
    admission_release = asyncio.Event()
    cancellation_tasks = [
        asyncio.create_task(
            client.disconnect(
                spec,
                disconnect_delays[index],
                admission_release,
            )
        )
        for index, spec in enumerate(cancellation_specs)
    ]
    admission_deadline = time.perf_counter() + args.cleanup_timeout_seconds
    admitted_cancel_requests = 0.0
    admission_metrics = pre_cancel_metrics
    while time.perf_counter() < admission_deadline:
        admission_metrics = await safe_metrics(
            client,
            writer,
            "adversarial-cancel-admission-poll",
        )
        admitted_cancel_requests = metric_delta(
            pre_cancel_metrics,
            admission_metrics,
            "mistralrs_prefix_cache_lookups_total",
        ) or 0.0
        if admitted_cancel_requests >= len(cancellation_specs):
            break
        await asyncio.sleep(args.cleanup_poll_seconds)
    cancel_admissions_ok = admitted_cancel_requests >= len(cancellation_specs)
    admission_release.set()
    cancellation_results = await asyncio.gather(*cancellation_tasks)
    for result in cancellation_results:
        await writer.emit("disconnect", phase="adversarial-cancellation", **result)
    disconnects_ok = all(
        result["outcome"] == "cancelled" and result["server_stream_accepted"]
        for result in cancellation_results
    )
    cancel_cleanup_ok, post_cancel_metrics, cancel_cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        pre_cancel_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-cancel-cleanup-poll",
        optional_gauges=DFLASH_ABORT_CLEANUP_GAUGES,
    )
    counter_deadline = time.perf_counter() + args.cleanup_timeout_seconds
    while True:
        disconnect_server_evidence = request_outcome_evidence(
            pre_cancel_metrics,
            post_cancel_metrics,
            "client_disconnected",
            len(cancellation_specs),
        )
        cancelled_sequences_evidence = labeled_counter_evidence(
            pre_cancel_metrics,
            post_cancel_metrics,
            "mistralrs_sequences_completed_total",
            "reason",
            "canceled",
            len(cancellation_specs),
        )
        if (
            disconnect_server_evidence["passed"]
            and cancelled_sequences_evidence["passed"]
        ) or time.perf_counter() >= counter_deadline:
            break
        await asyncio.sleep(args.cleanup_poll_seconds)
        post_cancel_metrics = await safe_metrics(
            client, writer, "adversarial-cancel-counter-poll"
        )
    cancel_admissions = admitted_cancel_requests
    cancel_kv_blocks_active_delta = metric_delta(
        pre_cancel_metrics,
        post_cancel_metrics,
        KV_CACHE_ACTIVE_GAUGE,
    )
    cancel_kv_blocks_prefix_cached_delta = metric_delta(
        pre_cancel_metrics,
        post_cancel_metrics,
        KV_CACHE_PREFIX_CACHED_GAUGE,
    )
    cancel_retry_specs = [
        RequestSpec(
            case_id=f"cancel-retry-{index:03d}",
            seed=spec.seed,
            max_tokens=args.max_tokens,
            prompt=spec.prompt,
            context_tokens=spec.context_tokens,
            tags=dict(spec.tags),
            extra=prompt_profile_extra(RETRIEVAL_PROMPT_PROFILE),
        )
        for index, spec in enumerate(cancellation_specs)
    ]
    cancel_retry, cancel_retry_summary = await run_batch(
        client,
        cancel_retry_specs,
        min(args.max_seqs, len(cancel_retry_specs)),
        writer,
        "adversarial-cancelled-prefix-retry",
        keep_output=True,
    )
    summaries.append(cancel_retry_summary)
    await record_retrieval_quality("adversarial-cancelled-prefix-retry", cancel_retry)
    cancel_retry_semantic_ok = all(
        validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )[0]
        for result in cancel_retry
    )
    cancel_retry_cleanup_ok, post_cancel_retry_metrics, cancel_retry_cleanup_detail = (
        await poll_for_cleanup(
            client,
            writer,
            post_cancel_metrics,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            "adversarial-cancel-retry-cleanup-poll",
        )
    )
    cancel_retry_prefix_cached_delta = metric_delta(
        post_cancel_metrics,
        post_cancel_retry_metrics,
        KV_CACHE_PREFIX_CACHED_GAUGE,
    )

    timeout_spec = retrieval_spec(
        client,
        longest,
        "timeout-long",
        args.seed + 40_000,
        args.max_tokens * 4,
        {"scenario": "timeout"},
    )
    timeout_spec.extra["ignore_eos"] = True
    pre_timeout_metrics = await safe_metrics(client, writer, "adversarial-pre-timeout")
    timeout_result = await client.stream_request(
        timeout_spec, timeout_seconds=args.timeout_test_seconds
    )
    await writer.emit(
        "request",
        phase="adversarial-timeout",
        concurrency=1,
        **timeout_result.record(True),
    )
    timeout_admission_deadline = time.perf_counter() + args.cleanup_timeout_seconds
    timeout_admissions = 0.0
    while time.perf_counter() < timeout_admission_deadline:
        timeout_admission_metrics = await safe_metrics(
            client,
            writer,
            "adversarial-timeout-admission-poll",
        )
        timeout_admissions = metric_delta(
            pre_timeout_metrics,
            timeout_admission_metrics,
            "mistralrs_prefix_cache_lookups_total",
        ) or 0.0
        if timeout_admissions >= 1:
            break
        await asyncio.sleep(args.cleanup_poll_seconds)
    timeout_cleanup_ok, post_timeout_metrics, timeout_cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        pre_timeout_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-timeout-cleanup-poll",
        optional_gauges=DFLASH_ABORT_CLEANUP_GAUGES,
    )
    timeout_response_started = (
        timeout_result.status_code is not None and timeout_result.status_code < 400
    )
    timeout_observed = (
        timeout_response_started and timeout_result.error_kind == "ReadTimeout"
    )
    timeout_admitted = timeout_admissions >= 1
    timeout_kv_blocks_active_delta = metric_delta(
        pre_timeout_metrics,
        post_timeout_metrics,
        KV_CACHE_ACTIVE_GAUGE,
    )
    timeout_kv_blocks_prefix_cached_delta = metric_delta(
        pre_timeout_metrics,
        post_timeout_metrics,
        KV_CACHE_PREFIX_CACHED_GAUGE,
    )
    timeout_counter_deadline = time.perf_counter() + args.cleanup_timeout_seconds
    while True:
        timeout_server_evidence = request_outcome_evidence(
            pre_timeout_metrics,
            post_timeout_metrics,
            "client_disconnected",
            1,
        )
        if timeout_server_evidence["passed"] or time.perf_counter() >= timeout_counter_deadline:
            break
        await asyncio.sleep(args.cleanup_poll_seconds)
        post_timeout_metrics = await safe_metrics(
            client, writer, "adversarial-timeout-counter-poll"
        )
    timeout_retry_spec = RequestSpec(
        case_id="timeout-long-retry",
        seed=timeout_spec.seed,
        max_tokens=args.max_tokens,
        prompt=timeout_spec.prompt,
        context_tokens=timeout_spec.context_tokens,
        tags=dict(timeout_spec.tags),
        extra=prompt_profile_extra(RETRIEVAL_PROMPT_PROFILE),
    )
    retry, retry_summary = await run_batch(
        client,
        [timeout_retry_spec],
        1,
        writer,
        "adversarial-retry",
        keep_output=True,
    )
    summaries.append(retry_summary)
    retry_semantic_ok = validate_retrieval_result(
        retry[0], client.tokenizer, args.max_repeated_ngram_ratio
    )[0]
    await record_runtime("adversarial-cancellation-timeout-retry", False)

    pre_burst_metrics = await safe_metrics(client, writer, "adversarial-pre-burst")
    burst_specs = [
        RequestSpec(
            case_id=f"burst-{index:04d}",
            seed=args.seed + 50_000 + index,
            max_tokens=args.burst_max_tokens,
            prompt=exact_context(
                client,
                args.context_lengths[index % len(args.context_lengths)],
                f"burst-{index}",
            ),
            context_tokens=args.context_lengths[index % len(args.context_lengths)],
            tags={"scenario": "burst"},
        )
        for index in range(args.burst_requests)
    ]
    burst_results, burst_summary = await run_batch(
        client,
        burst_specs,
        args.burst_requests,
        writer,
        "adversarial-burst",
        keep_output=False,
    )
    summaries.append(burst_summary)
    await record_quality("adversarial-burst", burst_results)
    burst_cleanup_ok, _, burst_cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        pre_burst_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-burst-cleanup",
    )
    await record_runtime("adversarial-burst", True)

    prefix_pressure = await run_prefix_pressure_workflow(args, client, writer)
    summaries.extend(prefix_pressure["summaries"])
    quality_checks.extend(prefix_pressure["quality_checks"])
    prefix_cache_stages = prefix_pressure["prefix_cache"]
    prefix_cache_stage_passed = prefix_pressure["passed"]
    prefix_correct = prefix_cache_stages["outputs_equal"]
    await record_runtime("adversarial-prefix-cache", False)

    churn_summaries = []
    churn_evidence = []
    for round_index in range(args.churn_rounds):
        for width in (args.max_seqs - 1, args.max_seqs, args.max_seqs + 1):
            churn_specs = [
                retrieval_spec(
                    client,
                    args.context_lengths[index % len(args.context_lengths)],
                    f"churn-r{round_index}-c{width}-{index}",
                    args.seed + 70_000 + round_index * 1_000 + width * 20 + index,
                    args.churn_max_tokens,
                    {
                        "scenario": "churn",
                        "round": round_index,
                        "width": width,
                    },
                )
                for index in range(width)
            ]
            phase = f"churn-r{round_index}-width{width}"
            pre_churn_metrics = await safe_metrics(client, writer, f"{phase}-start")
            batch_task = asyncio.create_task(
                run_batch(
                    client,
                    churn_specs,
                    width,
                    writer,
                    phase,
                    keep_output=True,
                )
            )
            await asyncio.sleep(0)
            queue_samples = []
            while not batch_task.done():
                snapshot = await safe_metrics(client, writer, f"{phase}-queue-poll")
                queue_samples.append(
                    metric_values(
                        snapshot,
                        (
                            "mistralrs_sequences_capacity",
                            "mistralrs_sequences_running",
                            "mistralrs_sequences_waiting",
                            "mistralrs_requests_pending_admission",
                            "http_requests_in_flight",
                        ),
                    )
                )
                await asyncio.sleep(args.overlap_queue_poll_seconds)
            churn_results, churn_summary = await batch_task
            churn_summaries.append(churn_summary)
            summaries.append(churn_summary)
            await record_retrieval_quality(phase, churn_results)
            churn_cleanup_ok, _, churn_cleanup = await poll_for_cleanup(
                client,
                writer,
                pre_churn_metrics,
                args.cleanup_timeout_seconds,
                args.cleanup_poll_seconds,
                f"{phase}-cleanup",
            )
            evidence = {
                **churn_capacity_evidence(
                    queue_samples,
                    width,
                    args.max_seqs,
                    sequence_capacity,
                    churn_cleanup_ok,
                    churn_summary["errors"],
                ),
                "round": round_index,
                "cleanup": churn_cleanup,
                "samples": queue_samples,
            }
            churn_evidence.append(evidence)
            await writer.emit("max_seqs_queue_evidence", **evidence)

    await record_runtime("adversarial-churn", True)

    max_seqs_queue_evidence = {
        "passed": (
            max_seqs_capacity_ok
            and any(item["queue_required"] for item in churn_evidence)
            and all(item["passed"] for item in churn_evidence)
        ),
        "configured_max_seqs": args.max_seqs,
        "server_sequence_capacity": sequence_capacity,
        "capacity_matches_16_boundary": max_seqs_capacity_ok,
        "batches": churn_evidence,
    }

    final_metrics = await safe_metrics(client, writer, "adversarial-end")
    running_after_cancel = metric_total(post_cancel_metrics, "mistralrs_sequences_running")
    waiting_after_cancel = metric_total(post_cancel_metrics, "mistralrs_sequences_waiting")
    metrics_delta = selected_metric_deltas(initial_metrics, final_metrics)
    adversarial_mtp = configured_speculative_evidence(
        initial_metrics,
        final_metrics,
        args,
        args.require_mtp,
    )
    adversarial_graph = cuda_graph_evidence(
        initial_metrics,
        final_metrics,
        initial_metrics,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    adversarial_memory = cuda_memory_pressure_evidence(
        initial_metrics,
        final_metrics,
        require_instrumentation=True,
    )
    kv_ownership = prefix_cached_ownership_evidence(
        cancel_kv_blocks_prefix_cached_delta,
        cancel_retry_prefix_cached_delta,
        timeout_kv_blocks_prefix_cached_delta,
    )
    acceptance_grade = acceptance_grade_evidence(
        args.acceptance_grade,
        args.require_mtp,
        args.expected_graph_components,
    )
    passed = (
        all(summary["errors"] == 0 for summary in summaries)
        and long_correctness["passed"]
        and long_resident_performance["passed"]
        and mixed_cohort["passed"]
        and mixed_throughput["passed"]
        and fairness_passed
        and overlap_evidence["passed"]
        and disconnects_ok
        and cancel_cleanup_ok
        and cancel_admissions_ok
        and disconnect_server_evidence["passed"]
        and cancelled_sequences_evidence["passed"]
        and cancel_retry_summary["errors"] == 0
        and cancel_retry_semantic_ok
        and cancel_retry_cleanup_ok
        and timeout_observed
        and timeout_admitted
        and timeout_server_evidence["passed"]
        and timeout_cleanup_ok
        and retry_semantic_ok
        and burst_cleanup_ok
        and kv_ownership["passed"]
        and prefix_cache_stage_passed
        and max_seqs_queue_evidence["passed"]
        and all(item["valid"] for item in quality_checks)
        and all(item["passed"] for item in runtime_evidence)
        and adversarial_mtp["passed"]
        and adversarial_graph["passed"]
        and adversarial_memory["passed"]
        and acceptance_grade["passed"]
    )
    return {
        "mode": "adversarial",
        "passed": passed,
        "summaries": summaries,
        "long_context_correctness": long_correctness,
        "long_resident_performance": long_resident_performance,
        "mixed_context_cohort": mixed_cohort,
        "throughput": mixed_throughput,
        "fairness": fairness_result,
        "overlap": overlap_evidence,
        "disconnects": cancellation_results,
        "disconnects_ok": disconnects_ok,
        "cancel_admissions": cancel_admissions,
        "cancel_admissions_ok": cancel_admissions_ok,
        "disconnect_server_evidence": disconnect_server_evidence,
        "cancelled_sequences_evidence": cancelled_sequences_evidence,
        "cancel_cleanup_ok": cancel_cleanup_ok,
        "cancel_cleanup_detail": cancel_cleanup_detail,
        "cancel_kv_cache_blocks_active_delta": cancel_kv_blocks_active_delta,
        "cancel_kv_cache_blocks_prefix_cached_delta": (
            cancel_kv_blocks_prefix_cached_delta
        ),
        "cancel_retry_semantic_ok": cancel_retry_semantic_ok,
        "cancel_retry_cleanup_ok": cancel_retry_cleanup_ok,
        "cancel_retry_cleanup_detail": cancel_retry_cleanup_detail,
        "cancel_retry_prefix_cached_delta": cancel_retry_prefix_cached_delta,
        "timeout_observed": timeout_observed,
        "timeout_response_started": timeout_response_started,
        "timeout_admitted": timeout_admitted,
        "timeout_server_evidence": timeout_server_evidence,
        "timeout_cleanup_ok": timeout_cleanup_ok,
        "timeout_cleanup_detail": timeout_cleanup_detail,
        "timeout_kv_cache_blocks_active_delta": timeout_kv_blocks_active_delta,
        "timeout_kv_cache_blocks_prefix_cached_delta": (
            timeout_kv_blocks_prefix_cached_delta
        ),
        "kv_ownership_gate_complete": kv_ownership["passed"],
        "kv_ownership": kv_ownership,
        "retry_succeeded": retry_summary["errors"] == 0 and retry_semantic_ok,
        "burst_cleanup_ok": burst_cleanup_ok,
        "burst_cleanup_detail": burst_cleanup_detail,
        "prefix_outputs_equal": prefix_correct,
        "prefix_pressure": prefix_pressure,
        "prefix_cache_stages": prefix_cache_stages,
        "quality_checks": quality_checks,
        "cleanup_ok": (
            cancel_cleanup_ok
            and cancel_retry_cleanup_ok
            and timeout_cleanup_ok
            and burst_cleanup_ok
            and prefix_pressure["cleanup_ok"]
        ),
        "post_cancel_sequences_running": running_after_cancel,
        "post_cancel_sequences_waiting": waiting_after_cancel,
        "metrics_delta": metrics_delta,
        "runtime_evidence": runtime_evidence,
        "mtp": adversarial_mtp,
        "cuda_graph": adversarial_graph,
        "cuda_memory": adversarial_memory,
        "acceptance_grade": args.acceptance_grade,
        "acceptance_grade_evidence": acceptance_grade,
        "churn_batches": len(churn_summaries),
        "max_seqs_queue_evidence": max_seqs_queue_evidence,
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def periodic_schedule(
    started: float,
    ended: float,
    interval_seconds: float,
    include_start: bool = False,
) -> list[float]:
    scheduled = []
    timestamp = started if include_start else started + interval_seconds
    while timestamp < ended - SCHEDULE_TIME_EPSILON_SECONDS:
        scheduled.append(timestamp)
        timestamp += interval_seconds
    return scheduled


def scheduled_observation_evidence(
    scheduled: Sequence[float],
    observed: Sequence[float],
    max_lateness_seconds: float,
) -> dict[str, Any]:
    paired_count = min(len(scheduled), len(observed))
    raw_lateness = [
        observed[index] - scheduled[index] for index in range(paired_count)
    ]
    lateness = [max(0.0, value) for value in raw_lateness]
    premature_observations = sum(
        value < -SCHEDULE_TIME_EPSILON_SECONDS for value in raw_lateness
    )
    observed_monotonic = all(
        current >= previous
        for previous, current in zip(observed, observed[1:])
    )
    exact_count = len(scheduled) == len(observed)
    lateness_summary = distribution(lateness)
    return {
        "passed": (
            bool(scheduled)
            and exact_count
            and premature_observations == 0
            and observed_monotonic
            and lateness_summary["max"] is not None
            and lateness_summary["max"] <= max_lateness_seconds
        ),
        "scheduled_count": len(scheduled),
        "observed_count": len(observed),
        "paired_count": paired_count,
        "missing_observations": max(0, len(scheduled) - len(observed)),
        "unexpected_observations": max(0, len(observed) - len(scheduled)),
        "exact_count": exact_count,
        "premature_observations": premature_observations,
        "observed_monotonic": observed_monotonic,
        "max_lateness_seconds": max_lateness_seconds,
        "lateness_seconds": lateness_summary,
        "first_scheduled": scheduled[0] if scheduled else None,
        "last_scheduled": scheduled[-1] if scheduled else None,
        "first_observed": observed[0] if observed else None,
        "last_observed": observed[-1] if observed else None,
    }


def comparison_window_coverage_evidence(
    started: float,
    planned_ended: float,
    actual_ended: float,
    window_seconds: float,
) -> dict[str, Any]:
    first_window = (started, started + window_seconds)
    final_window = (planned_ended - window_seconds, planned_ended)
    phase_seconds = planned_ended - started
    windows_non_overlapping = first_window[1] <= (
        final_window[0] + SCHEDULE_TIME_EPSILON_SECONDS
    )
    first_covered_seconds = max(
        0.0,
        min(actual_ended, first_window[1]) - first_window[0],
    )
    final_covered_seconds = max(
        0.0,
        min(actual_ended, final_window[1]) - final_window[0],
    )
    full_window_coverage = (
        first_covered_seconds + SCHEDULE_TIME_EPSILON_SECONDS >= window_seconds
        and final_covered_seconds + SCHEDULE_TIME_EPSILON_SECONDS >= window_seconds
    )
    return {
        "passed": (
            phase_seconds + SCHEDULE_TIME_EPSILON_SECONDS >= 2 * window_seconds
            and windows_non_overlapping
            and actual_ended + SCHEDULE_TIME_EPSILON_SECONDS >= planned_ended
            and full_window_coverage
        ),
        "requested_window_seconds": window_seconds,
        "planned_phase_seconds": phase_seconds,
        "actual_phase_seconds": actual_ended - started,
        "first_window": list(first_window),
        "final_window": list(final_window),
        "first_covered_seconds": first_covered_seconds,
        "final_covered_seconds": final_covered_seconds,
        "windows_non_overlapping": windows_non_overlapping,
        "full_window_coverage": full_window_coverage,
        "phase_reached_planned_end": (
            actual_ended + SCHEDULE_TIME_EPSILON_SECONDS >= planned_ended
        ),
    }


def acceptance_grade_evidence(
    enabled: bool,
    require_mtp: bool,
    expected_graph_components: Sequence[str],
) -> dict[str, Any]:
    required_components = {"target", "dflash"}
    configured_components = set(expected_graph_components)
    requirements_met = require_mtp and required_components.issubset(
        configured_components
    )
    return {
        "passed": not enabled or requirements_met,
        "enabled": enabled,
        "certification_complete": enabled and requirements_met,
        "require_mtp": require_mtp,
        "required_graph_components": sorted(required_components),
        "configured_graph_components": sorted(configured_components),
        "graph_components_complete": required_components.issubset(
            configured_components
        ),
    }


def request_decode_tok_s(result: RequestResult) -> float | None:
    if (
        not result.ok
        or result.ttft_seconds is None
        or result.completion_tokens <= 1
    ):
        return None
    decode_seconds = result.ended - result.started - result.ttft_seconds
    if decode_seconds <= 0:
        return None
    return (result.completion_tokens - 1) / decode_seconds


def fixed_length_completion_evidence(
    results: Sequence[RequestResult],
    expected_tokens: int,
    required: bool = True,
) -> dict[str, Any]:
    checks = [
        {
            "case_id": result.case_id,
            "passed": (
                result.ok
                and result.completion_tokens == expected_tokens
                and result.finish_reason == "length"
            ),
            "ok": result.ok,
            "completion_tokens": result.completion_tokens,
            "expected_completion_tokens": expected_tokens,
            "finish_reason": result.finish_reason,
        }
        for result in results
    ]
    observed_passed = bool(checks) and all(item["passed"] for item in checks)
    return {
        "passed": not required or observed_passed,
        "required": required,
        "observed_passed": observed_passed,
        "requests": len(checks),
        "expected_completion_tokens": expected_tokens,
        "failures": [item for item in checks if not item["passed"]],
    }


def production_output_quality_evidence(
    results: Sequence[RequestResult],
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
    fixed_output_length: bool,
) -> dict[str, Any]:
    fixed_length_traffic = [
        result
        for result in results
        if fixed_output_length and result.tags.get("role") == "traffic"
    ]
    quality_results = [
        result
        for result in results
        if not (fixed_output_length and result.tags.get("role") == "traffic")
    ]
    checks = [
        {"valid": valid, **detail}
        for result in quality_results
        for valid, detail in [
            validate_sampled_output(
                result,
                tokenizer,
                max_repeated_ngram_ratio,
            )
        ]
    ]
    return {
        "passed": all(check["valid"] for check in checks),
        "quality_output_contract": "normal_eos",
        "fixed_output_length_traffic": fixed_output_length,
        "checked_requests": len(checks),
        "checked_roles": dict(
            sorted(
                Counter(
                    str(result.tags.get("role")) for result in quality_results
                ).items()
            )
        ),
        "excluded_fixed_length_traffic_requests": len(fixed_length_traffic),
        "fixed_length_traffic_is_quality_evidence": False,
        "checks": checks,
        "failures": [check for check in checks if not check["valid"]],
    }


def fixed_seed_result_signature(result: RequestResult) -> tuple[str, int, str | None]:
    return (
        result.output_transcript,
        result.completion_tokens,
        result.finish_reason,
    )


def production_probe_evidence(
    results: Sequence[RequestResult],
    baseline: RequestResult,
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
    min_output_event_coverage: float,
    minimum_samples: int,
    max_ttft_seconds: float,
    max_tpot_seconds: float,
    max_latency_slowdown: float,
    max_schedule_lateness_seconds: float,
) -> dict[str, Any]:
    baseline_semantic_ok, baseline_semantic = validate_retrieval_result(
        baseline,
        tokenizer,
        max_repeated_ngram_ratio,
    )
    checks = []
    for result in results:
        semantic_ok, semantic = validate_retrieval_result(
            result,
            tokenizer,
            max_repeated_ngram_ratio,
        )
        ttft_ratio = ratio(result.ttft_seconds, baseline.ttft_seconds)
        tpot_ratio = ratio(result.tpot_seconds, baseline.tpot_seconds)
        transcript_exact = result.output_transcript == baseline.output_transcript
        completion_tokens_exact = (
            result.completion_tokens == baseline.completion_tokens
        )
        finish_reason_exact = result.finish_reason == baseline.finish_reason
        exact = transcript_exact and completion_tokens_exact and finish_reason_exact
        schedule_lateness_ok = (
            result.client_queue_seconds <= max_schedule_lateness_seconds
        )
        latency_ok = (
            result.ttft_seconds is not None
            and result.ttft_seconds <= max_ttft_seconds
            and result.tpot_seconds is not None
            and result.tpot_seconds <= max_tpot_seconds
            and ttft_ratio is not None
            and ttft_ratio <= max_latency_slowdown
            and tpot_ratio is not None
            and tpot_ratio <= max_latency_slowdown
            and schedule_lateness_ok
        )
        checks.append(
            {
                "case_id": result.case_id,
                "ok": result.ok,
                "exact_result": exact,
                "exact_transcript": transcript_exact,
                "completion_tokens_exact": completion_tokens_exact,
                "finish_reason_exact": finish_reason_exact,
                "semantic_ok": semantic_ok,
                "semantic": semantic,
                "ttft_seconds": result.ttft_seconds,
                "tpot_seconds": result.tpot_seconds,
                "ttft_ratio_to_isolated": ttft_ratio,
                "tpot_ratio_to_isolated": tpot_ratio,
                "decode_tok_s": request_decode_tok_s(result),
                "completion_tokens": result.completion_tokens,
                "finish_reason": result.finish_reason,
                "schedule_lateness_seconds": result.client_queue_seconds,
                "schedule_lateness_ok": schedule_lateness_ok,
                "latency_ok": latency_ok,
                "output_transcript_sha256": stable_hash(result.output_transcript),
            }
        )
    output_events = output_event_coverage_evidence(
        results,
        min_output_event_coverage,
    )
    complete = len(results) >= minimum_samples
    exact_transcripts = complete and all(
        item["exact_transcript"] for item in checks
    )
    exact_results = complete and all(item["exact_result"] for item in checks)
    semantic_ok = complete and all(item["semantic_ok"] for item in checks)
    latency_ok = complete and all(item["latency_ok"] for item in checks)
    return {
        "passed": (
            baseline_semantic_ok
            and exact_results
            and semantic_ok
            and latency_ok
            and output_events["output_event_coverage_ok"]
        ),
        "samples": len(results),
        "minimum_samples": minimum_samples,
        "exact_transcripts": exact_transcripts,
        "exact_results": exact_results,
        "semantic_ok": semantic_ok,
        "latency_ok": latency_ok,
        "max_ttft_seconds": max_ttft_seconds,
        "max_tpot_seconds": max_tpot_seconds,
        "max_latency_slowdown": max_latency_slowdown,
        "max_schedule_lateness_seconds": max_schedule_lateness_seconds,
        "baseline": {
            "case_id": baseline.case_id,
            "semantic_ok": baseline_semantic_ok,
            "semantic": baseline_semantic,
            "ttft_seconds": baseline.ttft_seconds,
            "tpot_seconds": baseline.tpot_seconds,
            "decode_tok_s": request_decode_tok_s(baseline),
            "completion_tokens": baseline.completion_tokens,
            "finish_reason": baseline.finish_reason,
            "output_transcript_sha256": stable_hash(baseline.output_transcript),
        },
        "checks": checks,
        "output_event_coverage": output_events,
    }


def production_semantic_sentinel_evidence(
    results: Sequence[RequestResult],
    context_lengths: Sequence[int],
    stages: Sequence[str],
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
    min_output_event_coverage: float,
    max_schedule_lateness_seconds: float,
) -> dict[str, Any]:
    expected = {
        (length, stage)
        for length in context_lengths
        for stage in stages
    }
    observed = Counter(
        (int(result.context_tokens or 0), str(result.tags.get("sentinel_stage")))
        for result in results
    )
    semantic_checks = []
    for result in results:
        valid, detail = validate_retrieval_result(
            result,
            tokenizer,
            max_repeated_ngram_ratio,
        )
        schedule_lateness_ok = (
            result.client_queue_seconds <= max_schedule_lateness_seconds
        )
        semantic_checks.append(
            {
                "valid": valid and schedule_lateness_ok,
                "semantic_valid": valid,
                **detail,
                "completion_tokens": result.completion_tokens,
                "finish_reason": result.finish_reason,
                "schedule_lateness_seconds": result.client_queue_seconds,
                "schedule_lateness_ok": schedule_lateness_ok,
            }
        )
    signatures_by_context = {
        str(length): {
            fixed_seed_result_signature(result)
            for result in results
            if result.context_tokens == length and result.ok
        }
        for length in context_lengths
    }
    coverage_complete = set(observed) == expected and all(
        observed[key] == 1 for key in expected
    )
    fixed_seed_exact = coverage_complete and all(
        len(signatures) == 1 for signatures in signatures_by_context.values()
    )
    output_events = output_event_coverage_evidence(
        results,
        min_output_event_coverage,
    )
    semantic_ok = len(semantic_checks) == len(expected) and all(
        item["semantic_valid"] for item in semantic_checks
    )
    schedule_lateness_ok = len(semantic_checks) == len(expected) and all(
        item["schedule_lateness_ok"] for item in semantic_checks
    )
    return {
        "passed": (
            coverage_complete
            and fixed_seed_exact
            and semantic_ok
            and schedule_lateness_ok
            and output_events["output_event_coverage_ok"]
        ),
        "expected": [
            {"context_tokens": length, "stage": stage}
            for length, stage in sorted(expected)
        ],
        "observed": [
            {
                "context_tokens": length,
                "stage": stage,
                "count": count,
            }
            for (length, stage), count in sorted(observed.items())
        ],
        "coverage_complete": coverage_complete,
        "fixed_seed_exact": fixed_seed_exact,
        "semantic_ok": semantic_ok,
        "schedule_lateness_ok": schedule_lateness_ok,
        "semantic_checks": semantic_checks,
        "max_schedule_lateness_seconds": max_schedule_lateness_seconds,
        "unique_result_signatures_by_context": {
            length: len(signatures)
            for length, signatures in signatures_by_context.items()
        },
        "output_event_coverage": output_events,
    }


def output_event_coverage_evidence(
    results: Sequence[RequestResult],
    min_output_event_coverage: float,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    per_request = []
    for result in successful:
        terminal_tokens = int(
            result.completion_tokens > 0 and result.finish_reason == "stop"
        )
        expected_tokens = result.completion_tokens - terminal_tokens
        if expected_tokens == 0 and result.streamed_output_tokens == 0:
            request_coverage = 1.0
        else:
            request_coverage = ratio(result.streamed_output_tokens, expected_tokens)
        request_ok = (
            request_coverage is not None
            and min_output_event_coverage
            <= request_coverage
            <= 2.0 - min_output_event_coverage
        )
        per_request.append(
            {
                "case_id": result.case_id,
                "observed_output_events": result.output_chunks,
                "observed_output_tokens": result.streamed_output_tokens,
                "reported_completion_tokens": result.completion_tokens,
                "unstreamed_terminal_tokens": terminal_tokens,
                "expected_streamable_tokens": expected_tokens,
                "output_event_coverage": request_coverage,
                "output_token_coverage": request_coverage,
                "passed": request_ok,
            }
        )
    observed_output_events = sum(item["observed_output_events"] for item in per_request)
    observed_output_tokens = sum(item["observed_output_tokens"] for item in per_request)
    reported_completion_tokens = sum(item["reported_completion_tokens"] for item in per_request)
    unstreamed_terminal_tokens = sum(item["unstreamed_terminal_tokens"] for item in per_request)
    expected_streamable_tokens = (
        reported_completion_tokens - unstreamed_terminal_tokens
    )
    if successful and expected_streamable_tokens == 0 and observed_output_tokens == 0:
        coverage = 1.0
    else:
        coverage = ratio(observed_output_tokens, expected_streamable_tokens)
    aggregate_coverage_ok = (
        coverage is not None
        and min_output_event_coverage
        <= coverage
        <= 2.0 - min_output_event_coverage
    )
    coverage_ok = bool(per_request) and aggregate_coverage_ok and all(
        item["passed"] for item in per_request
    )
    return {
        "observed_output_events": observed_output_events,
        "observed_output_tokens": observed_output_tokens,
        "reported_completion_tokens": reported_completion_tokens,
        "unstreamed_terminal_tokens": unstreamed_terminal_tokens,
        "expected_streamable_tokens": expected_streamable_tokens,
        "output_event_coverage": coverage,
        "output_token_coverage": coverage,
        "min_output_event_coverage": min_output_event_coverage,
        "aggregate_output_event_coverage_ok": aggregate_coverage_ok,
        "per_request_output_event_coverage": per_request,
        "output_event_coverage_ok": coverage_ok,
        "output_token_coverage_ok": coverage_ok,
    }


def throughput_evidence(
    summaries: Sequence[dict[str, Any]],
    thresholds: dict[int, float],
    min_scaling_efficiency: float,
    throughput_metric: str = "output_tok_s_common_wall",
) -> dict[str, Any]:
    actual = {
        int(summary["concurrency"]): summary.get(throughput_metric)
        for summary in summaries
    }
    measurements = {
        str(int(summary["concurrency"])): {
            "end_to_end_output_tok_s_common_wall": summary.get(
                "end_to_end_output_tok_s_common_wall",
                summary.get("output_tok_s_common_wall"),
            ),
            "decode_tok_s_active": summary.get("decode_tok_s_active"),
        }
        for summary in summaries
    }
    missing = sorted(set(thresholds) - set(actual))
    absolute = {
        str(concurrency): {
            "throughput_metric": throughput_metric,
            "actual_throughput": actual.get(concurrency),
            "actual_output_tok_s": actual.get(concurrency),
            "minimum_output_tok_s": minimum,
            "passed": (
                actual.get(concurrency) is not None
                and actual[concurrency] >= minimum
            ),
        }
        for concurrency, minimum in sorted(thresholds.items())
    }
    baseline_concurrency = min(thresholds)
    baseline_throughput = actual.get(baseline_concurrency)
    scaling = {}
    for concurrency in sorted(thresholds):
        if concurrency == baseline_concurrency:
            continue
        observed_ratio = ratio(actual.get(concurrency), baseline_throughput)
        ideal_ratio = concurrency / baseline_concurrency
        efficiency = ratio(observed_ratio, ideal_ratio)
        scaling[str(concurrency)] = {
            "throughput_metric": throughput_metric,
            "baseline_concurrency": baseline_concurrency,
            "observed_throughput_ratio": observed_ratio,
            "ideal_linear_ratio": ideal_ratio,
            "scaling_efficiency": efficiency,
            "minimum_scaling_efficiency": min_scaling_efficiency,
            "passed": (
                efficiency is not None and efficiency >= min_scaling_efficiency
            ),
        }
    return {
        "passed": (
            not missing
            and bool(absolute)
            and all(item["passed"] for item in absolute.values())
            and all(item["passed"] for item in scaling.values())
        ),
        "missing_concurrencies": missing,
        "throughput_metric": throughput_metric,
        "measurements": measurements,
        "absolute": absolute,
        "scaling": scaling,
        "baseline_concurrency": baseline_concurrency,
        "minimum_scaling_efficiency": min_scaling_efficiency,
    }


def exact_throughput_threshold_evidence(
    summaries: Sequence[dict[str, Any]],
    thresholds: dict[int, float],
    throughput_metric: str,
) -> dict[str, Any]:
    counts = Counter(int(summary["concurrency"]) for summary in summaries)
    actual = {
        int(summary["concurrency"]): summary.get(throughput_metric)
        for summary in summaries
    }
    expected = set(thresholds)
    observed = set(actual)
    measurements = {
        str(concurrency): {
            "actual": actual.get(concurrency),
            "minimum": minimum,
            "passed": (
                actual.get(concurrency) is not None
                and actual[concurrency] >= minimum
            ),
        }
        for concurrency, minimum in sorted(thresholds.items())
    }
    exact_cohort = expected == observed and all(counts[value] == 1 for value in expected)
    return {
        "passed": (
            exact_cohort
            and bool(measurements)
            and all(item["passed"] for item in measurements.values())
        ),
        "throughput_metric": throughput_metric,
        "expected_concurrencies": sorted(expected),
        "observed_concurrencies": sorted(observed),
        "summary_counts": {str(key): value for key, value in sorted(counts.items())},
        "exact_concurrency_cohort": exact_cohort,
        "measurements": measurements,
    }


def fairness_relative_slowdown_evidence(
    slowdowns: Sequence[float | None],
    maximum: float,
) -> dict[str, Any]:
    values = list(slowdowns)
    within_threshold = bool(values) and all(
        value is not None and value <= maximum for value in values
    )
    return {
        "passed": within_threshold,
        "gated": True,
        "maximum": maximum,
        "values": values,
        "within_threshold": within_threshold,
    }


def fairness_short_latency_evidence(
    results: Sequence[RequestResult],
    max_ttft_seconds: float,
    max_tpot_seconds: float,
) -> dict[str, Any]:
    requests = [
        {
            "case_id": result.case_id,
            "ok": result.ok,
            "ttft_seconds": result.ttft_seconds,
            "tpot_seconds": result.tpot_seconds,
            "ttft_passed": (
                result.ttft_seconds is not None
                and result.ttft_seconds <= max_ttft_seconds
            ),
            "tpot_passed": (
                result.tpot_seconds is not None
                and result.tpot_seconds <= max_tpot_seconds
            ),
        }
        for result in results
    ]
    return {
        "passed": bool(requests) and all(
            item["ok"] and item["ttft_passed"] and item["tpot_passed"]
            for item in requests
        ),
        "max_short_ttft_seconds": max_ttft_seconds,
        "max_short_tpot_seconds": max_tpot_seconds,
        "requests": requests,
    }


def serving_throughput_evidence(
    summaries: Sequence[dict[str, Any]],
    thresholds: dict[int, float],
    min_scaling_efficiency: float,
) -> dict[str, Any]:
    common_wall = throughput_evidence(
        summaries,
        thresholds,
        min_scaling_efficiency,
        "output_tok_s_common_wall",
    )
    decode_active = throughput_evidence(
        summaries,
        thresholds,
        min_scaling_efficiency,
        "decode_tok_s_active",
    )
    return {
        **common_wall,
        "passed": common_wall["passed"] and decode_active["passed"],
        "common_wall": common_wall,
        "decode_active": decode_active,
    }


def request_outcome_evidence(
    before: dict[str, float],
    after: dict[str, float],
    outcome: str,
    minimum_events: int,
) -> dict[str, Any]:
    counters = labeled_metric_deltas(before, after, REQUEST_OUTCOMES_COUNTER)
    matching = [item for item in counters if item["labels"].get("outcome") == outcome]
    observed = sum(item["delta"] for item in matching)
    instrumentation_present = any(
        key.split("{", 1)[0] == REQUEST_OUTCOMES_COUNTER
        for key in set(before) | set(after)
    )
    return {
        "passed": instrumentation_present and observed >= minimum_events,
        "metric": REQUEST_OUTCOMES_COUNTER,
        "outcome": outcome,
        "minimum_events": minimum_events,
        "observed_events": observed,
        "instrumentation_present": instrumentation_present,
        "matching_series": matching,
    }


def labeled_counter_evidence(
    before: dict[str, float],
    after: dict[str, float],
    metric: str,
    label: str,
    value: str,
    minimum_events: int,
) -> dict[str, Any]:
    counters = labeled_metric_deltas(before, after, metric)
    matching = [item for item in counters if item["labels"].get(label) == value]
    observed = sum(item["delta"] for item in matching)
    instrumentation_present = any(
        key.split("{", 1)[0] == metric for key in set(before) | set(after)
    )
    return {
        "passed": instrumentation_present and observed >= minimum_events,
        "metric": metric,
        "label": label,
        "value": value,
        "minimum_events": minimum_events,
        "observed_events": observed,
        "instrumentation_present": instrumentation_present,
        "matching_series": matching,
    }


def selected_metric_deltas(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float | None]:
    names = (
        "mistralrs_tokens_processed_total",
        "mistralrs_speculative_drafts_total",
        "mistralrs_speculative_draft_tokens_proposed_total",
        "mistralrs_speculative_draft_tokens_accepted_total",
        SPARSE_VERIFIER_GPU_COUNTER,
        SPARSE_VERIFIER_FALLBACK_COUNTER,
        "mistralrs_prefix_cache_lookups_total",
        "mistralrs_prefix_cache_hits_total",
        "mistralrs_prefix_cache_tokens_matched_total",
        "mistralrs_prefix_cache_tokens_reused_total",
        "mistralrs_prefix_cache_evictions_total",
        "mistralrs_paged_preemptions_total",
        "mistralrs_speculative_staged_drops_total",
        "mistralrs_encoder_cache_hits_total",
        "mistralrs_encoder_cache_misses_total",
        CUDA_GRAPH_EVENTS_COUNTER,
        CUDA_GRAPH_DISPATCH_COUNTER,
        CUDA_GRAPH_EVICTIONS_COUNTER,
        CUDA_MEMORY_MAINTENANCE_COUNTER,
        CUDA_MEMORY_PRESSURE_COUNTER,
        CUDA_MEMORY_RECLAIMED_BYTES_COUNTER,
        CUDA_PROMPT_BATCH_REDUCTIONS_COUNTER,
        CUDA_PROMPT_SEQUENCES_DEFERRED_COUNTER,
        CUDA_PROMPT_MEMORY_REJECTIONS_COUNTER,
        REQUEST_OUTCOMES_COUNTER,
        "mistralrs_sequences_completed_total",
        KV_CACHE_ACTIVE_GAUGE,
        KV_CACHE_PREFIX_CACHED_GAUGE,
        DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
        DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
    )
    return {name: metric_delta(before, after, name) for name in names}


def labeled_metric_deltas(
    before: dict[str, float], after: dict[str, float], metric: str
) -> list[dict[str, Any]]:
    keys = {
        key
        for key in set(before) | set(after)
        if key.split("{", 1)[0] == metric
    }
    results = []
    for key in sorted(keys):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', key))
        results.append(
            {
                "labels": labels,
                "delta": after.get(key, 0.0) - before.get(key, 0.0),
            }
        )
    return results


def labeled_metric_values(
    snapshot: dict[str, float], metric: str
) -> list[dict[str, Any]]:
    return [
        {
            "labels": dict(re.findall(r'(\w+)="([^"]*)"', key)),
            "value": value,
        }
        for key, value in sorted(snapshot.items())
        if key.split("{", 1)[0] == metric
    ]


def metric_family_present(snapshot: dict[str, float], metric: str) -> bool:
    return any(
        key.split("{", 1)[0] == metric
        for key in snapshot
    )


def cuda_memory_pressure_evidence(
    before: dict[str, float],
    after: dict[str, float],
    require_instrumentation: bool,
) -> dict[str, Any]:
    maintenance = labeled_metric_deltas(
        before,
        after,
        CUDA_MEMORY_MAINTENANCE_COUNTER,
    )
    pressure = labeled_metric_deltas(
        before,
        after,
        CUDA_MEMORY_PRESSURE_COUNTER,
    )
    maintenance_errors = sum(
        item["delta"]
        for item in maintenance
        if item["labels"].get("outcome") == "error"
    )
    counter_series = {
        CUDA_MEMORY_MAINTENANCE_COUNTER: maintenance,
        CUDA_MEMORY_PRESSURE_COUNTER: pressure,
    }
    negative_deltas = [
        {"metric": metric, **item}
        for metric, series in counter_series.items()
        for item in series
        if item["delta"] < 0
    ]
    scalar_deltas = {
        "reclaimed_bytes": metric_delta(
            before,
            after,
            CUDA_MEMORY_RECLAIMED_BYTES_COUNTER,
        ),
        "prompt_batch_reductions": metric_delta(
            before,
            after,
            CUDA_PROMPT_BATCH_REDUCTIONS_COUNTER,
        ),
        "prompt_sequences_deferred": metric_delta(
            before,
            after,
            CUDA_PROMPT_SEQUENCES_DEFERRED_COUNTER,
        ),
        "prompt_memory_rejections": metric_delta(
            before,
            after,
            CUDA_PROMPT_MEMORY_REJECTIONS_COUNTER,
        ),
        "graph_evictions": metric_delta(
            before,
            after,
            CUDA_GRAPH_EVICTIONS_COUNTER,
        ),
    }
    negative_deltas.extend(
        {"metric": name, "delta": value}
        for name, value in scalar_deltas.items()
        if value is not None and value < 0
    )
    reductions = scalar_deltas["prompt_batch_reductions"] or 0.0
    deferred = scalar_deltas["prompt_sequences_deferred"] or 0.0
    rejections = scalar_deltas["prompt_memory_rejections"] or 0.0
    pending = metric_total(after, CUDA_MEMORY_PENDING_GAUGE)
    instrumentation = {
        "maintenance_counter": metric_family_present(
            before, CUDA_MEMORY_MAINTENANCE_COUNTER
        )
        or metric_family_present(after, CUDA_MEMORY_MAINTENANCE_COUNTER),
        "pending_gauge": metric_family_present(after, CUDA_MEMORY_PENDING_GAUGE),
    }
    gates = {
        "instrumentation": (
            not require_instrumentation or all(instrumentation.values())
        ),
        "counter_monotonicity": not negative_deltas,
        "maintenance_errors": maintenance_errors == 0,
        "prompt_memory_rejections": rejections == 0,
        "maintenance_pending_clear": (
            pending == 0
            if instrumentation["pending_gauge"]
            else not require_instrumentation
        ),
        "deferred_sequence_accounting": deferred >= reductions,
    }
    return {
        "passed": all(gates.values()),
        "require_instrumentation": require_instrumentation,
        "instrumentation": instrumentation,
        "gates": gates,
        "maintenance_events": sum(item["delta"] for item in maintenance),
        "maintenance_errors": maintenance_errors,
        "maintenance": maintenance,
        "pressure_events": sum(item["delta"] for item in pressure),
        "pressure": pressure,
        "maintenance_pending": pending,
        "negative_counter_deltas": negative_deltas,
        **scalar_deltas,
    }


def sparse_verifier_evidence(
    before: dict[str, float],
    after: dict[str, float],
    required: bool,
    max_fallback_ratio: float,
    min_accounting_coverage: float,
) -> dict[str, Any]:
    drafts = metric_delta(before, after, "mistralrs_speculative_drafts_total")
    gpu_verified_raw = metric_delta(before, after, SPARSE_VERIFIER_GPU_COUNTER)
    cpu_fallbacks_raw = metric_delta(before, after, SPARSE_VERIFIER_FALLBACK_COUNTER)
    instrumentation_present = gpu_verified_raw is not None or cpu_fallbacks_raw is not None
    gpu_verified = gpu_verified_raw or 0.0
    cpu_fallbacks = cpu_fallbacks_raw or 0.0
    total = gpu_verified + cpu_fallbacks
    fallback_ratio = ratio(cpu_fallbacks, total)
    accounting_coverage = ratio(total, drafts)
    unaccounted = drafts - total if drafts is not None else None
    sane = (
        gpu_verified >= 0
        and cpu_fallbacks >= 0
        and (drafts is None or (drafts >= 0 and total <= drafts))
    )
    observed = total > 0
    fallback_ratio_passed = bool(
        not required
        or (fallback_ratio is not None and fallback_ratio <= max_fallback_ratio)
    )
    accounting_coverage_passed = bool(
        not required
        or (
            accounting_coverage is not None
            and accounting_coverage >= min_accounting_coverage
        )
    )
    return {
        "passed": (
            sane
            and (
                not required
                or (
                    instrumentation_present
                    and observed
                    and fallback_ratio_passed
                    and accounting_coverage_passed
                )
            )
        ),
        "required": required,
        "instrumentation_present": instrumentation_present,
        "observed": observed,
        "gpu_verified_sequences": gpu_verified,
        "cpu_fallback_sequences": cpu_fallbacks,
        "accounted_sequences": total,
        "total_sequences": total,
        "speculative_drafts": drafts,
        "unaccounted_sequences": unaccounted,
        "accounting_coverage": accounting_coverage,
        "minimum_accounting_coverage": min_accounting_coverage,
        "accounting_coverage_passed": accounting_coverage_passed,
        "fallback_ratio": fallback_ratio,
        "maximum_fallback_ratio": max_fallback_ratio,
        "fallback_ratio_passed": fallback_ratio_passed,
    }


def speculative_evidence(
    before: dict[str, float],
    after: dict[str, float],
    required: bool,
    min_acceptance_rate: float = DEFAULT_MIN_MTP_ACCEPTANCE_RATE,
    min_mean_advance: float = DEFAULT_MIN_MTP_MEAN_ADVANCE,
    min_proposal_depth: float = DEFAULT_MIN_MTP_PROPOSAL_DEPTH,
    max_sparse_fallback_ratio: float = DEFAULT_MAX_SPARSE_VERIFIER_FALLBACK_RATIO,
    min_sparse_accounting_coverage: float = (
        DEFAULT_MIN_SPARSE_VERIFIER_ACCOUNTING_COVERAGE
    ),
) -> dict[str, Any]:
    drafts = metric_delta(before, after, "mistralrs_speculative_drafts_total")
    proposed = metric_delta(
        before,
        after,
        "mistralrs_speculative_draft_tokens_proposed_total",
    )
    accepted = metric_delta(
        before,
        after,
        "mistralrs_speculative_draft_tokens_accepted_total",
    )
    per_position = []
    for item in labeled_metric_deltas(
        before,
        after,
        "mistralrs_speculative_draft_tokens_accepted_per_pos_total",
    ):
        raw_position = item["labels"].get("position")
        if raw_position is None:
            continue
        per_position.append(
            {
                "position": int(raw_position),
                "accepted_sequences": item["delta"],
                "survival_rate": ratio(item["delta"], drafts),
            }
        )
    per_position.sort(key=lambda item: item["position"])
    absent = drafts is None and proposed is None and accepted is None
    present = drafts is not None and proposed is not None and accepted is not None
    sane = (
        present
        and drafts >= 0
        and proposed >= 0
        and accepted >= 0
        and accepted <= proposed
        and all(item["accepted_sequences"] >= 0 for item in per_position)
        and all(
            left["accepted_sequences"] >= right["accepted_sequences"]
            for left, right in zip(per_position, per_position[1:])
        )
    )
    active = bool(drafts and proposed)
    acceptance_rate = ratio(accepted, proposed)
    proposal_depth = ratio(proposed, drafts)
    mean_accepted = ratio(accepted, drafts)
    mean_advance = 1.0 + mean_accepted if mean_accepted is not None else None
    performance_floors_passed = bool(
        not required
        or (
            acceptance_rate is not None
            and acceptance_rate >= min_acceptance_rate
            and mean_advance is not None
            and mean_advance >= min_mean_advance
            and proposal_depth is not None
            and proposal_depth >= min_proposal_depth
        )
    )
    sparse_verifier = sparse_verifier_evidence(
        before,
        after,
        required,
        max_sparse_fallback_ratio,
        min_sparse_accounting_coverage,
    )
    base_passed = bool((sane and (active or not required)) or (absent and not required))
    return {
        "passed": base_passed and performance_floors_passed and sparse_verifier["passed"],
        "required": required,
        "active": active,
        "base_passed": base_passed,
        "drafts": drafts,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "acceptance_rate": acceptance_rate,
        "mean_proposed_draft_tokens_per_draft": proposal_depth,
        "mean_accepted_draft_tokens_per_draft": mean_accepted,
        "mean_advance_tokens_per_target_step": mean_advance,
        "minimum_acceptance_rate": min_acceptance_rate,
        "minimum_mean_advance_tokens_per_target_step": min_mean_advance,
        "minimum_mean_proposed_draft_tokens_per_draft": min_proposal_depth,
        "performance_floors_passed": performance_floors_passed,
        "sparse_verifier": sparse_verifier,
        "accepted_per_position": per_position,
    }


def configured_speculative_evidence(
    before: dict[str, float],
    after: dict[str, float],
    args: argparse.Namespace,
    required: bool,
) -> dict[str, Any]:
    return speculative_evidence(
        before,
        after,
        required,
        args.min_mtp_acceptance_rate,
        args.min_mtp_mean_advance,
        args.min_mtp_proposal_depth,
        args.max_sparse_verifier_fallback_ratio,
        args.min_sparse_verifier_accounting_coverage,
    )


def target_only_speculative_evidence(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, Any]:
    evidence = speculative_evidence(before, after, False)
    present = all(
        evidence[name] is not None
        for name in ("drafts", "proposed_draft_tokens", "accepted_draft_tokens")
    )
    absent = all(
        evidence[name] is None
        for name in ("drafts", "proposed_draft_tokens", "accepted_draft_tokens")
    )
    inactive = absent or (
        present
        and all(
            evidence[name] == 0
            for name in ("drafts", "proposed_draft_tokens", "accepted_draft_tokens")
        )
    )
    return {
        **evidence,
        "passed": inactive,
        "instrumentation_present": present,
        "target_only_inactive": inactive,
        "inactive_evidence": "metrics_absent" if absent else "zero_deltas",
    }


def cuda_graph_evidence(
    before: dict[str, float],
    after: dict[str, float],
    startup: dict[str, float],
    expected_components: Sequence[str],
    min_replay_ratio: float,
    allowed_skip_reasons: Sequence[str] = DEFAULT_ALLOWED_CUDA_GRAPH_SKIP_REASONS,
) -> dict[str, Any]:
    dispatch_deltas = labeled_metric_deltas(
        before,
        after,
        CUDA_GRAPH_DISPATCH_COUNTER,
    )
    event_deltas = labeled_metric_deltas(
        before,
        after,
        CUDA_GRAPH_EVENTS_COUNTER,
    )
    failure_events = [
        item
        for item in event_deltas
        if item["labels"].get("outcome") == "failure"
        and item["delta"] > 0
    ]
    startup_failure_events = [
        item
        for item in labeled_metric_values(
            startup,
            CUDA_GRAPH_EVENTS_COUNTER,
        )
        if item["labels"].get("outcome") == "failure"
        and item["value"] > 0
    ]
    allowed_skip_reasons = set(allowed_skip_reasons)
    components = {}
    for component in expected_components:
        replay = sum(
            item["delta"]
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "replay"
        )
        eager = sum(
            item["delta"]
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "eager"
        )
        cache_population_eager = sum(
            item["delta"]
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "eager"
            and item["labels"].get("reason") == CUDA_GRAPH_CACHE_POPULATION_REASON
        )
        successful_captures = sum(
            item["delta"]
            for item in event_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("event") == "capture"
            and item["labels"].get("outcome") == "success"
        )
        accounted_cache_population = min(
            cache_population_eager,
            successful_captures,
        )
        unexpected_eager = eager - accounted_cache_population
        skipped = sum(
            item["delta"]
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "skipped"
        )
        allowed_skips = [
            item
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "skipped"
            and item["labels"].get("reason") in allowed_skip_reasons
            and item["delta"] > 0
        ]
        unexpected_skips = [
            item
            for item in dispatch_deltas
            if item["labels"].get("component") == component
            and item["labels"].get("mode") == "skipped"
            and item["labels"].get("reason") not in allowed_skip_reasons
            and item["delta"] > 0
        ]
        failures = sum(
            item["delta"]
            for item in failure_events
            if item["labels"].get("component") == component
        )
        startup_failures = sum(
            item["value"]
            for item in startup_failure_events
            if item["labels"].get("component") == component
        )
        eligible = replay + unexpected_eager
        replay_ratio = ratio(replay, eligible)
        dispatch_instrumentation_present = any(
            item["labels"].get("component") == component
            for item in dispatch_deltas
        )
        event_instrumentation_present = any(
            item["labels"].get("component") == component
            for item in event_deltas
        ) or any(
            item["labels"].get("component") == component
            for item in startup_failure_events
        )
        instrumentation_complete = (
            dispatch_instrumentation_present and event_instrumentation_present
        )
        components[component] = {
            "passed": (
                instrumentation_complete
                and eligible > 0
                and replay_ratio is not None
                and replay_ratio >= min_replay_ratio
                and unexpected_eager == 0
                and not unexpected_skips
                and failures == 0
                and startup_failures == 0
            ),
            "instrumentation_complete": instrumentation_complete,
            "dispatch_instrumentation_present": dispatch_instrumentation_present,
            "event_instrumentation_present": event_instrumentation_present,
            "eligible_dispatches": eligible,
            "replay_dispatches": replay,
            "eager_dispatches": eager,
            "cache_population_eager_dispatches": cache_population_eager,
            "successful_capture_events": successful_captures,
            "accounted_cache_population_dispatches": accounted_cache_population,
            "unexpected_eager_dispatches": unexpected_eager,
            "skipped_dispatches": skipped,
            "allowed_skipped_dispatches": sum(item["delta"] for item in allowed_skips),
            "unexpected_skipped_dispatches": sum(
                item["delta"] for item in unexpected_skips
            ),
            "allowed_skips": allowed_skips,
            "unexpected_skips": unexpected_skips,
            "eligible_replay_ratio": replay_ratio,
            "phase_failures": failures,
            "startup_cumulative_failures": startup_failures,
        }
    return {
        "passed": bool(components) and all(item["passed"] for item in components.values()),
        "expected_components": list(expected_components),
        "min_replay_ratio": min_replay_ratio,
        "components": components,
        "dispatch": dispatch_deltas,
        "event_deltas": event_deltas,
        "events": failure_events,
        "startup_failure_events": startup_failure_events,
        "allowed_skip_reasons": sorted(allowed_skip_reasons),
        "instrumentation_complete": bool(components)
        and all(item["instrumentation_complete"] for item in components.values()),
    }


def metric_values(
    snapshot: dict[str, float], names: Sequence[str]
) -> dict[str, float | None]:
    return {name: metric_total(snapshot, name) for name in names}


def cacheable_prefix_tokens(prompt_tokens: int, block_size_tokens: int) -> int:
    if prompt_tokens <= 1:
        return 0
    return (prompt_tokens - 1) // block_size_tokens * block_size_tokens


def eligible_prefix_reuse_tokens(
    prompt_tokens: int,
    block_size_tokens: int,
    replay_tokens: int,
) -> int:
    cached = cacheable_prefix_tokens(prompt_tokens, block_size_tokens)
    retained = max(0, cached - replay_tokens)
    return retained // block_size_tokens * block_size_tokens


def prefix_cache_evidence(
    before: dict[str, float],
    after: dict[str, float],
    expected_prompt_tokens: Sequence[int],
    min_match_fraction: float,
    block_size_tokens: int,
    replay_tokens_per_request: int,
) -> dict[str, Any]:
    prompt_tokens = list(expected_prompt_tokens)
    expected_requests = len(prompt_tokens)
    lookups = metric_delta(before, after, "mistralrs_prefix_cache_lookups_total")
    hits = metric_delta(before, after, "mistralrs_prefix_cache_hits_total")
    matched_tokens = metric_delta(
        before,
        after,
        "mistralrs_prefix_cache_tokens_matched_total",
    )
    reused_tokens = metric_delta(
        before,
        after,
        "mistralrs_prefix_cache_tokens_reused_total",
    )
    expected_cacheable_tokens = sum(
        cacheable_prefix_tokens(tokens, block_size_tokens) for tokens in prompt_tokens
    )
    expected_eligible_reuse_tokens = sum(
        eligible_prefix_reuse_tokens(
            tokens,
            block_size_tokens,
            replay_tokens_per_request,
        )
        for tokens in prompt_tokens
    )
    expected_reusable_requests = sum(
        eligible_prefix_reuse_tokens(
            tokens,
            block_size_tokens,
            replay_tokens_per_request,
        )
        > 0
        for tokens in prompt_tokens
    )
    minimum_matched_tokens = expected_cacheable_tokens * min_match_fraction
    minimum_reused_tokens = expected_eligible_reuse_tokens * min_match_fraction
    matched_prompt_fraction = ratio(matched_tokens, expected_cacheable_tokens)
    eligible_reuse_fraction = (
        1.0
        if expected_eligible_reuse_tokens == 0 and reused_tokens == 0
        else ratio(reused_tokens, expected_eligible_reuse_tokens)
    )
    reuse_contract_non_vacuous = (
        expected_requests > 0
        and expected_cacheable_tokens > 0
        and expected_eligible_reuse_tokens > 0
        and expected_reusable_requests > 0
    )
    return {
        "passed": (
            reuse_contract_non_vacuous
            and lookups is not None
            and hits is not None
            and matched_tokens is not None
            and reused_tokens is not None
            and lookups >= expected_requests
            and hits >= expected_reusable_requests
            and matched_tokens >= minimum_matched_tokens
            and reused_tokens >= minimum_reused_tokens
            and matched_prompt_fraction is not None
            and matched_prompt_fraction >= min_match_fraction
            and eligible_reuse_fraction is not None
            and eligible_reuse_fraction >= min_match_fraction
        ),
        "expected_requests": expected_requests,
        "expected_prompt_tokens": prompt_tokens,
        "expected_cacheable_tokens": expected_cacheable_tokens,
        "expected_eligible_reuse_tokens": expected_eligible_reuse_tokens,
        "expected_reusable_requests": expected_reusable_requests,
        "reuse_contract_non_vacuous": reuse_contract_non_vacuous,
        "block_size_tokens": block_size_tokens,
        "replay_tokens_per_request": replay_tokens_per_request,
        "minimum_matched_tokens": minimum_matched_tokens,
        "minimum_reused_tokens": minimum_reused_tokens,
        "min_match_fraction": min_match_fraction,
        "lookups": lookups,
        "hits": hits,
        "hit_rate": ratio(hits, lookups),
        "matched_tokens": matched_tokens,
        "reused_tokens": reused_tokens,
        "matched_prompt_fraction": matched_prompt_fraction,
        "eligible_reuse_fraction": eligible_reuse_fraction,
        "matched_reuse_fraction": ratio(reused_tokens, matched_tokens),
    }


async def resident_decode_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    await calibrate_prompt_profiles(client, writer, (CONTEXT_PROMPT_PROFILE,))
    prompts = {
        length: exact_context(client, length, f"resident-decode-{length}")
        for length in args.context_lengths
    }
    initial_metrics = await safe_metrics(client, writer, "resident-decode-start")

    warmup_specs = [
        RequestSpec(
            case_id=f"resident-warm-{length}",
            seed=args.seed + 100_000 + index,
            max_tokens=args.warmup_max_tokens,
            prompt=prompts[length],
            context_tokens=length,
            tags={"scenario": "resident_decode", "stage": "warm", "length": length},
            extra={"ignore_eos": True},
        )
        for index, length in enumerate(args.context_lengths)
    ]
    warmup_results, warmup_summary = await run_batch(
        client,
        warmup_specs,
        1,
        writer,
        "resident-decode-warm",
        keep_output=False,
    )
    warm_cleanup_ok, post_warm_metrics, warm_cleanup = await poll_for_cleanup(
        client,
        writer,
        initial_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "resident-decode-warm-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )

    verification_specs = [
        RequestSpec(
            case_id=f"resident-verify-{length}",
            seed=args.seed + 200_000 + index,
            max_tokens=args.warmup_max_tokens,
            prompt=prompts[length],
            context_tokens=length,
            tags={
                "scenario": "resident_decode",
                "stage": "verify",
                "length": length,
            },
            extra={"ignore_eos": True},
        )
        for index, length in enumerate(args.context_lengths)
    ]
    verification_results, verification_summary = await run_batch(
        client,
        verification_specs,
        1,
        writer,
        "resident-decode-verify",
        keep_output=False,
    )
    verification_cleanup_ok, resident_metrics, verification_cleanup = (
        await poll_for_cleanup(
            client,
            writer,
            post_warm_metrics,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            "resident-decode-verify-cleanup",
            RESIDENT_TRANSIENT_CLEANUP_GAUGES,
        )
    )
    verification_prefix = prefix_cache_evidence(
        post_warm_metrics,
        resident_metrics,
        [int(spec.context_tokens or 0) for spec in verification_specs],
        args.min_prefix_reuse_fraction,
        args.kv_block_size_tokens,
        args.speculative_prefix_replay_tokens,
    )
    residency_passed = (
        warmup_summary["errors"] == 0
        and verification_summary["errors"] == 0
        and warm_cleanup_ok
        and verification_cleanup_ok
        and verification_prefix["passed"]
    )
    residency = {
        "passed": residency_passed,
        "warmup": warmup_summary,
        "verification": verification_summary,
        "prefix_cache": verification_prefix,
        "warm_cleanup_ok": warm_cleanup_ok,
        "warm_cleanup": warm_cleanup,
        "verification_cleanup_ok": verification_cleanup_ok,
        "verification_cleanup": verification_cleanup,
        "warmup_prompt_tokens": [result.prompt_tokens for result in warmup_results],
        "verification_prompt_tokens": [
            result.prompt_tokens for result in verification_results
        ],
        "cleanup_gauges": {
            "before_warm": metric_values(initial_metrics, CLEANUP_GAUGES),
            "resident": metric_values(resident_metrics, CLEANUP_GAUGES),
        },
    }
    await writer.emit("resident_decode_residency", **residency)
    if not residency_passed:
        return {
            "mode": "resident-decode",
            "passed": False,
            "residency": residency,
            "phase_summaries": [],
            "exact_replay_cohorts": [],
            "exact_replays": [],
            "ordering_comparisons": [],
            "cross_concurrency_comparisons": [],
            "cross_concurrency_cohort": None,
            "fixed_seed_invariance": None,
            "final_c1_replay": None,
        }

    resident_lengths = (
        args.context_lengths * math.ceil(args.requests / len(args.context_lengths))
    )[: args.requests]
    specs = [
        RequestSpec(
            case_id=f"resident-decode-{index:03d}",
            seed=args.seed + index,
            max_tokens=args.max_tokens,
            prompt=prompts[length],
            context_tokens=length,
            tags={
                "scenario": "resident_decode",
                "stage": "measure",
                "length": length,
            },
            extra={"ignore_eos": True},
        )
        for index, length in enumerate(resident_lengths)
    ]
    common_specs = common_full_batch_specs(specs, args.concurrencies)
    cross_concurrency_cohort = common_full_batch_cohort_evidence(
        specs,
        common_specs,
        args.concurrencies,
    )
    await writer.emit(
        "resident_decode_cross_concurrency_cohort", **cross_concurrency_cohort
    )
    phase_runs: list[tuple[str, list[RequestResult], dict[str, Any]]] = []

    async def run_resident_batch(
        concurrency: int,
        phase: str,
        batch_specs: Sequence[RequestSpec],
        order: str,
        runtime_gated: bool = True,
    ) -> None:
        before = await safe_metrics(client, writer, f"{phase}-start")
        results, summary = await run_batch(
            client,
            batch_specs,
            concurrency,
            writer,
            phase,
            keep_output=True,
        )
        cleanup_ok, after, cleanup = await poll_for_cleanup(
            client,
            writer,
            before,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            f"{phase}-cleanup",
            RESIDENT_TRANSIENT_CLEANUP_GAUGES,
        )
        prefix = prefix_cache_evidence(
            before,
            after,
            [int(spec.context_tokens or 0) for spec in batch_specs],
            args.min_prefix_reuse_fraction,
            args.kv_block_size_tokens,
            args.speculative_prefix_replay_tokens,
        )
        mtp = configured_speculative_evidence(
            before,
            after,
            args,
            args.require_mtp,
        )
        graph = cuda_graph_evidence(
            before,
            after,
            initial_metrics,
            args.expected_graph_components,
            args.min_cuda_graph_replay_ratio,
        )
        quality = []
        for result in results:
            valid, detail = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {
                "phase": phase,
                "valid": valid,
                "requested_prompt_tokens": result.context_tokens,
                "server_prompt_tokens": result.prompt_tokens,
                "prompt_tokens_match": result.prompt_tokens == result.context_tokens,
                **detail,
            }
            quality.append(check)
            await writer.emit("resident_decode_quality", **check)
        summary.update(
            {
                "order": order,
                "runtime_gated": runtime_gated,
                "passed": (
                    summary["errors"] == 0
                    and cleanup_ok
                    and prefix["passed"]
                    and (
                        not runtime_gated
                        or (mtp["passed"] and graph["passed"])
                    )
                    and all(item["valid"] for item in quality)
                ),
                "prefix_cache": prefix,
                "mtp": mtp,
                "cuda_graph": graph,
                "cuda_graph_events": graph["events"],
                "metric_deltas": selected_metric_deltas(before, after),
                "cleanup_ok": cleanup_ok,
                "cleanup": cleanup,
                "cleanup_gauges": {
                    "before": metric_values(before, CLEANUP_GAUGES),
                    "after": metric_values(after, CLEANUP_GAUGES),
                },
                "quality": quality,
            }
        )
        phase_runs.append((phase, results, summary))
        await writer.emit("resident_decode_phase_summary", phase=phase, **summary)

    normal_phase_runs = []
    exact_replay_cohorts = []
    exact_replays = []
    ordering_comparisons = []
    for concurrency in args.concurrencies:
        measured_specs = full_batch_specs(specs, concurrency)
        await run_resident_batch(
            concurrency,
            f"resident-decode-c{concurrency}-stabilize",
            measured_specs,
            "stabilize",
            False,
        )
        normal_phase = f"resident-decode-c{concurrency}"
        await run_resident_batch(concurrency, normal_phase, measured_specs, "normal")
        normal_run = phase_runs[-1]
        normal_phase_runs.append(normal_run)

        exact_specs, exact_cohort = resident_exact_replay_cohort(
            measured_specs,
            concurrency,
        )
        exact_replay_cohorts.append(exact_cohort)
        await writer.emit(
            "resident_decode_exact_replay_cohort",
            phase=normal_phase,
            **exact_cohort,
        )
        if exact_cohort["normal_phase_reusable"]:
            exact_baseline_phase = normal_phase
            exact_baseline_results = results_for_specs(normal_run[1], exact_specs)
            exact_baseline_summary = normal_run[2]
        else:
            exact_baseline_phase = f"resident-decode-c{concurrency}-exact-baseline"
            await run_resident_batch(
                concurrency,
                exact_baseline_phase,
                exact_specs,
                "exact-baseline",
            )
            exact_baseline_results = phase_runs[-1][1]
            exact_baseline_summary = phase_runs[-1][2]

        replay_phase = f"resident-decode-c{concurrency}-exact-replay"
        await run_resident_batch(
            concurrency,
            replay_phase,
            exact_specs,
            "exact-replay",
        )
        replay_results = phase_runs[-1][1]
        replay_summary = phase_runs[-1][2]
        exact = exact_output_diagnostics(
            exact_baseline_results,
            replay_results,
            exact_baseline_phase,
            replay_phase,
            exact_specs,
        )
        statistical = compare_samples(
            replay_results,
            exact_baseline_results,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        semantic_passed = all(
            item["valid"]
            for item in exact_baseline_summary["quality"] + replay_summary["quality"]
        )
        fixed_seed_replay = fixed_seed_comparison_evidence(
            exact,
            statistical,
            semantic_passed,
        )
        replay_diagnostics = {
            "concurrency": concurrency,
            "cohort": exact_cohort,
            **fixed_seed_replay,
            "passed": exact_cohort["complete"] and fixed_seed_replay["passed"],
        }
        exact_replays.append(replay_diagnostics)
        await writer.emit("exact_replay_comparison", **replay_diagnostics)

        reverse_phase = f"resident-decode-c{concurrency}-reverse"
        await run_resident_batch(
            concurrency,
            reverse_phase,
            correctness_order_specs(measured_specs, True),
            "reverse",
        )
        reverse_results = phase_runs[-1][1]
        reverse_summary = phase_runs[-1][2]
        normal_ordering_results = results_for_specs(normal_run[1], measured_specs)
        exact = exact_output_diagnostics(
            normal_ordering_results,
            reverse_results,
            normal_phase,
            reverse_phase,
            measured_specs,
        )
        statistical = compare_samples(
            reverse_results,
            normal_ordering_results,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        semantic_passed = all(
            item["valid"]
            for item in normal_run[2]["quality"] + reverse_summary["quality"]
        )
        ordering_diagnostics = {
            "concurrency": concurrency,
            **fixed_seed_comparison_evidence(
                exact,
                statistical,
                semantic_passed,
                exact_gated=False,
            ),
        }
        ordering_comparisons.append(ordering_diagnostics)
        await writer.emit("exact_ordering_comparison", **ordering_diagnostics)

    baseline_phase, baseline_results, baseline_summary = normal_phase_runs[0]
    baseline_common_results = results_for_specs(baseline_results, common_specs)
    cross_concurrency_comparisons = []
    for phase, results, phase_summary in normal_phase_runs[1:]:
        common_results = results_for_specs(results, common_specs)
        semantic_passed = all(
            item["valid"]
            for item in baseline_summary["quality"] + phase_summary["quality"]
        )
        comparison = {
            "phase": phase,
            "cohort": cross_concurrency_cohort,
            **fixed_seed_comparison_evidence(
                exact_output_diagnostics(
                    baseline_common_results,
                    common_results,
                    baseline_phase,
                    phase,
                    common_specs,
                ),
                compare_samples(
                    common_results,
                    baseline_common_results,
                    client.tokenizer,
                    args.stat_max_ks,
                    args.stat_max_js,
                ),
                semantic_passed,
                exact_gated=False,
            ),
        }
        cross_concurrency_comparisons.append(comparison)
        await writer.emit("resident_decode_cross_concurrency", **comparison)

    fixed_seed_invariance = fixed_seed_invariance_evidence(
        exact_replays,
        ordering_comparisons,
        cross_concurrency_comparisons,
        len(args.concurrencies),
    )
    await writer.emit("resident_decode_fixed_seed_invariance", **fixed_seed_invariance)

    final_c1_replay = None
    if args.final_c1_replay and 1 in args.concurrencies:
        await run_resident_batch(
            1,
            "resident-decode-c1-final",
            specs,
            "final-replay",
        )
        final_phase, final_results, final_summary = phase_runs[-1]
        initial_c1 = next(
            results
            for phase, results, _ in phase_runs
            if phase == "resident-decode-c1"
        )
        exact = exact_output_diagnostics(
            initial_c1,
            final_results,
            "resident-decode-c1",
            final_phase,
            specs,
        )
        initial_c1_summary = next(
            summary
            for phase, _, summary in phase_runs
            if phase == "resident-decode-c1"
        )
        decode_throughput_ratio = ratio(
            final_summary["decode_tok_s_active"],
            initial_c1_summary["decode_tok_s_active"],
        )
        common_wall_throughput_ratio = ratio(
            final_summary["output_tok_s_common_wall"],
            initial_c1_summary["output_tok_s_common_wall"],
        )
        statistical = compare_samples(
            final_results,
            initial_c1,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        semantic_passed = all(
            item["valid"]
            for item in initial_c1_summary["quality"] + final_summary["quality"]
        )
        fixed_seed_replay = fixed_seed_comparison_evidence(
            exact,
            statistical,
            semantic_passed,
        )
        final_c1_replay = {
            **fixed_seed_replay,
            "passed": (
                final_summary["passed"]
                and fixed_seed_replay["passed"]
                and decode_throughput_ratio is not None
                and decode_throughput_ratio >= args.min_final_c1_throughput_ratio
            ),
            "throughput_metric": "decode_tok_s_active",
            "initial_decode_tok_s_active": initial_c1_summary["decode_tok_s_active"],
            "final_decode_tok_s_active": final_summary["decode_tok_s_active"],
            "initial_output_tok_s": initial_c1_summary["output_tok_s_common_wall"],
            "final_output_tok_s": final_summary["output_tok_s_common_wall"],
            "throughput_ratio": decode_throughput_ratio,
            "decode_throughput_ratio": decode_throughput_ratio,
            "common_wall_throughput_ratio": common_wall_throughput_ratio,
            "min_throughput_ratio": args.min_final_c1_throughput_ratio,
        }
        await writer.emit("resident_decode_c1_replay", **final_c1_replay)

    final_metrics = await safe_metrics(client, writer, "resident-decode-end")
    phase_summaries = [summary for _, _, summary in phase_runs]
    normal_phase_summaries = [summary for _, _, summary in normal_phase_runs]
    throughput = throughput_evidence(
        normal_phase_summaries,
        args.min_output_tok_s_by_concurrency,
        args.min_scaling_efficiency,
        "decode_tok_s_active",
    )
    await writer.emit("resident_decode_throughput_evidence", **throughput)
    passed = (
        residency_passed
        and throughput["passed"]
        and cross_concurrency_cohort["complete"]
        and fixed_seed_invariance["passed"]
        and all(summary["passed"] for summary in phase_summaries)
        and (final_c1_replay is None or final_c1_replay["passed"])
    )
    return {
        "mode": "resident-decode",
        "passed": passed,
        "production_sampling": asdict(client.policy),
        "context_lengths": list(args.context_lengths),
        "concurrencies": list(args.concurrencies),
        "requests_per_phase": len(specs),
        "residency": residency,
        "phase_summaries": phase_summaries,
        "exact_replay_cohorts": exact_replay_cohorts,
        "exact_replays": exact_replays,
        "ordering_comparisons": ordering_comparisons,
        "cross_concurrency_comparisons": cross_concurrency_comparisons,
        "cross_concurrency_cohort": cross_concurrency_cohort,
        "fixed_seed_invariance": fixed_seed_invariance,
        "throughput": throughput,
        "final_c1_replay": final_c1_replay,
        "metrics_delta": selected_metric_deltas(initial_metrics, final_metrics),
        "cleanup_gauges": metric_values(final_metrics, CLEANUP_GAUGES),
    }


def histogram_quantile_delta(
    before: dict[str, float],
    after: dict[str, float],
    histogram: str,
    quantile: float,
) -> float | None:
    buckets: dict[float, float] = Counter()
    bucket_name = f"{histogram}_bucket"
    for key, end_value in after.items():
        if key.split("{", 1)[0] != bucket_name:
            continue
        match = re.search(r'(?:\{|,)le="([^\"]+)"(?:,|})', key)
        if not match:
            continue
        raw_bound = match.group(1)
        bound = math.inf if raw_bound in ("+Inf", "Inf") else float(raw_bound)
        buckets[bound] += end_value - before.get(key, 0.0)
    if not buckets:
        return None
    total = buckets.get(math.inf, max(buckets.values()))
    if total <= 0:
        return None
    wanted = total * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for bound, count in sorted(buckets.items()):
        if count < wanted:
            previous_bound = bound
            previous_count = count
            continue
        if math.isinf(bound):
            return previous_bound
        bucket_count = count - previous_count
        if bucket_count <= 0:
            return bound
        fraction = (wanted - previous_count) / bucket_count
        return previous_bound + fraction * (bound - previous_bound)
    return None


def queue_histogram_summaries(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    histograms = set()
    for key in after:
        base_name = key.split("{", 1)[0]
        if base_name.endswith("_bucket") and any(
            term in base_name.lower() for term in ("queue", "admission", "waiting")
        ):
            histograms.add(base_name[:-7])
    return {
        name: {
            "p50": histogram_quantile_delta(before, after, name, 0.50),
            "p95": histogram_quantile_delta(before, after, name, 0.95),
            "p99": histogram_quantile_delta(before, after, name, 0.99),
        }
        for name in sorted(histograms)
    }


def histogram_observation_delta(
    before: dict[str, float],
    after: dict[str, float],
    histogram: str,
) -> float | None:
    count_name = f"{histogram}_count"
    end_count = metric_total(after, count_name)
    if end_count is not None:
        return end_count - (metric_total(before, count_name) or 0.0)
    for bound in ("+Inf", "Inf"):
        bucket = f'{histogram}_bucket{{le="{bound}"}}'
        end_bucket = metric_total(after, bucket)
        if end_bucket is not None:
            return end_bucket - (metric_total(before, bucket) or 0.0)
    return None


def queue_histogram_evidence(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, Any]:
    histograms = {}
    for name, quantiles in queue_histogram_summaries(before, after).items():
        observation_delta = histogram_observation_delta(before, after, name)
        quantiles_complete = all(
            value is not None and math.isfinite(value) and value >= 0
            for value in quantiles.values()
        )
        histograms[name] = {
            **quantiles,
            "observation_delta": observation_delta,
            "positive_observation_delta": (
                observation_delta is not None and observation_delta > 0
            ),
            "quantiles_complete": quantiles_complete,
            "passed": (
                observation_delta is not None
                and observation_delta > 0
                and quantiles_complete
            ),
        }
    return {
        "passed": bool(histograms) and all(
            item["passed"] for item in histograms.values()
        ),
        "histogram_count": len(histograms),
        "histograms": histograms,
    }


def parse_host_cpu_ticks(value: str) -> tuple[int, int]:
    row = next(
        (line for line in value.splitlines() if line.startswith("cpu ")),
        None,
    )
    if row is None:
        raise ValueError("/proc/stat does not contain an aggregate cpu row")
    fields = [int(field) for field in row.split()[1:]]
    if len(fields) < 4:
        raise ValueError("/proc/stat aggregate cpu row is incomplete")
    idle_ticks = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields[:8]), idle_ticks


def parse_process_cpu_ticks(value: str) -> int:
    command_end = value.rfind(")")
    if command_end < 0:
        raise ValueError("/proc process stat does not contain a command field")
    fields = value[command_end + 1 :].split()
    if len(fields) <= 12:
        raise ValueError("/proc process stat is incomplete")
    return int(fields[11]) + int(fields[12])


def cpu_utilization_samples(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
) -> tuple[list[float], list[float]]:
    host_samples = []
    process_samples = []
    for (previous_time, _, previous), (current_time, _, current) in zip(
        snapshots,
        snapshots[1:],
    ):
        elapsed = current_time - previous_time
        previous_total = previous.get("host_cpu_total_ticks")
        current_total = current.get("host_cpu_total_ticks")
        previous_idle = previous.get("host_cpu_idle_ticks")
        current_idle = current.get("host_cpu_idle_ticks")
        if None not in (previous_total, current_total, previous_idle, current_idle):
            total_delta = current_total - previous_total
            idle_delta = current_idle - previous_idle
            if total_delta > 0 and 0 <= idle_delta <= total_delta:
                host_samples.append(100.0 * (total_delta - idle_delta) / total_delta)
        previous_ticks = previous.get("process_cpu_ticks")
        current_ticks = current.get("process_cpu_ticks")
        ticks_per_second = current.get("process_cpu_clock_ticks_per_second")
        if (
            elapsed > 0
            and previous_ticks is not None
            and current_ticks is not None
            and ticks_per_second is not None
            and ticks_per_second > 0
            and current_ticks >= previous_ticks
        ):
            process_samples.append(
                100.0
                * (current_ticks - previous_ticks)
                / (ticks_per_second * elapsed)
            )
    return host_samples, process_samples


def summarize_telemetry(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]]
) -> dict[str, Any]:
    gauge_names = (
        "mistralrs_sequences_running",
        "mistralrs_sequences_waiting",
        "mistralrs_sequences_capacity",
        "mistralrs_requests_pending_admission",
        "mistralrs_kv_cache_blocks_used",
        KV_CACHE_ACTIVE_GAUGE,
        KV_CACHE_PREFIX_CACHED_GAUGE,
        "mistralrs_kv_cache_blocks_total",
        "mistralrs_recurrent_state_slots_used",
        "mistralrs_recurrent_state_slots_total",
        DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
        DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE,
        DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
        DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
    )
    gauges = {
        name: distribution(
            value
            for _, metrics, _ in snapshots
            if (value := metric_total(metrics, name)) is not None
        )
        for name in gauge_names
    }
    gpu_utilization = []
    gpu_memory = []
    gpu_power = []
    process_rss = []
    process_vmsize = []
    process_vmswap = []
    process_gpu_memory = []
    for _, _, process in snapshots:
        for gpu in process.get("gpus") or []:
            gpu_utilization.append(gpu["utilization_percent"])
            gpu_memory.append(gpu["memory_used_mib"])
            gpu_power.append(gpu["power_watts"])
        if process.get("process_vmrss_kib") is not None:
            process_rss.append(process["process_vmrss_kib"])
        if process.get("process_vmsize_kib") is not None:
            process_vmsize.append(process["process_vmsize_kib"])
        if process.get("process_vmswap_kib") is not None:
            process_vmswap.append(process["process_vmswap_kib"])
        if process.get("process_gpu_memory_used_mib") is not None:
            process_gpu_memory.append(process["process_gpu_memory_used_mib"])
    host_cpu, process_cpu = cpu_utilization_samples(snapshots)
    return {
        "gauges": gauges,
        "host_cpu_utilization_percent": distribution(host_cpu),
        "process_cpu_utilization_percent": distribution(process_cpu),
        "gpu_utilization_percent": distribution(gpu_utilization),
        "gpu_memory_used_mib": distribution(gpu_memory),
        "gpu_power_watts": distribution(gpu_power),
        "process_rss_kib": distribution(process_rss),
        "process_vmsize_kib": distribution(process_vmsize),
        "process_vmswap_kib": distribution(process_vmswap),
        "process_gpu_memory_used_mib": distribution(process_gpu_memory),
    }


def telemetry_evidence(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
    server_pid: int | None,
    min_samples: int,
    min_coverage: float,
    required_gauges: Sequence[str],
    cadence: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(snapshots)

    def coverage(count: int) -> float | None:
        return ratio(count, sample_count)

    metric_samples = sum(bool(metrics) for _, metrics, _ in snapshots)
    gpu_samples = sum(bool(process.get("gpus")) for _, _, process in snapshots)
    rss_samples = sum(
        process.get("process_vmrss_kib") is not None
        for _, _, process in snapshots
    )
    process_gpu_samples = sum(
        process.get("process_gpu_memory_used_mib") is not None
        for _, _, process in snapshots
    )
    server_process_samples = sum(
        process.get("process_is_mistralrs") is True
        for _, _, process in snapshots
    )
    host_cpu_samples = sum(
        process.get("host_cpu_total_ticks") is not None
        and process.get("host_cpu_idle_ticks") is not None
        for _, _, process in snapshots
    )
    process_cpu_samples = sum(
        process.get("process_cpu_ticks") is not None
        and process.get("process_cpu_clock_ticks_per_second") is not None
        for _, _, process in snapshots
    )
    host_cpu_utilization, process_cpu_utilization = cpu_utilization_samples(snapshots)
    cpu_interval_count = max(0, sample_count - 1)
    gauge_coverage = {}
    for gauge in required_gauges:
        present = sum(
            metric_total(metrics, gauge) is not None for _, metrics, _ in snapshots
        )
        value = coverage(present)
        gauge_coverage[gauge] = {
            "samples": present,
            "coverage": value,
            "passed": value is not None and value >= min_coverage,
        }
    metrics_coverage = coverage(metric_samples)
    gpu_coverage = coverage(gpu_samples)
    rss_coverage = coverage(rss_samples)
    process_gpu_coverage = coverage(process_gpu_samples)
    server_process_coverage = coverage(server_process_samples)
    host_cpu_coverage = coverage(host_cpu_samples)
    process_cpu_coverage = coverage(process_cpu_samples)
    host_cpu_utilization_coverage = ratio(
        len(host_cpu_utilization), cpu_interval_count
    )
    process_cpu_utilization_coverage = ratio(
        len(process_cpu_utilization), cpu_interval_count
    )
    return {
        "passed": (
            server_pid is not None
            and sample_count >= min_samples
            and metrics_coverage is not None
            and metrics_coverage >= min_coverage
            and gpu_coverage is not None
            and gpu_coverage >= min_coverage
            and process_gpu_coverage is not None
            and process_gpu_coverage >= min_coverage
            and rss_coverage is not None
            and rss_coverage >= min_coverage
            and server_process_coverage is not None
            and server_process_coverage >= min_coverage
            and host_cpu_coverage is not None
            and host_cpu_coverage >= min_coverage
            and process_cpu_coverage is not None
            and process_cpu_coverage >= min_coverage
            and host_cpu_utilization_coverage is not None
            and host_cpu_utilization_coverage >= min_coverage
            and process_cpu_utilization_coverage is not None
            and process_cpu_utilization_coverage >= min_coverage
            and cadence.get("passed") is True
            and all(item["passed"] for item in gauge_coverage.values())
        ),
        "server_pid": server_pid,
        "samples": sample_count,
        "minimum_samples": min_samples,
        "minimum_coverage": min_coverage,
        "metrics_samples": metric_samples,
        "metrics_coverage": metrics_coverage,
        "nvidia_smi_samples": gpu_samples,
        "nvidia_smi_coverage": gpu_coverage,
        "process_rss_samples": rss_samples,
        "process_rss_coverage": rss_coverage,
        "process_gpu_memory_samples": process_gpu_samples,
        "process_gpu_memory_coverage": process_gpu_coverage,
        "mistralrs_process_samples": server_process_samples,
        "mistralrs_process_coverage": server_process_coverage,
        "host_cpu_samples": host_cpu_samples,
        "host_cpu_coverage": host_cpu_coverage,
        "process_cpu_samples": process_cpu_samples,
        "process_cpu_coverage": process_cpu_coverage,
        "cpu_interval_count": cpu_interval_count,
        "host_cpu_utilization_samples": len(host_cpu_utilization),
        "host_cpu_utilization_coverage": host_cpu_utilization_coverage,
        "process_cpu_utilization_samples": len(process_cpu_utilization),
        "process_cpu_utilization_coverage": process_cpu_utilization_coverage,
        "cadence": cadence,
        "gauge_coverage": gauge_coverage,
    }


def memory_series_evidence(
    values: Sequence[float | None],
    min_coverage: float,
    max_final_growth: float,
    max_high_water_growth: float,
    max_final_growth_fraction: float | None = None,
) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    sample_count = len(values)
    observed_count = len(observed)
    observed_coverage = ratio(observed_count, sample_count)
    initial = values[0] if values else None
    final = values[-1] if values else None
    high_water = max(observed) if observed else None
    final_growth = final - initial if initial is not None and final is not None else None
    high_water_growth = (
        high_water - initial
        if initial is not None and high_water is not None
        else None
    )
    allowed_final_growth = max_final_growth
    if initial is not None and max_final_growth_fraction is not None:
        allowed_final_growth = max(
            allowed_final_growth,
            initial * max_final_growth_fraction,
        )
    return {
        "passed": (
            sample_count > 0
            and observed_coverage is not None
            and observed_coverage >= min_coverage
            and initial is not None
            and final is not None
            and final_growth is not None
            and final_growth <= allowed_final_growth
            and high_water_growth is not None
            and high_water_growth <= max_high_water_growth
        ),
        "samples": observed_count,
        "coverage": observed_coverage,
        "minimum_coverage": min_coverage,
        "initial_mib": initial,
        "final_mib": final,
        "high_water_mib": high_water,
        "final_growth_mib": final_growth,
        "maximum_final_growth_mib": allowed_final_growth,
        "maximum_final_growth_fraction": max_final_growth_fraction,
        "high_water_growth_mib": high_water_growth,
        "maximum_high_water_growth_mib": max_high_water_growth,
    }


def gauge_utilization_evidence(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
    used_gauge: str,
    total_gauge: str,
    min_coverage: float,
    max_utilization: float,
) -> dict[str, Any]:
    samples = []
    for _, metrics, _ in snapshots:
        used = metric_total(metrics, used_gauge)
        total = metric_total(metrics, total_gauge)
        samples.append(
            used / total
            if used is not None and total is not None and total > 0
            else None
        )
    observed = [value for value in samples if value is not None]
    observed_coverage = ratio(len(observed), len(samples))
    high_water = max(observed) if observed else None
    return {
        "passed": (
            bool(samples)
            and observed_coverage is not None
            and observed_coverage >= min_coverage
            and high_water is not None
            and high_water <= max_utilization
        ),
        "used_gauge": used_gauge,
        "total_gauge": total_gauge,
        "samples": len(observed),
        "coverage": observed_coverage,
        "minimum_coverage": min_coverage,
        "initial_utilization": samples[0] if samples else None,
        "final_utilization": samples[-1] if samples else None,
        "maximum_observed_utilization": high_water,
        "maximum_allowed_utilization": max_utilization,
    }


def final_resource_cleanup_evidence(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
    require_dflash_windowed_kv: bool,
    cleanup_gauges: Sequence[str] = CLEANUP_GAUGES,
) -> dict[str, Any]:
    if not snapshots:
        return {
            "passed": False,
            "gauges": {},
            "stable_capacities": {},
        }
    initial_metrics = snapshots[0][1]
    final_metrics = snapshots[-1][1]
    gauges = {}
    cleanup_gauges = tuple(cleanup_gauges)
    stable_capacity_gauges = (
        "mistralrs_kv_cache_blocks_total",
        "mistralrs_recurrent_state_slots_total",
    )
    if require_dflash_windowed_kv:
        cleanup_gauges = (
            *cleanup_gauges,
            DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
            DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
        )
        stable_capacity_gauges = (
            *stable_capacity_gauges,
            DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE,
            DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
        )
    for name in cleanup_gauges:
        initial = metric_total(initial_metrics, name)
        final = metric_total(final_metrics, name)
        gauges[name] = {
            "passed": initial is not None and final is not None and final <= initial,
            "initial": initial,
            "final": final,
        }
    stable_capacities = {}
    for name in stable_capacity_gauges:
        initial = metric_total(initial_metrics, name)
        final = metric_total(final_metrics, name)
        stable_capacities[name] = {
            "passed": initial is not None and final is not None and final == initial,
            "initial": initial,
            "final": final,
        }
    return {
        "passed": all(item["passed"] for item in gauges.values())
        and all(item["passed"] for item in stable_capacities.values()),
        "gauges": gauges,
        "stable_capacities": stable_capacities,
    }


def production_memory_evidence(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
    limits: ProductionMemoryLimits,
) -> dict[str, Any]:
    rss_values = [
        process.get("process_vmrss_kib") / 1024.0
        if process.get("process_vmrss_kib") is not None
        else None
        for _, _, process in snapshots
    ]
    process_gpu_values = [
        process.get("process_gpu_memory_used_mib")
        for _, _, process in snapshots
    ]
    process_gpu_coverage = ratio(
        sum(value is not None for value in process_gpu_values),
        len(process_gpu_values),
    )
    device_gpu_values = [
        sum(gpu["memory_used_mib"] for gpu in process.get("gpus") or [])
        if process.get("gpus")
        else None
        for _, _, process in snapshots
    ]
    process_rss = memory_series_evidence(
        rss_values,
        limits.min_coverage,
        limits.max_process_rss_drift_mib,
        limits.max_process_rss_high_water_mib,
        limits.max_process_rss_drift_fraction,
    )
    gpu_memory = memory_series_evidence(
        process_gpu_values,
        limits.min_coverage,
        limits.max_gpu_memory_drift_mib,
        limits.max_gpu_memory_high_water_mib,
    )
    gpu_memory["source"] = "server_pid_compute_process"
    gpu_memory["process_memory_coverage"] = process_gpu_coverage
    gpu_memory["whole_device_diagnostic"] = distribution(
        value for value in device_gpu_values if value is not None
    )
    gpu_memory["whole_device_fallback_used"] = False
    kv_blocks = gauge_utilization_evidence(
        snapshots,
        KV_CACHE_ACTIVE_GAUGE,
        "mistralrs_kv_cache_blocks_total",
        limits.min_coverage,
        limits.max_kv_block_utilization,
    )
    recurrent_slots = gauge_utilization_evidence(
        snapshots,
        "mistralrs_recurrent_state_slots_used",
        "mistralrs_recurrent_state_slots_total",
        limits.min_coverage,
        limits.max_recurrent_slot_utilization,
    )
    windowed_live_slots = gauge_utilization_evidence(
        snapshots,
        DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
        DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE,
        limits.min_coverage,
        MAX_WINDOWED_KV_SLOT_UTILIZATION,
    )
    windowed_checkpoint_slots = gauge_utilization_evidence(
        snapshots,
        DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
        DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE,
        limits.min_coverage,
        MAX_WINDOWED_KV_SLOT_UTILIZATION,
    )
    windowed_kv_slots = {
        "passed": (
            not limits.require_dflash_windowed_kv
            or (windowed_live_slots["passed"] and windowed_checkpoint_slots["passed"])
        ),
        "required": limits.require_dflash_windowed_kv,
        "live": windowed_live_slots,
        "checkpoint": windowed_checkpoint_slots,
    }
    final_cleanup = final_resource_cleanup_evidence(
        snapshots,
        limits.require_dflash_windowed_kv,
    )
    return {
        "passed": (
            process_rss["passed"]
            and gpu_memory["passed"]
            and kv_blocks["passed"]
            and recurrent_slots["passed"]
            and windowed_kv_slots["passed"]
            and final_cleanup["passed"]
        ),
        "process_rss": process_rss,
        "gpu_memory": gpu_memory,
        "kv_blocks": kv_blocks,
        "recurrent_slots": recurrent_slots,
        "windowed_kv_slots": windowed_kv_slots,
        "final_cleanup": final_cleanup,
    }


def multimodal_memory_evidence(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]],
    limits: ProductionMemoryLimits,
) -> dict[str, Any]:
    evidence = production_memory_evidence(snapshots, limits)
    final_cleanup = final_resource_cleanup_evidence(
        snapshots,
        False,
        MULTIMODAL_TRANSIENT_CLEANUP_GAUGES,
    )
    evidence["final_cleanup"] = final_cleanup
    evidence["passed"] = (
        evidence["process_rss"]["passed"]
        and evidence["gpu_memory"]["passed"]
        and evidence["kv_blocks"]["passed"]
        and evidence["recurrent_slots"]["passed"]
        and evidence["windowed_kv_slots"]["passed"]
        and final_cleanup["passed"]
    )
    return evidence


async def process_telemetry(server_pid: int | None) -> dict[str, Any]:
    telemetry: dict[str, Any] = {}
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"nvidia-smi exited with status {process.returncode}")
        gpu_rows = []
        for line in stdout.decode().splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 6:
                gpu_rows.append(
                    {
                        "index": int(fields[0]),
                        "utilization_percent": float(fields[1]),
                        "memory_used_mib": float(fields[2]),
                        "memory_total_mib": float(fields[3]),
                        "power_watts": float(fields[4]),
                        "temperature_c": float(fields[5]),
                    }
                )
        telemetry["gpus"] = gpu_rows
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        telemetry["gpus"] = []
        telemetry["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    if server_pid is not None:
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"nvidia-smi exited with status {process.returncode}")
            process_gpus = []
            for line in stdout.decode().splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) != 3:
                    continue
                try:
                    pid = int(fields[0])
                    memory_used_mib = float(fields[2])
                except ValueError:
                    continue
                if pid == server_pid:
                    process_gpus.append(
                        {
                            "pid": pid,
                            "gpu_uuid": fields[1],
                            "memory_used_mib": memory_used_mib,
                        }
                    )
            telemetry["process_gpus"] = process_gpus
            telemetry["process_gpu_memory_used_mib"] = (
                sum(gpu["memory_used_mib"] for gpu in process_gpus)
                if process_gpus
                else None
            )
        except (FileNotFoundError, RuntimeError) as exc:
            telemetry["process_gpus"] = []
            telemetry["process_gpu_memory_used_mib"] = None
            telemetry["nvidia_smi_process_error"] = f"{type(exc).__name__}: {exc}"
        status_path = Path(f"/proc/{server_pid}/status")
        cmdline_path = Path(f"/proc/{server_pid}/cmdline")
        stat_path = Path(f"/proc/{server_pid}/stat")
        try:
            status = status_path.read_text(encoding="utf-8")
            cmdline = cmdline_path.read_bytes()
            telemetry["process_is_mistralrs"] = b"mistralrs" in cmdline
            for key in ("VmRSS", "VmSize", "VmSwap"):
                match = re.search(rf"^{key}:\s+(\d+)\s+kB$", status, re.MULTILINE)
                telemetry[f"process_{key.lower()}_kib"] = int(match.group(1)) if match else None
        except OSError as exc:
            telemetry["process_error"] = f"{type(exc).__name__}: {exc}"
        try:
            telemetry["process_cpu_ticks"] = parse_process_cpu_ticks(
                stat_path.read_text(encoding="utf-8")
            )
            telemetry["process_cpu_clock_ticks_per_second"] = (
                PROCESS_CPU_CLOCK_TICKS_PER_SECOND
            )
        except (OSError, ValueError) as exc:
            telemetry["process_cpu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        host_total, host_idle = parse_host_cpu_ticks(
            Path("/proc/stat").read_text(encoding="utf-8")
        )
        telemetry["host_cpu_total_ticks"] = host_total
        telemetry["host_cpu_idle_ticks"] = host_idle
    except (OSError, ValueError) as exc:
        telemetry["host_cpu_error"] = f"{type(exc).__name__}: {exc}"
    return telemetry


async def telemetry_loop(
    client: SoakClient,
    writer: JsonlWriter,
    stop: asyncio.Event,
    server_pid: int | None,
    snapshots: list[tuple[float, dict[str, float], dict[str, Any]]],
    interval_seconds: float,
    scheduled_times: list[float],
    observed_times: list[float],
) -> float:
    next_scheduled_at = scheduled_times[-1] + interval_seconds

    async def collect(scheduled_at: float, terminal: bool) -> float:
        observation_started = time.perf_counter()
        scheduled_times.append(scheduled_at)
        observed_times.append(observation_started)
        metrics, process = await asyncio.gather(
            safe_metrics(client, writer, "production-telemetry"),
            process_telemetry(server_pid),
        )
        collected_at = time.perf_counter()
        snapshots.append((collected_at, metrics, process))
        await writer.emit(
            "telemetry",
            monotonic_seconds=collected_at,
            observation_started_monotonic_seconds=observation_started,
            scheduled_monotonic_seconds=scheduled_at,
            schedule_lateness_seconds=max(0.0, observation_started - scheduled_at),
            terminal=terminal,
            metrics=metrics,
            process=process,
        )
        return collected_at

    while True:
        scheduled_at = next_scheduled_at
        delay = scheduled_at - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        if stop.is_set():
            terminal_at = time.perf_counter()
            return await collect(terminal_at, True)
        await collect(scheduled_at, False)
        next_scheduled_at += interval_seconds


async def stream_request_with_slot(
    client: SoakClient,
    slots: asyncio.Semaphore,
    spec: RequestSpec,
    scheduled_at: float | None = None,
    retain_output_event_windows: Sequence[tuple[float, float]] | None = None,
) -> RequestResult:
    async with slots:
        return await client.stream_request(
            spec,
            scheduled_at=scheduled_at,
            retain_output_event_windows=retain_output_event_windows,
        )


@dataclass(frozen=True, slots=True)
class ProductionPhaseSlots:
    traffic: asyncio.Semaphore
    diagnostics: asyncio.Semaphore

    @classmethod
    def create(
        cls,
        traffic_concurrency: int,
        diagnostic_concurrency: int,
    ) -> ProductionPhaseSlots:
        return cls(
            traffic=asyncio.Semaphore(traffic_concurrency),
            diagnostics=asyncio.Semaphore(diagnostic_concurrency),
        )


async def wait_for_production_phase_tasks(
    tasks: Sequence[asyncio.Task[Any]],
) -> None:
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def choose_context(
    rng: random.Random,
    context_mix: Sequence[tuple[int, int]],
) -> int:
    return rng.choices(
        [item[0] for item in context_mix],
        weights=[item[1] for item in context_mix],
        k=1,
    )[0]


def allocate_prompt_pool_counts(
    context_mix: Sequence[tuple[int, int]],
    total_budget: int,
    per_context_cap: int,
) -> dict[int, int]:
    if total_budget < len(context_mix):
        raise ValueError(
            "resident prompt budget must fit at least one prompt per context length"
        )
    counts = {length: 1 for length, _ in context_mix}
    weights = {length: weight for length, weight in context_mix}
    target = min(total_budget, per_context_cap * len(context_mix))
    while sum(counts.values()) < target:
        candidates = [
            length for length in counts if counts[length] < per_context_cap
        ]
        if not candidates:
            break
        selected = max(candidates, key=lambda length: weights[length] / counts[length])
        counts[selected] += 1
    return counts


def load_quality_replay_cases(
    path: Path,
    case_ids: Sequence[str],
) -> tuple[list[QualityReplayCase], dict[str, Any]]:
    requested = list(case_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("quality replay case IDs must be non-empty and unique")
    records: dict[str, dict[str, Any]] = {}
    run_start: dict[str, Any] | None = None
    prewarm: dict[str, Any] | None = None
    context_prompt_overhead_tokens: int | None = None
    quality_failures: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            event = record.get("event")
            if event == "run_start":
                run_start = record
            elif event == "production_prewarm_summary":
                prewarm = record
            elif (
                event == "prompt_calibration"
                and record.get("profile") == CONTEXT_PROMPT_PROFILE
            ):
                context_prompt_overhead_tokens = int(record["overhead_tokens"])
            elif event == "production_phase_summary":
                quality_failures.update(
                    item["case_id"] for item in record.get("quality_failures") or ()
                )
            elif event == "request" and record.get("case_id") in requested:
                case_id = str(record["case_id"])
                if case_id in records:
                    raise RuntimeError(f"duplicate source request record for {case_id}")
                records[case_id] = record
    missing = [case_id for case_id in requested if case_id not in records]
    if missing:
        raise RuntimeError("missing source request records: " + ", ".join(missing))
    if run_start is None or run_start.get("mode") != "production":
        raise RuntimeError("quality replay source must be a production artifact")
    if prewarm is None:
        raise RuntimeError(
            "quality replay source is missing production prewarm evidence"
    )
    if context_prompt_overhead_tokens is None:
        raise RuntimeError(
            "quality replay source is missing context prompt calibration"
        )
    arguments = run_start.get("arguments") or {}
    source_seed = int(arguments["seed"])
    source_fixed_output_length = bool(arguments.get("fixed_output_length", False))
    context_mix = [tuple(map(int, item)) for item in arguments["context_mix"]]
    concurrencies = [int(value) for value in arguments["concurrencies"]]
    if len(set(concurrencies)) != len(concurrencies):
        raise RuntimeError(
            "quality replay source production concurrencies must be unique"
        )
    pool_counts = {
        int(length): int(count)
        for length, count in (prewarm.get("pool_counts") or {}).items()
    }
    if set(pool_counts) != {length for length, _ in context_mix}:
        raise RuntimeError("quality replay source has incomplete prompt pool counts")
    cases = []
    seed_provenance = []
    pattern = re.compile(
        r"^prod-c(?P<concurrency>\d+)-w(?P<worker>\d+)-r(?P<request>\d+)$"
    )
    for case_id in requested:
        match = pattern.fullmatch(case_id)
        if match is None:
            raise ValueError(f"unsupported production case ID: {case_id}")
        concurrency = int(match.group("concurrency"))
        worker_index = int(match.group("worker"))
        request_index = int(match.group("request"))
        if concurrency not in concurrencies:
            raise RuntimeError(
                f"source concurrency {concurrency} is not in run arguments"
            )
        phase_index = concurrencies.index(concurrency)
        rng = random.Random(source_seed + phase_index * 100_000 + worker_index)
        context_tokens = 0
        prompt_index = 0
        for _ in range(request_index + 1):
            context_tokens = choose_context(rng, context_mix)
            prompt_index = rng.randrange(pool_counts[context_tokens])
        source = records[case_id]
        reconstructed_seed = (
            source_seed
            + phase_index * 1_000_000
            + worker_index * 10_000
            + request_index
        )
        logged_seed = int(source.get("seed", -1))
        if source.get("tags", {}).get("role") != "traffic":
            raise RuntimeError(f"source request {case_id} is not production traffic")
        if logged_seed != reconstructed_seed:
            raise RuntimeError(
                f"source request {case_id} seed {logged_seed} does not match "
                f"reconstructed seed {reconstructed_seed}"
            )
        if int(source.get("context_tokens", -1)) != context_tokens:
            raise RuntimeError(f"source request {case_id} has an unexpected context")
        cases.append(
            QualityReplayCase(
                case_id=case_id,
                seed=logged_seed,
                concurrency=concurrency,
                worker_index=worker_index,
                request_index=request_index,
                context_tokens=context_tokens,
                prompt_index=prompt_index,
                prompt_label=f"production-{context_tokens}-{prompt_index}",
                source_output_transcript_sha256=str(
                    source.get("output_transcript_sha256") or ""
                ),
                source_quality_failure=(
                    not source_fixed_output_length and case_id in quality_failures
                ),
            )
        )
        seed_provenance.append(
            {
                "case_id": case_id,
                "logged_source_seed": logged_seed,
                "reconstructed_source_seed": reconstructed_seed,
                "matched": True,
            }
        )
    evidence = {
        "source_artifact": str(path),
        "source_seed": source_seed,
        "source_fixed_output_length": source_fixed_output_length,
        "source_traffic_quality_eligible": not source_fixed_output_length,
        "source_reported_quality_failures": [
            case_id for case_id in requested if case_id in quality_failures
        ],
        "source_concurrencies": concurrencies,
        "pool_counts": pool_counts,
        "context_prompt_overhead_tokens": context_prompt_overhead_tokens,
        "requested_case_ids": requested,
        "cases": [asdict(case) for case in cases],
        "seed_provenance": seed_provenance,
        "seed_provenance_complete": len(seed_provenance) == len(requested),
        "complete": len(cases) == len(requested),
    }
    return cases, evidence


def quality_replay_prompt_identity_evidence(
    specs: Sequence[RequestSpec],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    total_prompt_tokens = 0
    for spec in specs:
        if spec.context_tokens is None:
            raise ValueError("quality replay requests require context token counts")
        context_tokens = int(spec.context_tokens)
        total_prompt_tokens += context_tokens
        group = groups.setdefault(
            spec.prompt,
            {
                "context_tokens": context_tokens,
                "case_ids": [],
                "prompt_labels": set(),
            },
        )
        if group["context_tokens"] != context_tokens:
            raise ValueError(
                "identical quality replay prompts have inconsistent token counts"
            )
        group["case_ids"].append(spec.case_id)
        prompt_label = spec.tags.get("prompt_label")
        if prompt_label:
            group["prompt_labels"].add(str(prompt_label))

    identities = []
    for prompt, group in groups.items():
        identities.append(
            {
                "prompt_sha256": stable_hash(prompt),
                "prompt_labels": sorted(group["prompt_labels"]),
                "context_tokens": group["context_tokens"],
                "request_count": len(group["case_ids"]),
                "case_ids": list(group["case_ids"]),
            }
        )
    identities.sort(key=lambda item: item["prompt_sha256"])
    compulsory_miss_prompt_tokens = sum(
        identity["context_tokens"] for identity in identities
    )
    return {
        "request_count": len(specs),
        "prompt_identity_count": len(identities),
        "duplicate_prompt_requests": len(specs) - len(identities),
        "total_prompt_tokens": total_prompt_tokens,
        "compulsory_miss_prompt_tokens": compulsory_miss_prompt_tokens,
        "duplicate_prompt_tokens": (
            total_prompt_tokens - compulsory_miss_prompt_tokens
        ),
        "identities": identities,
    }


def make_quality_replay_specs(
    client: SoakClient,
    cases: Sequence[QualityReplayCase],
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> tuple[list[RequestSpec], dict[str, Any]]:
    if not cases or len(cases) > concurrency:
        raise ValueError("quality replay cases must fit in one concurrent batch")
    unique_cases = list({case.prompt_label: case for case in cases}.values())
    prompts = {
        case.prompt_label: exact_context(
            client,
            case.context_tokens,
            case.prompt_label,
        )
        for case in unique_cases
    }
    specs = [
        RequestSpec(
            case_id=case.case_id,
            seed=case.seed,
            max_tokens=max_tokens,
            prompt=prompts[case.prompt_label],
            context_tokens=case.context_tokens,
            tags={
                "scenario": "quality_replay",
                "role": "selected",
                "prompt_label": case.prompt_label,
                "source_quality_failure": case.source_quality_failure,
            },
            extra={"ignore_eos": False},
        )
        for case in cases
    ]
    for index in range(concurrency - len(specs)):
        case = unique_cases[index % len(unique_cases)]
        specs.append(
            RequestSpec(
                case_id=f"quality-replay-control-{index:02d}",
                seed=seed + QUALITY_REPLAY_CONTROL_SEED_OFFSET + index,
                max_tokens=max_tokens,
                prompt=prompts[case.prompt_label],
                context_tokens=case.context_tokens,
                tags={
                    "scenario": "quality_replay",
                    "role": "control",
                    "prompt_label": case.prompt_label,
                },
                extra={"ignore_eos": False},
            )
        )
    selected_identity_counts = Counter(case.prompt_label for case in cases)
    measurement_identities = quality_replay_prompt_identity_evidence(specs)
    evidence = {
        "concurrency": concurrency,
        "quality_output_contract": "normal_eos",
        "ignore_eos": False,
        "selected_cases": [case.case_id for case in cases],
        "selected_prompt_labels": [case.prompt_label for case in cases],
        "logical_prompt_identities": sorted(prompts),
        "logical_prompt_identity_count": len(prompts),
        "selected_duplicate_prompt_identities": [
            {
                "prompt_label": prompt_label,
                "request_count": count,
                "case_ids": [
                    case.case_id
                    for case in cases
                    if case.prompt_label == prompt_label
                ],
            }
            for prompt_label, count in sorted(selected_identity_counts.items())
            if count > 1
        ],
        "control_cases": len(specs) - len(cases),
        "measurement_cases": len(specs),
        "cold_compulsory_miss_requests": measurement_identities[
            "prompt_identity_count"
        ],
        "cold_duplicate_request_upper_bound": measurement_identities[
            "duplicate_prompt_requests"
        ],
        "measurement_prompt_identities": measurement_identities["identities"],
        "single_full_batch": len(specs) == concurrency,
    }
    return specs, evidence


def make_quality_replay_pressure_specs(
    client: SoakClient,
    wave: int,
    entries: int,
    context_tokens: int,
    max_tokens: int,
    seed: int,
) -> list[RequestSpec]:
    return [
        RequestSpec(
            case_id=f"quality-replay-pressure-w{wave:02d}-{index:02d}",
            seed=(seed + QUALITY_REPLAY_PRESSURE_SEED_OFFSET + wave * entries + index),
            max_tokens=max_tokens,
            prompt=exact_context(
                client,
                context_tokens,
                f"quality-replay-pressure-w{wave}-{index}",
            ),
            context_tokens=context_tokens,
            tags={
                "scenario": "quality_replay",
                "role": "pressure",
                "wave": wave,
            },
            extra={"ignore_eos": True},
        )
        for index in range(entries)
    ]


def quality_replay_prefix_state_evidence(
    before: dict[str, float],
    after: dict[str, float],
    specs: Sequence[RequestSpec],
    expectation: str,
    min_reuse_fraction: float,
) -> dict[str, Any]:
    identities = quality_replay_prompt_identity_evidence(specs)
    lookups = metric_delta(before, after, "mistralrs_prefix_cache_lookups_total")
    raw_hits = metric_delta(before, after, "mistralrs_prefix_cache_hits_total")
    raw_reused = metric_delta(
        before,
        after,
        "mistralrs_prefix_cache_tokens_reused_total",
    )
    hits = 0.0 if raw_hits is None and lookups is not None else raw_hits
    reused = 0.0 if raw_reused is None and lookups is not None else raw_reused
    request_count = identities["request_count"]
    identity_count = identities["prompt_identity_count"]
    duplicate_requests = identities["duplicate_prompt_requests"]
    expected_tokens = identities["total_prompt_tokens"]
    compulsory_miss_tokens = identities["compulsory_miss_prompt_tokens"]
    duplicate_tokens = identities["duplicate_prompt_tokens"]
    hit_rate = ratio(hits, lookups)
    reuse_fraction = ratio(reused, expected_tokens)
    misses = lookups - hits if lookups is not None and hits is not None else None
    observed = (
        lookups is not None
        and lookups >= request_count
        and hits is not None
        and hits >= 0
        and reused is not None
        and reused >= 0
        and hit_rate is not None
        and reuse_fraction is not None
    )
    if expectation == "hit":
        expected_hits = {
            "minimum": request_count * min_reuse_fraction,
            "maximum": request_count,
        }
        expected_misses = {
            "minimum": 0.0,
            "maximum": request_count * (1.0 - min_reuse_fraction),
        }
        expected_reused = {
            "minimum": expected_tokens * min_reuse_fraction,
            "maximum": expected_tokens,
        }
        state_matches = (
            observed
            and hit_rate >= min_reuse_fraction
            and reuse_fraction >= min_reuse_fraction
        )
    elif expectation == "miss":
        expected_hits = {
            "minimum": 0.0,
            "maximum": duplicate_requests
            + identity_count * (1.0 - min_reuse_fraction),
        }
        expected_misses = {
            "minimum": identity_count * min_reuse_fraction,
            "maximum": request_count,
        }
        expected_reused = {
            "minimum": 0.0,
            "maximum": duplicate_tokens
            + compulsory_miss_tokens * (1.0 - min_reuse_fraction),
        }
        state_matches = (
            observed
            and hits <= expected_hits["maximum"]
            and misses is not None
            and misses >= expected_misses["minimum"]
            and reused <= expected_reused["maximum"]
        )
    elif expectation == "either":
        expected_hits = {"minimum": 0.0, "maximum": request_count}
        expected_misses = {"minimum": 0.0, "maximum": request_count}
        expected_reused = {"minimum": 0.0, "maximum": expected_tokens}
        state_matches = observed
    else:
        raise ValueError(f"unsupported prefix state expectation: {expectation}")
    return {
        "passed": bool(state_matches),
        "expectation": expectation,
        "minimum_reuse_fraction": min_reuse_fraction,
        "lookups": lookups,
        "hits": hits,
        "misses": misses,
        "hits_counter_present": raw_hits is not None,
        "hit_rate": hit_rate,
        "expected_prompt_tokens": expected_tokens,
        "reused_tokens": reused,
        "reused_tokens_counter_present": raw_reused is not None,
        "reuse_fraction": reuse_fraction,
        "prompt_identity_count": identity_count,
        "duplicate_prompt_requests": duplicate_requests,
        "compulsory_miss_prompt_tokens": compulsory_miss_tokens,
        "duplicate_prompt_tokens": duplicate_tokens,
        "expected_hit_requests": expected_hits,
        "expected_miss_requests": expected_misses,
        "expected_reused_tokens": expected_reused,
        "prompt_identities": identities["identities"],
        "speculative_hits": metric_delta(
            before,
            after,
            "mistralrs_speculative_prefix_cache_hits_total",
        ),
        "speculative_misses": metric_delta(
            before,
            after,
            "mistralrs_speculative_prefix_cache_misses_total",
        ),
        "speculative_captures": metric_delta(
            before,
            after,
            "mistralrs_speculative_prefix_cache_captures_total",
        ),
    }


def quality_replay_source_output_evidence(
    results: Sequence[RequestResult],
    cases: Sequence[QualityReplayCase],
) -> dict[str, Any]:
    by_case = {result.case_id: result for result in results}
    checks = []
    for case in cases:
        result = by_case.get(case.case_id)
        observed = stable_hash(result.output_transcript) if result is not None else None
        checks.append(
            {
                "case_id": case.case_id,
                "source_sha256": case.source_output_transcript_sha256,
                "observed_sha256": observed,
                "source_quality_failure": case.source_quality_failure,
                "matched": bool(
                    result is not None
                    and result.ok
                    and case.source_output_transcript_sha256
                    and observed == case.source_output_transcript_sha256
                ),
            }
        )
    return {
        "complete": len(checks) == len(cases),
        "exact_matches": sum(check["matched"] for check in checks),
        "cases": checks,
    }


def quality_replay_exactness_evidence(
    cold_resident: dict[str, Any],
    resident_stable_reference: dict[str, Any],
    reversed_order: dict[str, Any],
    pressure_touches: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    required = [resident_stable_reference, reversed_order, *pressure_touches]
    return {
        "passed": all(comparison["passed"] for comparison in required),
        "cold_resident_diagnostic_only": True,
        "cold_resident": {**cold_resident, "gated": False},
        "resident_stable_reference": {
            **resident_stable_reference,
            "gated": True,
        },
        "reversed_order": {**reversed_order, "gated": True},
        "pressure_touches": [
            {**comparison, "gated": True} for comparison in pressure_touches
        ],
        "required_comparisons": len(required),
        "failed_required_comparisons": sum(
            not comparison["passed"] for comparison in required
        ),
    }


def cuda_graph_capture_quiescence_evidence(
    before: dict[str, float],
    after: dict[str, float],
    required_components: Sequence[str],
) -> dict[str, Any]:
    capture_deltas = [
        item
        for item in labeled_metric_deltas(
            before,
            after,
            CUDA_GRAPH_EVENTS_COUNTER,
        )
        if item["labels"].get("event") == "capture"
    ]
    capture_values = [
        item
        for snapshot in (before, after)
        for item in labeled_metric_values(snapshot, CUDA_GRAPH_EVENTS_COUNTER)
        if item["labels"].get("event") == "capture"
    ]
    components = {}
    for component in required_components:
        matching_deltas = [
            item
            for item in capture_deltas
            if item["labels"].get("component") == component
        ]
        instrumentation_present = any(
            item["labels"].get("component") == component
            for item in capture_values
        )
        components[component] = {
            "passed": instrumentation_present
            and all(item["delta"] == 0 for item in matching_deltas),
            "instrumentation_present": instrumentation_present,
            "capture_delta": sum(item["delta"] for item in matching_deltas),
            "series": matching_deltas,
        }
    return {
        "passed": bool(components)
        and all(component["passed"] for component in components.values()),
        "metric": CUDA_GRAPH_EVENTS_COUNTER,
        "required_components": list(required_components),
        "components": components,
        "capture_deltas": capture_deltas,
    }


def quality_replay_stability_evidence(
    attempts: Sequence[dict[str, Any]],
    maximum_attempts: int,
) -> dict[str, Any]:
    checks = [
        {
            **attempt,
            "passed": (
                attempt["stage_passed"]
                and attempt["exact_to_previous"]["passed"]
                and attempt["cuda_graph_captures"]["passed"]
            ),
        }
        for attempt in attempts
    ]
    converged = next((check for check in checks if check["passed"]), None)
    return {
        "passed": converged is not None,
        "maximum_attempts": maximum_attempts,
        "attempts_run": len(checks),
        "exhausted": len(checks) >= maximum_attempts and converged is None,
        "stable_reference_phase": (
            converged["phase"] if converged is not None else None
        ),
        "required_consecutive_exact_sets": 2,
        "required_zero_capture_components": (
            checks[-1]["cuda_graph_captures"]["required_components"]
            if checks
            else []
        ),
        "failure": (
            None
            if converged is not None
            else "resident outputs and CUDA graph captures did not converge"
        ),
        "attempts": checks,
    }


def quality_replay_pressure_plan_evidence(
    snapshot: dict[str, float],
    cases: Sequence[QualityReplayCase],
    selected_specs: Sequence[RequestSpec],
    config: QualityReplayPressureConfig,
) -> dict[str, Any]:
    logical_contexts = {case.prompt_label: case.context_tokens for case in cases}
    total_blocks = metric_total(snapshot, "mistralrs_kv_cache_blocks_total")
    owner_capacity = metric_total(
        snapshot,
        PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_GAUGE,
    )
    baseline_owner_entries = metric_total(
        snapshot,
        PAGED_RECURRENT_PREFIX_OWNERS_USED_GAUGE,
    )
    baseline_retained_blocks = metric_total(
        snapshot,
        PAGED_PREFIX_RETAINED_BLOCKS_GAUGE,
    )
    baseline_active_blocks = metric_total(snapshot, KV_CACHE_ACTIVE_GAUGE)
    selected_blocks = sum(
        math.ceil(tokens / config.block_size_tokens)
        for tokens in logical_contexts.values()
    )
    selected_active_context_blocks = sum(
        math.ceil(int(spec.context_tokens or 0) / config.block_size_tokens)
        for spec in selected_specs
    )
    selected_active_suffix_blocks = sum(
        math.ceil(spec.max_tokens / config.block_size_tokens)
        for spec in selected_specs
    )
    pressure_blocks_per_wave = config.entries * math.ceil(
        config.context_tokens / config.block_size_tokens
    )
    pressure_active_suffix_blocks = config.entries * math.ceil(
        config.max_tokens / config.block_size_tokens
    )
    pressure_blocks_per_entry = math.ceil(
        config.context_tokens / config.block_size_tokens
    )
    available_pressure_owner_entries = (
        max(
            0,
            math.floor(
                owner_capacity
                - len(logical_contexts)
            ),
        )
        if owner_capacity is not None and baseline_owner_entries is not None
        else None
    )
    elapsed_pressure_entries = max(0, config.waves - 1) * config.entries
    previous_pressure_entries = (
        min(elapsed_pressure_entries, available_pressure_owner_entries)
        if available_pressure_owner_entries is not None
        else None
    )
    retained_pressure_entries = (
        min(config.waves * config.entries, available_pressure_owner_entries)
        if available_pressure_owner_entries is not None
        else None
    )
    previous_pressure_blocks = (
        previous_pressure_entries * pressure_blocks_per_entry
        if previous_pressure_entries is not None
        else None
    )
    retained_pressure_blocks = (
        retained_pressure_entries * pressure_blocks_per_entry
        if retained_pressure_entries is not None
        else None
    )
    baseline_blocks = (baseline_retained_blocks or 0.0) + (
        baseline_active_blocks or 0.0
    )
    selected_cold_peak_blocks = (
        baseline_blocks
        + selected_active_context_blocks
        + selected_active_suffix_blocks
    )
    pressure_peak_blocks = (
        baseline_blocks
        + selected_blocks
        + (previous_pressure_blocks or 0)
        + pressure_blocks_per_wave
        + pressure_active_suffix_blocks
    )
    touch_peak_blocks = (
        baseline_blocks
        + selected_blocks
        + (retained_pressure_blocks or 0)
        + selected_active_suffix_blocks
    )
    peak_required_blocks = max(
        selected_cold_peak_blocks,
        pressure_peak_blocks,
        touch_peak_blocks,
    )
    cumulative_pressure_blocks = config.waves * pressure_blocks_per_wave
    logical_peak_entries = (
        (baseline_owner_entries or 0.0) + len(logical_contexts) + config.entries
    )
    headroom_blocks = (
        math.ceil(total_blocks * config.headroom_fraction)
        if total_blocks is not None
        else None
    )
    physical_budget = (
        total_blocks - headroom_blocks
        if total_blocks is not None and headroom_blocks is not None
        else None
    )
    baseline_instrumented = (
        baseline_owner_entries is not None
        and baseline_retained_blocks is not None
        and baseline_active_blocks is not None
    )
    owner_fit = (
        baseline_instrumented
        and owner_capacity is not None
        and logical_peak_entries <= owner_capacity
    )
    physical_fit = (
        baseline_instrumented
        and previous_pressure_blocks is not None
        and retained_pressure_blocks is not None
        and physical_budget is not None
        and peak_required_blocks <= physical_budget
    )
    cumulative_churn = (
        total_blocks is not None and cumulative_pressure_blocks > total_blocks
    )
    return {
        "passed": bool(owner_fit and physical_fit and cumulative_churn),
        "selected_logical_entries": len(logical_contexts),
        "baseline_owner_entries": baseline_owner_entries,
        "baseline_retained_blocks": baseline_retained_blocks,
        "baseline_active_blocks": baseline_active_blocks,
        "baseline_instrumented": baseline_instrumented,
        "pressure_entries_per_wave": config.entries,
        "pressure_waves": config.waves,
        "logical_peak_entries": logical_peak_entries,
        "owner_capacity": owner_capacity,
        "owner_capacity_available": owner_capacity is not None,
        "owner_fit": owner_fit,
        "selected_blocks": selected_blocks,
        "selected_active_context_blocks": selected_active_context_blocks,
        "selected_active_suffix_blocks": selected_active_suffix_blocks,
        "pressure_blocks_per_wave": pressure_blocks_per_wave,
        "pressure_blocks_per_entry": pressure_blocks_per_entry,
        "pressure_active_suffix_blocks": pressure_active_suffix_blocks,
        "available_pressure_owner_entries": available_pressure_owner_entries,
        "elapsed_pressure_entries": elapsed_pressure_entries,
        "previous_pressure_entries": previous_pressure_entries,
        "previous_pressure_blocks": previous_pressure_blocks,
        "retained_pressure_entries": retained_pressure_entries,
        "retained_pressure_blocks": retained_pressure_blocks,
        "selected_cold_peak_blocks": selected_cold_peak_blocks,
        "pressure_peak_blocks": pressure_peak_blocks,
        "touch_peak_blocks": touch_peak_blocks,
        "peak_retained_blocks": peak_required_blocks,
        "peak_required_blocks": peak_required_blocks,
        "total_blocks": total_blocks,
        "physical_headroom_fraction": config.headroom_fraction,
        "physical_headroom_blocks": headroom_blocks,
        "physical_budget_blocks": physical_budget,
        "physical_fit": physical_fit,
        "cumulative_pressure_blocks": cumulative_pressure_blocks,
        "cumulative_exceeds_physical_pool": cumulative_churn,
    }


async def quality_replay_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    cases, source = load_quality_replay_cases(
        args.source_production_artifact,
        args.case_ids,
    )
    client.prompt_overhead_tokens[CONTEXT_PROMPT_PROFILE] = int(
        source["context_prompt_overhead_tokens"]
    )
    specs, cohort = make_quality_replay_specs(
        client,
        cases,
        args.concurrency,
        args.max_tokens,
        args.seed,
    )
    initial = await safe_metrics(client, writer, "quality-replay-start")
    sequence_capacity = metric_total(initial, "mistralrs_sequences_capacity")
    prefix_cached = metric_total(initial, KV_CACHE_PREFIX_CACHED_GAUGE)
    retained_owners = metric_total(
        initial,
        PAGED_RECURRENT_PREFIX_OWNERS_USED_GAUGE,
    )
    empty_cache = prefix_cached == 0 and (
        retained_owners is None or retained_owners == 0
    )
    pressure_plan = quality_replay_pressure_plan_evidence(
        initial,
        cases,
        specs,
        QualityReplayPressureConfig(
            waves=args.pressure_waves,
            entries=args.pressure_entries,
            context_tokens=args.pressure_context_tokens,
            max_tokens=args.pressure_max_tokens,
            block_size_tokens=args.kv_block_size_tokens,
            headroom_fraction=args.prefix_pressure_kv_headroom_fraction,
        ),
    )
    sampling = production_sampling_policy_evidence(
        PART1_PRODUCTION_SAMPLING_POLICY,
        SamplingPolicy(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )
    await writer.emit(
        "quality_replay_cohort",
        source=source,
        cohort=cohort,
        pressure_plan=pressure_plan,
        production_sampling=sampling,
        server_sequence_capacity=sequence_capacity,
        initial_prefix_cached_blocks=prefix_cached,
        initial_retained_owners=retained_owners,
        empty_cache=empty_cache,
    )
    if not pressure_plan["passed"]:
        raise RuntimeError(
            "quality replay pressure must fit selected prefixes, retained prior pressure "
            "owners, and the active wave below logical and physical capacity while "
            "cumulative churn exceeds the physical pool"
        )

    quality_checks: list[dict[str, Any]] = []

    async def record_quality(
        phase: str,
        results: Sequence[RequestResult],
    ) -> list[dict[str, Any]]:
        checks = []
        for result in results:
            valid, detail = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            check = {"phase": phase, "valid": valid, **detail}
            checks.append(check)
            quality_checks.append(check)
            await writer.emit("quality_replay_quality", **check)
        return checks

    async def run_selected(
        phase: str,
        before: dict[str, float],
        expectation: str,
        request_specs: Sequence[RequestSpec],
    ) -> tuple[list[RequestResult], dict[str, Any], dict[str, float]]:
        results, summary = await run_batch(
            client,
            request_specs,
            args.concurrency,
            writer,
            phase,
            keep_output=True,
        )
        after = await safe_metrics(client, writer, f"{phase}-metrics")
        prefix = quality_replay_prefix_state_evidence(
            before,
            after,
            request_specs,
            expectation,
            args.min_prefix_reuse_fraction,
        )
        speculative_state_ok = (
            not args.require_mtp
            or expectation != "hit"
            or (
                prefix["speculative_hits"] is not None
                and prefix["speculative_hits"] >= len(request_specs)
                and prefix["speculative_misses"] == 0
            )
        )
        quality = await record_quality(phase, results)
        stage = {
            "passed": (
                summary["errors"] == 0
                and prefix["passed"]
                and speculative_state_ok
                and all(check["valid"] for check in quality)
            ),
            "phase": phase,
            "summary": summary,
            "prefix_cache": prefix,
            "speculative_state_ok": speculative_state_ok,
            "quality": quality,
            "source_outputs": quality_replay_source_output_evidence(results, cases),
        }
        await writer.emit("quality_replay_stage", **stage)
        return results, stage, after

    cold, cold_stage, after_cold = await run_selected(
        "quality-replay-cold",
        initial,
        "miss",
        specs,
    )
    resident, resident_stage, after_resident = await run_selected(
        "quality-replay-resident",
        after_cold,
        "hit",
        specs,
    )
    cold_resident_exact = exact_output_diagnostics(
        cold,
        resident,
        "quality-replay-cold",
        "quality-replay-resident",
        specs,
    )
    await writer.emit(
        "quality_replay_exact",
        gated=False,
        **cold_resident_exact,
    )

    stability_attempts = []
    stability_stages = []
    previous_results = resident
    previous_phase = "quality-replay-resident"
    stability_cursor = after_resident
    stable_reference = resident
    stable_reference_stage = resident_stage
    resident_stable_exact = cold_resident_exact
    for attempt_index in range(args.max_stability_passes):
        phase = f"quality-replay-stable-reference-{attempt_index + 1:02d}"
        current, stage, after_current = await run_selected(
            phase,
            stability_cursor,
            "hit",
            specs,
        )
        exact = exact_output_diagnostics(
            previous_results,
            current,
            previous_phase,
            phase,
            specs,
        )
        captures = cuda_graph_capture_quiescence_evidence(
            stability_cursor,
            after_current,
            args.expected_graph_components,
        )
        attempt = {
            "phase": phase,
            "stage_passed": stage["passed"],
            "exact_to_previous": exact,
            "cuda_graph_captures": captures,
        }
        stability_attempts.append(attempt)
        stability_stages.append(stage)
        await writer.emit(
            "quality_replay_stability_attempt",
            attempt=attempt_index + 1,
            maximum_attempts=args.max_stability_passes,
            **attempt,
        )
        await writer.emit(
            "quality_replay_exact",
            gated=True,
            **exact,
        )
        stability = quality_replay_stability_evidence(
            stability_attempts,
            args.max_stability_passes,
        )
        stable_reference = current
        stable_reference_stage = stage
        resident_stable_exact = exact
        stability_cursor = after_current
        previous_results = current
        previous_phase = phase
        if stability["passed"]:
            break
    stability = quality_replay_stability_evidence(
        stability_attempts,
        args.max_stability_passes,
    )
    await writer.emit("quality_replay_stability", **stability)
    if not stability["passed"]:
        components = ", ".join(args.expected_graph_components)
        raise RuntimeError(
            f"quality replay resident outputs and {components} CUDA graph captures "
            f"did not converge within {args.max_stability_passes} passes"
        )
    stable_reference_phase = str(stability["stable_reference_phase"])
    after_stable_reference = stability_cursor

    reversed_specs = list(reversed(specs))
    reversed_results, reversed_stage, after_reversed = await run_selected(
        "quality-replay-reversed-order",
        after_stable_reference,
        "hit",
        reversed_specs,
    )
    reversed_exact = exact_output_diagnostics(
        stable_reference,
        reversed_results,
        stable_reference_phase,
        "quality-replay-reversed-order",
        specs,
    )
    await writer.emit(
        "quality_replay_exact",
        gated=True,
        **reversed_exact,
    )

    cursor = after_reversed
    pressure_summaries = []
    touch_stages = []
    touch_exact = []
    pressure_results_ok = True
    last_pressure_metrics = after_reversed
    last_touch_results = stable_reference
    for wave in range(args.pressure_waves):
        pressure_phase = f"quality-replay-pressure-w{wave:02d}"
        pressure_specs = make_quality_replay_pressure_specs(
            client,
            wave,
            args.pressure_entries,
            args.pressure_context_tokens,
            args.pressure_max_tokens,
            args.seed,
        )
        pressure_results, pressure_summary = await run_batch(
            client,
            pressure_specs,
            min(args.concurrency, args.pressure_entries),
            writer,
            pressure_phase,
            keep_output=False,
        )
        pressure_completion = fixed_length_completion_evidence(
            pressure_results,
            args.pressure_max_tokens,
        )
        pressure_summary = {
            **pressure_summary,
            "fixed_length_completion": pressure_completion,
        }
        pressure_summaries.append(pressure_summary)
        pressure_results_ok = (
            pressure_results_ok and pressure_completion["passed"]
        )
        await writer.emit(
            "quality_replay_pressure_completion",
            phase=pressure_phase,
            **pressure_completion,
        )
        last_pressure_metrics = await safe_metrics(
            client,
            writer,
            f"{pressure_phase}-metrics",
        )
        touch_phase = (
            "quality-replay-after-pressure"
            if wave + 1 == args.pressure_waves
            else f"quality-replay-touch-w{wave:02d}"
        )
        last_touch_results, touch_stage, cursor = await run_selected(
            touch_phase,
            last_pressure_metrics,
            "hit",
            specs,
        )
        exact = exact_output_diagnostics(
            stable_reference,
            last_touch_results,
            stable_reference_phase,
            touch_phase,
            specs,
        )
        touch_stages.append(touch_stage)
        touch_exact.append(exact)
        await writer.emit("quality_replay_exact", gated=True, **exact)

    exactness = quality_replay_exactness_evidence(
        cold_resident_exact,
        resident_stable_exact,
        reversed_exact,
        touch_exact,
    )

    cleanup_ok, cleanup_metrics, cleanup = await poll_for_cleanup(
        client,
        writer,
        initial,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "quality-replay-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    allocation_pressure_revocations = sum(
        item["delta"]
        for item in labeled_metric_deltas(
            initial,
            cleanup_metrics,
            PAGED_PREFIX_RETENTION_PRESSURE_EVICTIONS_COUNTER,
        )
        if item["labels"].get("reason") == "allocation_pressure"
    )
    paired_residency = {
        "passed": (not args.require_mtp or allocation_pressure_revocations == 0),
        "allocation_pressure_revocations": allocation_pressure_revocations,
        "required": args.require_mtp,
    }
    checkpoint_retention = dflash_checkpoint_retention_evidence(
        {
            "before_cold": initial,
            "after_cold": after_cold,
            "after_hit": after_reversed,
            "after_pressure": last_pressure_metrics,
            "after_retry": cursor,
            "quiescent": cleanup_metrics,
        },
        cohort["logical_prompt_identity_count"]
        + args.pressure_waves * args.pressure_entries,
    )
    mtp = configured_speculative_evidence(
        initial,
        cleanup_metrics,
        args,
        args.require_mtp,
    )
    graph = cuda_graph_evidence(
        initial,
        cleanup_metrics,
        initial,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    memory = cuda_memory_pressure_evidence(
        initial,
        cleanup_metrics,
        require_instrumentation=True,
    )
    result = {
        "mode": "quality-replay",
        "passed": (
            sampling["passed"]
            and cohort["single_full_batch"]
            and sequence_capacity == args.concurrency
            and (empty_cache or not args.require_empty_prefix_cache)
            and pressure_plan["passed"]
            and cold_stage["passed"]
            and resident_stage["passed"]
            and all(stage["passed"] for stage in stability_stages)
            and stability["passed"]
            and reversed_stage["passed"]
            and exactness["passed"]
            and pressure_results_ok
            and all(stage["passed"] for stage in touch_stages)
            and paired_residency["passed"]
            and checkpoint_retention["passed"]
            and cleanup_ok
            and mtp["passed"]
            and graph["passed"]
            and memory["passed"]
        ),
        "source": source,
        "cohort": cohort,
        "production_sampling": sampling,
        "server_sequence_capacity": sequence_capacity,
        "empty_cache": empty_cache,
        "pressure_plan": pressure_plan,
        "cold": cold_stage,
        "resident": resident_stage,
        "stable_reference": stable_reference_stage,
        "stability": stability,
        "stability_stages": stability_stages,
        "reversed_order": reversed_stage,
        "cold_resident_exact": exactness["cold_resident"],
        "resident_stable_exact": exactness["resident_stable_reference"],
        "reversed_order_exact": exactness["reversed_order"],
        "exactness": exactness,
        "pressure_summaries": pressure_summaries,
        "touch_stages": touch_stages,
        "touch_exact": exactness["pressure_touches"],
        "after_pressure_source_outputs": quality_replay_source_output_evidence(
            last_touch_results,
            cases,
        ),
        "paired_residency": paired_residency,
        "checkpoint_retention": checkpoint_retention,
        "quality_checks": quality_checks,
        "quality_ok": all(check["valid"] for check in quality_checks),
        "cleanup_ok": cleanup_ok,
        "cleanup": cleanup,
        "mtp": mtp,
        "cuda_graph": graph,
        "cuda_memory": memory,
        "metric_deltas": selected_metric_deltas(initial, cleanup_metrics),
    }
    await writer.emit("quality_replay_summary", **result)
    return result


async def production_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    await calibrate_prompt_profiles(
        client,
        writer,
        (CONTEXT_PROMPT_PROFILE, RETRIEVAL_PROMPT_PROFILE),
    )
    lengths = tuple(length for length, _ in args.context_mix)
    capacity_metrics = await safe_metrics(client, writer, "production-capacity")
    sequence_capacity = metric_total(capacity_metrics, "mistralrs_sequences_capacity")
    preflight_process = await process_telemetry(args.server_pid)
    required_gauges = production_required_gauges(args.expected_graph_components)
    missing_gauges = [
        gauge
        for gauge in required_gauges
        if metric_total(capacity_metrics, gauge) is None
    ]
    preflight = {
        "passed": (
            not missing_gauges
            and preflight_process.get("process_is_mistralrs") is True
            and preflight_process.get("process_vmrss_kib") is not None
            and preflight_process.get("host_cpu_total_ticks") is not None
            and preflight_process.get("host_cpu_idle_ticks") is not None
            and preflight_process.get("process_cpu_ticks") is not None
            and bool(preflight_process.get("gpus"))
            and bool(preflight_process.get("process_gpus"))
            and preflight_process.get("process_gpu_memory_used_mib") is not None
        ),
        "server_pid": args.server_pid,
        "missing_required_gauges": missing_gauges,
        "process": preflight_process,
    }
    await writer.emit("production_telemetry_preflight", **preflight)
    if not preflight["passed"]:
        raise RuntimeError(
            "production telemetry preflight requires a live mistralrs --server-pid, "
            "readable host/process CPU and process RSS, and process-scoped nvidia-smi "
            "GPU data"
        )
    resident_prompt_budget = args.resident_prompt_budget
    if resident_prompt_budget is None:
        resident_prompt_budget = (
            int(sequence_capacity)
            if sequence_capacity is not None and sequence_capacity > 0
            else max(args.concurrencies)
        )
    pool_counts = allocate_prompt_pool_counts(
        args.context_mix,
        resident_prompt_budget,
        args.prompt_pool_size,
    )
    context_pools = {
        length: [
            exact_context(client, length, f"production-{length}-{index}")
            for index in range(pool_counts[length])
        ]
        for length in lengths
    }
    probe_template = retrieval_spec(
        client,
        min(lengths),
        "production-c1-probe",
        args.seed + 9_000_000,
        args.probe_max_tokens,
        {"scenario": "production", "role": "c1_probe"},
    )
    probe_template.extra["ignore_eos"] = False
    semantic_templates = {
        length: (
            probe_template
            if length == min(lengths)
            else retrieval_spec(
                client,
                length,
                f"production-semantic-{length}",
                args.seed + 9_000_000 + index,
                args.probe_max_tokens,
                {
                    "scenario": "production",
                    "role": "semantic_sentinel",
                    "context_tokens": length,
                },
            )
        )
        for index, length in enumerate(lengths)
    }
    for template in semantic_templates.values():
        template.extra["ignore_eos"] = False
    prewarm_metrics = await safe_metrics(client, writer, "production-prewarm-start")
    traffic_prewarm_specs = [
        RequestSpec(
            case_id=f"production-prewarm-{length}-{index}",
            seed=args.seed + 8_000_000 + offset,
            max_tokens=args.prewarm_max_tokens,
            prompt=prompt,
            context_tokens=length,
            tags={"scenario": "production", "role": "prewarm"},
            extra={"ignore_eos": True},
        )
        for offset, (length, index, prompt) in enumerate(
            (length, index, prompt)
            for length, prompts in context_pools.items()
            for index, prompt in enumerate(prompts)
        )
    ]
    sentinel_prewarm_specs = [
        RequestSpec(
            case_id=f"production-prewarm-semantic-{length}",
            seed=template.seed,
            max_tokens=args.prewarm_max_tokens,
            prompt=template.prompt,
            context_tokens=template.context_tokens,
            tags={"scenario": "production", "role": "sentinel_prewarm"},
            extra={**template.extra, "ignore_eos": True},
        )
        for length, template in semantic_templates.items()
    ]
    prewarm_specs = [*traffic_prewarm_specs, *sentinel_prewarm_specs]
    _, prewarm_summary = await run_batch(
        client,
        prewarm_specs,
        min(max(args.concurrencies), len(prewarm_specs)),
        writer,
        "production-prewarm",
        keep_output=False,
    )
    prewarm_cleanup_ok, post_prewarm_metrics, prewarm_cleanup = await poll_for_cleanup(
        client,
        writer,
        prewarm_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "production-prewarm-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    verification_specs = [
        RequestSpec(
            case_id=f"production-residency-{spec.case_id}",
            seed=spec.seed + 100_000,
            max_tokens=args.prewarm_max_tokens,
            prompt=spec.prompt,
            context_tokens=spec.context_tokens,
            tags={"scenario": "production", "role": "residency_verification"},
            extra=dict(spec.extra),
        )
        for spec in prewarm_specs
    ]
    _, verification_summary = await run_batch(
        client,
        verification_specs,
        min(max(args.concurrencies), len(verification_specs)),
        writer,
        "production-residency-verification",
        keep_output=False,
    )
    verification_cleanup_ok, verified_metrics, verification_cleanup = (
        await poll_for_cleanup(
            client,
            writer,
            post_prewarm_metrics,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            "production-residency-cleanup",
            RESIDENT_TRANSIENT_CLEANUP_GAUGES,
        )
    )
    residency_evidence = prefix_cache_evidence(
        post_prewarm_metrics,
        verified_metrics,
        [int(spec.context_tokens or 0) for spec in verification_specs],
        args.min_prefix_reuse_fraction,
        args.kv_block_size_tokens,
        args.speculative_prefix_replay_tokens,
    )
    prewarm_memory = cuda_memory_pressure_evidence(
        prewarm_metrics,
        verified_metrics,
        require_instrumentation=True,
    )
    prewarm_passed = (
        prewarm_summary["errors"] == 0
        and verification_summary["errors"] == 0
        and prewarm_cleanup_ok
        and verification_cleanup_ok
        and residency_evidence["passed"]
        and prewarm_memory["passed"]
    )
    prewarm = {
        "passed": prewarm_passed,
        "sequence_capacity": sequence_capacity,
        "resident_prompt_budget": resident_prompt_budget,
        "pool_counts": pool_counts,
        "prewarm": prewarm_summary,
        "verification": verification_summary,
        "prefix_cache": residency_evidence,
        "cuda_memory": prewarm_memory,
        "prewarm_cleanup": prewarm_cleanup,
        "verification_cleanup": verification_cleanup,
    }
    await writer.emit("production_prewarm_summary", **prewarm)
    if not prewarm_passed:
        raise RuntimeError(
            "production prewarm gate failed; sustained traffic was not started"
        )
    isolated_before_spec = RequestSpec(
        case_id="production-c1-isolated-before",
        seed=probe_template.seed,
        max_tokens=probe_template.max_tokens,
        prompt=probe_template.prompt,
        context_tokens=probe_template.context_tokens,
        tags={**probe_template.tags, "isolation": "before"},
        extra=dict(probe_template.extra),
    )
    isolated_before = await client.stream_request(isolated_before_spec)
    await writer.emit(
        "request",
        phase="production-c1-isolated-before",
        concurrency=1,
        **isolated_before.record(True),
    )
    isolated_before_evidence = production_probe_evidence(
        [isolated_before],
        isolated_before,
        client.tokenizer,
        args.max_repeated_ngram_ratio,
        args.min_output_event_coverage,
        1,
        args.max_probe_ttft_seconds,
        args.max_probe_tpot_seconds,
        1.0,
        args.max_schedule_lateness_seconds,
    )
    isolated_before_cleanup_ok, _, isolated_before_cleanup = await poll_for_cleanup(
        client,
        writer,
        verified_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "production-c1-isolated-before-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    stop = asyncio.Event()
    snapshots: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    initial_metrics, initial_process = await asyncio.gather(
        safe_metrics(client, writer, "production-start"),
        process_telemetry(args.server_pid),
    )
    run_started = time.perf_counter()
    snapshots.append((run_started, initial_metrics, initial_process))
    telemetry_schedule = [run_started]
    telemetry_observed = [run_started]
    telemetry_task = asyncio.create_task(
        telemetry_loop(
            client,
            writer,
            stop,
            args.server_pid,
            snapshots,
            args.telemetry_interval_seconds,
            telemetry_schedule,
            telemetry_observed,
        )
    )
    all_results: list[RequestResult] = []
    phase_summaries: list[dict[str, Any]] = []
    phase_duration = args.duration_seconds / len(args.concurrencies)
    telemetry_terminal_at: float | None = None

    async def run_phase(concurrency: int, phase_index: int) -> None:
        phase = f"production-c{concurrency}"
        phase_metrics_before = await safe_metrics(client, writer, f"{phase}-metrics-start")
        phase_start = time.perf_counter()
        phase_end = phase_start + phase_duration
        comparison_window = args.comparison_window_seconds
        probe_schedule = periodic_schedule(
            phase_start,
            phase_end,
            args.probe_interval_seconds,
        )
        retained_output_event_windows = (
            (phase_start, phase_start + comparison_window),
            (phase_end - comparison_window, phase_end),
            (phase_start, phase_end),
        )
        phase_results: list[RequestResult] = []
        phase_slots = ProductionPhaseSlots.create(
            concurrency,
            args.diagnostic_concurrency,
        )

        async def worker(worker_index: int) -> None:
            rng = random.Random(args.seed + phase_index * 100_000 + worker_index)
            request_index = 0
            while time.perf_counter() < phase_end:
                length = choose_context(rng, args.context_mix)
                prompt_pool = context_pools[length]
                spec = RequestSpec(
                    case_id=f"prod-c{concurrency}-w{worker_index}-r{request_index}",
                    seed=args.seed + phase_index * 1_000_000 + worker_index * 10_000 + request_index,
                    max_tokens=args.max_tokens,
                    prompt=prompt_pool[rng.randrange(len(prompt_pool))],
                    context_tokens=length,
                    tags={"scenario": "production", "role": "traffic", "concurrency": concurrency},
                    extra={"ignore_eos": args.fixed_output_length},
                )
                scheduled_at = time.perf_counter()
                result = await stream_request_with_slot(
                    client,
                    phase_slots.traffic,
                    spec,
                    scheduled_at=scheduled_at,
                    retain_output_event_windows=retained_output_event_windows,
                )
                phase_results.append(result)
                await writer.emit(
                    "request",
                    phase=phase,
                    concurrency=concurrency,
                    **result.record(False),
                )
                request_index += 1

        async def probes() -> None:
            for probe_index, scheduled_at in enumerate(probe_schedule):
                await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
                spec = RequestSpec(
                    case_id=f"probe-c{concurrency}-{probe_index:04d}",
                    seed=probe_template.seed,
                    max_tokens=probe_template.max_tokens,
                    prompt=probe_template.prompt,
                    context_tokens=probe_template.context_tokens,
                    tags={**probe_template.tags, "load": concurrency},
                    extra=dict(probe_template.extra),
                )
                result = await stream_request_with_slot(
                    client,
                    phase_slots.diagnostics,
                    spec,
                    scheduled_at=scheduled_at,
                )
                phase_results.append(result)
                await writer.emit(
                    "request",
                    phase=phase,
                    concurrency=concurrency,
                    **result.record(True),
                )

        async def semantic_sentinels() -> None:
            for stage, fraction in PRODUCTION_SENTINEL_STAGES:
                scheduled_at = phase_start + phase_duration * fraction
                await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
                specs = [
                    RequestSpec(
                        case_id=(
                            f"semantic-c{concurrency}-{stage}-{length}"
                        ),
                        seed=template.seed,
                        max_tokens=template.max_tokens,
                        prompt=template.prompt,
                        context_tokens=template.context_tokens,
                        tags={
                            **template.tags,
                            "role": "semantic_sentinel",
                            "load": concurrency,
                            "sentinel_stage": stage,
                        },
                        extra=dict(template.extra),
                    )
                    for length, template in semantic_templates.items()
                ]
                results = await asyncio.gather(
                    *(
                        stream_request_with_slot(
                            client,
                            phase_slots.diagnostics,
                            spec,
                            scheduled_at=scheduled_at,
                        )
                        for spec in specs
                    )
                )
                phase_results.extend(results)
                for result in results:
                    await writer.emit(
                        "request",
                        phase=phase,
                        concurrency=concurrency,
                        **result.record(True),
                    )

        await writer.emit(
            "production_phase_start",
            phase=phase,
            concurrency=concurrency,
            planned_seconds=phase_duration,
        )
        phase_tasks = [
            *(asyncio.create_task(worker(index)) for index in range(concurrency)),
            asyncio.create_task(probes()),
            asyncio.create_task(semantic_sentinels()),
        ]
        await wait_for_production_phase_tasks(phase_tasks)
        phase_completed = time.perf_counter()
        phase_metrics_after = await safe_metrics(client, writer, f"{phase}-metrics-end")
        wall = phase_completed - phase_start
        summary = summarize_batch(phase_results, wall, concurrency, phase)
        traffic = [result for result in phase_results if result.tags.get("role") == "traffic"]
        probes_only = [result for result in phase_results if result.tags.get("role") == "c1_probe"]
        semantic_only = [
            result
            for result in phase_results
            if result.tags.get("role") == "semantic_sentinel"
        ]
        quality_evidence = production_output_quality_evidence(
            phase_results,
            client.tokenizer,
            args.max_repeated_ngram_ratio,
            args.fixed_output_length,
        )
        quality = quality_evidence["checks"]
        summary["traffic"] = summarize_batch(traffic, wall, concurrency, f"{phase}-traffic")
        summary["probes"] = summarize_batch(probes_only, wall, 1, f"{phase}-probes")
        summary["semantic_sentinels"] = summarize_batch(
            semantic_only,
            wall,
            len(lengths),
            f"{phase}-semantic-sentinels",
        )
        summary["traffic_by_context"] = summarize_context_groups(
            traffic,
            wall,
            concurrency,
            f"{phase}-traffic",
        )
        summary["traffic_fixed_length"] = fixed_length_completion_evidence(
            traffic,
            args.max_tokens,
            args.fixed_output_length,
        )
        summary["quality_checked"] = len(quality)
        summary["quality_failures"] = quality_evidence["failures"]
        summary["quality_evidence"] = quality_evidence
        probe_cadence = scheduled_observation_evidence(
            probe_schedule,
            [
                result.started
                for result in sorted(probes_only, key=lambda item: item.case_id)
            ],
            args.max_schedule_lateness_seconds,
        )
        summary["probe_cadence"] = probe_cadence
        probe_evidence = production_probe_evidence(
            probes_only,
            isolated_before,
            client.tokenizer,
            args.max_repeated_ngram_ratio,
            args.min_output_event_coverage,
            len(probe_schedule),
            args.max_probe_ttft_seconds,
            args.max_probe_tpot_seconds,
            args.max_probe_latency_slowdown,
            args.max_schedule_lateness_seconds,
        )
        summary["probe_fixed_seed_samples"] = len(probes_only)
        summary["probe_fixed_seed_unique_outputs"] = len(
            {
                result.output_transcript
                for result in probes_only
                if result.ok
            }
        )
        summary["probe_fixed_seed_unique_results"] = len(
            {
                fixed_seed_result_signature(result)
                for result in probes_only
                if result.ok
            }
        )
        summary["probe_fixed_seed_exact"] = probe_evidence["exact_results"]
        summary["probe_semantic_checks"] = [
            item["semantic"] for item in probe_evidence["checks"]
        ]
        summary["probe_semantic_ok"] = probe_evidence["semantic_ok"]
        summary["probe_performance"] = probe_evidence
        summary["probe_exact_diagnostics_gated"] = True
        summary["semantic_sentinel_evidence"] = (
            production_semantic_sentinel_evidence(
                semantic_only,
                lengths,
                [stage for stage, _ in PRODUCTION_SENTINEL_STAGES],
                client.tokenizer,
                args.max_repeated_ngram_ratio,
                args.min_output_event_coverage,
                args.max_schedule_lateness_seconds,
            )
        )
        summary["first_vs_last_window"] = compare_time_windows(
            traffic,
            phase_start,
            phase_end,
            comparison_window,
            args.min_output_event_coverage,
            args.min_comparison_window_samples,
        )
        summary["comparison_window_coverage"] = (
            comparison_window_coverage_evidence(
                phase_start,
                phase_end,
                phase_completed,
                comparison_window,
            )
        )
        degradation = summary["first_vs_last_window"]["throughput_degradation_fraction"]
        summary["degradation_ok"] = (
            degradation is not None
            and degradation <= args.max_throughput_degradation_fraction
            and summary["first_vs_last_window"]["output_event_coverage_ok"]
            and summary["first_vs_last_window"]["window_sample_evidence"]["passed"]
            and summary["comparison_window_coverage"]["passed"]
        )
        summary["latency_degradation_ok"] = all(
            value is not None
            and value <= args.max_latency_degradation_fraction
            for value in summary["first_vs_last_window"][
                "latency_degradation_fractions"
            ].values()
        )
        summary["steady_state_output_tok_s"] = summary["first_vs_last_window"][
            "last_output_tok_s_stream_timeline"
        ]
        summary["bounded_phase_output_tok_s"] = summary["first_vs_last_window"][
            "full_phase_output_tok_s_stream_timeline"
        ]
        summary["quality_ok"] = quality_evidence["passed"]
        summary["prefix_cache"] = prefix_cache_evidence(
            phase_metrics_before,
            phase_metrics_after,
            [int(result.context_tokens or 0) for result in phase_results],
            args.min_prefix_reuse_fraction,
            args.kv_block_size_tokens,
            args.speculative_prefix_replay_tokens,
        )
        summary["mtp"] = configured_speculative_evidence(
            phase_metrics_before,
            phase_metrics_after,
            args,
            args.require_mtp,
        )
        summary["cuda_graph"] = cuda_graph_evidence(
            phase_metrics_before,
            phase_metrics_after,
            initial_metrics,
            args.expected_graph_components,
            args.min_cuda_graph_replay_ratio,
        )
        summary["cuda_memory"] = cuda_memory_pressure_evidence(
            phase_metrics_before,
            phase_metrics_after,
            require_instrumentation=True,
        )
        summary["queue_latency_evidence"] = queue_histogram_evidence(
            phase_metrics_before,
            phase_metrics_after,
        )
        summary["queue_latency_histograms"] = summary["queue_latency_evidence"][
            "histograms"
        ]
        summary["queue_latency_instrumentation_complete"] = summary[
            "queue_latency_evidence"
        ]["passed"]
        summary["probe_cadence_complete"] = summary["probe_cadence"][
            "passed"
        ]
        summary["metric_deltas"] = selected_metric_deltas(
            phase_metrics_before,
            phase_metrics_after,
        )
        summary["phase_metrics_ok"] = (
            summary["prefix_cache"]["passed"]
            and summary["mtp"]["passed"]
            and summary["cuda_graph"]["passed"]
            and summary["cuda_memory"]["passed"]
        )
        phase_summaries.append(summary)
        all_results.extend(phase_results)
        await writer.emit("production_phase_summary", phase=phase, **summary)

    try:
        for phase_index, concurrency in enumerate(args.concurrencies):
            await run_phase(concurrency, phase_index)
    finally:
        stop.set()
        telemetry_terminal_at = await telemetry_task
    telemetry_cadence = scheduled_observation_evidence(
        telemetry_schedule,
        telemetry_observed,
        args.max_schedule_lateness_seconds,
    )
    telemetry_cadence["terminal_sample_collected"] = telemetry_terminal_at is not None
    telemetry_cadence["terminal_sample_monotonic_seconds"] = telemetry_terminal_at
    telemetry_cadence["passed"] = (
        telemetry_cadence["passed"]
        and telemetry_cadence["terminal_sample_collected"]
    )
    await writer.emit("production_telemetry_cadence", **telemetry_cadence)
    pre_final_c1_cleanup_ok, _, pre_final_c1_cleanup = await poll_for_cleanup(
        client,
        writer,
        initial_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "production-pre-final-c1-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    isolated_after_spec = RequestSpec(
        case_id="production-c1-isolated-after",
        seed=probe_template.seed,
        max_tokens=probe_template.max_tokens,
        prompt=probe_template.prompt,
        context_tokens=probe_template.context_tokens,
        tags={**probe_template.tags, "isolation": "after"},
        extra=dict(probe_template.extra),
    )
    isolated_after = await client.stream_request(isolated_after_spec)
    await writer.emit(
        "request",
        phase="production-c1-isolated-after",
        concurrency=1,
        **isolated_after.record(True),
    )
    isolated_after_evidence = production_probe_evidence(
        [isolated_after],
        isolated_before,
        client.tokenizer,
        args.max_repeated_ngram_ratio,
        args.min_output_event_coverage,
        1,
        args.max_probe_ttft_seconds,
        args.max_probe_tpot_seconds,
        args.max_final_c1_latency_slowdown,
        args.max_schedule_lateness_seconds,
    )
    final_c1_decode_ratio = ratio(
        request_decode_tok_s(isolated_after),
        request_decode_tok_s(isolated_before),
    )
    isolated_c1_evidence = {
        "passed": (
            isolated_before_evidence["passed"]
            and isolated_after_evidence["passed"]
            and isolated_before_cleanup_ok
            and pre_final_c1_cleanup_ok
            and final_c1_decode_ratio is not None
            and final_c1_decode_ratio >= args.min_final_c1_decode_ratio
        ),
        "before": isolated_before_evidence,
        "after": isolated_after_evidence,
        "decode_throughput_ratio_after_over_before": final_c1_decode_ratio,
        "minimum_decode_throughput_ratio": args.min_final_c1_decode_ratio,
        "before_cleanup_ok": isolated_before_cleanup_ok,
        "before_cleanup": isolated_before_cleanup,
        "pre_after_cleanup_ok": pre_final_c1_cleanup_ok,
        "pre_after_cleanup": pre_final_c1_cleanup,
    }
    run_ended = time.perf_counter()
    cleanup_ok, final_metrics, cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        initial_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "production-final-cleanup",
        RESIDENT_TRANSIENT_CLEANUP_GAUGES,
    )
    final_process = await process_telemetry(args.server_pid)
    final_telemetry_timestamp = time.perf_counter()
    snapshots.append((final_telemetry_timestamp, final_metrics, final_process))
    metrics_delta = selected_metric_deltas(initial_metrics, final_metrics)
    queue_latency_evidence = queue_histogram_evidence(initial_metrics, final_metrics)
    mtp = configured_speculative_evidence(
        initial_metrics,
        final_metrics,
        args,
        args.require_mtp,
    )
    graph = cuda_graph_evidence(
        initial_metrics,
        final_metrics,
        initial_metrics,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    cuda_memory = cuda_memory_pressure_evidence(
        initial_metrics,
        final_metrics,
        require_instrumentation=True,
    )
    all_probes = [
        result for result in all_results if result.tags.get("role") == "c1_probe"
    ]
    all_probe_evidence = production_probe_evidence(
        all_probes,
        isolated_before,
        client.tokenizer,
        args.max_repeated_ngram_ratio,
        args.min_output_event_coverage,
        sum(
            summary["probe_cadence"]["scheduled_count"]
            for summary in phase_summaries
        ),
        args.max_probe_ttft_seconds,
        args.max_probe_tpot_seconds,
        args.max_probe_latency_slowdown,
        args.max_schedule_lateness_seconds,
    )
    probes_exact_across_load = all_probe_evidence["exact_results"]
    probes_semantic_across_load = all_probe_evidence["semantic_ok"]
    throughput = throughput_evidence(
        [
            {
                "concurrency": summary["concurrency"],
                "end_to_end_output_tok_s_common_wall": summary[
                    "steady_state_output_tok_s"
                ],
                "output_tok_s_common_wall": summary["steady_state_output_tok_s"],
                "decode_tok_s_active": summary["decode_tok_s_active"],
            }
            for summary in phase_summaries
        ],
        args.min_output_tok_s_by_concurrency,
        args.min_scaling_efficiency,
        "output_tok_s_common_wall",
    )
    telemetry_gate = telemetry_evidence(
        snapshots,
        args.server_pid,
        args.min_telemetry_samples,
        args.min_telemetry_coverage,
        required_gauges,
        telemetry_cadence,
    )
    memory_gate = production_memory_evidence(
        snapshots,
        ProductionMemoryLimits.from_args(args),
    )
    acceptance_grade = acceptance_grade_evidence(
        args.acceptance_grade,
        args.require_mtp,
        args.expected_graph_components,
    )
    passed = (
        prewarm_passed
        and throughput["passed"]
        and telemetry_gate["passed"]
        and memory_gate["passed"]
        and all(
            summary["errors"] == 0
            and summary["degradation_ok"]
            and summary["latency_degradation_ok"]
            and summary["quality_ok"]
            and summary["traffic_fixed_length"]["passed"]
            and summary["probe_performance"]["passed"]
            and summary["semantic_sentinel_evidence"]["passed"]
            and summary["queue_latency_instrumentation_complete"]
            and summary["probe_cadence_complete"]
            and summary["comparison_window_coverage"]["passed"]
            and summary["phase_metrics_ok"]
            for summary in phase_summaries
        )
        and bool(initial_metrics)
        and bool(final_metrics)
        and mtp["passed"]
        and graph["passed"]
        and cuda_memory["passed"]
        and all_probe_evidence["passed"]
        and isolated_c1_evidence["passed"]
        and queue_latency_evidence["passed"]
        and acceptance_grade["passed"]
        and cleanup_ok
    )
    return {
        "mode": "production",
        "passed": passed,
        "elapsed_seconds": run_ended - run_started,
        "prewarm": prewarm,
        "phase_summaries": phase_summaries,
        "throughput": throughput,
        "metrics_delta": metrics_delta,
        "queue_latency_histograms": queue_latency_evidence["histograms"],
        "queue_latency_evidence": queue_latency_evidence,
        "mtp": mtp,
        "mtp_acceptance_rate": mtp["acceptance_rate"],
        "mtp_mean_accepted_draft_tokens_per_draft": (
            mtp["mean_accepted_draft_tokens_per_draft"]
        ),
        "mtp_mean_advance_tokens_per_target_step": (
            mtp["mean_advance_tokens_per_target_step"]
        ),
        "telemetry_samples": len(snapshots),
        "telemetry": summarize_telemetry(snapshots),
        "telemetry_evidence": telemetry_gate,
        "telemetry_cadence": telemetry_cadence,
        "memory_evidence": memory_gate,
        "cuda_memory": cuda_memory,
        "cuda_graph": graph,
        "cuda_graph_events": graph["events"],
        "cuda_graph_ok": graph["passed"],
        "c1_probe_samples": len(all_probes),
        "c1_probes_exact_across_load": probes_exact_across_load,
        "c1_probe_exact_diagnostics_gated": True,
        "c1_probes_semantic_across_load": probes_semantic_across_load,
        "c1_loaded_probe_evidence": all_probe_evidence,
        "isolated_c1_evidence": isolated_c1_evidence,
        "acceptance_grade": args.acceptance_grade,
        "acceptance_grade_evidence": acceptance_grade,
        "cleanup_ok": cleanup_ok,
        "cleanup_detail": cleanup_detail,
    }


def compare_time_windows(
    results: Sequence[RequestResult],
    started: float,
    ended: float,
    window_seconds: float,
    min_output_event_coverage: float,
    minimum_samples: int = 0,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]

    def first_token_at(result: RequestResult) -> float | None:
        if result.ttft_seconds is None:
            return None
        return result.started + result.ttft_seconds

    first = [
        result
        for result in successful
        if (timestamp := first_token_at(result)) is not None
        and started <= timestamp < started + window_seconds
    ]
    last = [
        result
        for result in successful
        if (timestamp := first_token_at(result)) is not None
        and ended - window_seconds <= timestamp < ended
    ]

    def window_summary(items: Sequence[RequestResult], start: float, end: float) -> dict[str, Any]:
        return summarize_batch(items, max(0.0, end - start), 0, "window")

    first_summary = window_summary(first, started, min(ended, started + window_seconds))
    last_summary = window_summary(last, max(started, ended - window_seconds), ended)
    window_event_counts_available = all(
        len(result.output_event_window_counts) >= 3 for result in successful
    )
    if window_event_counts_available:
        first_event_count = sum(
            result.output_event_window_counts[0] for result in successful
        )
        last_event_count = sum(
            result.output_event_window_counts[1] for result in successful
        )
        full_phase_event_count = sum(
            result.output_event_window_counts[2] for result in successful
        )
    else:
        first_event_count = sum(
            started <= timestamp < started + window_seconds
            for result in successful
            for timestamp in result.output_event_times
        )
        last_event_count = sum(
            ended - window_seconds <= timestamp <= ended
            for result in successful
            for timestamp in result.output_event_times
        )
        full_phase_event_count = sum(
            started <= timestamp < ended
            for result in successful
            for timestamp in result.output_event_times
        )
    window_token_counts_available = all(
        len(result.output_token_window_counts) >= 3 for result in successful
    )
    timed_token_weights_complete = all(
        len(result.output_event_times) == len(result.output_event_token_counts)
        for result in successful
    )
    if window_token_counts_available:
        first_token_count = sum(
            result.output_token_window_counts[0] for result in successful
        )
        last_token_count = sum(
            result.output_token_window_counts[1] for result in successful
        )
        full_phase_token_count = sum(
            result.output_token_window_counts[2] for result in successful
        )
    elif timed_token_weights_complete:
        first_token_count = sum(
            token_count
            for result in successful
            for timestamp, token_count in zip(
                result.output_event_times,
                result.output_event_token_counts,
            )
            if started <= timestamp < started + window_seconds
        )
        last_token_count = sum(
            token_count
            for result in successful
            for timestamp, token_count in zip(
                result.output_event_times,
                result.output_event_token_counts,
            )
            if ended - window_seconds <= timestamp <= ended
        )
        full_phase_token_count = sum(
            token_count
            for result in successful
            for timestamp, token_count in zip(
                result.output_event_times,
                result.output_event_token_counts,
            )
            if started <= timestamp < ended
        )
    else:
        first_token_count = None
        last_token_count = None
        full_phase_token_count = None
    event_coverage = output_event_coverage_evidence(
        successful,
        min_output_event_coverage,
    )
    first_tps = ratio(first_token_count, window_seconds)
    last_tps = ratio(last_token_count, window_seconds)
    full_phase_seconds = ended - started
    full_phase_tps = ratio(full_phase_token_count, full_phase_seconds)
    latency_degradation_fractions = {}
    for name in ("ttft_seconds", "tpot_seconds", "client_queue_seconds"):
        for quantile in ("p95", "p99"):
            first_value = first_summary[name][quantile]
            last_value = last_summary[name][quantile]
            if first_value is not None and first_value > 0 and last_value is not None:
                degradation = last_value / first_value - 1.0
            elif first_value == 0 and last_value == 0:
                degradation = 0.0
            else:
                degradation = None
            latency_degradation_fractions[f"{name}_{quantile}"] = degradation
    window_sample_evidence = {
        "passed": len(first) >= minimum_samples and len(last) >= minimum_samples,
        "minimum_samples_per_window": minimum_samples,
        "first_samples": len(first),
        "last_samples": len(last),
    }
    return {
        "window_seconds": window_seconds,
        "first_window": [started, started + window_seconds],
        "last_window": [ended - window_seconds, ended],
        "latency_window_attribution": "first_token_timestamp",
        "full_phase_window": [started, ended],
        "first": first_summary,
        "last": last_summary,
        "first_timed_output_events": first_event_count,
        "last_timed_output_events": last_event_count,
        "full_phase_timed_output_events": full_phase_event_count,
        "first_timed_output_tokens": first_token_count,
        "last_timed_output_tokens": last_token_count,
        "full_phase_timed_output_tokens": full_phase_token_count,
        "window_counts_available": window_token_counts_available,
        "window_event_counts_available": window_event_counts_available,
        "window_token_counts_available": window_token_counts_available,
        "timed_token_weights_complete": timed_token_weights_complete,
        "first_output_tok_s_stream_timeline": first_tps,
        "last_output_tok_s_stream_timeline": last_tps,
        "full_phase_output_tok_s_stream_timeline": full_phase_tps,
        "timed_output_events": event_coverage["observed_output_events"],
        "timed_output_tokens": event_coverage["observed_output_tokens"],
        **event_coverage,
        "throughput_ratio_last_over_first": ratio(last_tps, first_tps),
        "throughput_degradation_fraction": (
            1.0 - last_tps / first_tps
            if first_tps is not None and first_tps > 0 and last_tps is not None
            else None
        ),
        "latency_degradation_fractions": latency_degradation_fractions,
        "window_sample_evidence": window_sample_evidence,
    }


def image_content(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{data}"


def multimodal_messages(image: str, nonce: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": nonce},
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def normalized_semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


def validate_image_output(
    result: RequestResult,
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
    required_phrases: Sequence[str],
    expected_attributes: Sequence[Sequence[str]],
) -> tuple[bool, dict[str, Any]]:
    sampled_output_valid, sampled_output = validate_sampled_output(
        result,
        tokenizer,
        max_repeated_ngram_ratio,
    )
    generated = result.reasoning_text + "\n" + result.output_text
    if result.tool_calls:
        generated += "\n" + json.dumps(result.tool_calls, sort_keys=True)
    normalized_output = normalized_semantic_text(generated)
    phrase_checks = [
        {
            "phrase": phrase,
            "matched": normalized_semantic_text(phrase) in normalized_output,
        }
        for phrase in required_phrases
    ]
    attribute_checks = []
    for alternatives in expected_attributes:
        matches = [
            phrase
            for phrase in alternatives
            if normalized_semantic_text(phrase) in normalized_output
        ]
        attribute_checks.append(
            {
                "alternatives": list(alternatives),
                "matches": matches,
                "matched": bool(matches),
            }
        )
    semantic_checks = (*phrase_checks, *attribute_checks)
    semantic_oracle_valid = bool(semantic_checks) and all(
        check["matched"] for check in semantic_checks
    )
    valid = sampled_output_valid and semantic_oracle_valid
    return valid, {
        **sampled_output,
        "sampled_output_valid": sampled_output_valid,
        "required_phrase_checks": phrase_checks,
        "expected_attribute_checks": attribute_checks,
        "semantic_oracle_valid": semantic_oracle_valid,
        "valid": valid,
    }


def load_summary_artifact(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        summary = None
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if record.get("event") == "run_summary":
                    summary = record
        if summary is None:
            raise RuntimeError(f"{path} does not contain a run_summary event")
        return summary
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not contain a summary object")
    return value


def comparable_server_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    build = (provenance.get("system_info") or {}).get("build") or {}
    process = provenance.get("process") or {}
    serve = process.get("serve_configuration") or {}
    gpu = provenance.get("gpu_driver") or {}
    realized = provenance.get("realized_kv_configuration") or {}
    serve_fields = (
        "subcommand",
        "model",
        "paged_attn",
        "pa_context_len",
        "pa_memory_mb",
        "pa_memory_fraction",
        "pa_block_size",
        "pa_cache_type",
        "max_seqs",
        "prefix_cache_n",
        "max_num_batched_tokens",
        "max_prefill_chunk_tokens",
        "max_decode_steps_before_prefill",
        "mtp_model",
        "mtp_n_predict",
        "mtp_draft_sampling",
    )
    return {
        "git_revision": build.get("git_revision"),
        "executable_sha256": process.get("executable_sha256"),
        "serve_configuration": {name: serve.get(name) for name in serve_fields},
        "gpus": sorted(
            (
                item.get("uuid"),
                item.get("name"),
                item.get("driver_version"),
                item.get("memory_total_mib"),
            )
            for item in gpu.get("gpus") or []
        ),
        "realized_kv_configuration": {
            name: realized.get(name)
            for name in (
                "blocks_total",
                "sequence_capacity",
                "recurrent_slots_total",
            )
        },
    }


def server_provenance_match_evidence(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_value = comparable_server_provenance(reference)
    candidate_value = comparable_server_provenance(candidate)
    fields = {
        name: {
            "passed": reference_value.get(name) == candidate_value.get(name),
            "reference": reference_value.get(name),
            "candidate": candidate_value.get(name),
        }
        for name in reference_value
    }
    complete = (
        (reference.get("evidence") or {}).get("complete") is True
        and (candidate.get("evidence") or {}).get("complete") is True
    )
    return {
        "passed": complete and all(item["passed"] for item in fields.values()),
        "provenance_complete": complete,
        "fields": fields,
    }


def text_prerequisite_evidence(
    paths: Sequence[Path],
    current_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_stages = {"part1", "adversarial", "production"}
    artifacts = []
    observed_stages: list[str] = []
    for path in paths:
        summary = load_summary_artifact(path)
        mode = summary.get("mode")
        stage = "part1" if mode in ("canary", "compare") else mode
        observed_stages.append(stage)
        acceptance_evidence = summary.get("acceptance_grade_evidence") or {}
        acceptance_complete = (
            summary.get("acceptance_grade") is True
            and acceptance_evidence.get("certification_complete") is True
        )
        mode_specific_complete = True
        if mode in ("canary", "compare"):
            mode_specific_complete = summary.get("coverage_complete") is True
        elif mode == "adversarial":
            mode_specific_complete = (
                (summary.get("max_seqs_queue_evidence") or {}).get("passed") is True
                and acceptance_complete
            )
        elif mode == "production":
            mode_specific_complete = (
                (summary.get("telemetry_evidence") or {}).get("passed") is True
                and acceptance_complete
            )
        artifact = {
            "path": str(path),
            "mode": mode,
            "stage": stage,
            "run_passed": summary.get("passed") is True,
            "mode_specific_complete": mode_specific_complete,
        }
        raw_provenance = summary.get("server_provenance")
        artifact["raw_server_provenance_present"] = isinstance(
            raw_provenance, dict
        )
        artifact["server_provenance_match"] = (
            server_provenance_match_evidence(raw_provenance, current_provenance)
            if isinstance(raw_provenance, dict)
            and current_provenance is not None
            else None
        )
        provenance_passed = (
            current_provenance is None
            or (
                artifact["raw_server_provenance_present"]
                and artifact["server_provenance_match"]["passed"]
            )
        )
        artifact["passed"] = (
            stage in required_stages
            and artifact["run_passed"]
            and mode_specific_complete
            and provenance_passed
        )
        artifacts.append(artifact)
    stage_counts = Counter(observed_stages)
    missing_stages = sorted(required_stages - set(observed_stages))
    duplicate_stages = sorted(
        stage for stage, count in stage_counts.items() if count > 1
    )
    return {
        "passed": (
            not missing_stages
            and not duplicate_stages
            and len(artifacts) == len(required_stages)
            and all(item["passed"] for item in artifacts)
        ),
        "required_stages": sorted(required_stages),
        "missing_stages": missing_stages,
        "duplicate_stages": duplicate_stages,
        "provenance_match_required": current_provenance is not None,
        "artifacts": artifacts,
    }


def fresh_multimodal_server_evidence(snapshot: dict[str, float]) -> dict[str, Any]:
    counters = {}
    for name in MULTIMODAL_FRESH_COUNTERS:
        observed = metric_total(snapshot, name)
        effective = observed if observed is not None else 0.0
        counters[name] = {
            "passed": effective == 0,
            "observed": observed,
            "effective": effective,
            "absent_means_zero": observed is None,
        }
    gauges = {
        name: {
            "passed": (value := metric_total(snapshot, name)) == 0,
            "observed": value,
        }
        for name in MULTIMODAL_FRESH_GAUGES
    }
    encoder_instrumentation = all(
        metric_total(snapshot, name) is not None
        for name in (
            "mistralrs_encoder_cache_hits_total",
            "mistralrs_encoder_cache_misses_total",
        )
    )
    return {
        "passed": (
            bool(snapshot)
            and encoder_instrumentation
            and all(item["passed"] for item in counters.values())
            and all(item["passed"] for item in gauges.values())
        ),
        "encoder_instrumentation_complete": encoder_instrumentation,
        "counters": counters,
        "gauges": gauges,
    }


def multimodal_capability_evidence(
    models: Sequence[dict[str, Any]],
    requested_model: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    candidates = list(models) if requested_model == "default" else [
        model for model in models if model.get("name") == requested_model
    ]
    vision_candidates = [
        model
        for model in candidates
        if "vision" in {
            str(modality).casefold()
            for modality in model.get("input_modalities") or []
        }
    ]
    sequence_capacity = metric_total(metrics, "mistralrs_sequences_capacity")
    return {
        "passed": bool(vision_candidates)
        and sequence_capacity is not None
        and sequence_capacity >= DEFAULT_MULTIMODAL_CONCURRENCY,
        "requested_model": requested_model,
        "candidate_models": candidates,
        "vision_models": vision_candidates,
        "sequence_capacity": sequence_capacity,
        "minimum_sequence_capacity": DEFAULT_MULTIMODAL_CONCURRENCY,
    }


def counter_delta_or_zero(
    before: dict[str, float], after: dict[str, float], metric: str
) -> float:
    return (metric_total(after, metric) or 0.0) - (
        metric_total(before, metric) or 0.0
    )


def encoder_cache_transition_evidence(
    before: dict[str, float],
    after: dict[str, float],
    expected: str,
) -> dict[str, Any]:
    hits = metric_delta(before, after, "mistralrs_encoder_cache_hits_total")
    misses = metric_delta(before, after, "mistralrs_encoder_cache_misses_total")
    paged_reused = counter_delta_or_zero(
        before,
        after,
        "mistralrs_prefix_cache_tokens_reused_total",
    )
    instrumentation_complete = hits is not None and misses is not None
    if expected == "cold":
        expected_transition = (
            instrumentation_complete
            and misses >= 1
            and hits == 0
        )
    elif expected == "hit":
        expected_transition = (
            instrumentation_complete
            and hits >= 1
            and misses == 0
        )
    else:
        raise ValueError(f"unsupported encoder-cache transition {expected!r}")
    return {
        "passed": expected_transition and paged_reused == 0,
        "expected": expected,
        "instrumentation_complete": instrumentation_complete,
        "encoder_cache_hits": hits,
        "encoder_cache_misses": misses,
        "paged_attention_tokens_reused": paged_reused,
        "paged_attention_reuse_absent": paged_reused == 0,
    }


async def poll_multimodal_transition(
    client: SoakClient,
    writer: JsonlWriter,
    baseline: dict[str, float],
    expected: str,
    timeout_seconds: float,
    poll_seconds: float,
    phase: str,
) -> tuple[bool, dict[str, float], dict[str, Any]]:
    deadline = time.perf_counter() + timeout_seconds
    last_snapshot: dict[str, float] = {}
    last_evidence: dict[str, Any] = {
        "passed": False,
        "expected": expected,
    }
    while time.perf_counter() < deadline:
        last_snapshot = await safe_metrics(client, writer, phase)
        last_evidence = encoder_cache_transition_evidence(
            baseline,
            last_snapshot,
            expected,
        )
        cleanup = cleanup_evidence(
            baseline,
            last_snapshot,
            {
                gauge: metric_total(baseline, gauge)
                for gauge in MULTIMODAL_TRANSIENT_CLEANUP_GAUGES
            },
            {
                gauge: metric_total(last_snapshot, gauge)
                for gauge in MULTIMODAL_TRANSIENT_CLEANUP_GAUGES
            },
        )
        quiescent = all(
            (value := metric_total(last_snapshot, gauge)) is not None
            and value <= (metric_total(baseline, gauge) or 0.0)
            for gauge in MULTIMODAL_TRANSIENT_CLEANUP_GAUGES
        )
        last_evidence["quiescent"] = quiescent
        last_evidence["cleanup"] = cleanup
        if last_evidence["passed"] and quiescent:
            return True, last_snapshot, last_evidence
        if (
            last_evidence.get("encoder_cache_hits", 0) not in (None, 0)
            and expected == "cold"
        ) or (
            last_evidence.get("encoder_cache_misses", 0) not in (None, 0)
            and expected == "hit"
        ) or last_evidence.get("paged_attention_tokens_reused", 0) != 0:
            break
        await asyncio.sleep(poll_seconds)
    return False, last_snapshot, last_evidence


def nominal_multimodal_phase_summary(
    results: Sequence[RequestResult],
    started: float,
    ended: float,
    drained_at: float,
    concurrency: int,
    name: str,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    offered = [
        result
        for result in successful
        if result.started - result.client_queue_seconds < ended
    ]
    first_token_in_window = sum(
        result.ttft_seconds is not None
        and result.started + result.ttft_seconds < ended
        for result in offered
    )
    token_counts_complete = all(
        len(result.output_token_window_counts) == 1 for result in results
    )
    output_tokens = (
        sum(result.output_token_window_counts[0] for result in successful)
        if token_counts_complete
        else None
    )
    wall_seconds = ended - started
    last_request_ended = max((result.ended for result in results), default=None)
    drain_complete = (
        last_request_ended is None
        or last_request_ended <= drained_at + SCHEDULE_TIME_EPSILON_SECONDS
    )
    return {
        **summarize_batch(offered, wall_seconds, concurrency, name),
        "submitted_requests": len(results),
        "successful_requests": len(successful),
        "offered_in_nominal_window_requests": len(offered),
        "first_token_in_nominal_window_requests": first_token_in_window,
        "nominal_window": [started, ended],
        "nominal_window_seconds": wall_seconds,
        "nominal_output_token_counts_complete": token_counts_complete,
        "nominal_output_tokens": output_tokens,
        "nominal_output_tok_s": ratio(output_tokens, wall_seconds),
        "offered_ttft_seconds": distribution(
            result.client_queue_seconds + result.ttft_seconds
            for result in offered
            if result.ttft_seconds is not None
        ),
        "requests_ending_after_nominal_window": sum(
            result.ended > ended for result in results
        ),
        "last_request_ended": last_request_ended,
        "drained_at": drained_at,
        "drain_complete": drain_complete,
    }


def multimodal_phase_performance_evidence(
    baseline: dict[str, Any],
    mixed_text: dict[str, Any],
    mixed_all: dict[str, Any],
    recovery: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    mixed_throughput_ratio = ratio(
        mixed_text.get("nominal_output_tok_s"),
        baseline.get("nominal_output_tok_s"),
    )
    mixed_tpot_ratio = ratio(
        (mixed_text.get("tpot_seconds") or {}).get("p95"),
        (baseline.get("tpot_seconds") or {}).get("p95"),
    )
    mixed_ttft_p99 = (mixed_all.get("offered_ttft_seconds") or {}).get("p99")
    recovery_throughput_ratio = ratio(
        recovery.get("nominal_output_tok_s"),
        baseline.get("nominal_output_tok_s"),
    )
    checks = {
        "mixed_throughput": {
            "passed": mixed_throughput_ratio is not None
            and mixed_throughput_ratio >= args.min_mixed_throughput_ratio,
            "observed_ratio": mixed_throughput_ratio,
            "minimum_ratio": args.min_mixed_throughput_ratio,
        },
        "mixed_tpot_p95": {
            "passed": mixed_tpot_ratio is not None
            and mixed_tpot_ratio <= args.max_mixed_tpot_ratio,
            "observed_ratio": mixed_tpot_ratio,
            "maximum_ratio": args.max_mixed_tpot_ratio,
        },
        "mixed_ttft_p99": {
            "passed": mixed_ttft_p99 is not None
            and mixed_ttft_p99 <= args.max_mixed_ttft_p99_seconds,
            "observed_seconds": mixed_ttft_p99,
            "maximum_seconds": args.max_mixed_ttft_p99_seconds,
            "includes_shared_slot_queue": True,
        },
        "recovery_throughput": {
            "passed": recovery_throughput_ratio is not None
            and recovery_throughput_ratio >= args.min_recovery_throughput_ratio,
            "observed_ratio": recovery_throughput_ratio,
            "minimum_ratio": args.min_recovery_throughput_ratio,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


async def multimodal_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
    server_provenance: dict[str, Any],
) -> dict[str, Any]:
    prerequisites = text_prerequisite_evidence(
        args.text_prerequisite_artifacts,
        server_provenance,
    )
    await writer.emit("multimodal_text_prerequisites", **prerequisites)
    if not prerequisites["passed"]:
        return {
            "mode": "multimodal",
            "passed": False,
            "text_prerequisites": prerequisites,
        }
    try:
        models = await client.ui_models()
    except Exception as exc:
        models = []
        await writer.emit(
            "multimodal_capability_error",
            error=f"{type(exc).__name__}: {exc}",
        )
    fresh_deadline = time.perf_counter() + args.cleanup_timeout_seconds
    initial_metrics: dict[str, float] = {}
    fresh_server = {"passed": False}
    while time.perf_counter() < fresh_deadline:
        initial_metrics = await safe_metrics(client, writer, "multimodal-fresh-start")
        fresh_server = fresh_multimodal_server_evidence(initial_metrics)
        if fresh_server["passed"]:
            break
        dirty = any(
            item.get("effective", item.get("observed")) not in (None, 0)
            for group in (fresh_server.get("counters", {}), fresh_server.get("gauges", {}))
            for item in group.values()
        )
        if dirty:
            break
        await asyncio.sleep(args.cleanup_poll_seconds)
    capability = multimodal_capability_evidence(
        models,
        client.model,
        initial_metrics,
    )
    preflight_process = await process_telemetry(args.server_pid)
    required_gauges = production_required_gauges(args.expected_graph_components)
    missing_gauges = [
        gauge
        for gauge in required_gauges
        if metric_total(initial_metrics, gauge) is None
    ]
    process_preflight = {
        "passed": (
            args.server_pid is not None
            and preflight_process.get("process_is_mistralrs") is True
            and preflight_process.get("process_vmrss_kib") is not None
            and preflight_process.get("host_cpu_total_ticks") is not None
            and preflight_process.get("host_cpu_idle_ticks") is not None
            and preflight_process.get("process_cpu_ticks") is not None
            and bool(preflight_process.get("gpus"))
            and bool(preflight_process.get("process_gpus"))
            and preflight_process.get("process_gpu_memory_used_mib") is not None
            and not missing_gauges
        ),
        "server_pid": args.server_pid,
        "missing_required_gauges": missing_gauges,
        "process": preflight_process,
    }
    preflight = {
        "passed": fresh_server["passed"]
        and capability["passed"]
        and process_preflight["passed"],
        "fresh_server": fresh_server,
        "capability": capability,
        "process_telemetry": process_preflight,
    }
    await writer.emit("multimodal_preflight", **preflight)
    if not preflight["passed"]:
        return {
            "mode": "multimodal",
            "passed": False,
            "text_prerequisites": prerequisites,
            "preflight": preflight,
        }

    image_path = Path(args.image)
    image_bytes_sha256 = hash_file(image_path) if image_path.is_file() else None
    image = image_content(str(args.image))
    image_seed = args.seed + 100_000
    cold_before = initial_metrics
    cold_spec = RequestSpec(
        case_id="multimodal-image-cold",
        seed=image_seed,
        max_tokens=args.image_max_tokens,
        messages=multimodal_messages(
            image,
            f"Cold encoder-cache nonce {stable_hash(str(image_seed))[:24]}.",
            args.image_prompt,
        ),
        tags={"scenario": "multimodal", "role": "image", "stage": "cold"},
    )
    cold, cold_summary = await run_batch(
        client, [cold_spec], 1, writer, "multimodal-image-cold", keep_output=True
    )
    cold_transition_ok, cold_after, cold_transition = await poll_multimodal_transition(
        client,
        writer,
        cold_before,
        "cold",
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "multimodal-image-cold-transition",
    )
    hit_spec = RequestSpec(
        case_id="multimodal-image-hit",
        seed=image_seed + 1,
        max_tokens=args.image_max_tokens,
        messages=multimodal_messages(
            image,
            f"Hit encoder-cache nonce {stable_hash(str(image_seed + 1))[:24]}.",
            args.image_prompt,
        ),
        tags={"scenario": "multimodal", "role": "image", "stage": "hit"},
    )
    hit, hit_summary = await run_batch(
        client, [hit_spec], 1, writer, "multimodal-image-hit", keep_output=True
    )
    hit_transition_ok, hit_after, hit_transition = await poll_multimodal_transition(
        client,
        writer,
        cold_after,
        "hit",
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "multimodal-image-hit-transition",
    )
    cache_proof = {
        "passed": (
            cold_summary["errors"] == 0
            and hit_summary["errors"] == 0
            and cold_transition_ok
            and hit_transition_ok
        ),
        "cold": cold_transition,
        "hit": hit_transition,
        "same_image_bytes_sha256": image_bytes_sha256,
        "new_nonce_per_request": True,
        "optional_exact_replay": {
            "equal": cold[0].output_transcript == hit[0].output_transcript,
            "used_as_cache_hit_proof": False,
        },
    }
    await writer.emit("multimodal_encoder_cache_evidence", **cache_proof)

    stop = asyncio.Event()
    snapshots: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    telemetry_initial_metrics = hit_after
    telemetry_initial_process = await process_telemetry(args.server_pid)
    traffic_started = time.perf_counter()
    snapshots.append(
        (traffic_started, telemetry_initial_metrics, telemetry_initial_process)
    )
    telemetry_schedule = [traffic_started]
    telemetry_observed = [traffic_started]
    telemetry_task = asyncio.create_task(
        telemetry_loop(
            client,
            writer,
            stop,
            args.server_pid,
            snapshots,
            args.telemetry_interval_seconds,
            telemetry_schedule,
            telemetry_observed,
        )
    )
    slots = asyncio.Semaphore(args.concurrency)
    phase_results: dict[str, list[RequestResult]] = {}
    phase_summaries: dict[str, dict[str, Any]] = {}
    phase_cleanups: dict[str, dict[str, Any]] = {}
    all_traffic_results: list[RequestResult] = []

    async def run_phase(name: str, inject_images: bool, phase_index: int) -> None:
        phase_metrics_before = await safe_metrics(client, writer, f"{name}-metrics-start")
        phase_start = time.perf_counter()
        phase_end = phase_start + args.phase_duration_seconds
        results: list[RequestResult] = []
        image_schedule: list[float] = []
        text_counter = 0

        async def text_worker(worker_index: int) -> None:
            nonlocal text_counter
            local_index = 0
            while time.perf_counter() < phase_end:
                sequence = text_counter
                text_counter += 1
                case_id = f"{name}-text-{sequence}"
                spec = RequestSpec(
                    case_id=case_id,
                    seed=args.seed + phase_index * 1_000_000 + sequence,
                    max_tokens=args.max_tokens,
                    prompt=(
                        f"Request nonce {stable_hash(case_id)[:24]}.\n"
                        f"{CANARY_PROMPTS[(worker_index + local_index) % len(CANARY_PROMPTS)]}"
                    ),
                    tags={
                        "scenario": "multimodal",
                        "role": "text",
                        "phase": name,
                    },
                    extra={"ignore_eos": True},
                )
                result = await stream_request_with_slot(
                    client,
                    slots,
                    spec,
                    scheduled_at=time.perf_counter(),
                    retain_output_event_windows=((phase_start, phase_end),),
                )
                results.append(result)
                await writer.emit(
                    "request",
                    phase=name,
                    concurrency=args.concurrency,
                    **result.record(False),
                )
                local_index += 1

        async def image_request(index: int, scheduled_at: float) -> None:
            case_id = f"{name}-image-{index}"
            spec = RequestSpec(
                case_id=case_id,
                seed=args.seed + 10_000_000 + index,
                max_tokens=args.image_max_tokens,
                messages=multimodal_messages(
                    image,
                    f"Mixed-load image nonce {stable_hash(case_id)[:24]}.",
                    args.image_prompt,
                ),
                tags={
                    "scenario": "multimodal",
                    "role": "image",
                    "phase": name,
                    "scheduled_monotonic_seconds": scheduled_at,
                },
            )
            result = await stream_request_with_slot(
                client,
                slots,
                spec,
                scheduled_at=scheduled_at,
                retain_output_event_windows=((phase_start, phase_end),),
            )
            results.append(result)
            await writer.emit(
                "request",
                phase=name,
                concurrency=args.concurrency,
                **result.record(True),
            )

        async def image_scheduler() -> None:
            tasks = []
            image_schedule.extend(
                periodic_schedule(
                    phase_start,
                    phase_end,
                    args.image_interval_seconds,
                    include_start=True,
                )
            )
            for index, scheduled_at in enumerate(image_schedule):
                delay = scheduled_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(
                    asyncio.create_task(image_request(index, scheduled_at))
                )
            if tasks:
                await asyncio.gather(*tasks)

        await writer.emit(
            "multimodal_phase_start",
            phase=name,
            concurrency=args.concurrency,
            planned_seconds=args.phase_duration_seconds,
            image_interval_seconds=(
                args.image_interval_seconds if inject_images else None
            ),
        )
        workers = [
            asyncio.create_task(text_worker(index))
            for index in range(args.concurrency)
        ]
        tasks: list[asyncio.Task[None]] = workers
        if inject_images:
            tasks = [*tasks, asyncio.create_task(image_scheduler())]
        await asyncio.gather(*tasks)
        phase_drained_at = time.perf_counter()
        cleanup_ok, phase_metrics_after, cleanup = await poll_for_cleanup(
            client,
            writer,
            phase_metrics_before,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            f"{name}-cleanup",
            MULTIMODAL_TRANSIENT_CLEANUP_GAUGES,
        )
        text_results = [
            result for result in results if result.tags.get("role") == "text"
        ]
        image_results = [
            result for result in results if result.tags.get("role") == "image"
        ]
        text_summary = nominal_multimodal_phase_summary(
            text_results,
            phase_start,
            phase_end,
            phase_drained_at,
            args.concurrency,
            f"{name}-text",
        )
        all_summary = nominal_multimodal_phase_summary(
            results,
            phase_start,
            phase_end,
            phase_drained_at,
            args.concurrency,
            name,
        )
        text_fixed_length = fixed_length_completion_evidence(
            text_results,
            args.max_tokens,
            True,
        )
        text_sample_count = sum(
            result.ok
            and result.completion_tokens == args.max_tokens
            and result.finish_reason == "length"
            and result.started - result.client_queue_seconds < phase_end
            for result in text_results
        )
        text_event_coverage = output_event_coverage_evidence(
            text_results,
            args.min_output_event_coverage,
        )
        ordered_image_results = sorted(
            image_results,
            key=lambda result: result.tags.get("scheduled_monotonic_seconds", math.inf),
        )
        image_schedule_evidence = (
            scheduled_observation_evidence(
                image_schedule,
                [result.started for result in ordered_image_results],
                args.max_schedule_lateness_seconds,
            )
            if inject_images
            else {
                "passed": not image_schedule and not image_results,
                "scheduled_count": len(image_schedule),
                "observed_count": len(image_results),
                "not_applicable": True,
            }
        )
        phase_integrity = {
            "passed": (
                text_summary["nominal_output_token_counts_complete"]
                and all_summary["nominal_output_token_counts_complete"]
                and text_summary["drain_complete"]
                and all_summary["drain_complete"]
                and text_event_coverage["output_token_coverage_ok"]
                and image_schedule_evidence["passed"]
            ),
            "nominal_output_token_counts_complete": (
                text_summary["nominal_output_token_counts_complete"]
                and all_summary["nominal_output_token_counts_complete"]
            ),
            "drain_complete": (
                text_summary["drain_complete"] and all_summary["drain_complete"]
            ),
            "text_output_event_coverage": text_event_coverage,
            "image_schedule": image_schedule_evidence,
        }
        summary = {
            "phase": name,
            "planned_window": [phase_start, phase_end],
            "drained_at": phase_drained_at,
            "drain_seconds": max(0.0, phase_drained_at - phase_end),
            "text": text_summary,
            "all": all_summary,
            "text_fixed_length": text_fixed_length,
            "full_length_text_requests_in_window": text_sample_count,
            "minimum_full_length_text_requests": args.min_text_requests_per_phase,
            "text_sample_count_passed": (
                text_sample_count >= args.min_text_requests_per_phase
            ),
            "image_requests": len(image_results),
            "phase_integrity": phase_integrity,
            "cleanup_ok": cleanup_ok,
            "cleanup": cleanup,
            "metric_deltas": selected_metric_deltas(
                phase_metrics_before,
                phase_metrics_after,
            ),
        }
        phase_results[name] = results
        phase_summaries[name] = summary
        phase_cleanups[name] = cleanup
        all_traffic_results.extend(results)
        await writer.emit("multimodal_phase_summary", **summary)

    telemetry_terminal_at: float | None = None
    try:
        await run_phase("multimodal-text-baseline", False, 0)
        await run_phase("multimodal-mixed", True, 1)
        await run_phase("multimodal-text-recovery", False, 2)
    finally:
        final_cleanup_ok, final_metrics, final_cleanup = await poll_for_cleanup(
            client,
            writer,
            telemetry_initial_metrics,
            args.cleanup_timeout_seconds,
            args.cleanup_poll_seconds,
            "multimodal-final-cleanup",
            MULTIMODAL_TRANSIENT_CLEANUP_GAUGES,
        )
        stop.set()
        telemetry_terminal_at = await telemetry_task

    telemetry_cadence = scheduled_observation_evidence(
        telemetry_schedule,
        telemetry_observed,
        args.max_schedule_lateness_seconds,
    )
    telemetry_cadence["terminal_sample_collected"] = telemetry_terminal_at is not None
    telemetry_cadence["terminal_sample_monotonic_seconds"] = telemetry_terminal_at
    telemetry_cadence["passed"] = (
        telemetry_cadence["passed"]
        and telemetry_cadence["terminal_sample_collected"]
    )
    required_gauges = production_required_gauges(args.expected_graph_components)
    telemetry_gate = telemetry_evidence(
        snapshots,
        args.server_pid,
        args.min_telemetry_samples,
        args.min_telemetry_coverage,
        required_gauges,
        telemetry_cadence,
    )
    memory_limits = ProductionMemoryLimits(
        min_coverage=args.min_telemetry_coverage,
        max_process_rss_drift_mib=args.max_process_rss_drift_mib,
        max_process_rss_drift_fraction=args.max_process_rss_drift_fraction,
        max_process_rss_high_water_mib=args.max_process_rss_high_water_mib,
        max_gpu_memory_drift_mib=args.max_gpu_memory_drift_mib,
        max_gpu_memory_high_water_mib=args.max_gpu_memory_high_water_mib,
        max_kv_block_utilization=args.max_kv_block_utilization,
        max_recurrent_slot_utilization=args.max_recurrent_slot_utilization,
        require_dflash_windowed_kv=False,
    )
    memory_gate = multimodal_memory_evidence(snapshots, memory_limits)

    text_checks = []
    image_checks = []
    for result in all_traffic_results:
        if result.tags.get("role") == "image":
            valid, evidence = validate_image_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
                args.image_required_phrases,
                args.image_expected_attributes,
            )
            image_checks.append(
                {"case_id": result.case_id, "valid": valid, **evidence}
            )
        else:
            valid, evidence = validate_sampled_output(
                result,
                client.tokenizer,
                args.max_repeated_ngram_ratio,
            )
            text_checks.append(
                {"case_id": result.case_id, "valid": valid, **evidence}
            )
    for result in (*cold, *hit):
        valid, evidence = validate_image_output(
            result,
            client.tokenizer,
            args.max_repeated_ngram_ratio,
            args.image_required_phrases,
            args.image_expected_attributes,
        )
        image_checks.append(
            {"case_id": result.case_id, "valid": valid, **evidence}
        )
    mixed_image_results = [
        result
        for result in phase_results["multimodal-mixed"]
        if result.tags.get("role") == "image"
    ]
    semantic_image_requests = sum(result.ok for result in mixed_image_results)
    quality = {
        "passed": (
            bool(text_checks)
            and all(check["valid"] for check in text_checks)
            and len(image_checks) >= args.min_image_requests + 2
            and all(check["valid"] for check in image_checks)
            and semantic_image_requests >= args.min_image_requests
            and all(
                summary["text_fixed_length"]["passed"]
                and summary["text_sample_count_passed"]
                and summary["phase_integrity"]["passed"]
                for summary in phase_summaries.values()
            )
        ),
        "text_checks": text_checks,
        "image_checks": image_checks,
        "semantic_image_requests_in_mixed_phase": semantic_image_requests,
        "minimum_semantic_image_requests": args.min_image_requests,
    }
    performance = multimodal_phase_performance_evidence(
        phase_summaries["multimodal-text-baseline"]["text"],
        phase_summaries["multimodal-mixed"]["text"],
        phase_summaries["multimodal-mixed"]["all"],
        phase_summaries["multimodal-text-recovery"]["text"],
        args,
    )
    mtp = configured_speculative_evidence(
        initial_metrics,
        final_metrics,
        args,
        args.require_mtp,
    )
    graph = cuda_graph_evidence(
        initial_metrics,
        final_metrics,
        initial_metrics,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    cuda_memory = cuda_memory_pressure_evidence(
        initial_metrics,
        final_metrics,
        require_instrumentation=True,
    )
    sampling = production_sampling_policy_evidence(
        PART1_PRODUCTION_SAMPLING_POLICY,
        client.policy,
    )
    acceptance = acceptance_grade_evidence(
        args.acceptance_grade,
        args.require_mtp,
        args.expected_graph_components,
    )
    phase_cleanup_complete = all(
        summary["cleanup_ok"] for summary in phase_summaries.values()
    )
    passed = (
        preflight["passed"]
        and cache_proof["passed"]
        and quality["passed"]
        and performance["passed"]
        and phase_cleanup_complete
        and final_cleanup_ok
        and telemetry_gate["passed"]
        and memory_gate["passed"]
        and mtp["passed"]
        and graph["passed"]
        and cuda_memory["passed"]
        and sampling["passed"]
        and acceptance["passed"]
    )
    return {
        "mode": "multimodal",
        "passed": passed,
        "text_prerequisites": prerequisites,
        "preflight": preflight,
        "encoder_cache": cache_proof,
        "image_oracle": {
            "path": str(args.image),
            "sha256": image_bytes_sha256,
            "required_phrases": args.image_required_phrases,
            "expected_attributes": args.image_expected_attributes,
        },
        "quality": quality,
        "performance": performance,
        "phases": phase_summaries,
        "cold": cold_summary,
        "hit": hit_summary,
        "metrics_delta": selected_metric_deltas(initial_metrics, final_metrics),
        "mtp": mtp,
        "cuda_graph": graph,
        "cuda_memory": cuda_memory,
        "telemetry": summarize_telemetry(snapshots),
        "telemetry_evidence": telemetry_gate,
        "telemetry_cadence": telemetry_cadence,
        "memory_evidence": memory_gate,
        "production_sampling": sampling,
        "acceptance_grade": args.acceptance_grade,
        "acceptance_grade_evidence": acceptance,
        "phase_cleanup_complete": phase_cleanup_complete,
        "phase_cleanups": phase_cleanups,
        "final_cleanup_ok": final_cleanup_ok,
        "final_cleanup": final_cleanup,
    }


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "unused"))
    parser.add_argument("--tokenizer", help="tokenizer.json path or Hugging Face tokenizer ID")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument(
        "--server-pid",
        type=int,
        help="PID of the exact server process, used for binary and command provenance",
    )
    parser.add_argument(
        "--require-server-provenance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "fail unless git SHA, binary, serve command, GPU driver, and KV configuration "
            "are captured; production, prefix-pressure, and multimodal always require this"
        ),
    )
    parser.add_argument("--output", type=Path, help="JSONL evidence path")
    return parser


def add_speculative_evidence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-mtp-acceptance-rate",
        type=float,
        default=DEFAULT_MIN_MTP_ACCEPTANCE_RATE,
        help="minimum accepted/proposed draft-token ratio when MTP is required",
    )
    parser.add_argument(
        "--min-mtp-mean-advance",
        type=float,
        default=DEFAULT_MIN_MTP_MEAN_ADVANCE,
        help="minimum mean output tokens advanced per target verification step",
    )
    parser.add_argument(
        "--min-mtp-proposal-depth",
        type=float,
        default=DEFAULT_MIN_MTP_PROPOSAL_DEPTH,
        help="minimum mean proposed draft tokens per speculative sequence",
    )
    parser.add_argument(
        "--max-sparse-verifier-fallback-ratio",
        type=float,
        default=DEFAULT_MAX_SPARSE_VERIFIER_FALLBACK_RATIO,
        help="maximum CPU fallback fraction for sparse speculative verification",
    )
    parser.add_argument(
        "--min-sparse-verifier-accounting-coverage",
        type=float,
        default=DEFAULT_MIN_SPARSE_VERIFIER_ACCOUNTING_COVERAGE,
        help=(
            "minimum fraction of speculative rows accounted for by CUDA sparse "
            "verification or its explicit CPU fallback"
        ),
    )


def add_prefix_pressure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefix-pressure-namespace",
        help="unique prompt namespace; defaults to the non-overwritable evidence artifact identity",
    )
    parser.add_argument("--prefix-context-tokens", type=int, default=8_192)
    parser.add_argument(
        "--min-prefix-reuse-fraction",
        type=float,
        default=DEFAULT_MIN_PREFIX_REUSE_FRACTION,
    )
    parser.add_argument("--prefix-pressure-entries", type=int, default=24)
    parser.add_argument(
        "--prefix-pressure-context-tokens", type=int, default=100_000
    )
    parser.add_argument(
        "--prefix-pressure-capacity-fraction",
        type=float,
        default=DEFAULT_PREFIX_PRESSURE_CAPACITY_FRACTION,
    )
    parser.add_argument(
        "--prefix-pressure-kv-headroom-fraction",
        type=float,
        default=DEFAULT_PREFIX_PRESSURE_KV_HEADROOM_FRACTION,
    )
    parser.add_argument(
        "--prefix-pressure-max-entries",
        type=int,
        default=DEFAULT_PREFIX_PRESSURE_MAX_ENTRIES,
    )
    parser.add_argument(
        "--kv-block-size-tokens", type=int, default=DEFAULT_KV_BLOCK_SIZE_TOKENS
    )
    parser.add_argument(
        "--speculative-prefix-replay-tokens",
        type=int,
        default=DEFAULT_SPECULATIVE_PREFIX_REPLAY_TOKENS,
    )
    parser.add_argument("--prefix-pressure-max-tokens", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production soak testing for mistral.rs and other OpenAI-compatible servers"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = common_parser()

    canary = subparsers.add_parser("canary", parents=[common])
    add_speculative_evidence_args(canary)
    canary.add_argument(
        "--sampling-policy",
        choices=PART1_SAMPLING_POLICIES,
        default=PART1_PRODUCTION_SAMPLING_POLICY,
        help=(
            "named Part 1 sampling contract; production requires the default "
            "temperature/top-p/top-k/min-p/repetition-penalty values"
        ),
    )
    canary.add_argument("--concurrencies", type=parse_int_list, default=DEFAULT_CONCURRENCIES)
    canary.add_argument("--requests", type=int, default=16)
    canary.add_argument("--max-tokens", type=int, default=128)
    canary.add_argument("--max-length-tokens", type=int, default=17)
    canary.add_argument("--skip-edge-cases", action="store_true")
    canary.add_argument("--require-part1-complete", action="store_true")
    canary.add_argument("--require-mtp", action="store_true")
    canary.add_argument("--eos-token-id", type=int)
    canary.add_argument("--reference-url", help="target-only server for statistical comparison")
    canary.add_argument("--stat-max-ks", type=float, default=0.35)
    canary.add_argument("--stat-max-js", type=float, default=0.20)
    canary.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target",),
    )
    canary.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    canary.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )

    sweep = subparsers.add_parser("sweep", parents=[common])
    sweep.add_argument("--concurrencies", type=parse_int_list, default=DEFAULT_CONCURRENCIES)
    sweep.add_argument("--requests", type=int, default=32)
    sweep.add_argument("--context-tokens", type=int)
    sweep.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    resident_decode = subparsers.add_parser("resident-decode", parents=[common])
    add_speculative_evidence_args(resident_decode)
    resident_decode.add_argument(
        "--context-lengths",
        type=parse_int_list,
        default=DEFAULT_RESIDENT_CONTEXT_LENGTHS,
    )
    resident_decode.add_argument(
        "--concurrencies",
        type=parse_int_list,
        default=DEFAULT_RESIDENT_CONCURRENCIES,
    )
    resident_decode.add_argument(
        "--requests", type=int, default=DEFAULT_RESIDENT_REQUESTS
    )
    resident_decode.add_argument("--max-tokens", type=int, default=128)
    resident_decode.add_argument("--warmup-max-tokens", type=int, default=1)
    resident_decode.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    resident_decode.add_argument(
        "--cleanup-poll-seconds",
        type=float,
        default=DEFAULT_CLEANUP_POLL_SECONDS,
    )
    resident_decode.add_argument("--stat-max-ks", type=float, default=0.60)
    resident_decode.add_argument("--stat-max-js", type=float, default=0.35)
    resident_decode.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    resident_decode.add_argument(
        "--min-prefix-reuse-fraction",
        type=float,
        default=DEFAULT_MIN_PREFIX_REUSE_FRACTION,
    )
    resident_decode.add_argument(
        "--kv-block-size-tokens", type=int, default=DEFAULT_KV_BLOCK_SIZE_TOKENS
    )
    resident_decode.add_argument(
        "--speculative-prefix-replay-tokens",
        type=int,
        default=DEFAULT_SPECULATIVE_PREFIX_REPLAY_TOKENS,
    )
    resident_decode.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target",),
    )
    resident_decode.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    resident_decode.add_argument(
        "--min-final-c1-throughput-ratio",
        type=float,
        default=0.95,
    )
    resident_decode.add_argument(
        "--final-c1-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    resident_decode.add_argument("--require-mtp", action="store_true")
    resident_decode.add_argument(
        "--min-output-tok-s-by-concurrency",
        type=parse_concurrency_thresholds,
        required=True,
    )
    resident_decode.add_argument(
        "--min-scaling-efficiency",
        type=float,
        required=True,
    )

    adversarial = subparsers.add_parser("adversarial", parents=[common])
    add_speculative_evidence_args(adversarial)
    adversarial.add_argument(
        "--context-lengths", type=parse_int_list, default=DEFAULT_CONTEXT_LENGTHS
    )
    adversarial.add_argument(
        "--long-correctness-context-lengths",
        type=parse_int_list,
        default=DEFAULT_LONG_CORRECTNESS_LENGTHS,
    )
    adversarial.add_argument(
        "--long-correctness-concurrencies",
        type=parse_int_list,
        default=DEFAULT_LONG_CORRECTNESS_CONCURRENCIES,
    )
    adversarial.add_argument("--long-correctness-max-tokens", type=int, default=64)
    adversarial.add_argument("--long-correctness-stat-max-ks", type=float, default=0.60)
    adversarial.add_argument("--long-correctness-stat-max-js", type=float, default=0.35)
    adversarial.add_argument(
        "--long-resident-max-tokens",
        type=int,
        default=DEFAULT_ADVERSARIAL_LONG_RESIDENT_MAX_TOKENS,
    )
    adversarial.add_argument(
        "--min-long-resident-decode-tok-s-by-concurrency",
        type=parse_concurrency_thresholds,
        default=DEFAULT_ADVERSARIAL_LONG_RESIDENT_MIN_DECODE_TOK_S,
    )
    adversarial.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    adversarial.add_argument("--max-seqs", type=int, default=16)
    adversarial.add_argument("--max-tokens", type=int, default=128)
    adversarial.add_argument("--mixed-requests", type=int, default=16)
    adversarial.add_argument(
        "--throughput-concurrencies",
        type=parse_int_list,
        default=DEFAULT_RESIDENT_CONCURRENCIES,
    )
    adversarial.add_argument(
        "--min-output-tok-s-by-concurrency",
        type=parse_concurrency_thresholds,
        required=True,
    )
    adversarial.add_argument(
        "--min-scaling-efficiency",
        type=float,
        required=True,
    )
    adversarial.add_argument("--overlap-short-requests", type=int, default=8)
    adversarial.add_argument(
        "--overlap-baseline-seconds",
        type=float,
        default=DEFAULT_OVERLAP_BASELINE_SECONDS,
    )
    adversarial.add_argument(
        "--overlap-queue-poll-seconds",
        type=float,
        default=DEFAULT_OVERLAP_QUEUE_POLL_SECONDS,
    )
    adversarial.add_argument(
        "--min-overlap-baseline-completions",
        type=int,
        default=DEFAULT_MIN_OVERLAP_BASELINE_COMPLETIONS,
    )
    adversarial.add_argument(
        "--min-overlap-decode-events-per-second",
        type=float,
        required=True,
    )
    adversarial.add_argument(
        "--min-overlap-decode-throughput-ratio",
        type=float,
        required=True,
    )
    adversarial.add_argument(
        "--max-overlap-decode-gap-seconds",
        type=float,
        default=DEFAULT_MAX_OVERLAP_DECODE_GAP_SECONDS,
    )
    adversarial.add_argument(
        "--max-overlap-prefill-ttft-seconds",
        type=float,
        default=DEFAULT_MAX_OVERLAP_PREFILL_TTFT_SECONDS,
    )
    adversarial.add_argument(
        "--min-output-event-coverage",
        type=float,
        default=DEFAULT_MIN_OUTPUT_EVENT_COVERAGE,
    )
    adversarial.add_argument("--cancel-requests", type=int, default=8)
    adversarial.add_argument("--cancel-context-tokens", type=int, default=32_768)
    adversarial.add_argument("--disconnect-after-seconds", type=float, default=0.25)
    adversarial.add_argument(
        "--cleanup-timeout-seconds", type=float, default=DEFAULT_CLEANUP_TIMEOUT_SECONDS
    )
    adversarial.add_argument(
        "--cleanup-poll-seconds", type=float, default=DEFAULT_CLEANUP_POLL_SECONDS
    )
    adversarial.add_argument("--timeout-test-seconds", type=float, default=0.05)
    adversarial.add_argument(
        "--fairness-max-slowdown", type=float, default=DEFAULT_FAIRNESS_MAX_SLOWDOWN
    )
    adversarial.add_argument(
        "--fairness-max-short-ttft-seconds",
        type=float,
        default=DEFAULT_FAIRNESS_MAX_SHORT_TTFT_SECONDS,
    )
    adversarial.add_argument(
        "--fairness-max-short-tpot-seconds",
        type=float,
        default=DEFAULT_FAIRNESS_MAX_SHORT_TPOT_SECONDS,
    )
    adversarial.add_argument(
        "--fairness-stagger-seconds", type=float, default=DEFAULT_FAIRNESS_STAGGER_SECONDS
    )
    adversarial.add_argument("--burst-requests", type=int, default=32)
    adversarial.add_argument("--burst-max-tokens", type=int, default=64)
    add_prefix_pressure_args(adversarial)
    adversarial.add_argument(
        "--min-overlap-baseline-events",
        type=int,
        default=DEFAULT_MIN_OVERLAP_BASELINE_EVENTS,
    )
    adversarial.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target", "dflash"),
    )
    adversarial.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    adversarial.add_argument(
        "--churn-rounds", type=int, default=DEFAULT_ADVERSARIAL_CHURN_ROUNDS
    )
    adversarial.add_argument("--churn-max-tokens", type=int, default=64)
    adversarial.add_argument(
        "--acceptance-grade",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    adversarial.add_argument(
        "--require-mtp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    prefix_pressure = subparsers.add_parser(
        "prefix-pressure",
        parents=[common],
        help="run only the cold-hit-capacity-eviction-retry memory-pressure gate",
    )
    prefix_pressure.add_argument("--max-seqs", type=int, default=16)
    prefix_pressure.add_argument("--max-tokens", type=int, default=128)
    prefix_pressure.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    prefix_pressure.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    prefix_pressure.add_argument(
        "--cleanup-poll-seconds",
        type=float,
        default=DEFAULT_CLEANUP_POLL_SECONDS,
    )
    add_prefix_pressure_args(prefix_pressure)

    quality_replay = subparsers.add_parser("quality-replay", parents=[common])
    add_speculative_evidence_args(quality_replay)
    quality_replay.add_argument(
        "--source-production-artifact",
        type=Path,
        required=True,
    )
    quality_replay.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        required=True,
    )
    quality_replay.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_CONCURRENCY,
    )
    quality_replay.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    quality_replay.add_argument(
        "--pressure-waves",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_PRESSURE_WAVES,
        help="waves must cumulatively churn more blocks than the physical KV pool",
    )
    quality_replay.add_argument(
        "--pressure-entries",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_PRESSURE_ENTRIES,
        help=(
            "distinct entries per wave; selected identities plus one wave must fit "
            "the server's retained-prefix owner capacity"
        ),
    )
    quality_replay.add_argument(
        "--pressure-context-tokens",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_PRESSURE_CONTEXT_TOKENS,
        help=(
            "context size is runtime-gated so selected identities, retained prior "
            "pressure owners, and the active wave fit the physical KV pool with "
            "configured headroom"
        ),
    )
    quality_replay.add_argument(
        "--pressure-max-tokens",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_PRESSURE_MAX_TOKENS,
    )
    quality_replay.add_argument(
        "--max-stability-passes",
        type=int,
        default=DEFAULT_QUALITY_REPLAY_MAX_STABILITY_PASSES,
        help=(
            "maximum resident passes allowed to reach consecutive exact outputs with "
            "zero CUDA graph captures"
        ),
    )
    quality_replay.add_argument(
        "--prefix-pressure-kv-headroom-fraction",
        type=float,
        default=DEFAULT_PREFIX_PRESSURE_KV_HEADROOM_FRACTION,
    )
    quality_replay.add_argument(
        "--min-prefix-reuse-fraction",
        type=float,
        default=DEFAULT_MIN_PREFIX_REUSE_FRACTION,
    )
    quality_replay.add_argument(
        "--kv-block-size-tokens",
        type=int,
        default=DEFAULT_KV_BLOCK_SIZE_TOKENS,
    )
    quality_replay.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    quality_replay.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    quality_replay.add_argument(
        "--cleanup-poll-seconds",
        type=float,
        default=DEFAULT_CLEANUP_POLL_SECONDS,
    )
    quality_replay.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target",),
    )
    quality_replay.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    quality_replay.add_argument("--require-mtp", action="store_true")
    quality_replay.add_argument(
        "--require-empty-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    production = subparsers.add_parser("production", parents=[common])
    add_speculative_evidence_args(production)
    production.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_PRODUCTION_DURATION_SECONDS,
    )
    production.add_argument("--concurrencies", type=parse_int_list, default=(8, 16))
    production.add_argument(
        "--min-output-tok-s-by-concurrency",
        type=parse_concurrency_thresholds,
        default=DEFAULT_PRODUCTION_MIN_OUTPUT_TOK_S_BY_CONCURRENCY,
    )
    production.add_argument(
        "--min-scaling-efficiency",
        type=float,
        required=True,
    )
    production.add_argument("--context-mix", type=parse_context_mix, default=DEFAULT_CONTEXT_MIX)
    production.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    production.add_argument("--prompt-pool-size", type=int, default=16)
    production.add_argument("--resident-prompt-budget", type=int)
    production.add_argument("--prewarm-max-tokens", type=int, default=8)
    production.add_argument(
        "--min-prefix-reuse-fraction",
        type=float,
        default=DEFAULT_MIN_PREFIX_REUSE_FRACTION,
    )
    production.add_argument(
        "--kv-block-size-tokens", type=int, default=DEFAULT_KV_BLOCK_SIZE_TOKENS
    )
    production.add_argument(
        "--speculative-prefix-replay-tokens",
        type=int,
        default=DEFAULT_SPECULATIVE_PREFIX_REPLAY_TOKENS,
    )
    production.add_argument(
        "--fixed-output-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    production.add_argument("--probe-max-tokens", type=int, default=128)
    production.add_argument("--max-probe-ttft-seconds", type=float, required=True)
    production.add_argument("--max-probe-tpot-seconds", type=float, required=True)
    production.add_argument(
        "--max-probe-latency-slowdown",
        type=float,
        default=DEFAULT_MAX_PROBE_LATENCY_SLOWDOWN,
    )
    production.add_argument(
        "--max-final-c1-latency-slowdown",
        type=float,
        default=DEFAULT_MAX_FINAL_C1_LATENCY_SLOWDOWN,
    )
    production.add_argument(
        "--min-final-c1-decode-ratio",
        type=float,
        default=DEFAULT_MIN_FINAL_C1_DECODE_RATIO,
    )
    production.add_argument(
        "--max-schedule-lateness-seconds",
        type=float,
        default=DEFAULT_MAX_SCHEDULE_LATENESS_SECONDS,
    )
    production.add_argument(
        "--probe-interval-seconds", type=float, default=DEFAULT_PROBE_INTERVAL_SECONDS
    )
    production.add_argument(
        "--diagnostic-concurrency",
        type=int,
        default=DEFAULT_PRODUCTION_DIAGNOSTIC_CONCURRENCY,
    )
    production.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=DEFAULT_TELEMETRY_INTERVAL_SECONDS,
    )
    production.add_argument(
        "--comparison-window-seconds",
        type=float,
        default=DEFAULT_COMPARISON_WINDOW_SECONDS,
    )
    production.add_argument(
        "--min-comparison-window-samples",
        type=int,
        default=DEFAULT_MIN_COMPARISON_WINDOW_SAMPLES,
    )
    production.add_argument(
        "--max-throughput-degradation-fraction",
        type=float,
        default=DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION,
    )
    production.add_argument(
        "--max-latency-degradation-fraction",
        type=float,
        default=DEFAULT_MAX_LATENCY_DEGRADATION_FRACTION,
    )
    production.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    production.add_argument(
        "--min-output-event-coverage",
        type=float,
        default=DEFAULT_MIN_OUTPUT_EVENT_COVERAGE,
    )
    production.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target", "dflash"),
    )
    production.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    production.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    production.add_argument(
        "--cleanup-poll-seconds",
        type=float,
        default=DEFAULT_CLEANUP_POLL_SECONDS,
    )
    production.add_argument(
        "--min-telemetry-samples",
        type=int,
        default=DEFAULT_MIN_TELEMETRY_SAMPLES,
    )
    production.add_argument(
        "--min-telemetry-coverage",
        type=float,
        default=DEFAULT_MIN_TELEMETRY_COVERAGE,
    )
    production.add_argument(
        "--max-process-rss-drift-mib",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB,
        help="maximum final server RSS growth, before the relative allowance",
    )
    production.add_argument(
        "--max-process-rss-drift-fraction",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_DRIFT_FRACTION,
        help="maximum final server RSS growth as a fraction of initial RSS",
    )
    production.add_argument(
        "--max-process-rss-high-water-mib",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_HIGH_WATER_MIB,
        help="maximum sampled server RSS growth above the initial reading",
    )
    production.add_argument(
        "--max-gpu-memory-drift-mib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_DRIFT_MIB,
        help="maximum final server GPU-memory growth",
    )
    production.add_argument(
        "--max-gpu-memory-high-water-mib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB,
        help="maximum sampled server GPU-memory growth above the initial reading",
    )
    production.add_argument(
        "--max-kv-block-utilization",
        type=float,
        default=DEFAULT_MAX_KV_BLOCK_UTILIZATION,
        help="maximum sampled active/total PagedAttention block ratio",
    )
    production.add_argument(
        "--max-recurrent-slot-utilization",
        type=float,
        default=DEFAULT_MAX_RECURRENT_SLOT_UTILIZATION,
        help="maximum sampled used/total recurrent-state slot ratio",
    )
    production.add_argument(
        "--acceptance-grade",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    production.add_argument(
        "--require-mtp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    multimodal = subparsers.add_parser("multimodal", parents=[common])
    add_speculative_evidence_args(multimodal)
    multimodal.add_argument(
        "--image",
        default=DEFAULT_MULTIMODAL_IMAGE,
        help="local oracle image path",
    )
    multimodal.add_argument(
        "--image-prompt",
        default=DEFAULT_MULTIMODAL_IMAGE_PROMPT,
    )
    multimodal.add_argument(
        "--image-required-phrase",
        dest="image_required_phrases",
        action="append",
        type=parse_nonempty_phrase,
        default=list(DEFAULT_MULTIMODAL_REQUIRED_PHRASES),
        help="case-insensitive phrase required in every image response; repeatable",
    )
    multimodal.add_argument(
        "--image-expected-attribute",
        dest="image_expected_attributes",
        action="append",
        type=parse_phrase_alternatives,
        default=list(DEFAULT_MULTIMODAL_EXPECTED_ATTRIBUTES),
        help=(
            "case-insensitive visual-attribute phrases; separate acceptable alternatives "
            "with | and repeat the flag for each required attribute"
        ),
    )
    multimodal.add_argument(
        "--phase-duration-seconds",
        type=float,
        default=DEFAULT_MULTIMODAL_PHASE_DURATION_SECONDS,
    )
    multimodal.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_MULTIMODAL_CONCURRENCY,
    )
    multimodal.add_argument(
        "--image-interval-seconds",
        type=float,
        default=DEFAULT_MULTIMODAL_IMAGE_INTERVAL_SECONDS,
    )
    multimodal.add_argument("--max-tokens", type=int, default=128)
    multimodal.add_argument("--image-max-tokens", type=int, default=128)
    multimodal.add_argument(
        "--min-text-requests-per-phase",
        type=int,
        default=DEFAULT_MULTIMODAL_MIN_TEXT_REQUESTS_PER_PHASE,
    )
    multimodal.add_argument(
        "--min-image-requests",
        type=int,
        default=DEFAULT_MULTIMODAL_MIN_IMAGE_REQUESTS,
    )
    multimodal.add_argument(
        "--min-mixed-throughput-ratio",
        type=float,
        default=DEFAULT_MULTIMODAL_MIN_MIXED_THROUGHPUT_RATIO,
    )
    multimodal.add_argument(
        "--max-mixed-tpot-ratio",
        type=float,
        default=DEFAULT_MULTIMODAL_MAX_MIXED_TPOT_RATIO,
    )
    multimodal.add_argument(
        "--max-mixed-ttft-p99-seconds",
        type=float,
        default=DEFAULT_MULTIMODAL_MAX_MIXED_TTFT_P99_SECONDS,
    )
    multimodal.add_argument(
        "--min-recovery-throughput-ratio",
        type=float,
        default=DEFAULT_MULTIMODAL_MIN_RECOVERY_THROUGHPUT_RATIO,
    )
    multimodal.add_argument(
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    multimodal.add_argument(
        "--min-output-event-coverage",
        type=float,
        default=DEFAULT_MIN_OUTPUT_EVENT_COVERAGE,
        help="minimum per-request and aggregate streamed-token accounting coverage",
    )
    multimodal.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target", "dflash"),
    )
    multimodal.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    multimodal.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    multimodal.add_argument(
        "--cleanup-poll-seconds",
        type=float,
        default=DEFAULT_CLEANUP_POLL_SECONDS,
    )
    multimodal.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=DEFAULT_TELEMETRY_INTERVAL_SECONDS,
    )
    multimodal.add_argument(
        "--min-telemetry-samples",
        type=int,
        default=DEFAULT_MIN_TELEMETRY_SAMPLES,
    )
    multimodal.add_argument(
        "--min-telemetry-coverage",
        type=float,
        default=DEFAULT_MIN_TELEMETRY_COVERAGE,
    )
    multimodal.add_argument(
        "--max-schedule-lateness-seconds",
        type=float,
        default=DEFAULT_MAX_SCHEDULE_LATENESS_SECONDS,
    )
    multimodal.add_argument(
        "--max-process-rss-drift-mib",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB,
    )
    multimodal.add_argument(
        "--max-process-rss-drift-fraction",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_DRIFT_FRACTION,
    )
    multimodal.add_argument(
        "--max-process-rss-high-water-mib",
        type=float,
        default=DEFAULT_MAX_PROCESS_RSS_HIGH_WATER_MIB,
    )
    multimodal.add_argument(
        "--max-gpu-memory-drift-mib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_DRIFT_MIB,
    )
    multimodal.add_argument(
        "--max-gpu-memory-high-water-mib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB,
    )
    multimodal.add_argument(
        "--max-kv-block-utilization",
        type=float,
        default=DEFAULT_MAX_KV_BLOCK_UTILIZATION,
    )
    multimodal.add_argument(
        "--max-recurrent-slot-utilization",
        type=float,
        default=DEFAULT_MAX_RECURRENT_SLOT_UTILIZATION,
    )
    multimodal.add_argument(
        "--acceptance-grade",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    multimodal.add_argument(
        "--require-mtp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    multimodal.add_argument(
        "--text-prerequisite-artifacts",
        type=Path,
        nargs="+",
        required=True,
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate-phase", default="canary-c1-normal")
    compare.add_argument("--reference-phase", default="canary-c1-normal")
    compare.add_argument("--tokenizer", help="tokenizer.json path or Hugging Face tokenizer ID")
    compare.add_argument("--stat-max-ks", type=float, default=0.35)
    compare.add_argument("--stat-max-js", type=float, default=0.20)
    compare.add_argument("--require-part1-complete", action="store_true")
    compare.add_argument("--output", type=Path, help="JSONL comparison evidence path")
    return parser


def validate_prefix_pressure_args(args: argparse.Namespace) -> None:
    for name in (
        "max_seqs",
        "max_tokens",
        "prefix_context_tokens",
        "prefix_pressure_entries",
        "prefix_pressure_context_tokens",
        "prefix_pressure_max_entries",
        "prefix_pressure_max_tokens",
        "kv_block_size_tokens",
        "cleanup_timeout_seconds",
        "cleanup_poll_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.speculative_prefix_replay_tokens < 0:
        raise ValueError("--speculative-prefix-replay-tokens cannot be negative")
    if args.prefix_pressure_max_entries < args.prefix_pressure_entries:
        raise ValueError(
            "--prefix-pressure-max-entries must be at least --prefix-pressure-entries"
        )
    if args.prefix_pressure_capacity_fraction <= 1:
        raise ValueError("--prefix-pressure-capacity-fraction must be greater than 1")
    if not 0 < args.prefix_pressure_kv_headroom_fraction < 1:
        raise ValueError(
            "--prefix-pressure-kv-headroom-fraction must be greater than 0 and less than 1"
        )
    if not 0 < args.min_prefix_reuse_fraction <= 1:
        raise ValueError(
            "--min-prefix-reuse-fraction must be greater than 0 and at most 1"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "compare":
        if args.stat_max_ks <= 0 or args.stat_max_js <= 0:
            raise ValueError("statistical thresholds must be positive")
        return
    if args.mode in ("adversarial", "production", "multimodal") and args.require_mtp is None:
        args.require_mtp = args.acceptance_grade
    positive_fields = (
        "timeout",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.server_pid is not None and args.server_pid <= 0:
        raise ValueError("--server-pid must be positive")
    if args.require_server_provenance and args.server_pid is None:
        raise ValueError("--require-server-provenance requires --server-pid")
    if not 0 <= args.min_p <= 1:
        raise ValueError("--min-p must be between 0 and 1")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be greater than 0 and at most 1")
    if hasattr(args, "min_mtp_acceptance_rate"):
        if not 0 < args.min_mtp_acceptance_rate <= 1:
            raise ValueError("--min-mtp-acceptance-rate must be greater than 0 and at most 1")
        if args.min_mtp_mean_advance <= 1:
            raise ValueError("--min-mtp-mean-advance must be greater than 1")
        if args.min_mtp_proposal_depth <= 0:
            raise ValueError("--min-mtp-proposal-depth must be positive")
        if not 0 <= args.max_sparse_verifier_fallback_ratio <= 1:
            raise ValueError(
                "--max-sparse-verifier-fallback-ratio must be between 0 and 1"
            )
        if not 0 < args.min_sparse_verifier_accounting_coverage <= 1:
            raise ValueError(
                "--min-sparse-verifier-accounting-coverage must be greater than 0 "
                "and at most 1"
            )
    if hasattr(args, "requests") and args.requests <= 0:
        raise ValueError("--requests must be positive")
    if hasattr(args, "max_tokens") and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if (
        hasattr(args, "max_repeated_ngram_ratio")
        and not 0 <= args.max_repeated_ngram_ratio <= 1
    ):
        raise ValueError("--max-repeated-ngram-ratio must be between 0 and 1")
    if args.mode == "canary" and (args.stat_max_ks <= 0 or args.stat_max_js <= 0):
        raise ValueError("statistical thresholds must be positive")
    if args.mode == "canary":
        if args.max_length_tokens <= 0:
            raise ValueError("--max-length-tokens must be positive")
        sampling_policy = production_sampling_policy_evidence(
            args.sampling_policy,
            SamplingPolicy(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            ),
        )
        if (
            args.sampling_policy == PART1_PRODUCTION_SAMPLING_POLICY
            and not sampling_policy["passed"]
        ):
            mismatches = ", ".join(
                field
                for field, matches in sampling_policy["matches"].items()
                if not matches
            )
            raise ValueError(
                "--sampling-policy production requires the production sampling values; "
                f"mismatched fields: {mismatches}"
            )
        if (
            args.require_part1_complete
            and args.sampling_policy != PART1_PRODUCTION_SAMPLING_POLICY
        ):
            raise ValueError(
                "--require-part1-complete requires --sampling-policy production"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        if args.require_mtp and set(args.expected_graph_components) != {"target", "dflash"}:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
    if args.mode in (
        "adversarial",
        "prefix-pressure",
        "production",
        "quality-replay",
        "resident-decode",
    ) and not args.tokenizer:
        raise ValueError(f"{args.mode} mode requires --tokenizer")
    if args.mode == "prefix-pressure":
        validate_prefix_pressure_args(args)
        if args.server_pid is None:
            raise ValueError("prefix-pressure mode requires --server-pid")
    if args.mode == "quality-replay":
        for name in (
            "concurrency",
            "max_tokens",
            "pressure_waves",
            "pressure_entries",
            "pressure_context_tokens",
            "pressure_max_tokens",
            "max_stability_passes",
            "kv_block_size_tokens",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.server_pid is None:
            raise ValueError("quality-replay mode requires --server-pid")
        if not args.source_production_artifact.is_file():
            raise ValueError("--source-production-artifact must be an existing file")
        if len(set(args.case_ids)) != len(args.case_ids):
            raise ValueError("--case-id values must be unique")
        if len(args.case_ids) > args.concurrency:
            raise ValueError("quality replay cases must fit in one concurrent batch")
        if not 0 < args.min_prefix_reuse_fraction <= 1:
            raise ValueError(
                "--min-prefix-reuse-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.prefix_pressure_kv_headroom_fraction < 1:
            raise ValueError(
                "--prefix-pressure-kv-headroom-fraction must be greater than 0 "
                "and less than 1"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        if args.require_mtp and set(args.expected_graph_components) != {
            "target",
            "dflash",
        }:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
        sampling = production_sampling_policy_evidence(
            PART1_PRODUCTION_SAMPLING_POLICY,
            SamplingPolicy(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            ),
        )
        if not sampling["passed"]:
            raise ValueError("quality-replay requires the production sampling values")
    if args.mode == "resident-decode":
        for name in (
            "requests",
            "warmup_max_tokens",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "stat_max_ks",
            "stat_max_js",
            "kv_block_size_tokens",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.speculative_prefix_replay_tokens < 0:
            raise ValueError("--speculative-prefix-replay-tokens cannot be negative")
        if args.requests < max(args.concurrencies):
            raise ValueError(
                "--requests must be at least the maximum resident-decode concurrency"
            )
        if len(set(args.concurrencies)) != len(args.concurrencies):
            raise ValueError("--concurrencies must be unique")
        if not 0 < args.min_prefix_reuse_fraction <= 1:
            raise ValueError(
                "--min-prefix-reuse-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        if not 0 < args.min_final_c1_throughput_ratio <= 1:
            raise ValueError(
                "--min-final-c1-throughput-ratio must be greater than 0 and at most 1"
            )
        if set(args.min_output_tok_s_by_concurrency) != set(args.concurrencies):
            raise ValueError(
                "--min-output-tok-s-by-concurrency must cover every resident concurrency exactly"
            )
        if len(args.concurrencies) < 2 or args.min_scaling_efficiency <= 0:
            raise ValueError(
                "resident throughput gating requires at least two concurrencies and a positive "
                "--min-scaling-efficiency"
            )
        if args.require_mtp and set(args.expected_graph_components) != {"target", "dflash"}:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
    if args.mode == "adversarial":
        validate_prefix_pressure_args(args)
        if args.acceptance_grade and args.server_pid is None:
            raise ValueError("acceptance-grade adversarial mode requires --server-pid")
        if len(set(args.context_lengths)) != len(args.context_lengths):
            raise ValueError("--context-lengths must be unique")
        if len(set(args.long_correctness_context_lengths)) != len(
            args.long_correctness_context_lengths
        ):
            raise ValueError("--long-correctness-context-lengths must be unique")
        required_contexts = set(DEFAULT_CONTEXT_LENGTHS)
        if not required_contexts.issubset(args.context_lengths):
            raise ValueError(
                "--context-lengths must include 1024,8192,32768,100000"
            )
        if args.max_seqs != 16:
            raise ValueError("the adversarial boundary contract requires --max-seqs 16")
        missing = {
            args.cancel_context_tokens,
            args.prefix_context_tokens,
        } - set(args.context_lengths)
        if missing:
            values = ", ".join(str(value) for value in sorted(missing))
            raise ValueError(f"context lengths must include cancellation/prefix lengths: {values}")
        if 3 not in args.long_correctness_concurrencies:
            raise ValueError("--long-correctness-concurrencies must include 3")
        required_long_contexts = (
            set(DEFAULT_LONG_CORRECTNESS_LENGTHS)
            if args.acceptance_grade
            else {60_000, 65_536, 100_000}
        )
        if not required_long_contexts.issubset(args.long_correctness_context_lengths):
            values = ",".join(str(value) for value in sorted(required_long_contexts))
            raise ValueError(
                "--long-correctness-context-lengths must include " + values
            )
        if "target" not in args.expected_graph_components:
            raise ValueError(
                "cold long-context graph coverage requires target graph instrumentation"
            )
        if args.long_correctness_stat_max_ks <= 0 or args.long_correctness_stat_max_js <= 0:
            raise ValueError("long-context statistical thresholds must be positive")
        if len(set(args.long_correctness_concurrencies)) != len(
            args.long_correctness_concurrencies
        ):
            raise ValueError("--long-correctness-concurrencies must be unique")
        if max(args.long_correctness_concurrencies) > len(
            args.long_correctness_context_lengths
        ):
            raise ValueError(
                "long-context correctness concurrency cannot exceed the number of "
                "requested correctness cases"
            )
        if set(args.min_long_resident_decode_tok_s_by_concurrency) != {1, 3}:
            raise ValueError(
                "--min-long-resident-decode-tok-s-by-concurrency must cover "
                "concurrencies 1 and 3 exactly"
            )
        for name in (
            "max_seqs",
            "mixed_requests",
            "overlap_short_requests",
            "cancel_requests",
            "burst_requests",
            "churn_rounds",
            "churn_max_tokens",
            "burst_max_tokens",
            "long_correctness_max_tokens",
            "long_resident_max_tokens",
            "min_overlap_baseline_completions",
            "min_overlap_baseline_events",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.long_resident_max_tokens <= 1:
            raise ValueError("--long-resident-max-tokens must be greater than 1")
        if (
            args.acceptance_grade
            and args.churn_rounds < DEFAULT_ADVERSARIAL_CHURN_ROUNDS
        ):
            raise ValueError(
                "acceptance-grade --churn-rounds must be at least "
                f"{DEFAULT_ADVERSARIAL_CHURN_ROUNDS}"
            )
        if (
            args.acceptance_grade
            and args.long_resident_max_tokens
            < DEFAULT_ADVERSARIAL_LONG_RESIDENT_MAX_TOKENS
        ):
            raise ValueError(
                "acceptance-grade --long-resident-max-tokens must be at least "
                f"{DEFAULT_ADVERSARIAL_LONG_RESIDENT_MAX_TOKENS}"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        for name in (
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "timeout_test_seconds",
            "disconnect_after_seconds",
            "fairness_max_slowdown",
            "fairness_max_short_ttft_seconds",
            "fairness_max_short_tpot_seconds",
            "fairness_stagger_seconds",
            "overlap_baseline_seconds",
            "overlap_queue_poll_seconds",
            "min_overlap_decode_events_per_second",
            "min_overlap_decode_throughput_ratio",
            "max_overlap_decode_gap_seconds",
            "max_overlap_prefill_ttft_seconds",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if (
            args.acceptance_grade
            and args.max_overlap_decode_gap_seconds
            > DEFAULT_MAX_OVERLAP_DECODE_GAP_SECONDS
        ):
            raise ValueError(
                "acceptance-grade --max-overlap-decode-gap-seconds must be at most "
                f"{DEFAULT_MAX_OVERLAP_DECODE_GAP_SECONDS}"
            )
        if (
            args.acceptance_grade
            and args.max_overlap_prefill_ttft_seconds
            > DEFAULT_MAX_OVERLAP_PREFILL_TTFT_SECONDS
        ):
            raise ValueError(
                "acceptance-grade --max-overlap-prefill-ttft-seconds must be at most "
                f"{DEFAULT_MAX_OVERLAP_PREFILL_TTFT_SECONDS}"
            )
        if len(set(args.throughput_concurrencies)) != len(
            args.throughput_concurrencies
        ):
            raise ValueError("--throughput-concurrencies must be unique")
        if args.acceptance_grade and set(DEFAULT_RESIDENT_CONCURRENCIES) - set(
            args.throughput_concurrencies
        ):
            raise ValueError(
                "acceptance-grade --throughput-concurrencies must include 1,3,8,16"
            )
        if set(args.min_output_tok_s_by_concurrency) != set(
            args.throughput_concurrencies
        ):
            raise ValueError(
                "--min-output-tok-s-by-concurrency must cover every adversarial throughput "
                "concurrency exactly"
            )
        if args.mixed_requests < max(args.throughput_concurrencies):
            raise ValueError(
                "--mixed-requests must be at least the maximum throughput concurrency"
            )
        minimum_context_cases = args.mixed_requests // len(args.context_lengths)
        for concurrency in args.throughput_concurrencies:
            if not any(
                requests_per_context * len(args.context_lengths) % concurrency == 0
                for requests_per_context in range(1, minimum_context_cases + 1)
            ):
                raise ValueError(
                    "mixed requests cannot form a context-balanced full batch at "
                    f"concurrency {concurrency}"
                )
        if args.min_scaling_efficiency <= 0:
            raise ValueError("--min-scaling-efficiency must be positive")
        if not 0 < args.min_output_event_coverage <= 1:
            raise ValueError(
                "--min-output-event-coverage must be greater than 0 and at most 1"
            )
        if args.overlap_short_requests >= args.max_seqs:
            raise ValueError("--overlap-short-requests must leave one slot for the large prefill")
        if args.cancel_requests > args.max_seqs:
            raise ValueError(
                "--cancel-requests cannot exceed --max-seqs for post-admission cancellation"
            )
        if args.require_mtp and set(args.expected_graph_components) != {"target", "dflash"}:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
        if not acceptance_grade_evidence(
            args.acceptance_grade,
            args.require_mtp,
            args.expected_graph_components,
        )["passed"]:
            raise ValueError(
                "--acceptance-grade adversarial runs require MTP and "
                "--expected-graph-components target,dflash"
            )
    if args.mode == "production":
        for name in (
            "duration_seconds",
            "probe_interval_seconds",
            "diagnostic_concurrency",
            "telemetry_interval_seconds",
            "comparison_window_seconds",
            "min_comparison_window_samples",
            "prompt_pool_size",
            "prewarm_max_tokens",
            "probe_max_tokens",
            "max_probe_ttft_seconds",
            "max_probe_tpot_seconds",
            "max_probe_latency_slowdown",
            "max_final_c1_latency_slowdown",
            "max_schedule_lateness_seconds",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "min_telemetry_samples",
            "kv_block_size_tokens",
            "max_process_rss_drift_mib",
            "max_process_rss_high_water_mib",
            "max_gpu_memory_drift_mib",
            "max_gpu_memory_high_water_mib",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.speculative_prefix_replay_tokens < 0:
            raise ValueError("--speculative-prefix-replay-tokens cannot be negative")
        if not 0 < args.min_final_c1_decode_ratio <= 1:
            raise ValueError(
                "--min-final-c1-decode-ratio must be greater than 0 and at most 1"
            )
        if not 0 <= args.max_throughput_degradation_fraction <= 1:
            raise ValueError(
                "--max-throughput-degradation-fraction must be between 0 and 1"
            )
        if not 0 <= args.max_latency_degradation_fraction <= 1:
            raise ValueError(
                "--max-latency-degradation-fraction must be between 0 and 1"
            )
        if (
            args.acceptance_grade
            and args.max_throughput_degradation_fraction
            > DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION
        ):
            raise ValueError(
                "acceptance-grade production requires "
                "--max-throughput-degradation-fraction <= "
                f"{DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION}"
            )
        if (
            args.acceptance_grade
            and args.max_latency_degradation_fraction
            > DEFAULT_MAX_LATENCY_DEGRADATION_FRACTION
        ):
            raise ValueError(
                "acceptance-grade production requires "
                "--max-latency-degradation-fraction <= "
                f"{DEFAULT_MAX_LATENCY_DEGRADATION_FRACTION}"
            )
        if args.resident_prompt_budget is not None and args.resident_prompt_budget <= 0:
            raise ValueError("--resident-prompt-budget must be positive")
        if not 0 < args.min_prefix_reuse_fraction <= 1:
            raise ValueError(
                "--min-prefix-reuse-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.min_output_event_coverage <= 1:
            raise ValueError(
                "--min-output-event-coverage must be greater than 0 and at most 1"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        if not 0 < args.min_telemetry_coverage <= 1:
            raise ValueError(
                "--min-telemetry-coverage must be greater than 0 and at most 1"
            )
        if (
            args.acceptance_grade
            and args.min_telemetry_coverage < DEFAULT_MIN_TELEMETRY_COVERAGE
        ):
            raise ValueError(
                "acceptance-grade production requires --min-telemetry-coverage >= "
                f"{DEFAULT_MIN_TELEMETRY_COVERAGE}"
            )
        if not 0 < args.max_process_rss_drift_fraction <= 1:
            raise ValueError(
                "--max-process-rss-drift-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.max_kv_block_utilization <= 1:
            raise ValueError(
                "--max-kv-block-utilization must be greater than 0 and at most 1"
            )
        if not 0 < args.max_recurrent_slot_utilization <= 1:
            raise ValueError(
                "--max-recurrent-slot-utilization must be greater than 0 and at most 1"
            )
        if len(set(args.concurrencies)) != len(args.concurrencies):
            raise ValueError("--concurrencies must be unique")
        if not {8, 16}.issubset(args.concurrencies):
            raise ValueError("--concurrencies must include 8 and 16")
        if (
            args.acceptance_grade
            and args.min_comparison_window_samples
            < DEFAULT_MIN_COMPARISON_WINDOW_SAMPLES
        ):
            raise ValueError(
                "acceptance-grade production requires at least "
                f"{DEFAULT_MIN_COMPARISON_WINDOW_SAMPLES} comparison-window samples"
            )
        if args.acceptance_grade and not args.fixed_output_length:
            raise ValueError(
                "acceptance-grade production requires --fixed-output-length"
            )
        if args.server_pid is None:
            raise ValueError("production mode requires --server-pid")
        minimum_phase_seconds = (
            args.probe_interval_seconds * MIN_PRODUCTION_PROBES_PER_PHASE
        )
        phase_seconds = args.duration_seconds / len(args.concurrencies)
        if phase_seconds <= minimum_phase_seconds:
            raise ValueError(
                "each production concurrency phase must be longer than "
                f"{MIN_PRODUCTION_PROBES_PER_PHASE} probe intervals; increase "
                "--duration-seconds or decrease --probe-interval-seconds"
            )
        if (
            phase_seconds + SCHEDULE_TIME_EPSILON_SECONDS
            < 2 * args.comparison_window_seconds
        ):
            raise ValueError(
                "each production concurrency phase must cover complete, "
                "non-overlapping first and final comparison windows"
            )
        if set(args.min_output_tok_s_by_concurrency) != set(args.concurrencies):
            raise ValueError(
                "--min-output-tok-s-by-concurrency must cover every production concurrency "
                "exactly"
            )
        if args.min_scaling_efficiency <= 0:
            raise ValueError("--min-scaling-efficiency must be positive")
        if args.require_mtp and set(args.expected_graph_components) != {"target", "dflash"}:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
        if not acceptance_grade_evidence(
            args.acceptance_grade,
            args.require_mtp,
            args.expected_graph_components,
        )["passed"]:
            raise ValueError(
                "--acceptance-grade production runs require MTP and "
                "--expected-graph-components target,dflash"
            )
    if args.mode == "multimodal":
        for name in (
            "phase_duration_seconds",
            "concurrency",
            "image_interval_seconds",
            "max_tokens",
            "image_max_tokens",
            "min_text_requests_per_phase",
            "min_image_requests",
            "min_mixed_throughput_ratio",
            "max_mixed_tpot_ratio",
            "max_mixed_ttft_p99_seconds",
            "min_recovery_throughput_ratio",
            "min_output_event_coverage",
            "min_cuda_graph_replay_ratio",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "telemetry_interval_seconds",
            "min_telemetry_samples",
            "min_telemetry_coverage",
            "max_schedule_lateness_seconds",
            "max_process_rss_drift_mib",
            "max_process_rss_high_water_mib",
            "max_gpu_memory_drift_mib",
            "max_gpu_memory_high_water_mib",
            "max_kv_block_utilization",
            "max_recurrent_slot_utilization",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if not 0 < args.max_process_rss_drift_fraction <= 1:
            raise ValueError(
                "--max-process-rss-drift-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.min_telemetry_coverage <= 1:
            raise ValueError(
                "--min-telemetry-coverage must be greater than 0 and at most 1"
            )
        if not 0 < args.min_output_event_coverage <= 1:
            raise ValueError(
                "--min-output-event-coverage must be greater than 0 and at most 1"
            )
        if not 0 < args.max_kv_block_utilization <= 1:
            raise ValueError(
                "--max-kv-block-utilization must be greater than 0 and at most 1"
            )
        if not 0 < args.max_recurrent_slot_utilization <= 1:
            raise ValueError(
                "--max-recurrent-slot-utilization must be greater than 0 and at most 1"
            )
        if not args.image_required_phrases and not args.image_expected_attributes:
            raise ValueError(
                "multimodal mode requires --image-required-phrase or "
                "--image-expected-attribute"
            )
        if args.server_pid is None:
            raise ValueError("multimodal mode requires a fresh --server-pid")
        if args.require_mtp and set(args.expected_graph_components) != {"target", "dflash"}:
            raise ValueError(
                "--require-mtp requires --expected-graph-components target,dflash"
            )
        if args.acceptance_grade:
            if not args.tokenizer:
                raise ValueError(
                    "acceptance-grade multimodal requires --tokenizer for exact "
                    "nominal-window token accounting"
                )
            if args.concurrency != DEFAULT_MULTIMODAL_CONCURRENCY:
                raise ValueError(
                    "acceptance-grade multimodal requires --concurrency "
                    f"{DEFAULT_MULTIMODAL_CONCURRENCY}"
                )
            if args.phase_duration_seconds < DEFAULT_MULTIMODAL_PHASE_DURATION_SECONDS:
                raise ValueError(
                    "acceptance-grade multimodal requires --phase-duration-seconds >= "
                    f"{DEFAULT_MULTIMODAL_PHASE_DURATION_SECONDS}"
                )
            if args.image_interval_seconds > DEFAULT_MULTIMODAL_IMAGE_INTERVAL_SECONDS:
                raise ValueError(
                    "acceptance-grade multimodal requires --image-interval-seconds <= "
                    f"{DEFAULT_MULTIMODAL_IMAGE_INTERVAL_SECONDS}"
                )
            if len(
                periodic_schedule(
                    0.0,
                    args.phase_duration_seconds,
                    args.image_interval_seconds,
                    include_start=True,
                )
            ) < DEFAULT_MULTIMODAL_MIN_IMAGE_REQUESTS:
                raise ValueError(
                    "acceptance-grade multimodal must schedule at least "
                    f"{DEFAULT_MULTIMODAL_MIN_IMAGE_REQUESTS} image requests"
                )
            lower_bounds = {
                "max_tokens": 128,
                "image_max_tokens": 128,
                "min_text_requests_per_phase": DEFAULT_MULTIMODAL_MIN_TEXT_REQUESTS_PER_PHASE,
                "min_image_requests": DEFAULT_MULTIMODAL_MIN_IMAGE_REQUESTS,
                "min_mixed_throughput_ratio": DEFAULT_MULTIMODAL_MIN_MIXED_THROUGHPUT_RATIO,
                "min_recovery_throughput_ratio": DEFAULT_MULTIMODAL_MIN_RECOVERY_THROUGHPUT_RATIO,
                "min_output_event_coverage": DEFAULT_MIN_OUTPUT_EVENT_COVERAGE,
                "min_cuda_graph_replay_ratio": DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
                "min_telemetry_coverage": DEFAULT_MIN_TELEMETRY_COVERAGE,
                "min_telemetry_samples": DEFAULT_MIN_TELEMETRY_SAMPLES,
                "min_mtp_acceptance_rate": DEFAULT_MIN_MTP_ACCEPTANCE_RATE,
                "min_mtp_mean_advance": DEFAULT_MIN_MTP_MEAN_ADVANCE,
                "min_mtp_proposal_depth": DEFAULT_MIN_MTP_PROPOSAL_DEPTH,
                "min_sparse_verifier_accounting_coverage": (
                    DEFAULT_MIN_SPARSE_VERIFIER_ACCOUNTING_COVERAGE
                ),
            }
            for name, minimum in lower_bounds.items():
                if getattr(args, name) < minimum:
                    raise ValueError(
                        "acceptance-grade multimodal requires "
                        f"--{name.replace('_', '-')} >= {minimum}"
                    )
            upper_bounds = {
                "max_mixed_tpot_ratio": DEFAULT_MULTIMODAL_MAX_MIXED_TPOT_RATIO,
                "max_mixed_ttft_p99_seconds": DEFAULT_MULTIMODAL_MAX_MIXED_TTFT_P99_SECONDS,
                "cleanup_timeout_seconds": DEFAULT_CLEANUP_TIMEOUT_SECONDS,
                "cleanup_poll_seconds": DEFAULT_CLEANUP_POLL_SECONDS,
                "telemetry_interval_seconds": DEFAULT_TELEMETRY_INTERVAL_SECONDS,
                "max_schedule_lateness_seconds": DEFAULT_MAX_SCHEDULE_LATENESS_SECONDS,
                "max_process_rss_drift_mib": DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB,
                "max_process_rss_drift_fraction": DEFAULT_MAX_PROCESS_RSS_DRIFT_FRACTION,
                "max_process_rss_high_water_mib": DEFAULT_MAX_PROCESS_RSS_HIGH_WATER_MIB,
                "max_gpu_memory_drift_mib": DEFAULT_MAX_GPU_MEMORY_DRIFT_MIB,
                "max_gpu_memory_high_water_mib": DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB,
                "max_kv_block_utilization": DEFAULT_MAX_KV_BLOCK_UTILIZATION,
                "max_recurrent_slot_utilization": DEFAULT_MAX_RECURRENT_SLOT_UTILIZATION,
                "max_repeated_ngram_ratio": DEFAULT_MAX_REPEATED_NGRAM_RATIO,
                "max_sparse_verifier_fallback_ratio": (
                    DEFAULT_MAX_SPARSE_VERIFIER_FALLBACK_RATIO
                ),
            }
            for name, maximum in upper_bounds.items():
                if getattr(args, name) > maximum:
                    raise ValueError(
                        "acceptance-grade multimodal requires "
                        f"--{name.replace('_', '-')} <= {maximum}"
                    )
            oracle = Path(DEFAULT_MULTIMODAL_IMAGE).resolve()
            if Path(args.image).resolve() != oracle or not oracle.is_file():
                raise ValueError(
                    "acceptance-grade multimodal requires --image "
                    f"{DEFAULT_MULTIMODAL_IMAGE}"
                )
            required_phrases = {
                normalized_semantic_text(value)
                for value in args.image_required_phrases
            }
            if not {
                normalized_semantic_text(value)
                for value in DEFAULT_MULTIMODAL_REQUIRED_PHRASES
            }.issubset(required_phrases):
                raise ValueError(
                    "acceptance-grade multimodal cannot weaken the image OCR oracle"
                )
            configured_attributes = [
                {normalized_semantic_text(value) for value in alternatives}
                for alternatives in args.image_expected_attributes
            ]
            for required in DEFAULT_MULTIMODAL_EXPECTED_ATTRIBUTES:
                required_values = {
                    normalized_semantic_text(value) for value in required
                }
                if not any(required_values.issubset(values) for values in configured_attributes):
                    raise ValueError(
                        "acceptance-grade multimodal cannot weaken the image color oracle"
                    )
            sampling = production_sampling_policy_evidence(
                PART1_PRODUCTION_SAMPLING_POLICY,
                SamplingPolicy(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                ),
            )
            if not sampling["passed"]:
                raise ValueError(
                    "acceptance-grade multimodal requires the production sampling values"
                )
            if not acceptance_grade_evidence(
                args.acceptance_grade,
                args.require_mtp,
                args.expected_graph_components,
            )["passed"]:
                raise ValueError(
                    "acceptance-grade multimodal runs require MTP and "
                    "--expected-graph-components target,dflash"
                )


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output:
        jsonl = args.output
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        jsonl = DEFAULT_OUTPUT_ROOT / f"{stamp}-{args.mode}.jsonl"
    summary = jsonl.with_suffix(".summary.json")
    existing = [path for path in (jsonl, summary) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise ValueError(f"refusing to overwrite existing output files: {joined}")
    return jsonl, summary


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def serialized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = {key: json_compatible(value) for key, value in vars(args).items()}
    if "api_key" in values:
        values["api_key"] = "<redacted>"
    return values


async def async_main(args: argparse.Namespace) -> int:
    validate_args(args)
    tokenizer = TokenizerAdapter(args.tokenizer) if args.tokenizer else None
    if args.mode == "compare":
        jsonl_path, summary_path = output_paths(args)
        writer = JsonlWriter(jsonl_path)
        started = time.perf_counter()
        await writer.emit(
            "run_start",
            mode=args.mode,
            arguments=serialized_arguments(args),
        )
        try:
            summary = await compare_mode(args, tokenizer, writer)
            summary["elapsed_seconds"] = time.perf_counter() - started
            summary["evidence_jsonl"] = str(jsonl_path)
            await writer.emit("run_summary", **summary)
        except Exception as exc:
            summary = {
                "mode": args.mode,
                "passed": False,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "evidence_jsonl": str(jsonl_path),
            }
            await writer.emit("run_error", **summary)
        finally:
            writer.close()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Evidence: {jsonl_path}")
        print(f"Summary:  {summary_path}")
        return 0 if summary.get("passed") else 1
    policy = SamplingPolicy(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )
    jsonl_path, summary_path = output_paths(args)
    if (
        hasattr(args, "prefix_pressure_namespace")
        and args.prefix_pressure_namespace is None
    ):
        path_hash = stable_hash(str(jsonl_path.resolve()))[:12]
        args.prefix_pressure_namespace = f"{jsonl_path.stem}-{path_hash}"
    writer = JsonlWriter(jsonl_path)
    client = SoakClient(
        args.base_url,
        args.model,
        args.api_key,
        args.timeout,
        policy,
        tokenizer,
    )
    server_provenance = await collect_server_provenance(client, args.server_pid)
    provenance_required = server_provenance_required(args)
    server_provenance["required"] = provenance_required
    await writer.emit(
        "run_start",
        mode=args.mode,
        arguments=serialized_arguments(args),
        policy=asdict(policy),
        server_provenance=server_provenance,
    )
    started = time.perf_counter()
    summary: dict[str, Any]
    try:
        if (
            provenance_required
            and not server_provenance["evidence"]["complete"]
        ):
            failed = [
                name
                for name, passed in server_provenance["evidence"]["checks"].items()
                if not passed
            ]
            raise RuntimeError(
                "required server provenance is incomplete: " + ", ".join(failed)
            )
        if args.mode == "canary":
            summary = await canary_mode(args, client, writer)
        elif args.mode == "sweep":
            summary = await sweep_mode(args, client, writer)
        elif args.mode == "resident-decode":
            summary = await resident_decode_mode(args, client, writer)
        elif args.mode == "adversarial":
            summary = await adversarial_mode(args, client, writer)
        elif args.mode == "prefix-pressure":
            summary = await prefix_pressure_mode(args, client, writer)
        elif args.mode == "quality-replay":
            summary = await quality_replay_mode(args, client, writer)
        elif args.mode == "production":
            summary = await production_mode(args, client, writer)
        elif args.mode == "multimodal":
            summary = await multimodal_mode(
                args,
                client,
                writer,
                server_provenance,
            )
        else:
            raise RuntimeError(f"unsupported mode {args.mode}")
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["evidence_jsonl"] = str(jsonl_path)
        summary["server_provenance"] = server_provenance
        await writer.emit("run_summary", **summary)
    except Exception as exc:
        summary = {
            "mode": args.mode,
            "passed": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "evidence_jsonl": str(jsonl_path),
            "server_provenance": server_provenance,
        }
        await writer.emit("run_error", **summary)
    finally:
        await client.close()
        writer.close()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evidence: {jsonl_path}")
    print(f"Summary:  {summary_path}")
    return 0 if summary.get("passed") else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
