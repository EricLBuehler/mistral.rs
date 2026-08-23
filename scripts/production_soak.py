#!/usr/bin/env python3
"""Production correctness, scheduler, and endurance testing for an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import mimetypes
import os
import random
import re
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
DEFAULT_CONTEXT_MIX = ((1_024, 45), (8_192, 30), (32_768, 20), (100_000, 5))
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 5.0
DEFAULT_PROBE_INTERVAL_SECONDS = 30.0
DEFAULT_COMPARISON_WINDOW_SECONDS = 3_600.0
DEFAULT_OUTPUT_ROOT = Path("artifacts/production_soak")
OUTPUT_PREVIEW_CHARS = 256
DEFAULT_MAX_REPEATED_NGRAM_RATIO = 0.20
DEFAULT_FAIRNESS_MAX_SLOWDOWN = 3.0
DEFAULT_FAIRNESS_STAGGER_SECONDS = 0.05
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 30.0
DEFAULT_CLEANUP_POLL_SECONDS = 0.25
DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION = 0.05
DEFAULT_MIN_PREFIX_REUSE_FRACTION = 0.98
DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO = 0.98
DEFAULT_PREFIX_PRESSURE_CAPACITY_FRACTION = 1.10
DEFAULT_KV_BLOCK_SIZE_TOKENS = 32
DEFAULT_PREFIX_PRESSURE_MAX_ENTRIES = 128
DEFAULT_MIN_OUTPUT_EVENT_COVERAGE = 0.98
RETRIEVAL_PADDING_SLACK_TOKENS = 64
RETRIEVAL_SOURCE_MARGIN_TOKENS = 256
RETRIEVAL_INITIAL_SAMPLE_ROWS = 512
RETRIEVAL_PADDING_PROBES = (1, 2, 8)
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
EXACT_CONTEXT_SUFFIX = "\nEnd of deterministic production-soak context.\n"
PROMETHEUS_METRIC_PREFIXES = ("mistralrs_", "http_requests_in_flight")
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
        required = target_tokens + 8_192
        repeats = max(1, math.ceil(required / len(body_tokens)))
        source_ids = self.encode(CONTEXT_PARAGRAPH * repeats)
        estimated_body_tokens = target_tokens - fixed_tokens
        low = max(1, estimated_body_tokens - 4_096)
        high = min(len(source_ids), estimated_body_tokens + 4_096)
        candidates = [estimated_body_tokens]
        for offset in range(1, high - low + 1):
            if estimated_body_tokens - offset >= low:
                candidates.append(estimated_body_tokens - offset)
            if estimated_body_tokens + offset <= high:
                candidates.append(estimated_body_tokens + offset)
        for candidate_length in candidates:
            text = prefix + self.decode(source_ids[:candidate_length]) + EXACT_CONTEXT_SUFFIX
            if self.count(text) == target_tokens:
                return text
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
        best_below: tuple[int, str, int] | None = None
        for _ in range(RETRIEVAL_LENGTH_REFINEMENT_STEPS):
            if candidate_tokens in visited:
                break
            visited.add(candidate_tokens)
            text = build(candidate_tokens)
            actual = self.count(text)
            if actual == target_tokens:
                return text, expected
            if actual < target_tokens and (
                best_below is None or actual > best_below[2]
            ):
                best_below = (candidate_tokens, text, actual)
            candidate_tokens = max(
                1,
                min(len(filler_ids), candidate_tokens + target_tokens - actual),
            )
        if best_below is None:
            raise RuntimeError(
                f"could not construct a retrieval prompt below {target_tokens} tokens"
            )
        candidate_tokens, text, actual = best_below
        deficit = target_tokens - actual

        probes = sorted({min(deficit, count) for count in RETRIEVAL_PADDING_PROBES})
        for unit in RETRIEVAL_PADDING_CANDIDATES:
            if self.count(unit) != 1:
                continue
            if any(
                self.count(build(candidate_tokens, unit * count)) != actual + count
                for count in probes
            ):
                continue
            text = build(candidate_tokens, unit * deficit)
            if self.count(text) == target_tokens:
                return text, expected
        raise RuntimeError(
            f"could not find stable one-token padding for a {target_tokens}-token retrieval prompt"
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
    output_event_times: list[float] = field(default_factory=list)
    output_event_window_counts: list[int] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return self.ended - self.started

    def record(self, keep_output: bool) -> dict[str, Any]:
        value = asdict(self)
        output_event_times = value.pop("output_event_times")
        value["retained_output_event_timestamps"] = len(output_event_times)
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
        output_event_window_counts = (
            [0] * len(retain_output_event_windows)
            if retain_output_event_windows is not None
            else []
        )
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
                        now = time.perf_counter()
                        if first_token is None:
                            first_token = now
                        if retain_output_event_windows is None:
                            token_times.append(now)
                        else:
                            for index, (start, end) in enumerate(
                                retain_output_event_windows
                            ):
                                if start <= now and (
                                    now < end
                                    or index + 1 == len(retain_output_event_windows)
                                ):
                                    output_event_window_counts[index] += 1
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
            output_event_times=token_times,
            output_event_window_counts=output_event_window_counts,
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
            "tags": spec.tags,
        }

    async def disconnect(self, spec: RequestSpec, after_seconds: float) -> dict[str, Any]:
        started = time.perf_counter()

        async def consume() -> None:
            async with self.http.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json=self.payload(spec, stream=True),
            ) as response:
                response.raise_for_status()
                async for _ in response.aiter_bytes():
                    pass

        task = asyncio.create_task(consume())
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
            "error": error,
            "context_tokens": spec.context_tokens,
        }

    async def metrics(self) -> dict[str, float]:
        response = await self.http.get(f"{self.root_base}/metrics", timeout=30.0)
        response.raise_for_status()
        return parse_prometheus(response.text)


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
    matches = [value for key, value in snapshot.items() if key.split("{", 1)[0] == metric]
    return sum(matches) if matches else None


def metric_delta(before: dict[str, float], after: dict[str, float], metric: str) -> float | None:
    start = metric_total(before, metric)
    end = metric_total(after, metric)
    if start is None or end is None:
        return None
    return end - start


def summarize_batch(
    results: Sequence[RequestResult],
    wall_seconds: float,
    concurrency: int,
    name: str,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    output_tokens = sum(result.completion_tokens for result in successful)
    return {
        "name": name,
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "output_tokens": output_tokens,
        "output_tok_s_common_wall": output_tokens / wall_seconds if wall_seconds > 0 else None,
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


def repeated_ngram_ratio(tokens: Sequence[str], width: int = 4) -> float:
    if len(tokens) < width:
        return 0.0
    ngrams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


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
) -> dict[str, Any]:
    reference_by_key = {
        (result.case_id, result.seed): result for result in reference if result.ok
    }
    candidate_by_key = {
        (result.case_id, result.seed): result for result in candidate if result.ok
    }
    reference_keys = set(reference_by_key)
    candidate_keys = set(candidate_by_key)
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
    complete = not missing_candidate and not missing_reference
    return {
        "reference_phase": reference_phase,
        "candidate_phase": candidate_phase,
        "passed": complete and not mismatches,
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
        "mismatches": mismatches,
    }


def validate_sampled_output(
    result: RequestResult,
    tokenizer: TokenizerAdapter | None,
    max_repeated_ngram_ratio: float,
) -> tuple[bool, dict[str, Any]]:
    generated = result.reasoning_text + result.output_text
    if result.tool_calls:
        generated += json.dumps(result.tool_calls, sort_keys=True)
    units = text_units(generated, tokenizer)
    repetition = repeated_ngram_ratio(units)
    valid = (
        result.ok
        and result.completion_tokens > 0
        and result.output_chunks > 0
        and result.finish_reason is not None
        and bool(units)
        and repetition <= max_repeated_ngram_ratio
    )
    return valid, {
        "case_id": result.case_id,
        "seed": result.seed,
        "ok": result.ok,
        "nonempty": bool(units),
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "repeated_ngram_ratio": repetition,
        "max_repeated_ngram_ratio": max_repeated_ngram_ratio,
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
    passed = (
        compatibility["compatible"]
        and statistical["passed"]
        and bool(candidate_canary_passed)
        and bool(reference_canary_passed)
    )
    summary = {
        "mode": "compare",
        "passed": passed,
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "candidate_phase": args.candidate_phase,
        "reference_phase": args.reference_phase,
        "candidate_canary_passed": candidate_canary_passed,
        "reference_canary_passed": reference_canary_passed,
        "compatibility": compatibility,
        "statistical_comparison": statistical,
        "shared_fixed_seed_cases": len(shared),
        "exact_fixed_seed_matches": exact,
        "exact_fixed_seed_match_rate": exact / len(shared) if shared else None,
    }
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
    specs = make_canary_specs(args)
    baseline: list[RequestResult] | None = None
    baseline_phase = ""
    summaries: list[dict[str, Any]] = []
    all_candidate: list[RequestResult] = []
    candidate_normal_by_concurrency: dict[int, list[RequestResult]] = {}
    cross_phase_results: list[tuple[str, list[RequestResult]]] = []
    exact_replays: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    for concurrency in args.concurrencies:
        normal_results: list[RequestResult] | None = None
        normal_phase = ""
        for order in ("normal", "normal-replay", "reverse"):
            ordered = specs if order == "normal" else list(reversed(specs))
            if order == "normal-replay":
                ordered = specs
            phase = f"canary-c{concurrency}-{order}"
            results, summary = await run_batch(
                client, ordered, concurrency, writer, phase, keep_output=True
            )
            summary["order"] = order
            summaries.append(summary)
            for result in results:
                valid, detail = validate_sampled_output(
                    result, client.tokenizer, args.max_repeated_ngram_ratio
                )
                quality_checks.append({"phase": phase, "valid": valid, **detail})
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
                replay = exact_output_diagnostics(
                    normal_results, results, normal_phase, phase
                )
                exact_replays.append(replay)
                await writer.emit("exact_replay_comparison", **replay)
            else:
                cross_phase_results.append((phase, results))
    if baseline is None:
        raise RuntimeError("canary produced no baseline")

    cross_phase_comparisons = []
    fixed_seed_mismatches = []
    for phase, results in cross_phase_results:
        if phase == baseline_phase:
            continue
        exact = exact_output_diagnostics(baseline, results, baseline_phase, phase)
        statistical = compare_samples(
            results,
            baseline,
            client.tokenizer,
            args.stat_max_ks,
            args.stat_max_js,
        )
        comparison = {
            "phase": phase,
            "exact_diagnostics": exact,
            "statistical_comparison": statistical,
            "passed": exact["passed"] and statistical["passed"],
        }
        cross_phase_comparisons.append(comparison)
        fixed_seed_mismatches.extend(
            {"phase": phase, **mismatch} for mismatch in exact["mismatches"]
        )
        await writer.emit("cross_shape_comparison", **comparison)
    edge_summary = {"passed": True, "edges": {}}
    if not args.skip_edge_cases:
        edge_summary = await run_edge_cases(client, args, writer)
    statistical = None
    statistical_by_concurrency: list[dict[str, Any]] = []
    reference_summaries: list[dict[str, Any]] = []
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
    request_errors = sum(summary["errors"] for summary in summaries)
    passed = (
        request_errors == 0
        and all(item["passed"] for item in exact_replays)
        and all(item["passed"] for item in cross_phase_comparisons)
        and all(item["valid"] for item in quality_checks)
        and edge_summary["passed"]
        and (statistical is None or statistical["passed"])
        and all(item["passed"] for item in statistical_by_concurrency)
    )
    return {
        "mode": "canary",
        "passed": passed,
        "production_sampling": asdict(client.policy),
        "summaries": summaries,
        "exact_replays": exact_replays,
        "cross_phase_comparisons": cross_phase_comparisons,
        "fixed_seed_mismatches": fixed_seed_mismatches,
        "quality_checks": quality_checks,
        "edge_cases": edge_summary,
        "reference_summaries": reference_summaries,
        "statistical_comparison": statistical,
        "statistical_comparison_by_concurrency": statistical_by_concurrency,
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
    baseline, baseline_summary = await run_batch(
        client,
        specs,
        1,
        writer,
        "long-correctness-c1-baseline",
        keep_output=True,
    )
    summaries = [baseline_summary]
    exact_replays: list[dict[str, Any]] = []
    cross_phase_results: list[tuple[str, list[RequestResult]]] = []
    semantic_checks: list[dict[str, Any]] = []
    for result in baseline:
        valid, detail = validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )
        semantic_checks.append({"phase": "c1-baseline", "valid": valid, **detail})

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
        "long-correctness-c1-baseline",
        replay_phase,
    )
    exact_replays.append(replay_diagnostics)
    await writer.emit("exact_replay_comparison", **replay_diagnostics)
    for result in replay:
        valid, detail = validate_retrieval_result(
            result, client.tokenizer, args.max_repeated_ngram_ratio
        )
        semantic_checks.append({"phase": replay_phase, "valid": valid, **detail})

    for concurrency in args.long_correctness_concurrencies:
        normal_results: list[RequestResult] | None = None
        normal_phase = ""
        for order in ("normal", "normal-replay", "reverse"):
            ordered = specs if order == "normal" else list(reversed(specs))
            if order == "normal-replay":
                ordered = specs
            phase = f"long-correctness-c{concurrency}-{order}"
            results, summary = await run_batch(
                client, ordered, concurrency, writer, phase, keep_output=True
            )
            summaries.append(summary)
            if order == "normal":
                normal_results = results
                normal_phase = phase
                cross_phase_results.append((phase, results))
            elif order == "normal-replay":
                if normal_results is None:
                    raise RuntimeError("normal long-context phase must precede its exact replay")
                replay_diagnostics = exact_output_diagnostics(
                    normal_results, results, normal_phase, phase
                )
                exact_replays.append(replay_diagnostics)
                await writer.emit("exact_replay_comparison", **replay_diagnostics)
            else:
                cross_phase_results.append((phase, results))
            for result in results:
                valid, detail = validate_retrieval_result(
                    result, client.tokenizer, args.max_repeated_ngram_ratio
                )
                semantic_checks.append({"phase": phase, "valid": valid, **detail})

    cross_phase_comparisons = []
    for phase, results in cross_phase_results:
        exact = exact_output_diagnostics(
            baseline, results, "long-correctness-c1-baseline", phase
        )
        statistical = compare_samples(
            results,
            baseline,
            client.tokenizer,
            args.long_correctness_stat_max_ks,
            args.long_correctness_stat_max_js,
        )
        comparison = {
            "phase": phase,
            "exact_diagnostics": exact,
            "statistical_comparison": statistical,
            "passed": exact["passed"] and statistical["passed"],
        }
        cross_phase_comparisons.append(comparison)
        await writer.emit("cross_shape_comparison", **comparison)

    passed = (
        all(summary["errors"] == 0 for summary in summaries)
        and all(item["passed"] for item in exact_replays)
        and all(item["passed"] for item in cross_phase_comparisons)
        and all(item["valid"] for item in semantic_checks)
    )
    result = {
        "passed": passed,
        "lengths": list(args.long_correctness_context_lengths),
        "concurrencies": list(args.long_correctness_concurrencies),
        "summaries": summaries,
        "exact_replays": exact_replays,
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
    "http_requests_in_flight",
)
RESIDENT_TRANSIENT_CLEANUP_GAUGES = (
    "mistralrs_sequences_running",
    "mistralrs_sequences_waiting",
    "mistralrs_requests_pending_admission",
    "mistralrs_recurrent_state_slots_used",
    "http_requests_in_flight",
)

CANCELLATION_OWNERSHIP_METRIC_LIMITATION = (
    "mistralrs_kv_cache_blocks_used includes valid prefix-cache retention. The engine still "
    "needs separate active-sequence and prefix-owned KV block gauges to prove KV ownership "
    "cleanup after cancellation."
)


async def poll_for_cleanup(
    client: SoakClient,
    writer: JsonlWriter,
    baseline: dict[str, float],
    timeout_seconds: float,
    poll_seconds: float,
    phase: str,
    gauges: Sequence[str] = CLEANUP_GAUGES,
) -> tuple[bool, dict[str, float], dict[str, Any]]:
    baseline_values = {name: metric_total(baseline, name) for name in gauges}
    missing_baseline = [name for name, value in baseline_values.items() if value is None]
    if missing_baseline:
        return False, {}, {"missing_baseline_metrics": missing_baseline}
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
            return True, last_snapshot, {
                "baseline": baseline_values,
                "observed": last_values,
            }
        await asyncio.sleep(poll_seconds)
    return False, last_snapshot, {
        "baseline": baseline_values,
        "observed": last_values,
        "timeout_seconds": timeout_seconds,
    }


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

    async def record_runtime(phase: str, gated: bool) -> None:
        nonlocal runtime_cursor
        after = await safe_metrics(client, writer, f"{phase}-runtime-end")
        evidence = {
            "phase": phase,
            "gated": gated,
            "mtp": speculative_evidence(
                runtime_cursor,
                after,
                gated and "dflash" in args.expected_graph_components,
            ),
            "cuda_graph": cuda_graph_evidence(
                runtime_cursor,
                after,
                initial_metrics,
                args.expected_graph_components,
                args.min_cuda_graph_replay_ratio,
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
            )
        )
        runtime_evidence.append(evidence)
        await writer.emit("adversarial_runtime_summary", **evidence)
        runtime_cursor = after

    long_correctness = await run_long_context_correctness(args, client, writer)
    await record_runtime("adversarial-long-correctness", True)

    mixed_lengths = (
        args.context_lengths * math.ceil(args.mixed_requests / len(args.context_lengths))
    )[: args.mixed_requests]
    mixed_specs = [
        RequestSpec(
            case_id=f"mixed-{index:03d}-{length}",
            seed=args.seed + index,
            max_tokens=args.max_tokens,
            prompt=exact_context(client, length, f"mixed-{index}-{length}"),
            context_tokens=length,
            tags={"scenario": "mixed"},
        )
        for index, length in enumerate(mixed_lengths)
    ]
    mixed_results, summary = await run_batch(
        client,
        mixed_specs,
        min(args.max_seqs, len(mixed_specs)),
        writer,
        "adversarial-mixed-context",
        keep_output=True,
    )
    summaries.append(summary)
    await record_quality("adversarial-mixed-context", mixed_results)
    await record_runtime("adversarial-mixed-context", True)

    short_spec = retrieval_spec(
        client,
        shortest,
        "fairness-short-b",
        args.seed + 10_000,
        args.max_tokens,
        {"scenario": "fairness", "role": "short"},
    )
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
            longest,
            f"fairness-long-a-{order}",
            args.seed + 10_001 + index,
            args.max_tokens,
            {"scenario": "fairness", "role": "long", "order": order},
        )
        for index, order in enumerate(("a_then_b", "b_then_a"))
    }
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
            }
        )
    fairness_slowdowns = [
        value
        for run in fairness_runs
        for value in (run["short_ttft_slowdown"], run["short_tpot_slowdown"])
    ]
    fairness_semantic_results = [
        warmup_short,
        isolated_short,
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
        and all(run["short_equal_to_isolated"] for run in fairness_runs)
        and all(value is not None and value <= args.fairness_max_slowdown for value in fairness_slowdowns)
        and all(fairness_semantic)
    )
    fairness_result = {
        "passed": fairness_passed,
        "max_slowdown": args.fairness_max_slowdown,
        "stagger_seconds": args.fairness_stagger_seconds,
        "warmup_equal_to_isolated": (
            warmup_short.output_transcript == isolated_short.output_transcript
        ),
        "isolated_short_ttft_seconds": isolated_short.ttft_seconds,
        "isolated_short_tpot_seconds": isolated_short.tpot_seconds,
        "orders": fairness_runs,
    }
    await writer.emit("fairness_summary", **fairness_result)
    await record_runtime("adversarial-fairness", True)

    overlap_specs = [
        RequestSpec(
            case_id="overlap-large-prefill",
            seed=args.seed + 20_000,
            max_tokens=args.max_tokens,
            prompt=exact_context(client, longest, "overlap-large-prefill"),
            context_tokens=longest,
            tags={"scenario": "overlap", "role": "large_prefill"},
        )
    ] + [
        RequestSpec(
            case_id=f"overlap-decode-{index:03d}",
            seed=args.seed + 20_001 + index,
            max_tokens=args.max_tokens,
            prompt=exact_context(client, shortest, f"overlap-decode-{index}"),
            context_tokens=shortest,
            tags={"scenario": "overlap", "role": "decode"},
        )
        for index in range(args.overlap_short_requests)
    ]
    overlap_results, overlap_summary = await run_batch(
        client,
        overlap_specs,
        min(args.max_seqs, len(overlap_specs)),
        writer,
        "adversarial-prefill-overlap",
        keep_output=False,
    )
    summaries.append(overlap_summary)
    await record_quality("adversarial-prefill-overlap", overlap_results)
    await record_runtime("adversarial-prefill-overlap", True)

    pre_cancel_metrics = await safe_metrics(client, writer, "adversarial-pre-cancel")
    cancellation_specs = [
        RequestSpec(
            case_id=f"disconnect-{index:03d}",
            seed=args.seed + 30_000 + index,
            max_tokens=args.max_tokens * 4,
            prompt=exact_context(
                client, args.cancel_context_tokens, f"disconnect-{index}"
            ),
            context_tokens=args.cancel_context_tokens,
            tags={"scenario": "disconnect"},
        )
        for index in range(args.cancel_requests)
    ]
    cancellation_rng = random.Random(args.seed + 30_000)
    disconnect_delays = [
        args.disconnect_after_seconds * cancellation_rng.uniform(0.5, 2.0)
        for _ in cancellation_specs
    ]
    cancellation_results = await asyncio.gather(
        *(
            client.disconnect(spec, disconnect_delays[index])
            for index, spec in enumerate(cancellation_specs)
        )
    )
    for result in cancellation_results:
        await writer.emit("disconnect", phase="adversarial-cancellation", **result)
    disconnects_ok = all(result["outcome"] == "cancelled" for result in cancellation_results)
    cancel_cleanup_ok, post_cancel_metrics, cancel_cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        pre_cancel_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-cancel-cleanup-poll",
    )
    cancel_admissions = metric_delta(
        pre_cancel_metrics, post_cancel_metrics, "mistralrs_prefix_cache_lookups_total"
    )
    cancel_admissions_ok = (
        cancel_admissions is not None and cancel_admissions >= len(cancellation_specs)
    )
    cancel_kv_blocks_delta = metric_delta(
        pre_cancel_metrics,
        post_cancel_metrics,
        "mistralrs_kv_cache_blocks_used",
    )

    timeout_spec = retrieval_spec(
        client,
        longest,
        "timeout-large",
        args.seed + 40_000,
        args.max_tokens,
        {"scenario": "timeout"},
    )
    pre_timeout_metrics = await safe_metrics(client, writer, "adversarial-pre-timeout")
    timeout_result = await client.stream_request(
        timeout_spec, timeout_seconds=args.timeout_test_seconds
    )
    await writer.emit(
        "request",
        phase="adversarial-timeout",
        concurrency=1,
        **timeout_result.record(False),
    )
    timeout_cleanup_ok, post_timeout_metrics, timeout_cleanup_detail = await poll_for_cleanup(
        client,
        writer,
        pre_timeout_metrics,
        args.cleanup_timeout_seconds,
        args.cleanup_poll_seconds,
        "adversarial-timeout-cleanup-poll",
    )
    timeout_admissions = metric_delta(
        pre_timeout_metrics, post_timeout_metrics, "mistralrs_prefix_cache_lookups_total"
    )
    timeout_observed = timeout_result.error_kind == "ReadTimeout"
    timeout_admitted = timeout_admissions is not None and timeout_admissions >= 1
    timeout_kv_blocks_delta = metric_delta(
        pre_timeout_metrics,
        post_timeout_metrics,
        "mistralrs_kv_cache_blocks_used",
    )
    retry, retry_summary = await run_batch(
        client, [timeout_spec], 1, writer, "adversarial-retry", keep_output=True
    )
    summaries.append(retry_summary)
    retry_semantic_ok = validate_retrieval_result(
        retry[0], client.tokenizer, args.max_repeated_ngram_ratio
    )[0]
    await record_runtime("adversarial-cancellation-timeout-retry", False)

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
    await record_runtime("adversarial-burst", True)

    prefix_prompt = contexts[args.prefix_context_tokens]
    prefix_seed = args.seed + 60_000
    pre_prefix_metrics = await safe_metrics(client, writer, "prefix-before-cold")
    prefix_spec = RequestSpec(
        case_id="prefix-cold",
        seed=prefix_seed,
        max_tokens=args.max_tokens,
        prompt=prefix_prompt,
        context_tokens=args.prefix_context_tokens,
        tags={"scenario": "prefix", "stage": "cold"},
    )
    cold, cold_summary = await run_batch(
        client, [prefix_spec], 1, writer, "prefix-cold", keep_output=True
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
        client, [hit_spec], 1, writer, "prefix-hit", keep_output=True
    )
    summaries.append(hit_summary)
    await record_quality("prefix-hit", hit)
    post_hit_metrics = await safe_metrics(client, writer, "prefix-after-hit")
    hit_evidence = prefix_cache_evidence(
        post_cold_metrics,
        post_hit_metrics,
        1,
        args.prefix_context_tokens,
        args.min_prefix_reuse_fraction,
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
    eviction_specs = [
        RequestSpec(
            case_id=f"prefix-pressure-{index:04d}",
            seed=args.seed + 61_000 + index,
            max_tokens=args.prefix_pressure_max_tokens,
            prompt=exact_context(
                client,
                args.prefix_pressure_context_tokens,
                f"prefix-pressure-{index}",
            ),
            context_tokens=args.prefix_pressure_context_tokens,
            tags={"scenario": "prefix", "stage": "pressure"},
        )
        for index in range(pressure_entries)
    ]
    pressure_results, pressure_summary = await run_batch(
        client,
        eviction_specs,
        min(args.max_seqs, len(eviction_specs)),
        writer,
        "prefix-pressure",
        keep_output=False,
    )
    summaries.append(pressure_summary)
    await record_quality("prefix-pressure", pressure_results)
    post_pressure_metrics = await safe_metrics(client, writer, "prefix-after-pressure-load")
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
        client, [after_spec], 1, writer, "prefix-after-pressure", keep_output=True
    )
    summaries.append(after_summary)
    await record_quality("prefix-after-pressure", after)
    post_after_metrics = await safe_metrics(client, writer, "prefix-after-evicted-retry")
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
    prefix_cache_stage_passed = (
        prefix_correct
        and cold_miss_observed
        and hit_evidence["passed"]
        and pressure_reached_eviction
        and target_was_evicted
    )
    prefix_cache_stages = {
        "passed": prefix_cache_stage_passed,
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
        "pressure_evictions": pressure_evictions,
        "pressure_reached_eviction": pressure_reached_eviction,
        "target_retry_reused_tokens": after_reused_tokens,
        "target_retry_reuse_fraction": after_reuse_fraction,
        "target_was_evicted": target_was_evicted,
        "outputs_equal": prefix_correct,
    }
    await writer.emit("prefix_cache_stage_summary", **prefix_cache_stages)
    await record_runtime("adversarial-prefix-cache", False)

    churn_summaries = []
    for round_index in range(args.churn_rounds):
        for width in (args.max_seqs - 1, args.max_seqs, args.max_seqs + 1):
            churn_specs = [
                RequestSpec(
                    case_id=f"churn-r{round_index}-c{width}-{index}",
                    seed=args.seed + 70_000 + round_index * 1_000 + width * 20 + index,
                    max_tokens=args.churn_max_tokens,
                    prompt=exact_context(
                        client,
                        args.context_lengths[index % len(args.context_lengths)],
                        f"churn-r{round_index}-c{width}-{index}",
                    ),
                    context_tokens=args.context_lengths[index % len(args.context_lengths)],
                    tags={"scenario": "churn", "round": round_index, "width": width},
                )
                for index in range(width)
            ]
            churn_results, churn_summary = await run_batch(
                client,
                churn_specs,
                width,
                writer,
                f"churn-r{round_index}-width{width}",
                keep_output=False,
            )
            churn_summaries.append(churn_summary)
            summaries.append(churn_summary)
            await record_quality(
                f"churn-r{round_index}-width{width}",
                churn_results,
            )

    await record_runtime("adversarial-churn", True)

    final_metrics = await safe_metrics(client, writer, "adversarial-end")
    running_after_cancel = metric_total(post_cancel_metrics, "mistralrs_sequences_running")
    waiting_after_cancel = metric_total(post_cancel_metrics, "mistralrs_sequences_waiting")
    metrics_delta = selected_metric_deltas(initial_metrics, final_metrics)
    adversarial_mtp = speculative_evidence(
        initial_metrics,
        final_metrics,
        "dflash" in args.expected_graph_components,
    )
    adversarial_graph = cuda_graph_evidence(
        initial_metrics,
        final_metrics,
        initial_metrics,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    passed = (
        all(summary["errors"] == 0 for summary in summaries)
        and long_correctness["passed"]
        and fairness_passed
        and disconnects_ok
        and cancel_cleanup_ok
        and cancel_admissions_ok
        and timeout_observed
        and timeout_admitted
        and timeout_cleanup_ok
        and retry_semantic_ok
        and prefix_cache_stage_passed
        and all(item["valid"] for item in quality_checks)
        and all(item["passed"] for item in runtime_evidence)
        and adversarial_mtp["passed"]
        and adversarial_graph["passed"]
    )
    return {
        "mode": "adversarial",
        "passed": passed,
        "summaries": summaries,
        "long_context_correctness": long_correctness,
        "fairness": fairness_result,
        "disconnects": cancellation_results,
        "disconnects_ok": disconnects_ok,
        "cancel_admissions": cancel_admissions,
        "cancel_admissions_ok": cancel_admissions_ok,
        "cancel_cleanup_ok": cancel_cleanup_ok,
        "cancel_cleanup_detail": cancel_cleanup_detail,
        "cancel_kv_cache_blocks_used_delta": cancel_kv_blocks_delta,
        "timeout_observed": timeout_observed,
        "timeout_admitted": timeout_admitted,
        "timeout_cleanup_ok": timeout_cleanup_ok,
        "timeout_cleanup_detail": timeout_cleanup_detail,
        "timeout_kv_cache_blocks_used_delta": timeout_kv_blocks_delta,
        "kv_ownership_gate_complete": False,
        "kv_ownership_core_metric_needed": CANCELLATION_OWNERSHIP_METRIC_LIMITATION,
        "retry_succeeded": retry_summary["errors"] == 0 and retry_semantic_ok,
        "prefix_outputs_equal": prefix_correct,
        "prefix_cache_stages": prefix_cache_stages,
        "quality_checks": quality_checks,
        "cleanup_ok": cancel_cleanup_ok and timeout_cleanup_ok,
        "post_cancel_sequences_running": running_after_cancel,
        "post_cancel_sequences_waiting": waiting_after_cancel,
        "metrics_delta": metrics_delta,
        "runtime_evidence": runtime_evidence,
        "mtp": adversarial_mtp,
        "cuda_graph": adversarial_graph,
        "churn_batches": len(churn_summaries),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def selected_metric_deltas(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float | None]:
    names = (
        "mistralrs_tokens_processed_total",
        "mistralrs_speculative_drafts_total",
        "mistralrs_speculative_draft_tokens_proposed_total",
        "mistralrs_speculative_draft_tokens_accepted_total",
        "mistralrs_prefix_cache_lookups_total",
        "mistralrs_prefix_cache_hits_total",
        "mistralrs_prefix_cache_tokens_matched_total",
        "mistralrs_prefix_cache_tokens_reused_total",
        "mistralrs_prefix_cache_evictions_total",
        "mistralrs_paged_preemptions_total",
        "mistralrs_speculative_staged_drops_total",
        "mistralrs_encoder_cache_hits_total",
        "mistralrs_encoder_cache_misses_total",
        "mistralrs_cuda_graph_events_total",
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


def speculative_evidence(
    before: dict[str, float],
    after: dict[str, float],
    required: bool,
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
    return {
        "passed": bool((sane and (active or not required)) or (absent and not required)),
        "required": required,
        "active": active,
        "drafts": drafts,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "acceptance_rate": ratio(accepted, proposed),
        "mean_accepted_draft_length": ratio(accepted, drafts),
        "mean_advance_length_with_bonus": (
            1.0 + accepted / drafts if accepted is not None and drafts else None
        ),
        "accepted_per_position": per_position,
    }


CUDA_GRAPH_METRIC_LIMITATION = (
    "The current engine metric does not count every target Ok(None) graph miss or "
    "ineligible eager decode. A per-decode replay/eager outcome counter is still required "
    "for complete graph coverage."
)


def cuda_graph_evidence(
    before: dict[str, float],
    after: dict[str, float],
    startup: dict[str, float],
    expected_components: Sequence[str],
    min_replay_ratio: float,
) -> dict[str, Any]:
    deltas = labeled_metric_deltas(
        before,
        after,
        "mistralrs_cuda_graph_events_total",
    )
    startup_values = labeled_metric_values(
        startup,
        "mistralrs_cuda_graph_events_total",
    )
    components = {}
    for component in expected_components:
        replay = sum(
            item["delta"]
            for item in deltas
            if item["labels"].get("component") == component
            and item["labels"].get("event") == "replay"
            and item["labels"].get("outcome") == "success"
        )
        eager = sum(
            item["delta"]
            for item in deltas
            if item["labels"].get("component") == component
            and item["labels"].get("event") == "eager_fallback"
            and item["labels"].get("outcome") == "success"
        )
        failures = sum(
            item["delta"]
            for item in deltas
            if item["labels"].get("component") == component
            and item["labels"].get("outcome") == "failure"
        )
        startup_failures = sum(
            item["value"]
            for item in startup_values
            if item["labels"].get("component") == component
            and item["labels"].get("outcome") == "failure"
        )
        replay_ratio = ratio(replay, replay + eager)
        components[component] = {
            "passed": (
                replay > 0
                and replay_ratio is not None
                and replay_ratio >= min_replay_ratio
                and failures == 0
                and startup_failures == 0
            ),
            "replay_successes": replay,
            "recorded_eager_fallback_successes": eager,
            "replay_ratio_over_recorded_outcomes": replay_ratio,
            "failures": failures,
            "startup_cumulative_failures": startup_failures,
        }
    return {
        "passed": bool(components) and all(item["passed"] for item in components.values()),
        "expected_components": list(expected_components),
        "min_replay_ratio": min_replay_ratio,
        "components": components,
        "events": deltas,
        "instrumentation_complete": False,
        "core_metric_needed": CUDA_GRAPH_METRIC_LIMITATION,
    }


def metric_values(
    snapshot: dict[str, float], names: Sequence[str]
) -> dict[str, float | None]:
    return {name: metric_total(snapshot, name) for name in names}


def prefix_cache_evidence(
    before: dict[str, float],
    after: dict[str, float],
    expected_requests: int,
    expected_prompt_tokens: int,
    min_reuse_fraction: float,
) -> dict[str, Any]:
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
    minimum_reused_tokens = expected_prompt_tokens * min_reuse_fraction
    return {
        "passed": (
            lookups is not None
            and hits is not None
            and matched_tokens is not None
            and reused_tokens is not None
            and lookups >= expected_requests
            and hits >= expected_requests
            and reused_tokens >= minimum_reused_tokens
            and ratio(reused_tokens, matched_tokens) is not None
            and ratio(reused_tokens, matched_tokens) >= min_reuse_fraction
        ),
        "expected_requests": expected_requests,
        "expected_prompt_tokens": expected_prompt_tokens,
        "minimum_reused_tokens": minimum_reused_tokens,
        "min_reuse_fraction": min_reuse_fraction,
        "lookups": lookups,
        "hits": hits,
        "hit_rate": ratio(hits, lookups),
        "matched_tokens": matched_tokens,
        "reused_tokens": reused_tokens,
        "reused_prompt_fraction": ratio(reused_tokens, expected_prompt_tokens),
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
        len(verification_specs),
        sum(spec.context_tokens for spec in verification_specs),
        args.min_prefix_reuse_fraction,
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
            "cross_shape_comparisons": [],
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
    phase_runs: list[tuple[str, list[RequestResult], dict[str, Any]]] = []

    async def run_resident_batch(concurrency: int, phase: str) -> None:
        before = await safe_metrics(client, writer, f"{phase}-start")
        results, summary = await run_batch(
            client,
            specs,
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
            len(specs),
            sum(spec.context_tokens for spec in specs),
            args.min_prefix_reuse_fraction,
        )
        mtp = speculative_evidence(
            before,
            after,
            "dflash" in args.expected_graph_components,
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
                "passed": (
                    summary["errors"] == 0
                    and cleanup_ok
                    and prefix["passed"]
                    and mtp["passed"]
                    and graph["passed"]
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

    for concurrency in args.concurrencies:
        await run_resident_batch(concurrency, f"resident-decode-c{concurrency}")

    baseline_phase, baseline_results, _ = phase_runs[0]
    cross_shape_comparisons = []
    for phase, results, _ in phase_runs[1:]:
        comparison = {
            "phase": phase,
            "exact_diagnostics": exact_output_diagnostics(
                baseline_results,
                results,
                baseline_phase,
                phase,
            ),
            "statistical_comparison": compare_samples(
                results,
                baseline_results,
                client.tokenizer,
                args.stat_max_ks,
                args.stat_max_js,
            ),
        }
        comparison["passed"] = (
            comparison["exact_diagnostics"]["passed"]
            and comparison["statistical_comparison"]["passed"]
        )
        cross_shape_comparisons.append(comparison)
        await writer.emit("resident_decode_cross_shape", **comparison)

    final_c1_replay = None
    if args.final_c1_replay and 1 in args.concurrencies:
        await run_resident_batch(1, "resident-decode-c1-final")
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
        )
        initial_c1_summary = next(
            summary
            for phase, _, summary in phase_runs
            if phase == "resident-decode-c1"
        )
        throughput_ratio = ratio(
            final_summary["output_tok_s_common_wall"],
            initial_c1_summary["output_tok_s_common_wall"],
        )
        final_c1_replay = {
            "passed": (
                exact["passed"]
                and final_summary["passed"]
                and throughput_ratio is not None
                and throughput_ratio >= args.min_final_c1_throughput_ratio
            ),
            "exact_diagnostics": exact,
            "initial_output_tok_s": initial_c1_summary["output_tok_s_common_wall"],
            "final_output_tok_s": final_summary["output_tok_s_common_wall"],
            "throughput_ratio": throughput_ratio,
            "min_throughput_ratio": args.min_final_c1_throughput_ratio,
        }
        await writer.emit("resident_decode_c1_replay", **final_c1_replay)

    final_metrics = await safe_metrics(client, writer, "resident-decode-end")
    phase_summaries = [summary for _, _, summary in phase_runs]
    passed = (
        residency_passed
        and all(summary["passed"] for summary in phase_summaries)
        and all(item["passed"] for item in cross_shape_comparisons)
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
        "cross_shape_comparisons": cross_shape_comparisons,
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
    histograms = {
        key.split("{", 1)[0][:-7]
        for key in after
        if key.split("{", 1)[0].endswith("_bucket")
        and any(term in key.lower() for term in ("queue", "admission", "waiting"))
    }
    return {
        name: {
            "p50": histogram_quantile_delta(before, after, name, 0.50),
            "p95": histogram_quantile_delta(before, after, name, 0.95),
            "p99": histogram_quantile_delta(before, after, name, 0.99),
        }
        for name in sorted(histograms)
    }


def summarize_telemetry(
    snapshots: Sequence[tuple[float, dict[str, float], dict[str, Any]]]
) -> dict[str, Any]:
    gauge_names = (
        "mistralrs_sequences_running",
        "mistralrs_sequences_waiting",
        "mistralrs_sequences_capacity",
        "mistralrs_requests_pending_admission",
        "mistralrs_kv_cache_blocks_used",
        "mistralrs_kv_cache_blocks_total",
        "mistralrs_recurrent_state_slots_used",
        "mistralrs_recurrent_state_slots_total",
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
    for _, _, process in snapshots:
        for gpu in process.get("gpus") or []:
            gpu_utilization.append(gpu["utilization_percent"])
            gpu_memory.append(gpu["memory_used_mib"])
            gpu_power.append(gpu["power_watts"])
        if process.get("process_vmrss_kib") is not None:
            process_rss.append(process["process_vmrss_kib"])
    return {
        "gauges": gauges,
        "gpu_utilization_percent": distribution(gpu_utilization),
        "gpu_memory_used_mib": distribution(gpu_memory),
        "gpu_power_watts": distribution(gpu_power),
        "process_rss_kib": distribution(process_rss),
    }


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
    except (FileNotFoundError, ValueError):
        telemetry["gpus"] = []
    if server_pid is not None:
        status_path = Path(f"/proc/{server_pid}/status")
        try:
            status = status_path.read_text(encoding="utf-8")
            for key in ("VmRSS", "VmSize", "VmSwap"):
                match = re.search(rf"^{key}:\s+(\d+)\s+kB$", status, re.MULTILINE)
                telemetry[f"process_{key.lower()}_kib"] = int(match.group(1)) if match else None
        except OSError as exc:
            telemetry["process_error"] = f"{type(exc).__name__}: {exc}"
    return telemetry


async def telemetry_loop(
    client: SoakClient,
    writer: JsonlWriter,
    stop: asyncio.Event,
    interval_seconds: float,
    server_pid: int | None,
    snapshots: list[tuple[float, dict[str, float], dict[str, Any]]],
) -> None:
    while not stop.is_set():
        timestamp = time.perf_counter()
        metrics, process = await asyncio.gather(
            safe_metrics(client, writer, "production-telemetry"),
            process_telemetry(server_pid),
        )
        snapshots.append((timestamp, metrics, process))
        await writer.emit(
            "telemetry",
            monotonic_seconds=timestamp,
            metrics=metrics,
            process=process,
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass


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


async def production_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    await calibrate_prompt_profiles(client, writer, (CONTEXT_PROMPT_PROFILE,))
    lengths = tuple(length for length, _ in args.context_mix)
    capacity_metrics = await safe_metrics(client, writer, "production-capacity")
    sequence_capacity = metric_total(capacity_metrics, "mistralrs_sequences_capacity")
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
    prewarm_metrics = await safe_metrics(client, writer, "production-prewarm-start")
    prewarm_specs = [
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
            extra={"ignore_eos": True},
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
        len(verification_specs),
        sum(spec.context_tokens or 0 for spec in verification_specs),
        args.min_prefix_reuse_fraction,
    )
    prewarm_passed = (
        prewarm_summary["errors"] == 0
        and verification_summary["errors"] == 0
        and prewarm_cleanup_ok
        and verification_cleanup_ok
        and residency_evidence["passed"]
    )
    prewarm = {
        "passed": prewarm_passed,
        "sequence_capacity": sequence_capacity,
        "resident_prompt_budget": resident_prompt_budget,
        "pool_counts": pool_counts,
        "prewarm": prewarm_summary,
        "verification": verification_summary,
        "prefix_cache": residency_evidence,
        "prewarm_cleanup": prewarm_cleanup,
        "verification_cleanup": verification_cleanup,
    }
    await writer.emit("production_prewarm_summary", **prewarm)
    stop = asyncio.Event()
    snapshots: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    initial_metrics, initial_process = await asyncio.gather(
        safe_metrics(client, writer, "production-start"),
        process_telemetry(args.server_pid),
    )
    run_started = time.perf_counter()
    snapshots.append((run_started, initial_metrics, initial_process))
    telemetry_task = asyncio.create_task(
        telemetry_loop(
            client,
            writer,
            stop,
            args.telemetry_interval_seconds,
            args.server_pid,
            snapshots,
        )
    )
    all_results: list[RequestResult] = []
    phase_summaries: list[dict[str, Any]] = []
    phase_duration = args.duration_seconds / len(args.concurrencies)

    async def run_phase(concurrency: int, phase_index: int) -> None:
        phase = f"production-c{concurrency}"
        phase_metrics_before = await safe_metrics(client, writer, f"{phase}-metrics-start")
        phase_start = time.perf_counter()
        phase_end = phase_start + phase_duration
        comparison_window = min(args.comparison_window_seconds, phase_duration / 2.0)
        retained_output_event_windows = (
            (phase_start, phase_start + comparison_window),
            (phase_end - comparison_window, phase_end),
        )
        phase_results: list[RequestResult] = []

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
                result = await client.stream_request(
                    spec,
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
            probe_index = 0
            next_probe = phase_start + args.probe_interval_seconds
            while next_probe < phase_end:
                await asyncio.sleep(max(0.0, next_probe - time.perf_counter()))
                if time.perf_counter() >= phase_end:
                    break
                spec = RequestSpec(
                    case_id=f"probe-c{concurrency}-{probe_index:04d}",
                    seed=args.seed + 9_000_000,
                    max_tokens=args.probe_max_tokens,
                    prompt=context_pools[min(lengths)][0],
                    context_tokens=min(lengths),
                    tags={"scenario": "production", "role": "c1_probe", "load": concurrency},
                    extra={"ignore_eos": args.fixed_output_length},
                )
                result = await client.stream_request(spec, scheduled_at=next_probe)
                phase_results.append(result)
                await writer.emit(
                    "request",
                    phase=phase,
                    concurrency=concurrency,
                    **result.record(True),
                )
                probe_index += 1
                next_probe += args.probe_interval_seconds
                now = time.perf_counter()
                while next_probe <= now:
                    next_probe += args.probe_interval_seconds

        await writer.emit(
            "production_phase_start",
            phase=phase,
            concurrency=concurrency,
            planned_seconds=phase_duration,
        )
        workers = [asyncio.create_task(worker(index)) for index in range(concurrency)]
        probe_task = asyncio.create_task(probes())
        await asyncio.gather(*workers, probe_task)
        phase_metrics_after = await safe_metrics(client, writer, f"{phase}-metrics-end")
        wall = time.perf_counter() - phase_start
        summary = summarize_batch(phase_results, wall, concurrency, phase)
        traffic = [result for result in phase_results if result.tags.get("role") == "traffic"]
        probes_only = [result for result in phase_results if result.tags.get("role") == "c1_probe"]
        quality = [
            {
                "valid": valid,
                **detail,
            }
            for result in phase_results
            for valid, detail in [
                validate_sampled_output(
                    result,
                    client.tokenizer,
                    args.max_repeated_ngram_ratio,
                )
            ]
        ]
        summary["traffic"] = summarize_batch(traffic, wall, concurrency, f"{phase}-traffic")
        summary["probes"] = summarize_batch(probes_only, wall, 1, f"{phase}-probes")
        summary["traffic_by_context"] = summarize_context_groups(
            traffic,
            wall,
            concurrency,
            f"{phase}-traffic",
        )
        summary["quality_checked"] = len(quality)
        summary["quality_failures"] = [item for item in quality if not item["valid"]]
        probe_transcripts = {
            result.output_transcript for result in probes_only if result.ok
        }
        summary["probe_fixed_seed_samples"] = len(probes_only)
        summary["probe_fixed_seed_unique_outputs"] = len(probe_transcripts)
        summary["probe_fixed_seed_exact"] = (
            len(probes_only) >= 2
            and len(probe_transcripts) == 1
            and all(result.ok for result in probes_only)
        )
        summary["first_vs_last_window"] = compare_time_windows(
            traffic,
            phase_start,
            phase_end,
            comparison_window,
            args.min_output_event_coverage,
        )
        degradation = summary["first_vs_last_window"]["throughput_degradation_fraction"]
        summary["degradation_ok"] = (
            degradation is not None
            and degradation <= args.max_throughput_degradation_fraction
            and summary["first_vs_last_window"]["output_event_coverage_ok"]
        )
        summary["quality_ok"] = all(item["valid"] for item in quality)
        summary["prefix_cache"] = prefix_cache_evidence(
            phase_metrics_before,
            phase_metrics_after,
            len(phase_results),
            sum(result.context_tokens or 0 for result in phase_results),
            args.min_prefix_reuse_fraction,
        )
        summary["mtp"] = speculative_evidence(
            phase_metrics_before,
            phase_metrics_after,
            "dflash" in args.expected_graph_components,
        )
        summary["cuda_graph"] = cuda_graph_evidence(
            phase_metrics_before,
            phase_metrics_after,
            initial_metrics,
            args.expected_graph_components,
            args.min_cuda_graph_replay_ratio,
        )
        summary["queue_latency_histograms"] = queue_histogram_summaries(
            phase_metrics_before,
            phase_metrics_after,
        )
        summary["metric_deltas"] = selected_metric_deltas(
            phase_metrics_before,
            phase_metrics_after,
        )
        summary["phase_metrics_ok"] = (
            summary["prefix_cache"]["passed"]
            and summary["mtp"]["passed"]
            and summary["cuda_graph"]["passed"]
        )
        phase_summaries.append(summary)
        all_results.extend(phase_results)
        await writer.emit("production_phase_summary", phase=phase, **summary)

    try:
        for phase_index, concurrency in enumerate(args.concurrencies):
            await run_phase(concurrency, phase_index)
    finally:
        stop.set()
        await telemetry_task
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
    snapshots.append((run_ended, final_metrics, final_process))
    metrics_delta = selected_metric_deltas(initial_metrics, final_metrics)
    queue_latency = queue_histogram_summaries(initial_metrics, final_metrics)
    mtp = speculative_evidence(
        initial_metrics,
        final_metrics,
        "dflash" in args.expected_graph_components,
    )
    graph = cuda_graph_evidence(
        initial_metrics,
        final_metrics,
        initial_metrics,
        args.expected_graph_components,
        args.min_cuda_graph_replay_ratio,
    )
    all_probes = [
        result for result in all_results if result.tags.get("role") == "c1_probe"
    ]
    probe_transcripts = {
        result.output_transcript for result in all_probes if result.ok
    }
    probes_exact_across_load = (
        len(all_probes) >= 2
        and len(probe_transcripts) == 1
        and all(result.ok for result in all_probes)
    )
    passed = (
        prewarm_passed
        and all(
            summary["errors"] == 0
            and summary["degradation_ok"]
            and summary["quality_ok"]
            and summary["probe_fixed_seed_exact"]
            and summary["phase_metrics_ok"]
            for summary in phase_summaries
        )
        and bool(initial_metrics)
        and bool(final_metrics)
        and mtp["passed"]
        and graph["passed"]
        and probes_exact_across_load
        and cleanup_ok
    )
    return {
        "mode": "production",
        "passed": passed,
        "elapsed_seconds": run_ended - run_started,
        "prewarm": prewarm,
        "phase_summaries": phase_summaries,
        "metrics_delta": metrics_delta,
        "queue_latency_histograms": queue_latency,
        "mtp": mtp,
        "mtp_acceptance_rate": mtp["acceptance_rate"],
        "mtp_accepted_length": mtp["mean_advance_length_with_bonus"],
        "telemetry_samples": len(snapshots),
        "telemetry": summarize_telemetry(snapshots),
        "cuda_graph": graph,
        "cuda_graph_events": graph["events"],
        "cuda_graph_ok": graph["passed"],
        "c1_probe_samples": len(all_probes),
        "c1_probes_exact_across_load": probes_exact_across_load,
        "cleanup_ok": cleanup_ok,
        "cleanup_detail": cleanup_detail,
    }


def compare_time_windows(
    results: Sequence[RequestResult],
    started: float,
    ended: float,
    window_seconds: float,
    min_output_event_coverage: float,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    first = [
        result for result in results if started <= result.ended <= started + window_seconds
    ]
    last = [result for result in results if ended - window_seconds <= result.ended <= ended]

    def window_summary(items: Sequence[RequestResult], start: float, end: float) -> dict[str, Any]:
        return summarize_batch(items, max(0.0, end - start), 0, "window")

    first_summary = window_summary(first, started, min(ended, started + window_seconds))
    last_summary = window_summary(last, max(started, ended - window_seconds), ended)
    window_counts_available = all(
        len(result.output_event_window_counts) >= 2 for result in successful
    )
    if window_counts_available:
        first_event_count = sum(
            result.output_event_window_counts[0] for result in successful
        )
        last_event_count = sum(
            result.output_event_window_counts[1] for result in successful
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
    timed_events = sum(result.output_chunks for result in successful)
    completion_tokens = sum(result.completion_tokens for result in successful)
    event_coverage = ratio(timed_events, completion_tokens)
    event_coverage_ok = (
        event_coverage is not None
        and min_output_event_coverage
        <= event_coverage
        <= 2.0 - min_output_event_coverage
    )
    first_tps = first_event_count / window_seconds
    last_tps = last_event_count / window_seconds
    return {
        "window_seconds": window_seconds,
        "first": first_summary,
        "last": last_summary,
        "first_timed_output_events": first_event_count,
        "last_timed_output_events": last_event_count,
        "window_counts_available": window_counts_available,
        "first_output_tok_s_stream_timeline": first_tps,
        "last_output_tok_s_stream_timeline": last_tps,
        "timed_output_events": timed_events,
        "reported_completion_tokens": completion_tokens,
        "output_event_coverage": event_coverage,
        "min_output_event_coverage": min_output_event_coverage,
        "output_event_coverage_ok": event_coverage_ok,
        "throughput_ratio_last_over_first": ratio(last_tps, first_tps),
        "throughput_degradation_fraction": (
            1.0 - last_tps / first_tps if first_tps else None
        ),
    }


def image_content(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{data}"


def multimodal_messages(image: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def multimodal_mode(
    args: argparse.Namespace,
    client: SoakClient,
    writer: JsonlWriter,
) -> dict[str, Any]:
    image = image_content(args.image)
    before = await safe_metrics(client, writer, "multimodal-start")
    text_specs = [
        RequestSpec(
            case_id=f"multimodal-text-baseline-{index}",
            seed=args.seed + index,
            max_tokens=args.max_tokens,
            prompt=CANARY_PROMPTS[index % len(CANARY_PROMPTS)],
            tags={"scenario": "multimodal", "role": "text_baseline"},
        )
        for index in range(args.concurrency)
    ]
    _, baseline_summary = await run_batch(
        client,
        text_specs,
        args.concurrency,
        writer,
        "multimodal-text-baseline",
        keep_output=False,
    )
    image_seed = args.seed + 100_000
    image_prompt = args.image_prompt
    cold_spec = RequestSpec(
        case_id="image-cold",
        seed=image_seed,
        max_tokens=args.max_tokens,
        messages=multimodal_messages(image, image_prompt),
        tags={"scenario": "multimodal", "role": "image", "stage": "cold"},
    )
    cold, cold_summary = await run_batch(
        client, [cold_spec], 1, writer, "multimodal-image-cold", keep_output=True
    )
    variant_spec = RequestSpec(
        case_id="image-cache-variant",
        seed=image_seed + 1,
        max_tokens=args.max_tokens,
        messages=multimodal_messages(image, args.image_variant_prompt),
        tags={"scenario": "multimodal", "role": "image", "stage": "variant"},
    )
    _, variant_summary = await run_batch(
        client, [variant_spec], 1, writer, "multimodal-image-variant", keep_output=True
    )
    repeat_spec = RequestSpec(
        case_id="image-repeat",
        seed=image_seed,
        max_tokens=args.max_tokens,
        messages=multimodal_messages(image, image_prompt),
        tags={"scenario": "multimodal", "role": "image", "stage": "repeat"},
    )
    repeat, repeat_summary = await run_batch(
        client, [repeat_spec], 1, writer, "multimodal-image-repeat", keep_output=True
    )
    mixed_summaries = []
    for round_index in range(args.mixed_rounds):
        specs = [
            RequestSpec(
                case_id=f"mixed-image-r{round_index}",
                seed=args.seed + 200_000 + round_index,
                max_tokens=args.max_tokens,
                messages=multimodal_messages(image, args.image_variant_prompt),
                tags={"scenario": "multimodal", "role": "image", "round": round_index},
            )
        ] + [
            RequestSpec(
                case_id=f"mixed-text-r{round_index}-{index}",
                seed=args.seed + 210_000 + round_index * 100 + index,
                max_tokens=args.max_tokens,
                prompt=CANARY_PROMPTS[index % len(CANARY_PROMPTS)],
                tags={"scenario": "multimodal", "role": "text", "round": round_index},
            )
            for index in range(max(1, args.concurrency - 1))
        ]
        _, summary = await run_batch(
            client,
            specs,
            args.concurrency,
            writer,
            f"multimodal-mixed-r{round_index}",
            keep_output=False,
        )
        mixed_summaries.append(summary)
    after = await safe_metrics(client, writer, "multimodal-end")
    metrics_delta = selected_metric_deltas(before, after)
    outputs_equal = cold[0].output_text == repeat[0].output_text
    all_summaries = [baseline_summary, cold_summary, variant_summary, repeat_summary, *mixed_summaries]
    encoder_hits = metrics_delta.get("mistralrs_encoder_cache_hits_total")
    passed = all(summary["errors"] == 0 for summary in all_summaries) and outputs_equal
    if encoder_hits is not None:
        passed = passed and encoder_hits >= 1
    return {
        "mode": "multimodal",
        "passed": passed,
        "fixed_seed_repeat_equal": outputs_equal,
        "baseline": baseline_summary,
        "cold": cold_summary,
        "variant": variant_summary,
        "repeat": repeat_summary,
        "mixed": mixed_summaries,
        "metrics_delta": metrics_delta,
        "encoder_cache_observed": encoder_hits is not None,
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
    parser.add_argument("--output", type=Path, help="JSONL evidence path")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production soak testing for mistral.rs and other OpenAI-compatible servers"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = common_parser()

    canary = subparsers.add_parser("canary", parents=[common])
    canary.add_argument("--concurrencies", type=parse_int_list, default=DEFAULT_CONCURRENCIES)
    canary.add_argument("--requests", type=int, default=16)
    canary.add_argument("--max-tokens", type=int, default=128)
    canary.add_argument("--max-length-tokens", type=int, default=17)
    canary.add_argument("--skip-edge-cases", action="store_true")
    canary.add_argument("--eos-token-id", type=int)
    canary.add_argument("--reference-url", help="target-only server for statistical comparison")
    canary.add_argument("--stat-max-ks", type=float, default=0.35)
    canary.add_argument("--stat-max-js", type=float, default=0.20)
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

    adversarial = subparsers.add_parser("adversarial", parents=[common])
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
        "--max-repeated-ngram-ratio",
        type=float,
        default=DEFAULT_MAX_REPEATED_NGRAM_RATIO,
    )
    adversarial.add_argument("--max-seqs", type=int, default=16)
    adversarial.add_argument("--max-tokens", type=int, default=128)
    adversarial.add_argument("--mixed-requests", type=int, default=16)
    adversarial.add_argument("--overlap-short-requests", type=int, default=8)
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
        "--fairness-stagger-seconds", type=float, default=DEFAULT_FAIRNESS_STAGGER_SECONDS
    )
    adversarial.add_argument("--burst-requests", type=int, default=32)
    adversarial.add_argument("--burst-max-tokens", type=int, default=64)
    adversarial.add_argument("--prefix-context-tokens", type=int, default=8_192)
    adversarial.add_argument(
        "--min-prefix-reuse-fraction",
        type=float,
        default=DEFAULT_MIN_PREFIX_REUSE_FRACTION,
    )
    adversarial.add_argument("--prefix-pressure-entries", type=int, default=24)
    adversarial.add_argument(
        "--prefix-pressure-context-tokens", type=int, default=100_000
    )
    adversarial.add_argument(
        "--prefix-pressure-capacity-fraction",
        type=float,
        default=DEFAULT_PREFIX_PRESSURE_CAPACITY_FRACTION,
    )
    adversarial.add_argument(
        "--prefix-pressure-max-entries",
        type=int,
        default=DEFAULT_PREFIX_PRESSURE_MAX_ENTRIES,
    )
    adversarial.add_argument(
        "--kv-block-size-tokens", type=int, default=DEFAULT_KV_BLOCK_SIZE_TOKENS
    )
    adversarial.add_argument("--prefix-pressure-max-tokens", type=int, default=8)
    adversarial.add_argument(
        "--expected-graph-components",
        type=parse_graph_components,
        default=("target",),
    )
    adversarial.add_argument(
        "--min-cuda-graph-replay-ratio",
        type=float,
        default=DEFAULT_MIN_CUDA_GRAPH_REPLAY_RATIO,
    )
    adversarial.add_argument("--churn-rounds", type=int, default=3)
    adversarial.add_argument("--churn-max-tokens", type=int, default=64)

    production = subparsers.add_parser("production", parents=[common])
    production.add_argument("--duration-seconds", type=float, default=3_600.0)
    production.add_argument("--concurrencies", type=parse_int_list, default=(8, 16))
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
        "--fixed-output-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    production.add_argument("--probe-max-tokens", type=int, default=128)
    production.add_argument(
        "--probe-interval-seconds", type=float, default=DEFAULT_PROBE_INTERVAL_SECONDS
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
        "--max-throughput-degradation-fraction",
        type=float,
        default=DEFAULT_MAX_THROUGHPUT_DEGRADATION_FRACTION,
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
    production.add_argument("--server-pid", type=int)

    multimodal = subparsers.add_parser("multimodal", parents=[common])
    multimodal.add_argument("--image", required=True, help="image URL, data URL, or local path")
    multimodal.add_argument("--image-prompt", default="Describe this image precisely.")
    multimodal.add_argument(
        "--image-variant-prompt", default="List the main visible objects and colors."
    )
    multimodal.add_argument("--concurrency", type=int, default=4)
    multimodal.add_argument("--mixed-rounds", type=int, default=4)
    multimodal.add_argument("--max-tokens", type=int, default=128)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate-phase", default="canary-c1-normal")
    compare.add_argument("--reference-phase", default="canary-c1-normal")
    compare.add_argument("--tokenizer", help="tokenizer.json path or Hugging Face tokenizer ID")
    compare.add_argument("--stat-max-ks", type=float, default=0.35)
    compare.add_argument("--stat-max-js", type=float, default=0.20)
    compare.add_argument("--output", type=Path, help="JSONL comparison evidence path")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "compare":
        if args.stat_max_ks <= 0 or args.stat_max_js <= 0:
            raise ValueError("statistical thresholds must be positive")
        return
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
    if not 0 <= args.min_p <= 1:
        raise ValueError("--min-p must be between 0 and 1")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be greater than 0 and at most 1")
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
    if args.mode in ("adversarial", "production", "resident-decode") and not args.tokenizer:
        raise ValueError(f"{args.mode} mode requires --tokenizer")
    if args.mode == "resident-decode":
        for name in (
            "requests",
            "warmup_max_tokens",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "stat_max_ks",
            "stat_max_js",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.requests < max(args.concurrencies):
            raise ValueError(
                "--requests must be at least the maximum resident-decode concurrency"
            )
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
    if args.mode == "adversarial":
        missing = {
            args.cancel_context_tokens,
            args.prefix_context_tokens,
        } - set(args.context_lengths)
        if missing:
            values = ", ".join(str(value) for value in sorted(missing))
            raise ValueError(f"context lengths must include cancellation/prefix lengths: {values}")
        if 3 not in args.long_correctness_concurrencies:
            raise ValueError("--long-correctness-concurrencies must include 3")
        if args.long_correctness_stat_max_ks <= 0 or args.long_correctness_stat_max_js <= 0:
            raise ValueError("long-context statistical thresholds must be positive")
        for name in (
            "max_seqs",
            "mixed_requests",
            "overlap_short_requests",
            "cancel_requests",
            "burst_requests",
            "prefix_pressure_entries",
            "prefix_pressure_context_tokens",
            "prefix_pressure_max_entries",
            "kv_block_size_tokens",
            "churn_rounds",
            "long_correctness_max_tokens",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.prefix_pressure_max_entries < args.prefix_pressure_entries:
            raise ValueError(
                "--prefix-pressure-max-entries must be at least --prefix-pressure-entries"
            )
        if args.prefix_pressure_capacity_fraction <= 1:
            raise ValueError("--prefix-pressure-capacity-fraction must be greater than 1")
        if not 0 < args.min_prefix_reuse_fraction <= 1:
            raise ValueError(
                "--min-prefix-reuse-fraction must be greater than 0 and at most 1"
            )
        if not 0 < args.min_cuda_graph_replay_ratio <= 1:
            raise ValueError(
                "--min-cuda-graph-replay-ratio must be greater than 0 and at most 1"
            )
        for name in (
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
            "timeout_test_seconds",
            "fairness_max_slowdown",
            "fairness_stagger_seconds",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "production":
        for name in (
            "duration_seconds",
            "probe_interval_seconds",
            "telemetry_interval_seconds",
            "comparison_window_seconds",
            "prompt_pool_size",
            "prewarm_max_tokens",
            "cleanup_timeout_seconds",
            "cleanup_poll_seconds",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if not 0 <= args.max_throughput_degradation_fraction <= 1:
            raise ValueError(
                "--max-throughput-degradation-fraction must be between 0 and 1"
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
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
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
    writer = JsonlWriter(jsonl_path)
    client = SoakClient(
        args.base_url,
        args.model,
        args.api_key,
        args.timeout,
        policy,
        tokenizer,
    )
    await writer.emit(
        "run_start",
        mode=args.mode,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        policy=asdict(policy),
    )
    started = time.perf_counter()
    summary: dict[str, Any]
    try:
        if args.mode == "canary":
            summary = await canary_mode(args, client, writer)
        elif args.mode == "sweep":
            summary = await sweep_mode(args, client, writer)
        elif args.mode == "resident-decode":
            summary = await resident_decode_mode(args, client, writer)
        elif args.mode == "adversarial":
            summary = await adversarial_mode(args, client, writer)
        elif args.mode == "production":
            summary = await production_mode(args, client, writer)
        elif args.mode == "multimodal":
            summary = await multimodal_mode(args, client, writer)
        else:
            raise RuntimeError(f"unsupported mode {args.mode}")
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
