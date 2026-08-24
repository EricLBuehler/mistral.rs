import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from scripts import production_soak as soak


def request_result(
    *,
    completion_tokens: int,
    output_chunks: int,
    finish_reason: str,
    ok: bool = True,
    output_text: str = "output",
    reasoning_text: str = "",
    case_id: str = "coverage",
    seed: int = 1,
    started: float = 0.0,
    ended: float = 1.0,
    ttft_seconds: float | None = 0.1,
    tpot_seconds: float | None = 0.01,
    client_queue_seconds: float = 0.0,
    prompt_tokens: int = 8,
    context_tokens: int = 8,
    output_transcript: str = "output",
    output_event_times: list[float] | None = None,
    output_event_token_counts: list[int] | None = None,
    streamed_output_tokens: int | None = None,
    output_event_window_counts: list[int] | None = None,
    output_token_window_counts: list[int] | None = None,
    tags: dict | None = None,
) -> soak.RequestResult:
    event_times = output_event_times or []
    return soak.RequestResult(
        case_id=case_id,
        seed=seed,
        ok=ok,
        status_code=200,
        started=started,
        ended=ended,
        ttft_seconds=ttft_seconds,
        tpot_seconds=tpot_seconds,
        client_queue_seconds=client_queue_seconds,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        finish_reason=finish_reason,
        output_text=output_text,
        reasoning_text=reasoning_text,
        tool_calls=[],
        output_transcript=output_transcript,
        output_chunks=output_chunks,
        stream_done=True,
        usage_received=True,
        request_id="request",
        error=None,
        error_kind=None,
        context_tokens=context_tokens,
        tags=tags or {},
        streamed_output_tokens=(
            output_chunks
            if streamed_output_tokens is None
            else streamed_output_tokens
        ),
        output_event_times=event_times,
        output_event_token_counts=(
            [1] * len(event_times)
            if output_event_token_counts is None
            else output_event_token_counts
        ),
        output_event_window_counts=output_event_window_counts or [],
        output_token_window_counts=output_token_window_counts or [],
    )


class TokenizerExactTextTests(unittest.TestCase):
    class BoundaryExpansionTokenizer(soak.TokenizerAdapter):
        body_marker = "\x1e"

        def __init__(self, expansion: int = 5_000, body_token_width: int = 1) -> None:
            self.expansion = expansion
            self.body_token_width = body_token_width
            self.encode_calls = 0
            self.source_builds = 0
            self._exact_text_source_cache = {}

        def encode(self, text: str) -> list[int]:
            self.encode_calls += 1
            if text.startswith(soak.CONTEXT_PARAGRAPH) and len(text) > len(
                soak.CONTEXT_PARAGRAPH
            ):
                self.source_builds += 1
            body_tokens = text.count(self.body_marker)
            ordinary_tokens = len(text) - body_tokens
            expansion = self.expansion if body_tokens else 0
            return [0] * (
                ordinary_tokens + body_tokens * self.body_token_width + expansion
            )

        def decode(self, token_ids) -> str:
            return self.body_marker * len(token_ids)

    class ContractingSourceTokenizer(BoundaryExpansionTokenizer):
        def encode(self, text: str) -> list[int]:
            if text.startswith(soak.CONTEXT_PARAGRAPH) and len(text) > len(
                soak.CONTEXT_PARAGRAPH
            ):
                self.encode_calls += 1
                self.source_builds += 1
                return [0] * (len(text) // 2)
            return super().encode(text)

    def test_exact_text_corrects_large_round_trip_boundary_drift(self) -> None:
        tokenizer = self.BoundaryExpansionTokenizer()
        label = "large-boundary-drift"
        prefix = f"Production soak case {label}.\n"
        fixed_tokens = tokenizer.count(prefix + soak.EXACT_CONTEXT_SUFFIX)
        target_tokens = fixed_tokens + 6_000

        text = tokenizer.exact_text(target_tokens, label)

        self.assertEqual(tokenizer.count(text), target_tokens)
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(text.endswith(soak.EXACT_CONTEXT_SUFFIX))
        self.assertLessEqual(tokenizer.encode_calls, 7)

    def test_exact_text_uses_bounded_padding_for_token_count_gaps(self) -> None:
        tokenizer = self.BoundaryExpansionTokenizer(
            expansion=0,
            body_token_width=2,
        )
        label = "two-token-body"
        prefix = f"Production soak case {label}.\n"
        fixed_tokens = tokenizer.count(prefix + soak.EXACT_CONTEXT_SUFFIX)
        target_tokens = fixed_tokens + 201

        text = tokenizer.exact_text(target_tokens, label)

        self.assertEqual(tokenizer.count(text), target_tokens)
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(text.endswith(soak.EXACT_CONTEXT_SUFFIX))
        self.assertLess(tokenizer.encode_calls, 80)

    def test_exact_text_reuses_encoded_context_source(self) -> None:
        tokenizer = self.BoundaryExpansionTokenizer()

        first = tokenizer.exact_text(10_000, "source-cache-a")
        second = tokenizer.exact_text(10_000, "source-cache-b")

        self.assertEqual(tokenizer.count(first), 10_000)
        self.assertEqual(tokenizer.count(second), 10_000)
        self.assertEqual(tokenizer.source_builds, 1)

    def test_exact_text_grows_source_after_cross_boundary_contraction(self) -> None:
        tokenizer = self.ContractingSourceTokenizer(expansion=0)
        label = "contracting-source"

        text = tokenizer.exact_text(10_000, label)

        self.assertEqual(tokenizer.count(text), 10_000)
        self.assertGreaterEqual(tokenizer.source_builds, 2)
        self.assertLessEqual(tokenizer.source_builds, 3)


class SampledOutputQualityTests(unittest.TestCase):
    def test_coherent_restatement_does_not_trigger_loop_gate(self) -> None:
        reasoning = """We need answer user's request. Need parse user: long repeated deterministic production-soak context, ends with "This is deterministic production-so..." then "End of deterministic production-soak context.
Respond with varied original prose without quoting or repeating the context."

User wants: "Production soak case resident-decode-60000.
This is deterministic production-soak context... repeated. ... End of deterministic production-soak context.
Respond with varied original prose without quoting or repeating the context."

We must respond with varied original prose without quoting/repeating context. Need likely produce original prose, not repeat. Need maybe acknowledge"""
        valid, evidence = soak.validate_sampled_output(
            request_result(
                completion_tokens=128,
                output_chunks=128,
                finish_reason="length",
                output_text="",
                reasoning_text=reasoning,
                output_transcript=reasoning,
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
        )

        self.assertGreater(evidence["repeated_ngram_ratio"], 0.20)
        self.assertLess(evidence["excess_repeated_ngram_ratio"], 0.20)
        self.assertTrue(evidence["repetition_valid"])
        self.assertTrue(evidence["reasoning_nonempty"])
        self.assertFalse(evidence["content_nonempty"])
        self.assertTrue(valid)

    def test_severe_repeated_motif_is_rejected(self) -> None:
        output = "alpha beta gamma delta " * 12
        valid, evidence = soak.validate_sampled_output(
            request_result(
                completion_tokens=48,
                output_chunks=48,
                finish_reason="length",
                output_text=output,
                output_transcript=output,
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
        )

        self.assertFalse(valid)
        self.assertTrue(evidence["degeneration_detected"])
        self.assertTrue(
            evidence["channel_repetition"]["content"]["periodic_loop"]["detected"]
        )

    def test_tail_loop_after_coherent_prose_is_rejected(self) -> None:
        output = (
            "A healthy service keeps request state isolated while load changes. "
            "Cancellation and retry paths must release their resources. "
            + "late loop motif here " * 10
        )
        valid, evidence = soak.validate_sampled_output(
            request_result(
                completion_tokens=64,
                output_chunks=64,
                finish_reason="length",
                output_text=output,
                output_transcript=output,
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
        )

        self.assertFalse(valid)
        self.assertTrue(
            evidence["channel_repetition"]["content"]["tail_periodic_loop"][
                "detected"
            ]
        )

    def test_one_coherent_repetition_is_allowed(self) -> None:
        sentence = "The scheduler remains stable under changing load. "
        output = sentence * 2
        valid, evidence = soak.validate_sampled_output(
            request_result(
                completion_tokens=16,
                output_chunks=16,
                finish_reason="length",
                output_text=output,
                output_transcript=output,
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
        )

        self.assertTrue(valid)
        self.assertTrue(evidence["content_nonempty"])
        self.assertFalse(evidence["reasoning_nonempty"])
        self.assertFalse(evidence["degeneration_detected"])

    def test_short_output_has_stable_degeneration_evidence(self) -> None:
        evidence = soak.repetition_evidence(["brief", "answer"], 0.20)

        self.assertTrue(evidence["valid"])
        self.assertEqual(evidence["repeated_ngram_ratio"], 0.0)
        self.assertEqual(evidence["excess_repeated_ngram_ratio"], 0.0)
        self.assertEqual(evidence["periodic_loop"]["span_tokens"], 0)


class OutputEventCoverageTests(unittest.TestCase):
    def test_empty_stream_delta_does_not_invent_a_token(self) -> None:
        self.assertEqual(
            soak.stream_delta_token_count("", None, None, None),
            0,
        )
        self.assertEqual(
            soak.stream_delta_token_count(
                None,
                None,
                [{"function": {"arguments": "{}"}}],
                None,
            ),
            1,
        )

    def test_coalesced_chunks_use_streamed_token_weight(self) -> None:
        evidence = soak.output_event_coverage_evidence(
            [
                request_result(
                    completion_tokens=41,
                    output_chunks=2,
                    streamed_output_tokens=40,
                    finish_reason="stop",
                )
            ],
            0.98,
        )

        self.assertEqual(evidence["observed_output_events"], 2)
        self.assertEqual(evidence["observed_output_tokens"], 40)
        self.assertEqual(evidence["output_token_coverage"], 1.0)
        self.assertTrue(evidence["output_event_coverage_ok"])

    def test_stop_excludes_one_unstreamed_terminal_token(self) -> None:
        evidence = soak.output_event_coverage_evidence(
            [
                request_result(
                    completion_tokens=41,
                    output_chunks=40,
                    finish_reason="stop",
                )
            ],
            0.98,
        )

        self.assertEqual(evidence["reported_completion_tokens"], 41)
        self.assertEqual(evidence["unstreamed_terminal_tokens"], 1)
        self.assertEqual(evidence["expected_streamable_tokens"], 40)
        self.assertEqual(evidence["output_event_coverage"], 1.0)
        self.assertTrue(evidence["output_event_coverage_ok"])

    def test_length_keeps_full_reported_token_denominator(self) -> None:
        evidence = soak.output_event_coverage_evidence(
            [
                request_result(
                    completion_tokens=41,
                    output_chunks=40,
                    finish_reason="length",
                )
            ],
            0.98,
        )

        self.assertEqual(evidence["unstreamed_terminal_tokens"], 0)
        self.assertEqual(evidence["expected_streamable_tokens"], 41)
        self.assertAlmostEqual(evidence["output_event_coverage"], 40 / 41)
        self.assertFalse(evidence["output_event_coverage_ok"])

    def test_production_windows_use_streamable_denominator(self) -> None:
        evidence = soak.compare_time_windows(
            [
                request_result(
                    completion_tokens=41,
                    output_chunks=40,
                    finish_reason="stop",
                )
            ],
            started=0.0,
            ended=2.0,
            window_seconds=1.0,
            min_output_event_coverage=0.98,
        )

        self.assertEqual(evidence["reported_completion_tokens"], 41)
        self.assertEqual(evidence["expected_streamable_tokens"], 40)
        self.assertTrue(evidence["output_event_coverage_ok"])

    def test_timeline_rates_weight_coalesced_chunks_by_tokens(self) -> None:
        evidence = soak.compare_time_windows(
            [
                request_result(
                    completion_tokens=31,
                    output_chunks=2,
                    streamed_output_tokens=30,
                    finish_reason="stop",
                    started=0.0,
                    ended=2.0,
                    output_event_times=[0.5, 1.5],
                    output_event_token_counts=[10, 20],
                )
            ],
            started=0.0,
            ended=2.0,
            window_seconds=1.0,
            min_output_event_coverage=0.98,
        )

        self.assertEqual(evidence["first_timed_output_events"], 1)
        self.assertEqual(evidence["last_timed_output_events"], 1)
        self.assertEqual(evidence["first_timed_output_tokens"], 10)
        self.assertEqual(evidence["last_timed_output_tokens"], 20)
        self.assertEqual(evidence["first_output_tok_s_stream_timeline"], 10.0)
        self.assertEqual(evidence["last_output_tok_s_stream_timeline"], 20.0)

    def test_aggregate_coverage_cannot_hide_a_bad_request(self) -> None:
        evidence = soak.output_event_coverage_evidence(
            [
                request_result(
                    case_id="missing-events",
                    completion_tokens=10,
                    output_chunks=5,
                    finish_reason="length",
                ),
                request_result(
                    case_id="extra-events",
                    completion_tokens=10,
                    output_chunks=15,
                    finish_reason="length",
                ),
            ],
            0.98,
        )

        self.assertEqual(evidence["output_event_coverage"], 1.0)
        self.assertTrue(evidence["aggregate_output_event_coverage_ok"])
        self.assertFalse(evidence["output_event_coverage_ok"])
        self.assertEqual(
            [item["passed"] for item in evidence["per_request_output_event_coverage"]],
            [False, False],
        )

    def test_latency_windows_use_first_token_time(self) -> None:
        evidence = soak.compare_time_windows(
            [
                request_result(
                    case_id="first",
                    completion_tokens=10,
                    output_chunks=10,
                    finish_reason="length",
                    started=0.0,
                    ended=9.0,
                    ttft_seconds=0.2,
                ),
                request_result(
                    case_id="last",
                    completion_tokens=10,
                    output_chunks=10,
                    finish_reason="length",
                    started=8.0,
                    ended=8.5,
                    ttft_seconds=0.2,
                ),
            ],
            started=0.0,
            ended=10.0,
            window_seconds=2.0,
            min_output_event_coverage=0.98,
        )

        self.assertEqual(evidence["latency_window_attribution"], "first_token_timestamp")
        self.assertEqual(evidence["first"]["requests"], 1)
        self.assertEqual(evidence["last"]["requests"], 1)

    def test_production_windows_gate_sample_count_and_p95_p99_degradation(self) -> None:
        results = []
        for window, offset in (("first", 0.0), ("last", 10.0)):
            for index in range(32):
                started = offset + 0.1 + index * 0.1
                slowdown = 1.0 if window == "first" else 1.1
                results.append(
                    request_result(
                        case_id=f"{window}-{index}",
                        completion_tokens=2,
                        output_chunks=1,
                        streamed_output_tokens=2,
                        finish_reason="length",
                        started=started,
                        ended=started + 0.2,
                        ttft_seconds=0.1 * slowdown,
                        tpot_seconds=0.01 * slowdown,
                        client_queue_seconds=0.01 * slowdown,
                        output_event_times=[started + 0.1 * slowdown],
                        output_event_token_counts=[2],
                    )
                )

        evidence = soak.compare_time_windows(
            results,
            started=0.0,
            ended=20.0,
            window_seconds=10.0,
            min_output_event_coverage=0.98,
            minimum_samples=32,
        )

        self.assertTrue(evidence["window_sample_evidence"]["passed"])
        self.assertEqual(
            set(evidence["latency_degradation_fractions"]),
            {
                "ttft_seconds_p95",
                "ttft_seconds_p99",
                "tpot_seconds_p95",
                "tpot_seconds_p99",
                "client_queue_seconds_p95",
                "client_queue_seconds_p99",
            },
        )
        self.assertTrue(
            all(
                value is not None and value <= 0.20
                for value in evidence["latency_degradation_fractions"].values()
            )
        )

        evidence = soak.compare_time_windows(
            results[:-1],
            started=0.0,
            ended=20.0,
            window_seconds=10.0,
            min_output_event_coverage=0.98,
            minimum_samples=32,
        )
        self.assertFalse(evidence["window_sample_evidence"]["passed"])


class StreamingTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_timeout_closes_streaming_response(self) -> None:
        class TimeoutStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                raise httpx.ReadTimeout("timed out while reading stream")
                yield b""

            async def aclose(self) -> None:
                return None

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(json.loads(request.content)["stream"])
            return httpx.Response(
                200,
                headers={"x-request-id": "timeout-request"},
                stream=TimeoutStream(),
            )

        client = soak.SoakClient(
            "http://soak.test",
            "model",
            "token",
            10.0,
            soak.SamplingPolicy(),
            None,
        )
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client.stream_request(
                soak.RequestSpec(
                    case_id="timeout",
                    seed=1,
                    max_tokens=64,
                    prompt="prompt",
                ),
                timeout_seconds=0.05,
            )
        finally:
            await client.close()

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.error_kind, "ReadTimeout")
        self.assertEqual(result.request_id, "timeout-request")


class OverlapIntervalTests(unittest.TestCase):
    def test_complete_window_includes_inflight_request_tokens(self) -> None:
        results = [
            request_result(
                case_id="completed-before-prefill",
                completion_tokens=2,
                output_chunks=1,
                streamed_output_tokens=4,
                finish_reason="stop",
                started=0.0,
                ended=1.8,
                output_event_times=[1.5],
                output_event_token_counts=[4],
            ),
            request_result(
                case_id="completed-after-prefill",
                completion_tokens=3,
                output_chunks=2,
                streamed_output_tokens=7,
                finish_reason="stop",
                started=1.0,
                ended=2.8,
                output_event_times=[1.7, 2.4],
                output_event_token_counts=[3, 4],
            ),
        ]

        baseline, overlapped = soak.overlap_window_evidence(
            results,
            baseline_started=1.0,
            prefill_started=2.0,
            prefill_first_token=2.6,
        )

        self.assertEqual(baseline["completed_requests"], 1)
        self.assertEqual(baseline["output_events"], 2)
        self.assertEqual(baseline["output_tokens"], 7)
        self.assertIsNotNone(overlapped)
        self.assertEqual(overlapped["output_events"], 1)
        self.assertEqual(overlapped["output_tokens"], 4)

    def test_baseline_interval_excludes_worker_ramp(self) -> None:
        results = [
            request_result(
                case_id="ramp",
                completion_tokens=2,
                output_chunks=2,
                finish_reason="length",
                started=0.0,
                ended=1.5,
                output_event_times=[0.5, 1.0],
            ),
            request_result(
                case_id="baseline",
                completion_tokens=3,
                output_chunks=3,
                finish_reason="length",
                started=1.5,
                ended=3.0,
                output_event_times=[2.1, 2.5, 2.9],
            ),
        ]

        evidence = soak.output_interval_evidence(results, 2.0, 3.1)

        self.assertEqual(evidence["completed_requests"], 1)
        self.assertEqual(evidence["output_events"], 3)
        self.assertEqual(evidence["output_tokens"], 3)
        self.assertAlmostEqual(evidence["seconds"], 1.1)


class PrefixReuseTests(unittest.TestCase):
    def test_replay_eligibility_is_block_aligned(self) -> None:
        self.assertEqual(soak.cacheable_prefix_tokens(8_192, 32), 8_160)
        self.assertEqual(soak.eligible_prefix_reuse_tokens(8_192, 32, 2_048), 6_112)
        self.assertEqual(soak.eligible_prefix_reuse_tokens(2_048, 32, 2_048), 0)
        self.assertEqual(soak.eligible_prefix_reuse_tokens(1_024, 32, 2_048), 0)

    def test_prefix_gate_uses_only_replay_eligible_tokens_for_reuse(self) -> None:
        before = {
            "mistralrs_prefix_cache_lookups_total": 0.0,
            "mistralrs_prefix_cache_hits_total": 0.0,
            "mistralrs_prefix_cache_tokens_matched_total": 0.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }
        after = {
            "mistralrs_prefix_cache_lookups_total": 2.0,
            "mistralrs_prefix_cache_hits_total": 1.0,
            "mistralrs_prefix_cache_tokens_matched_total": 10_176.0,
            "mistralrs_prefix_cache_tokens_reused_total": 6_112.0,
        }

        evidence = soak.prefix_cache_evidence(
            before,
            after,
            [8_192, 2_048],
            0.98,
            32,
            2_048,
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["expected_cacheable_tokens"], 10_176)
        self.assertEqual(evidence["expected_eligible_reuse_tokens"], 6_112)
        self.assertEqual(evidence["expected_reusable_requests"], 1)

    def test_prefix_gate_rejects_vacuous_reuse_contract(self) -> None:
        before = {
            "mistralrs_prefix_cache_lookups_total": 0.0,
            "mistralrs_prefix_cache_hits_total": 0.0,
            "mistralrs_prefix_cache_tokens_matched_total": 0.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }
        after = {
            "mistralrs_prefix_cache_lookups_total": 2.0,
            "mistralrs_prefix_cache_hits_total": 0.0,
            "mistralrs_prefix_cache_tokens_matched_total": 3_008.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }

        evidence = soak.prefix_cache_evidence(
            before,
            after,
            [1_024, 2_048],
            0.98,
            32,
            4_096,
        )

        self.assertFalse(evidence["reuse_contract_non_vacuous"])
        self.assertFalse(evidence["passed"])

    def test_prefix_gate_rejects_missing_reuse(self) -> None:
        before = {
            "mistralrs_prefix_cache_lookups_total": 0.0,
            "mistralrs_prefix_cache_hits_total": 0.0,
            "mistralrs_prefix_cache_tokens_matched_total": 0.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }
        after = {
            "mistralrs_prefix_cache_lookups_total": 1.0,
            "mistralrs_prefix_cache_hits_total": 0.0,
            "mistralrs_prefix_cache_tokens_matched_total": 8_160.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }

        evidence = soak.prefix_cache_evidence(
            before,
            after,
            [8_192],
            0.98,
            32,
            2_048,
        )

        self.assertTrue(evidence["reuse_contract_non_vacuous"])
        self.assertFalse(evidence["passed"])


class PrefixPressurePlanTests(unittest.TestCase):
    @staticmethod
    def metrics(
        total_blocks: float,
        active_blocks: float = 0.0,
        prefix_cached_blocks: float = 0.0,
    ) -> dict[str, float]:
        return {
            "mistralrs_kv_cache_blocks_total": total_blocks,
            soak.KV_CACHE_ACTIVE_GAUGE: active_blocks,
            soak.KV_CACHE_PREFIX_CACHED_GAUGE: prefix_cached_blocks,
        }

    @staticmethod
    def config(**overrides: int | float) -> soak.PrefixPressureConfig:
        values = {
            "entries": 24,
            "max_sequences": 16,
            "context_tokens": 100_000,
            "max_completion_tokens": 8,
            "block_size_tokens": 32,
            "kv_headroom_fraction": 0.10,
            **overrides,
        }
        return soak.PrefixPressureConfig(**values)

    def test_active_concurrency_fits_observed_kv_capacity_with_headroom(self) -> None:
        plan = soak.prefix_pressure_plan(
            self.metrics(27_831, prefix_cached_blocks=256),
            self.config(),
        )

        self.assertEqual(plan["blocks_per_request"], 3_126)
        self.assertEqual(plan["headroom_blocks"], 2_784)
        self.assertEqual(plan["capacity_concurrency"], 8)
        self.assertEqual(plan["concurrency"], 8)
        self.assertLessEqual(
            plan["active_working_set_blocks"],
            plan["active_budget_blocks"],
        )

    def test_observed_active_blocks_reduce_pressure_concurrency(self) -> None:
        plan = soak.prefix_pressure_plan(
            self.metrics(31_045, active_blocks=4_000),
            self.config(),
        )

        self.assertEqual(plan["capacity_concurrency"], 7)
        self.assertEqual(plan["concurrency"], 7)

    def test_cached_prefixes_remain_reclaimable_for_eviction_pressure(self) -> None:
        plan = soak.prefix_pressure_plan(
            self.metrics(31_045, prefix_cached_blocks=27_000),
            self.config(),
        )

        self.assertEqual(plan["concurrency"], 8)
        self.assertEqual(plan["entries"], 24)
        self.assertEqual(plan["prefix_cached_blocks_observed_reclaimable"], 27_000)

    def test_configured_limits_cap_capacity_concurrency(self) -> None:
        plan = soak.prefix_pressure_plan(
            self.metrics(100_000),
            self.config(entries=5, max_sequences=8),
        )

        self.assertGreater(plan["capacity_concurrency"], 8)
        self.assertEqual(plan["concurrency"], 5)

    def test_rejects_capacity_that_cannot_fit_one_request(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot fit one active request"):
            soak.prefix_pressure_plan(self.metrics(3_000), self.config())


class QualityReplayTests(unittest.TestCase):
    @staticmethod
    def pressure_cases() -> list[soak.QualityReplayCase]:
        return [
            soak.QualityReplayCase(
                case_id=str(index),
                seed=index,
                concurrency=16,
                worker_index=index,
                request_index=0,
                context_tokens=context,
                prompt_index=index,
                prompt_label=f"prompt-{index}",
                source_output_transcript_sha256="hash",
                source_quality_failure=False,
            )
            for index, context in enumerate((1_024, 8_192, 8_192, 32_768))
        ]

    @staticmethod
    def pressure_specs(
        cases: list[soak.QualityReplayCase],
    ) -> list[soak.RequestSpec]:
        return [
            soak.RequestSpec(
                case_id=f"selected-{index}",
                seed=index,
                max_tokens=256,
                context_tokens=cases[index % len(cases)].context_tokens,
            )
            for index in range(16)
        ]

    @staticmethod
    def pressure_config(
        context_tokens: int = 8_192,
    ) -> soak.QualityReplayPressureConfig:
        return soak.QualityReplayPressureConfig(
            waves=8,
            entries=16,
            context_tokens=context_tokens,
            max_tokens=8,
            block_size_tokens=32,
            headroom_fraction=0.10,
        )

    @staticmethod
    def pressure_metrics(
        owner_capacity: int = 20,
        owner_used: int = 0,
        retained_blocks: int = 0,
    ) -> dict[str, float]:
        return {
            "mistralrs_kv_cache_blocks_total": 27_080,
            soak.PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_GAUGE: owner_capacity,
            soak.PAGED_RECURRENT_PREFIX_OWNERS_USED_GAUGE: owner_used,
            soak.PAGED_PREFIX_RETAINED_BLOCKS_GAUGE: retained_blocks,
            soak.KV_CACHE_ACTIVE_GAUGE: 0,
        }

    def test_source_cases_reconstruct_production_rng_identity(self) -> None:
        source_seed = 20260841
        concurrency = 16
        worker = 13
        request_index = 2
        phase_index = 1
        context_mix = [[1_024, 45], [8_192, 30], [32_768, 20], [100_000, 5]]
        pool_counts = {"1024": 7, "8192": 5, "32768": 3, "100000": 1}
        rng = soak.random.Random(source_seed + phase_index * 100_000 + worker)
        context_tokens = 0
        prompt_index = 0
        for _ in range(request_index + 1):
            context_tokens = soak.choose_context(rng, context_mix)
            prompt_index = rng.randrange(pool_counts[str(context_tokens)])
        case_id = f"prod-c{concurrency}-w{worker}-r{request_index}"
        request_seed = (
            source_seed + phase_index * 1_000_000 + worker * 10_000 + request_index
        )
        records = [
            {
                "event": "run_start",
                "mode": "production",
                "arguments": {
                    "seed": source_seed,
                    "context_mix": context_mix,
                    "concurrencies": [8, 16],
                    "fixed_output_length": False,
                },
            },
            {
                "event": "prompt_calibration",
                "profile": soak.CONTEXT_PROMPT_PROFILE,
                "overhead_tokens": 17,
            },
            {
                "event": "production_prewarm_summary",
                "pool_counts": pool_counts,
            },
            {
                "event": "request",
                "case_id": case_id,
                "seed": request_seed,
                "context_tokens": context_tokens,
                "output_transcript_sha256": "source-hash",
                "tags": {"role": "traffic"},
            },
            {
                "event": "production_phase_summary",
                "quality_failures": [{"case_id": case_id}],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            cases, evidence = soak.load_quality_replay_cases(path, [case_id])

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].context_tokens, context_tokens)
        self.assertEqual(cases[0].prompt_index, prompt_index)
        self.assertEqual(
            cases[0].prompt_label,
            f"production-{context_tokens}-{prompt_index}",
        )
        self.assertEqual(cases[0].seed, request_seed)
        self.assertTrue(cases[0].source_quality_failure)
        self.assertEqual(evidence["context_prompt_overhead_tokens"], 17)
        self.assertTrue(evidence["seed_provenance_complete"])
        self.assertEqual(
            evidence["seed_provenance"],
            [
                {
                    "case_id": case_id,
                    "logged_source_seed": request_seed,
                    "reconstructed_source_seed": request_seed,
                    "matched": True,
                }
            ],
        )
        with patch.object(
            soak, "exact_context", side_effect=lambda _, __, label: label
        ):
            specs, _ = soak.make_quality_replay_specs(
                object(), cases, 1, 256, source_seed
            )
        self.assertEqual(specs[0].seed, request_seed)
        self.assertFalse(specs[0].extra["ignore_eos"])

        records[0]["arguments"]["fixed_output_length"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed-source.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            fixed_cases, fixed_evidence = soak.load_quality_replay_cases(
                path,
                [case_id],
            )

        self.assertFalse(fixed_cases[0].source_quality_failure)
        self.assertFalse(fixed_evidence["source_traffic_quality_eligible"])
        self.assertEqual(
            fixed_evidence["source_reported_quality_failures"],
            [case_id],
        )

        records[0]["arguments"]["fixed_output_length"] = False
        records[0]["arguments"]["concurrencies"] = [8, 16, 16]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-concurrency-source.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "concurrencies must be unique"):
                soak.load_quality_replay_cases(path, [case_id])

        records[0]["arguments"]["concurrencies"] = [8, 16]
        records[3]["seed"] = request_seed + 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not match reconstructed seed"):
                soak.load_quality_replay_cases(path, [case_id])

    def test_replay_cohort_pads_one_full_batch_without_new_prompt_identities(
        self,
    ) -> None:
        cases = [
            soak.QualityReplayCase(
                case_id="a",
                seed=1,
                concurrency=16,
                worker_index=0,
                request_index=0,
                context_tokens=1_024,
                prompt_index=0,
                prompt_label="production-1024-0",
                source_output_transcript_sha256="a",
                source_quality_failure=True,
            ),
            soak.QualityReplayCase(
                case_id="b",
                seed=2,
                concurrency=16,
                worker_index=1,
                request_index=0,
                context_tokens=8_192,
                prompt_index=1,
                prompt_label="production-8192-1",
                source_output_transcript_sha256="b",
                source_quality_failure=False,
            ),
            soak.QualityReplayCase(
                case_id="c",
                seed=3,
                concurrency=16,
                worker_index=2,
                request_index=0,
                context_tokens=1_024,
                prompt_index=0,
                prompt_label="production-1024-0",
                source_output_transcript_sha256="c",
                source_quality_failure=False,
            ),
        ]
        with patch.object(
            soak, "exact_context", side_effect=lambda _, __, label: label
        ):
            specs, evidence = soak.make_quality_replay_specs(
                object(), cases, 16, 256, 100
            )

        self.assertEqual(len(specs), 16)
        self.assertEqual(evidence["logical_prompt_identity_count"], 2)
        self.assertEqual(evidence["cold_compulsory_miss_requests"], 2)
        self.assertEqual(evidence["cold_duplicate_request_upper_bound"], 14)
        self.assertEqual(
            evidence["selected_duplicate_prompt_identities"],
            [
                {
                    "prompt_label": "production-1024-0",
                    "request_count": 2,
                    "case_ids": ["a", "c"],
                }
            ],
        )
        self.assertEqual(
            {spec.prompt for spec in specs},
            {"production-1024-0", "production-8192-1"},
        )
        self.assertTrue(all(spec.extra["ignore_eos"] is False for spec in specs))
        self.assertEqual(evidence["quality_output_contract"], "normal_eos")
        self.assertTrue(evidence["single_full_batch"])

    def test_cold_prefix_evidence_accounts_for_duplicate_prompt_identities(
        self,
    ) -> None:
        specs = [
            soak.RequestSpec(
                case_id="a-0",
                seed=1,
                max_tokens=8,
                prompt="prompt-a",
                context_tokens=100,
                tags={"prompt_label": "a"},
            ),
            soak.RequestSpec(
                case_id="a-1",
                seed=2,
                max_tokens=8,
                prompt="prompt-a",
                context_tokens=100,
                tags={"prompt_label": "a"},
            ),
            soak.RequestSpec(
                case_id="b-0",
                seed=3,
                max_tokens=8,
                prompt="prompt-b",
                context_tokens=200,
                tags={"prompt_label": "b"},
            ),
        ]
        before = {"mistralrs_prefix_cache_lookups_total": 0.0}
        all_miss = {"mistralrs_prefix_cache_lookups_total": 3.0}

        evidence = soak.quality_replay_prefix_state_evidence(
            before,
            all_miss,
            specs,
            "miss",
            0.98,
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["hits"], 0.0)
        self.assertEqual(evidence["misses"], 3.0)
        self.assertFalse(evidence["hits_counter_present"])
        self.assertFalse(evidence["reused_tokens_counter_present"])
        self.assertEqual(evidence["prompt_identity_count"], 2)
        self.assertEqual(evidence["duplicate_prompt_requests"], 1)
        self.assertEqual(evidence["compulsory_miss_prompt_tokens"], 300)
        self.assertEqual(evidence["duplicate_prompt_tokens"], 100)

        duplicate_hit = {
            "mistralrs_prefix_cache_lookups_total": 3.0,
            "mistralrs_prefix_cache_hits_total": 1.0,
            "mistralrs_prefix_cache_tokens_reused_total": 100.0,
        }
        self.assertTrue(
            soak.quality_replay_prefix_state_evidence(
                before,
                duplicate_hit,
                specs,
                "miss",
                0.98,
            )["passed"]
        )

        unexpected_hit = {
            "mistralrs_prefix_cache_lookups_total": 3.0,
            "mistralrs_prefix_cache_hits_total": 2.0,
            "mistralrs_prefix_cache_tokens_reused_total": 300.0,
        }
        self.assertFalse(
            soak.quality_replay_prefix_state_evidence(
                before,
                unexpected_hit,
                specs,
                "miss",
                0.98,
            )["passed"]
        )

    def test_missing_hit_counter_remains_a_failed_hit_expectation(self) -> None:
        specs = [
            soak.RequestSpec(
                case_id="a",
                seed=1,
                max_tokens=8,
                prompt="prompt-a",
                context_tokens=100,
            )
        ]
        evidence = soak.quality_replay_prefix_state_evidence(
            {"mistralrs_prefix_cache_lookups_total": 0.0},
            {"mistralrs_prefix_cache_lookups_total": 1.0},
            specs,
            "hit",
            0.98,
        )

        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["hits"], 0.0)

    def test_exactness_gates_only_equivalent_resident_paths(self) -> None:
        failed_cold_transition = {"passed": False, "candidate_phase": "resident"}
        stable = {"passed": True, "candidate_phase": "stable-reference"}
        reversed_order = {"passed": True, "candidate_phase": "reversed-order"}
        touches = [
            {"passed": True, "candidate_phase": f"touch-{index}"}
            for index in range(3)
        ]

        evidence = soak.quality_replay_exactness_evidence(
            failed_cold_transition,
            stable,
            reversed_order,
            touches,
        )

        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["cold_resident_diagnostic_only"])
        self.assertFalse(evidence["cold_resident"]["gated"])
        self.assertTrue(evidence["resident_stable_reference"]["gated"])
        self.assertTrue(evidence["reversed_order"]["gated"])
        self.assertEqual(evidence["required_comparisons"], 5)

        touches[-1]["passed"] = False
        failed = soak.quality_replay_exactness_evidence(
            failed_cold_transition,
            stable,
            reversed_order,
            touches,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failed_required_comparisons"], 1)

    def test_reversed_request_order_preserves_fixed_seed_comparison(self) -> None:
        specs = [
            soak.RequestSpec(case_id="a", seed=1, max_tokens=8),
            soak.RequestSpec(case_id="b", seed=2, max_tokens=8),
        ]
        stable = [
            request_result(
                case_id=spec.case_id,
                seed=spec.seed,
                completion_tokens=4,
                output_chunks=4,
                finish_reason="stop",
                output_transcript=f"transcript-{spec.case_id}",
            )
            for spec in specs
        ]

        evidence = soak.exact_output_diagnostics(
            stable,
            list(reversed(stable)),
            "stable-reference",
            "reversed-order",
            specs,
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["exact_matches"], 2)

    def test_stability_requires_exact_outputs_and_zero_graph_captures(self) -> None:
        def capture_key(component: str) -> str:
            return (
                f'{soak.CUDA_GRAPH_EVENTS_COUNTER}{{component="{component}",'
                'event="capture",outcome="success"}'
            )

        before = {
            capture_key("target"): 3.0,
            capture_key("dflash"): 2.0,
        }
        quiet = dict(before)
        target_capture = {**before, capture_key("target"): 4.0}
        quiet_evidence = soak.cuda_graph_capture_quiescence_evidence(
            before,
            quiet,
            ("target", "dflash"),
        )
        capture_evidence = soak.cuda_graph_capture_quiescence_evidence(
            before,
            target_capture,
            ("target", "dflash"),
        )

        self.assertTrue(quiet_evidence["passed"])
        self.assertFalse(capture_evidence["passed"])
        self.assertEqual(
            capture_evidence["components"]["target"]["capture_delta"],
            1.0,
        )

        attempts = [
            {
                "phase": "capture",
                "stage_passed": True,
                "exact_to_previous": {"passed": True},
                "cuda_graph_captures": capture_evidence,
            },
            {
                "phase": "changed-output",
                "stage_passed": True,
                "exact_to_previous": {"passed": False},
                "cuda_graph_captures": quiet_evidence,
            },
            {
                "phase": "stable",
                "stage_passed": True,
                "exact_to_previous": {"passed": True},
                "cuda_graph_captures": quiet_evidence,
            },
        ]
        convergence = soak.quality_replay_stability_evidence(attempts, 4)

        self.assertTrue(convergence["passed"])
        self.assertEqual(convergence["stable_reference_phase"], "stable")
        self.assertEqual(convergence["required_consecutive_exact_sets"], 2)
        self.assertEqual(
            convergence["required_zero_capture_components"],
            ["target", "dflash"],
        )

        failed = soak.quality_replay_stability_evidence(attempts[:2], 2)
        self.assertFalse(failed["passed"])
        self.assertTrue(failed["exhausted"])
        self.assertIsNotNone(failed["failure"])

    def test_pressure_requests_require_fixed_length_completion(self) -> None:
        with patch.object(
            soak,
            "exact_context",
            side_effect=lambda _, __, label: label,
        ):
            specs = soak.make_quality_replay_pressure_specs(
                object(),
                wave=0,
                entries=2,
                context_tokens=8_192,
                max_tokens=8,
                seed=1,
            )
        results = [
            request_result(
                case_id=spec.case_id,
                seed=spec.seed,
                completion_tokens=8,
                output_chunks=8,
                finish_reason="length",
            )
            for spec in specs
        ]

        self.assertTrue(all(spec.extra["ignore_eos"] for spec in specs))
        self.assertTrue(soak.fixed_length_completion_evidence(results, 8)["passed"])
        results[0].completion_tokens = 7
        results[1].finish_reason = "stop"
        evidence = soak.fixed_length_completion_evidence(results, 8)
        self.assertFalse(evidence["passed"])
        self.assertEqual(len(evidence["failures"]), 2)

    def test_pressure_plan_churns_pool_without_overcommitting_paired_footprint(
        self,
    ) -> None:
        cases = self.pressure_cases()
        specs = self.pressure_specs(cases)
        evidence = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(),
            cases,
            specs,
            self.pressure_config(),
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["logical_peak_entries"], 20)
        self.assertEqual(evidence["previous_pressure_blocks"], 4_096)
        self.assertLess(
            evidence["peak_required_blocks"],
            evidence["physical_budget_blocks"],
        )
        self.assertGreater(evidence["cumulative_pressure_blocks"], 27_080)

        owner_overcommit = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(owner_capacity=19),
            cases,
            specs,
            self.pressure_config(),
        )
        self.assertFalse(owner_overcommit["passed"])
        self.assertFalse(owner_overcommit["owner_fit"])

        physical_overcommit = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(),
            cases,
            specs,
            self.pressure_config(100_000),
        )
        self.assertFalse(physical_overcommit["passed"])
        self.assertFalse(physical_overcommit["physical_fit"])

    def test_pressure_plan_accounts_for_retained_previous_wave(self) -> None:
        cases = self.pressure_cases()
        evidence = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(),
            cases,
            self.pressure_specs(cases),
            self.pressure_config(32_000),
        )

        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["pressure_blocks_per_wave"], 16_000)
        self.assertEqual(evidence["previous_pressure_blocks"], 16_000)
        self.assertGreater(
            evidence["pressure_peak_blocks"],
            evidence["physical_budget_blocks"],
        )

    def test_pressure_plan_accounts_for_all_capacity_retained_prior_owners(
        self,
    ) -> None:
        cases = self.pressure_cases()
        evidence = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(owner_capacity=60),
            cases,
            self.pressure_specs(cases),
            self.pressure_config(),
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["available_pressure_owner_entries"], 56)
        self.assertEqual(evidence["elapsed_pressure_entries"], 112)
        self.assertEqual(evidence["previous_pressure_entries"], 56)
        self.assertEqual(evidence["previous_pressure_blocks"], 14_336)
        self.assertEqual(evidence["retained_pressure_entries"], 56)

        overcommitted = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(owner_capacity=100),
            cases,
            self.pressure_specs(cases),
            self.pressure_config(),
        )
        self.assertFalse(overcommitted["physical_fit"])
        self.assertEqual(overcommitted["previous_pressure_entries"], 96)

    def test_pressure_plan_accounts_for_nonempty_baseline(self) -> None:
        cases = self.pressure_cases()
        specs = self.pressure_specs(cases)
        owner_overcommit = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(owner_used=1),
            cases,
            specs,
            self.pressure_config(),
        )
        physical_overcommit = soak.quality_replay_pressure_plan_evidence(
            self.pressure_metrics(owner_capacity=40, retained_blocks=16_000),
            cases,
            specs,
            self.pressure_config(),
        )

        self.assertFalse(owner_overcommit["owner_fit"])
        self.assertEqual(owner_overcommit["logical_peak_entries"], 21)
        self.assertFalse(physical_overcommit["physical_fit"])
        self.assertGreater(
            physical_overcommit["peak_required_blocks"],
            physical_overcommit["physical_budget_blocks"],
        )


class CudaGraphEvidenceTests(unittest.TestCase):
    @staticmethod
    def key(metric: str, **labels: str) -> str:
        rendered = ",".join(f'{name}="{value}"' for name, value in labels.items())
        return f"{metric}{{{rendered}}}"

    def healthy_metrics(self) -> dict[str, float]:
        return {
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="replay",
                reason="cache_hit",
            ): 100.0,
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="skipped",
                reason="prefill",
            ): 20.0,
            self.key(
                soak.CUDA_GRAPH_EVENTS_COUNTER,
                component="target",
                event="replay",
                outcome="success",
            ): 100.0,
        }

    def test_documented_prefill_skips_are_allowed(self) -> None:
        after = self.healthy_metrics()

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 0.98)

        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["instrumentation_complete"])
        self.assertEqual(
            evidence["components"]["target"]["allowed_skipped_dispatches"],
            20.0,
        )

    def test_unexpected_skipped_dispatch_is_gated(self) -> None:
        after = self.healthy_metrics()
        after[
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="skipped",
                reason="incompatible_shape",
            )
        ] = 1.0

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 0.98)

        self.assertFalse(evidence["passed"])
        self.assertEqual(
            evidence["components"]["target"]["unexpected_skipped_dispatches"],
            1.0,
        )

    def test_cache_population_eager_is_accounted_by_successful_capture(self) -> None:
        after = self.healthy_metrics()
        after[
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="eager",
                reason=soak.CUDA_GRAPH_CACHE_POPULATION_REASON,
            )
        ] = 4.0
        after[
            self.key(
                soak.CUDA_GRAPH_EVENTS_COUNTER,
                component="target",
                event="capture",
                outcome="success",
            )
        ] = 4.0

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 1.0)
        target = evidence["components"]["target"]

        self.assertTrue(evidence["passed"])
        self.assertEqual(target["eager_dispatches"], 4.0)
        self.assertEqual(target["accounted_cache_population_dispatches"], 4.0)
        self.assertEqual(target["unexpected_eager_dispatches"], 0.0)
        self.assertEqual(target["eligible_replay_ratio"], 1.0)

    def test_unaccounted_cache_population_and_other_eager_are_gated(self) -> None:
        after = self.healthy_metrics()
        after[
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="eager",
                reason=soak.CUDA_GRAPH_CACHE_POPULATION_REASON,
            )
        ] = 4.0
        after[
            self.key(
                soak.CUDA_GRAPH_EVENTS_COUNTER,
                component="target",
                event="capture",
                outcome="success",
            )
        ] = 3.0
        after[
            self.key(
                soak.CUDA_GRAPH_DISPATCH_COUNTER,
                component="target",
                mode="eager",
                reason="replay_error",
            )
        ] = 1.0

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 0.98)
        target = evidence["components"]["target"]

        self.assertFalse(evidence["passed"])
        self.assertEqual(target["accounted_cache_population_dispatches"], 3.0)
        self.assertEqual(target["unexpected_eager_dispatches"], 2.0)

    def test_capture_failure_remains_a_hard_failure(self) -> None:
        after = self.healthy_metrics()
        after[
            self.key(
                soak.CUDA_GRAPH_EVENTS_COUNTER,
                component="target",
                event="capture",
                outcome="failure",
            )
        ] = 1.0

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 0.98)

        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["components"]["target"]["phase_failures"], 1.0)

    def test_dispatch_without_event_instrumentation_is_rejected(self) -> None:
        after = self.healthy_metrics()
        after = {
            key: value
            for key, value in after.items()
            if not key.startswith(soak.CUDA_GRAPH_EVENTS_COUNTER)
        }

        evidence = soak.cuda_graph_evidence({}, after, after, ("target",), 0.98)

        self.assertFalse(evidence["instrumentation_complete"])
        self.assertFalse(evidence["passed"])


class SpeculativeEvidenceTests(unittest.TestCase):
    @staticmethod
    def metrics(
        *,
        drafts: float = 10.0,
        proposed: float = 70.0,
        accepted: float = 35.0,
        gpu_verified: float | None = 10.0,
        cpu_fallbacks: float | None = None,
    ) -> dict[str, float]:
        metrics = {
            "mistralrs_speculative_drafts_total": drafts,
            "mistralrs_speculative_draft_tokens_proposed_total": proposed,
            "mistralrs_speculative_draft_tokens_accepted_total": accepted,
        }
        if gpu_verified is not None:
            metrics[soak.SPARSE_VERIFIER_GPU_COUNTER] = gpu_verified
        if cpu_fallbacks is not None:
            metrics[soak.SPARSE_VERIFIER_FALLBACK_COUNTER] = cpu_fallbacks
        return metrics

    def evidence(self, after: dict[str, float]) -> dict:
        return soak.speculative_evidence(
            {},
            after,
            True,
            min_acceptance_rate=0.05,
            min_mean_advance=1.10,
            min_proposal_depth=2.0,
            max_sparse_fallback_ratio=0.01,
        )

    def test_healthy_mtp_and_sparse_verifier_pass(self) -> None:
        evidence = self.evidence(self.metrics())

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["acceptance_rate"], 0.5)
        self.assertEqual(evidence["mean_proposed_draft_tokens_per_draft"], 7.0)
        self.assertEqual(evidence["mean_advance_tokens_per_target_step"], 4.5)
        self.assertEqual(evidence["sparse_verifier"]["fallback_ratio"], 0.0)

    def test_mtp_quality_floors_are_non_vacuous(self) -> None:
        zero_acceptance = self.evidence(self.metrics(accepted=0.0))
        shallow_proposals = self.evidence(
            self.metrics(proposed=10.0, accepted=5.0)
        )

        self.assertFalse(zero_acceptance["performance_floors_passed"])
        self.assertFalse(zero_acceptance["passed"])
        self.assertFalse(shallow_proposals["performance_floors_passed"])
        self.assertFalse(shallow_proposals["passed"])

    def test_sparse_fallback_ratio_and_instrumentation_are_gated(self) -> None:
        excessive_fallback = self.evidence(
            self.metrics(gpu_verified=98.0, cpu_fallbacks=2.0)
        )
        missing_instrumentation = self.evidence(
            self.metrics(gpu_verified=None, cpu_fallbacks=None)
        )

        self.assertEqual(
            excessive_fallback["sparse_verifier"]["fallback_ratio"],
            0.02,
        )
        self.assertFalse(excessive_fallback["passed"])
        self.assertFalse(
            missing_instrumentation["sparse_verifier"]["instrumentation_present"]
        )
        self.assertFalse(missing_instrumentation["passed"])

    def test_unaccounted_cpu_verification_is_gated(self) -> None:
        evidence = self.evidence(
            self.metrics(drafts=100.0, proposed=700.0, accepted=350.0, gpu_verified=1.0)
        )

        self.assertEqual(evidence["sparse_verifier"]["fallback_ratio"], 0.0)
        self.assertEqual(evidence["sparse_verifier"]["accounting_coverage"], 0.01)
        self.assertEqual(evidence["sparse_verifier"]["unaccounted_sequences"], 99.0)
        self.assertFalse(evidence["sparse_verifier"]["accounting_coverage_passed"])
        self.assertFalse(evidence["sparse_verifier"]["passed"])
        self.assertFalse(evidence["passed"])


class MetricSelectorTests(unittest.TestCase):
    def test_metric_total_filters_labels_independent_of_series_order(self) -> None:
        snapshot = {
            'mistralrs_windowed_kv_slots_used{pool="live",component="dflash"}': 2.0,
            'mistralrs_windowed_kv_slots_used{component="other",pool="live"}': 5.0,
            'mistralrs_windowed_kv_slots_used{component="dflash",pool="checkpoint"}': 3.0,
        }

        self.assertEqual(
            soak.metric_total(
                snapshot,
                soak.DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE,
            ),
            2.0,
        )
        self.assertEqual(
            soak.metric_total(snapshot, soak.WINDOWED_KV_SLOTS_USED_GAUGE),
            10.0,
        )

    def test_dflash_production_requires_both_windowed_pools(self) -> None:
        gauges = soak.production_required_gauges(("target", "dflash"))

        self.assertIn(soak.DFLASH_WINDOWED_KV_LIVE_SLOTS_USED_GAUGE, gauges)
        self.assertIn(soak.DFLASH_WINDOWED_KV_LIVE_SLOTS_TOTAL_GAUGE, gauges)
        self.assertIn(soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE, gauges)
        self.assertIn(soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE, gauges)
        self.assertEqual(
            soak.production_required_gauges(("target",)),
            soak.REQUIRED_PRODUCTION_GAUGES,
        )


class QueueLatencyEvidenceTests(unittest.TestCase):
    @staticmethod
    def histogram(count: float, bucket_deltas: tuple[float, float, float]) -> dict[str, float]:
        return {
            'mistralrs_scheduler_queue_seconds_bucket{le="0.1"}': bucket_deltas[0],
            'mistralrs_scheduler_queue_seconds_bucket{le="1"}': bucket_deltas[1],
            'mistralrs_scheduler_queue_seconds_bucket{le="+Inf"}': bucket_deltas[2],
            "mistralrs_scheduler_queue_seconds_count": count,
        }

    def test_positive_observations_and_complete_quantiles_pass(self) -> None:
        before = self.histogram(25.0, (10.0, 20.0, 25.0))
        after = self.histogram(30.0, (12.0, 24.0, 30.0))

        evidence = soak.queue_histogram_evidence(before, after)
        histogram = evidence["histograms"]["mistralrs_scheduler_queue_seconds"]

        self.assertTrue(evidence["passed"])
        self.assertEqual(histogram["observation_delta"], 5.0)
        self.assertTrue(histogram["quantiles_complete"])
        self.assertIsNotNone(histogram["p99"])

    def test_stale_or_incomplete_histograms_fail(self) -> None:
        before = self.histogram(25.0, (10.0, 20.0, 25.0))
        stale = soak.queue_histogram_evidence(before, dict(before))
        missing_observation_counter = {
            key: value + 1.0
            for key, value in before.items()
            if not key.endswith("_count") and 'le="+Inf"' not in key
        }

        self.assertFalse(stale["passed"])
        self.assertEqual(
            stale["histograms"]["mistralrs_scheduler_queue_seconds"][
                "observation_delta"
            ],
            0.0,
        )
        self.assertFalse(
            soak.queue_histogram_evidence({}, missing_observation_counter)["passed"]
        )


class CpuTelemetryTests(unittest.TestCase):
    @staticmethod
    def process(
        host_total: int,
        host_idle: int,
        process_ticks: int,
    ) -> dict:
        return {
            "host_cpu_total_ticks": host_total,
            "host_cpu_idle_ticks": host_idle,
            "process_cpu_ticks": process_ticks,
            "process_cpu_clock_ticks_per_second": 100,
            "process_is_mistralrs": True,
            "process_vmrss_kib": 1_024,
            "process_gpu_memory_used_mib": 100.0,
            "gpus": [
                {
                    "utilization_percent": 50.0,
                    "memory_used_mib": 100.0,
                    "power_watts": 200.0,
                }
            ],
        }

    def test_proc_cpu_parsers_handle_aggregate_and_spaced_command(self) -> None:
        self.assertEqual(
            soak.parse_host_cpu_ticks(
                "cpu  100 10 20 30 5 2 3 4 90 9\ncpu0 1 2 3 4"
            ),
            (174, 35),
        )
        fields = ["S", *(["0"] * 12)]
        fields[11] = "120"
        fields[12] = "30"
        self.assertEqual(
            soak.parse_process_cpu_ticks(
                f"123 (mistralrs worker) {' '.join(fields)}"
            ),
            150,
        )

    def test_summary_reports_host_and_server_process_cpu(self) -> None:
        snapshots = [
            (0.0, {"mistralrs_sequences_running": 1.0}, self.process(1_000, 400, 100)),
            (1.0, {"mistralrs_sequences_running": 2.0}, self.process(1_100, 420, 150)),
        ]
        cadence = soak.scheduled_observation_evidence(
            [0.0, 1.0],
            [0.0, 1.0],
            0.1,
        )

        summary = soak.summarize_telemetry(snapshots)
        evidence = soak.telemetry_evidence(
            snapshots,
            42,
            2,
            1.0,
            ("mistralrs_sequences_running",),
            cadence,
        )

        self.assertEqual(summary["host_cpu_utilization_percent"]["mean"], 80.0)
        self.assertEqual(summary["process_cpu_utilization_percent"]["mean"], 50.0)
        self.assertTrue(evidence["passed"])

        missed_cadence = soak.scheduled_observation_evidence(
            [0.0, 1.0],
            [0.0],
            0.1,
        )
        self.assertFalse(
            soak.telemetry_evidence(
                snapshots,
                42,
                2,
                1.0,
                ("mistralrs_sequences_running",),
                missed_cadence,
            )["passed"]
        )

        snapshots[-1][2].pop("process_cpu_ticks")
        self.assertFalse(
            soak.telemetry_evidence(
                snapshots,
                42,
                2,
                1.0,
                ("mistralrs_sequences_running",),
                cadence,
            )["passed"]
        )

    def test_process_scoped_gpu_coverage_is_required(self) -> None:
        snapshots = [
            (0.0, {"mistralrs_sequences_running": 1.0}, self.process(1_000, 400, 100)),
            (1.0, {"mistralrs_sequences_running": 1.0}, self.process(1_100, 420, 150)),
        ]
        snapshots[-1][2]["process_gpu_memory_used_mib"] = None
        cadence = soak.scheduled_observation_evidence(
            [0.0, 1.0],
            [0.0, 1.0],
            0.1,
        )

        evidence = soak.telemetry_evidence(
            snapshots,
            42,
            2,
            0.95,
            ("mistralrs_sequences_running",),
            cadence,
        )

        self.assertEqual(evidence["process_gpu_memory_coverage"], 0.5)
        self.assertFalse(evidence["passed"])


class CadenceAndWindowEvidenceTests(unittest.TestCase):
    def test_periodic_schedule_and_observations_are_exactly_accounted(self) -> None:
        scheduled = soak.periodic_schedule(0.0, 12.0, 5.0, include_start=True)

        self.assertEqual(scheduled, [0.0, 5.0, 10.0])
        self.assertTrue(
            soak.scheduled_observation_evidence(
                scheduled,
                [0.0, 5.1, 10.2],
                0.25,
            )["passed"]
        )
        self.assertFalse(
            soak.scheduled_observation_evidence(
                scheduled,
                [0.0, 5.1],
                0.25,
            )["passed"]
        )
        self.assertFalse(
            soak.scheduled_observation_evidence(
                scheduled,
                [0.0, 5.1, 10.5],
                0.25,
            )["passed"]
        )

    def test_comparison_windows_require_real_full_non_overlapping_coverage(self) -> None:
        complete = soak.comparison_window_coverage_evidence(
            0.0,
            7_200.0,
            7_200.0,
            3_600.0,
        )
        ended_early = soak.comparison_window_coverage_evidence(
            0.0,
            7_200.0,
            7_199.0,
            3_600.0,
        )
        overlapping = soak.comparison_window_coverage_evidence(
            0.0,
            3_600.0,
            3_600.0,
            3_600.0,
        )

        self.assertTrue(complete["passed"])
        self.assertFalse(ended_early["passed"])
        self.assertFalse(overlapping["passed"])


class ProductionRuntimeCoordinationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_prewarm_stops_before_production_traffic(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "production",
                "--server-pid",
                "42",
                "--tokenizer",
                "tokenizer.json",
                "--min-scaling-efficiency",
                "0.5",
                "--max-probe-ttft-seconds",
                "1.0",
                "--max-probe-tpot-seconds",
                "0.05",
            ]
        )
        client = type(
            "Client",
            (),
            {
                "tokenizer": object(),
                "stream_request": AsyncMock(),
            },
        )()
        writer = type("Writer", (), {"emit": AsyncMock()})()
        clean_metrics = {"mistralrs_sequences_running": 0.0}
        cleanup = {"passed": True}

        def retrieval(
            _client,
            length,
            label,
            seed,
            max_tokens,
            tags,
        ):
            return soak.RequestSpec(
                case_id=label,
                seed=seed,
                max_tokens=max_tokens,
                prompt=f"prompt-{length}",
                context_tokens=length,
                tags=tags,
            )

        process = {
            "process_is_mistralrs": True,
            "process_vmrss_kib": 1,
            "host_cpu_total_ticks": 1,
            "host_cpu_idle_ticks": 1,
            "process_cpu_ticks": 1,
            "gpus": [{}],
            "process_gpus": [{}],
            "process_gpu_memory_used_mib": 1.0,
        }
        run_batch = AsyncMock(
            side_effect=[
                ([], {"errors": 0}),
                ([], {"errors": 0}),
            ]
        )
        poll_for_cleanup = AsyncMock(
            side_effect=[
                (True, clean_metrics, cleanup),
                (True, clean_metrics, cleanup),
            ]
        )
        with (
            patch.object(soak, "calibrate_prompt_profiles", new=AsyncMock()),
            patch.object(
                soak,
                "safe_metrics",
                new=AsyncMock(
                    side_effect=[
                        {"mistralrs_sequences_capacity": 16.0},
                        clean_metrics,
                    ]
                ),
            ),
            patch.object(soak, "process_telemetry", new=AsyncMock(return_value=process)),
            patch.object(soak, "production_required_gauges", return_value=()),
            patch.object(soak, "exact_context", return_value="prompt"),
            patch.object(soak, "retrieval_spec", side_effect=retrieval),
            patch.object(soak, "run_batch", new=run_batch),
            patch.object(soak, "poll_for_cleanup", new=poll_for_cleanup),
            patch.object(
                soak,
                "prefix_cache_evidence",
                return_value={"passed": False},
            ),
            patch.object(
                soak,
                "cuda_memory_pressure_evidence",
                return_value={"passed": True},
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "production prewarm gate failed",
            ):
                await soak.production_mode(args, client, writer)

        self.assertEqual(run_batch.await_count, 2)
        client.stream_request.assert_not_awaited()
        final_emit = writer.emit.await_args_list[-1]
        event = final_emit.args[0]
        fields = final_emit.kwargs
        self.assertEqual(event, "production_prewarm_summary")
        self.assertFalse(fields["passed"])

    async def test_diagnostics_do_not_consume_traffic_slots(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.active = {"traffic": 0, "diagnostic": 0}
                self.maximum = {"traffic": 0, "diagnostic": 0}
                self.maximum_total = 0
                self.overlap = asyncio.Event()
                self.release = asyncio.Event()
                self.scheduled_at = []

            async def stream_request(self, spec, *, scheduled_at=None, **_kwargs):
                role = spec.tags["role"]
                self.scheduled_at.append(scheduled_at)
                self.active[role] += 1
                self.maximum[role] = max(self.maximum[role], self.active[role])
                self.maximum_total = max(self.maximum_total, sum(self.active.values()))
                if self.active == {"traffic": 2, "diagnostic": 1}:
                    self.overlap.set()
                try:
                    await self.release.wait()
                    return request_result(
                        case_id=spec.case_id,
                        completion_tokens=spec.max_tokens,
                        output_chunks=spec.max_tokens,
                        finish_reason="length",
                        tags=spec.tags,
                    )
                finally:
                    self.active[role] -= 1

        client = Client()
        slots = soak.ProductionPhaseSlots.create(2, 1)
        traffic = [
            soak.RequestSpec(
                f"traffic-{index}",
                index,
                8,
                prompt="prompt",
                tags={"role": "traffic"},
            )
            for index in range(2)
        ]
        diagnostics = [
            soak.RequestSpec(
                f"diagnostic-{index}",
                index,
                8,
                prompt="prompt",
                tags={"role": "diagnostic"},
            )
            for index in range(2)
        ]
        scheduled_at = 123.0
        tasks = [
            *(
                asyncio.create_task(
                    soak.stream_request_with_slot(
                        client,
                        slots.traffic,
                        spec,
                        scheduled_at=scheduled_at,
                    )
                )
                for spec in traffic
            ),
            *(
                asyncio.create_task(
                    soak.stream_request_with_slot(
                        client,
                        slots.diagnostics,
                        spec,
                        scheduled_at=scheduled_at,
                    )
                )
                for spec in diagnostics
            ),
        ]

        await asyncio.wait_for(client.overlap.wait(), timeout=1.0)
        self.assertEqual(client.active, {"traffic": 2, "diagnostic": 1})
        client.release.set()
        await soak.wait_for_production_phase_tasks(tasks)

        self.assertEqual(client.maximum, {"traffic": 2, "diagnostic": 1})
        self.assertEqual(client.maximum_total, 3)
        self.assertEqual(client.scheduled_at, [scheduled_at] * 4)
        self.assertTrue(all(task.done() for task in tasks))

    async def test_production_phase_task_failure_cleans_up_siblings(self) -> None:
        blocker_started = asyncio.Event()
        blocker_cancelled = asyncio.Event()

        async def blocker() -> None:
            blocker_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                blocker_cancelled.set()

        async def fail() -> None:
            await blocker_started.wait()
            raise RuntimeError("phase failed")

        tasks = [asyncio.create_task(blocker()), asyncio.create_task(fail())]

        with self.assertRaisesRegex(RuntimeError, "phase failed"):
            await soak.wait_for_production_phase_tasks(tasks)

        self.assertTrue(blocker_cancelled.is_set())
        self.assertTrue(all(task.done() for task in tasks))

    async def test_telemetry_continues_until_stop_and_collects_terminal_sample(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.records = []

            async def emit(self, event, **record) -> None:
                self.records.append((event, record))

        writer = Writer()
        stop = asyncio.Event()
        started = soak.time.perf_counter()
        snapshots = []
        scheduled = [started]
        observed = [started]
        with (
            patch.object(
                soak,
                "safe_metrics",
                new=AsyncMock(return_value={"mistralrs_sequences_running": 0.0}),
            ),
            patch.object(
                soak,
                "process_telemetry",
                new=AsyncMock(return_value={"process_gpu_memory_used_mib": 1.0}),
            ),
        ):
            task = asyncio.create_task(
                soak.telemetry_loop(
                    object(),
                    writer,
                    stop,
                    42,
                    snapshots,
                    0.01,
                    scheduled,
                    observed,
                )
            )
            await asyncio.sleep(0.025)
            stop.set()
            terminal_at = await task

        self.assertGreaterEqual(len(snapshots), 3)
        self.assertEqual(len(scheduled), len(observed))
        self.assertEqual(terminal_at, snapshots[-1][0])
        self.assertTrue(writer.records[-1][1]["terminal"])


class ServerProvenanceTests(unittest.TestCase):
    def test_command_capture_redacts_secrets_and_extracts_serving_config(self) -> None:
        argv = [
            "/srv/mistralrs",
            "serve",
            "-m",
            "Qwen/Qwen3.8-27B-FP8",
            "--mtp-model=incoai/Qwen3.8-27B-DFlash2",
            "--pa-cache-type",
            "f8e4m3",
            "--paged-attn=on",
            "--max-seqs",
            "16",
            "--prefix-cache-n",
            "20",
            "--mtp-n-predict=7",
            "--mtp-draft-sampling",
            "stochastic",
            "--api-key",
            "secret-value",
            "--token-source=also-secret",
        ]

        redacted = soak.redact_server_argv(argv)
        config = soak.parse_serve_configuration(argv)

        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("also-secret", " ".join(redacted))
        self.assertEqual(config["subcommand"], "serve")
        self.assertEqual(config["model"], "Qwen/Qwen3.8-27B-FP8")
        self.assertEqual(config["mtp_model"], "incoai/Qwen3.8-27B-DFlash2")
        self.assertEqual(config["pa_cache_type"], "f8e4m3")
        self.assertEqual(config["paged_attn"], "on")
        self.assertEqual(config["max_seqs"], "16")
        self.assertEqual(config["prefix_cache_n"], "20")
        self.assertEqual(config["mtp_n_predict"], "7")
        self.assertEqual(config["mtp_draft_sampling"], "stochastic")

    def test_complete_provenance_requires_exact_git_binary_gpu_and_kv(self) -> None:
        provenance = {
            "server_pid": 42,
            "system_info": {"build": {"git_revision": "a" * 40}},
            "process": {
                "process_is_mistralrs": True,
                "executable": "/srv/mistralrs",
                "executable_sha256": "b" * 64,
                "command_sha256": "c" * 64,
                "serve_configuration": {
                    "subcommand": "serve",
                    "paged_attn": "auto",
                    "pa_cache_type": "f8e4m3",
                },
            },
            "gpu_driver": {
                "available": True,
                "gpus": [
                    {
                        "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                        "driver_version": "580.105.08",
                    }
                ],
                "server_process_gpus": [
                    {"pid": 42, "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"}
                ],
            },
            "realized_kv_configuration": {
                "blocks_total": 1_024.0,
                "sequence_capacity": 16.0,
            },
        }

        evidence = soak.server_provenance_evidence(provenance)

        self.assertTrue(evidence["complete"])
        provenance["system_info"]["build"]["git_revision"] = "unknown"
        self.assertFalse(soak.server_provenance_evidence(provenance)["complete"])
        provenance["system_info"]["build"]["git_revision"] = "a" * 40
        provenance["process"]["serve_configuration"]["pa_cache_type"] = "auto"
        self.assertFalse(soak.server_provenance_evidence(provenance)["complete"])

    def test_serialized_arguments_never_persist_api_key(self) -> None:
        args = soak.build_parser().parse_args(
            ["canary", "--api-key", "top-secret"]
        )

        self.assertEqual(soak.serialized_arguments(args)["api_key"], "<redacted>")

    def test_acceptance_adversarial_requires_complete_provenance_preflight(self) -> None:
        parser = soak.build_parser()
        acceptance = parser.parse_args(
            [
                "adversarial",
                "--tokenizer",
                "tokenizer.json",
                "--server-pid",
                "1",
                "--min-output-tok-s-by-concurrency",
                "1:1,3:1,8:1,16:1",
                "--min-scaling-efficiency",
                "0.1",
                "--min-overlap-decode-events-per-second",
                "1",
                "--min-overlap-decode-throughput-ratio",
                "0.1",
            ]
        )
        diagnostic = parser.parse_args(
            [
                "adversarial",
                "--tokenizer",
                "tokenizer.json",
                "--no-acceptance-grade",
                "--no-require-mtp",
                "--throughput-concurrencies",
                "4",
                "--min-output-tok-s-by-concurrency",
                "4:1",
                "--min-scaling-efficiency",
                "0.1",
                "--min-overlap-decode-events-per-second",
                "1",
                "--min-overlap-decode-throughput-ratio",
                "0.1",
            ]
        )

        self.assertTrue(soak.server_provenance_required(acceptance))
        self.assertFalse(soak.server_provenance_required(diagnostic))


class CudaMemoryPressureEvidenceTests(unittest.TestCase):
    @staticmethod
    def metrics(
        *,
        maintenance: float,
        pending: float,
        pressure: float = 0.0,
        reclaimed: float = 0.0,
        reductions: float = 0.0,
        deferred: float = 0.0,
        rejections: float = 0.0,
        errors: float = 0.0,
    ) -> dict[str, float]:
        return {
            (
                'mistralrs_cuda_memory_maintenance_total{device="cuda[0]",'
                'reason="prompt_boundary",action="maintain",outcome="ok"}'
            ): maintenance,
            (
                'mistralrs_cuda_memory_maintenance_total{device="cuda[0]",'
                'reason="prompt_boundary",action="maintain",outcome="error"}'
            ): errors,
            'mistralrs_cuda_memory_pressure_total{device="cuda[0]",level="graph"}': pressure,
            'mistralrs_cuda_memory_maintenance_pending{device="cuda[0]"}': pending,
            soak.CUDA_MEMORY_RECLAIMED_BYTES_COUNTER: reclaimed,
            soak.CUDA_PROMPT_BATCH_REDUCTIONS_COUNTER: reductions,
            soak.CUDA_PROMPT_SEQUENCES_DEFERRED_COUNTER: deferred,
            soak.CUDA_PROMPT_MEMORY_REJECTIONS_COUNTER: rejections,
            soak.CUDA_GRAPH_EVICTIONS_COUNTER: 0.0,
        }

    def test_recovery_activity_is_reported_without_failing(self) -> None:
        before = self.metrics(maintenance=10.0, pending=0.0)
        after = self.metrics(
            maintenance=16.0,
            pending=0.0,
            pressure=2.0,
            reclaimed=1024.0,
            reductions=2.0,
            deferred=3.0,
        )
        after[soak.CUDA_GRAPH_EVICTIONS_COUNTER] = 4.0

        evidence = soak.cuda_memory_pressure_evidence(before, after, True)

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["pressure_events"], 2.0)
        self.assertEqual(evidence["prompt_batch_reductions"], 2.0)
        self.assertEqual(evidence["prompt_sequences_deferred"], 3.0)
        self.assertEqual(evidence["graph_evictions"], 4.0)

    def test_errors_rejections_pending_and_bad_accounting_are_gated(self) -> None:
        before = self.metrics(maintenance=10.0, pending=0.0)
        cases = (
            self.metrics(maintenance=11.0, pending=0.0, errors=1.0),
            self.metrics(maintenance=11.0, pending=0.0, rejections=1.0),
            self.metrics(maintenance=11.0, pending=1.0),
            self.metrics(maintenance=11.0, pending=0.0, reductions=2.0, deferred=1.0),
        )

        for after in cases:
            with self.subTest(after=after):
                self.assertFalse(
                    soak.cuda_memory_pressure_evidence(before, after, True)["passed"]
                )

    def test_required_instrumentation_cannot_be_missing(self) -> None:
        evidence = soak.cuda_memory_pressure_evidence({}, {}, True)

        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["gates"]["instrumentation"])


class ProductionMemoryEvidenceTests(unittest.TestCase):
    @staticmethod
    def metrics(
        *,
        active_blocks: float = 0.0,
        recurrent_used: float = 0.0,
        kv_total: float = 100.0,
        recurrent_total: float = 32.0,
    ) -> dict[str, float]:
        return {
            "mistralrs_sequences_running": 0.0,
            "mistralrs_sequences_waiting": 0.0,
            "mistralrs_requests_pending_admission": 0.0,
            "mistralrs_recurrent_state_slots_used": recurrent_used,
            "mistralrs_recurrent_state_slots_total": recurrent_total,
            soak.KV_CACHE_ACTIVE_GAUGE: active_blocks,
            "mistralrs_kv_cache_blocks_total": kv_total,
            "http_requests_in_flight": 0.0,
        }

    @staticmethod
    def process(
        rss_mib: float,
        process_gpu_mib: float | None,
        device_gpu_mib: float,
    ) -> dict:
        return {
            "process_vmrss_kib": rss_mib * 1024.0,
            "process_gpu_memory_used_mib": process_gpu_mib,
            "gpus": [{"memory_used_mib": device_gpu_mib}],
        }

    def healthy_snapshots(self) -> list[tuple[float, dict, dict]]:
        return [
            (0.0, self.metrics(), self.process(10_000.0, 90_000.0, 91_000.0)),
            (
                1.0,
                self.metrics(active_blocks=50.0, recurrent_used=16.0),
                self.process(10_300.0, 90_500.0, 93_500.0),
            ),
            (2.0, self.metrics(), self.process(10_100.0, 90_100.0, 91_100.0)),
        ]

    def dflash_snapshots(self) -> list[tuple[float, dict, dict]]:
        snapshots = self.healthy_snapshots()
        for index, (_, metrics, _) in enumerate(snapshots):
            live_used = 8.0 if index == 1 else 0.0
            checkpoint_used = 4.0 if index == 1 else 0.0
            metrics[
                'mistralrs_windowed_kv_slots_used{pool="live",component="dflash"}'
            ] = live_used
            metrics[
                'mistralrs_windowed_kv_slots_total{pool="live",component="dflash"}'
            ] = 16.0
            metrics[
                'mistralrs_windowed_kv_slots_used{pool="checkpoint",component="dflash"}'
            ] = checkpoint_used
            metrics[
                'mistralrs_windowed_kv_slots_total{pool="checkpoint",component="dflash"}'
            ] = 8.0
        return snapshots

    @staticmethod
    def evidence(snapshots: list[tuple[float, dict, dict]]) -> dict:
        return soak.production_memory_evidence(
            snapshots,
            soak.ProductionMemoryLimits(
                min_coverage=0.95,
                max_process_rss_drift_mib=512.0,
                max_process_rss_drift_fraction=0.10,
                max_process_rss_high_water_mib=2_048.0,
                max_gpu_memory_drift_mib=512.0,
                max_gpu_memory_high_water_mib=2_048.0,
                max_kv_block_utilization=0.95,
                max_recurrent_slot_utilization=0.95,
            ),
        )

    def test_stable_memory_and_final_cleanup_pass(self) -> None:
        evidence = self.evidence(self.healthy_snapshots())

        self.assertTrue(evidence["passed"])
        self.assertEqual(
            evidence["gpu_memory"]["source"],
            "server_pid_compute_process",
        )
        self.assertEqual(evidence["kv_blocks"]["maximum_observed_utilization"], 0.5)
        self.assertTrue(evidence["final_cleanup"]["passed"])

    def test_dflash_windowed_pool_occupancy_and_cleanup_pass(self) -> None:
        evidence = soak.production_memory_evidence(
            self.dflash_snapshots(),
            soak.ProductionMemoryLimits(require_dflash_windowed_kv=True),
        )

        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["windowed_kv_slots"]["required"])
        self.assertEqual(
            evidence["windowed_kv_slots"]["live"][
                "maximum_observed_utilization"
            ],
            0.5,
        )
        self.assertEqual(
            evidence["windowed_kv_slots"]["checkpoint"][
                "maximum_observed_utilization"
            ],
            0.5,
        )

    def test_required_dflash_windowed_pool_instrumentation_is_gated(self) -> None:
        evidence = soak.production_memory_evidence(
            self.healthy_snapshots(),
            soak.ProductionMemoryLimits(require_dflash_windowed_kv=True),
        )

        self.assertFalse(evidence["windowed_kv_slots"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_dflash_checkpoint_slot_leak_is_gated(self) -> None:
        snapshots = self.dflash_snapshots()
        snapshots[-1][1][
            'mistralrs_windowed_kv_slots_used{pool="checkpoint",component="dflash"}'
        ] = 1.0

        evidence = soak.production_memory_evidence(
            snapshots,
            soak.ProductionMemoryLimits(require_dflash_windowed_kv=True),
        )

        self.assertFalse(evidence["final_cleanup"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_transient_gpu_high_water_is_gated_even_after_recovery(self) -> None:
        snapshots = self.healthy_snapshots()
        snapshots[1][2]["process_gpu_memory_used_mib"] = 93_000.0

        evidence = self.evidence(snapshots)

        self.assertFalse(evidence["gpu_memory"]["passed"])
        self.assertEqual(evidence["gpu_memory"]["final_growth_mib"], 100.0)
        self.assertEqual(evidence["gpu_memory"]["high_water_growth_mib"], 3_000.0)
        self.assertFalse(evidence["passed"])

    def test_final_rss_drift_is_gated(self) -> None:
        snapshots = self.healthy_snapshots()
        snapshots[-1][2]["process_vmrss_kib"] = 11_500.0 * 1024.0

        evidence = self.evidence(snapshots)

        self.assertEqual(evidence["process_rss"]["final_growth_mib"], 1_500.0)
        self.assertFalse(evidence["process_rss"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_resource_pool_high_water_is_gated(self) -> None:
        snapshots = self.healthy_snapshots()
        snapshots[1] = (
            snapshots[1][0],
            self.metrics(active_blocks=96.0, recurrent_used=31.0),
            snapshots[1][2],
        )

        evidence = self.evidence(snapshots)

        self.assertFalse(evidence["kv_blocks"]["passed"])
        self.assertFalse(evidence["recurrent_slots"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_full_recurrent_capacity_is_healthy_by_default(self) -> None:
        snapshots = self.healthy_snapshots()
        for index, (timestamp, metrics, process) in enumerate(snapshots):
            metrics["mistralrs_recurrent_state_slots_total"] = 17.0
            metrics["mistralrs_recurrent_state_slots_used"] = (
                17.0 if index == 1 else 1.0
            )
            snapshots[index] = (timestamp, metrics, process)

        evidence = soak.production_memory_evidence(
            snapshots,
            soak.ProductionMemoryLimits(),
        )

        self.assertEqual(
            evidence["recurrent_slots"]["maximum_observed_utilization"],
            1.0,
        )
        self.assertTrue(evidence["recurrent_slots"]["passed"])
        self.assertTrue(evidence["passed"])

    def test_pid_gpu_memory_is_preferred_over_whole_device_noise(self) -> None:
        evidence = self.evidence(self.healthy_snapshots())

        self.assertTrue(evidence["gpu_memory"]["passed"])
        self.assertEqual(
            evidence["gpu_memory"]["source"],
            "server_pid_compute_process",
        )

    def test_whole_device_memory_does_not_replace_missing_pid_metric(self) -> None:
        snapshots = self.healthy_snapshots()
        for _, _, process in snapshots:
            process["process_gpu_memory_used_mib"] = None
        snapshots[1][2]["gpus"][0]["memory_used_mib"] = 91_500.0

        evidence = self.evidence(snapshots)

        self.assertFalse(evidence["gpu_memory"]["passed"])
        self.assertFalse(evidence["passed"])
        self.assertEqual(
            evidence["gpu_memory"]["source"],
            "server_pid_compute_process",
        )
        self.assertFalse(evidence["gpu_memory"]["whole_device_fallback_used"])
        self.assertEqual(
            evidence["gpu_memory"]["whole_device_diagnostic"]["max"],
            91_500.0,
        )

    def test_final_resource_leak_is_gated(self) -> None:
        snapshots = self.healthy_snapshots()
        snapshots[-1] = (
            snapshots[-1][0],
            self.metrics(active_blocks=1.0, recurrent_used=1.0),
            snapshots[-1][2],
        )

        evidence = self.evidence(snapshots)

        self.assertFalse(evidence["final_cleanup"]["passed"])
        self.assertFalse(evidence["passed"])


class BatchMeasurementTests(unittest.TestCase):
    @staticmethod
    def specs(count: int = 16) -> list[soak.RequestSpec]:
        return [
            soak.RequestSpec(
                case_id=f"case-{index}",
                messages=[],
                max_tokens=1,
                seed=index,
            )
            for index in range(count)
        ]

    def test_c3_measurement_excludes_partial_drain_batch(self) -> None:
        specs = self.specs()

        measured = soak.full_batch_specs(specs, 3)

        self.assertEqual(len(measured), 15)
        self.assertEqual(len(measured) % 3, 0)

    def test_mixed_specs_have_independent_prompt_contexts(self) -> None:
        class Tokenizer:
            @staticmethod
            def retrieval_text(content_tokens: int, label: str):
                return f"prompt-{content_tokens}-{label}", f"answer-{label}"

        class Client:
            tokenizer = Tokenizer()

            @staticmethod
            def calibrated_content_tokens(profile: str, prompt_tokens: int) -> int:
                self = profile
                return prompt_tokens

        specs = soak.adversarial_mixed_specs(
            Client(),
            (1_024, 8_192, 32_768, 100_000),
            16,
            100,
            64,
        )
        evidence = soak.request_cohort_uniqueness_evidence(specs)

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["requests"], 16)
        self.assertEqual(evidence["unique_prompt_hashes"], 16)
        self.assertEqual(len({spec.case_id for spec in specs}), 16)
        self.assertEqual(
            [spec.context_tokens for spec in specs[:4]],
            [1_024, 8_192, 32_768, 100_000],
        )

    def test_mixed_cohort_rejects_reused_prompts(self) -> None:
        specs = [
            soak.RequestSpec(
                case_id=f"case-{index}",
                messages=[],
                max_tokens=1,
                seed=index,
                prompt="same prompt",
            )
            for index in range(4)
        ]

        evidence = soak.request_cohort_uniqueness_evidence(specs)

        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["unique_prompt_hashes"], 1)

    def test_long_resident_cohort_uses_unique_prewarmed_prompts(self) -> None:
        class Tokenizer:
            @staticmethod
            def exact_text(content_tokens: int, label: str) -> str:
                return f"prompt-{content_tokens}-{label}"

        class Client:
            tokenizer = Tokenizer()

            @staticmethod
            def calibrated_content_tokens(profile: str, prompt_tokens: int) -> int:
                self = profile
                return prompt_tokens

        measured, warm = soak.adversarial_long_resident_cohorts(
            Client(),
            100,
            64,
            {"ignore_eos": True},
        )
        uniqueness = soak.request_cohort_uniqueness_evidence(measured)

        self.assertTrue(uniqueness["passed"])
        self.assertEqual(len(measured), 3)
        self.assertEqual([spec.prompt for spec in warm], [spec.prompt for spec in measured])
        self.assertEqual([spec.max_tokens for spec in measured], [64, 64, 64])
        self.assertEqual([spec.max_tokens for spec in warm], [1, 1, 1])
        self.assertEqual(
            [spec.context_tokens for spec in measured],
            [soak.ADVERSARIAL_LONG_RESIDENT_CONTEXT_TOKENS] * 3,
        )

    def test_long_resident_completion_gate_requires_full_length(self) -> None:
        complete = [
            request_result(
                case_id=f"case-{index}",
                seed=index,
                completion_tokens=64,
                output_chunks=64,
                finish_reason="length",
            )
            for index in range(3)
        ]

        self.assertTrue(soak.full_length_completion_evidence(complete, 64)["passed"])
        complete[1].completion_tokens = 63
        self.assertFalse(soak.full_length_completion_evidence(complete, 64)["passed"])
        complete[1].completion_tokens = 64
        complete[1].finish_reason = "stop"
        self.assertFalse(soak.full_length_completion_evidence(complete, 64)["passed"])

    def test_long_resident_gate_uses_exact_decode_thresholds(self) -> None:
        summaries = [
            {"concurrency": 1, "decode_tok_s_active": 204.6},
            {"concurrency": 3, "decode_tok_s_active": 319.3},
        ]

        evidence = soak.exact_throughput_threshold_evidence(
            summaries,
            {1: 190.0, 3: 300.0},
            "decode_tok_s_active",
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["throughput_metric"], "decode_tok_s_active")
        self.assertFalse(
            soak.exact_throughput_threshold_evidence(
                summaries[:1],
                {1: 190.0, 3: 300.0},
                "decode_tok_s_active",
            )["passed"]
        )
        self.assertFalse(
            soak.exact_throughput_threshold_evidence(
                [summaries[0], summaries[0], summaries[1]],
                {1: 190.0, 3: 300.0},
                "decode_tok_s_active",
            )["passed"]
        )

    def test_fairness_uses_absolute_short_latency_slos(self) -> None:
        healthy = request_result(
            completion_tokens=8,
            output_chunks=8,
            finish_reason="stop",
            ttft_seconds=0.182,
            tpot_seconds=0.02,
        )

        evidence = soak.fairness_short_latency_evidence([healthy], 1.0, 0.05)

        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["requests"][0]["ttft_passed"])
        self.assertTrue(evidence["requests"][0]["tpot_passed"])

    def test_fairness_absolute_short_latency_slos_reject_regressions(self) -> None:
        slow_ttft = request_result(
            completion_tokens=8,
            output_chunks=8,
            finish_reason="stop",
            ttft_seconds=1.01,
            tpot_seconds=0.02,
        )
        slow_tpot = request_result(
            completion_tokens=8,
            output_chunks=8,
            finish_reason="stop",
            ttft_seconds=0.2,
            tpot_seconds=0.051,
        )

        self.assertFalse(
            soak.fairness_short_latency_evidence([slow_ttft], 1.0, 0.05)["passed"]
        )
        self.assertFalse(
            soak.fairness_short_latency_evidence([slow_tpot], 1.0, 0.05)["passed"]
        )
        failed = request_result(
            completion_tokens=8,
            output_chunks=8,
            finish_reason="stop",
            ok=False,
            ttft_seconds=0.2,
            tpot_seconds=0.02,
        )
        self.assertFalse(
            soak.fairness_short_latency_evidence([failed], 1.0, 0.05)["passed"]
        )

    def test_fairness_relative_slowdown_is_gated(self) -> None:
        healthy = soak.fairness_relative_slowdown_evidence(
            [1.2, 2.9],
            3.0,
        )
        regressed = soak.fairness_relative_slowdown_evidence(
            [1.2, 3.1],
            3.0,
        )

        self.assertTrue(healthy["passed"])
        self.assertTrue(healthy["gated"])
        self.assertFalse(regressed["passed"])
        self.assertFalse(
            soak.fairness_relative_slowdown_evidence([], 3.0)["passed"]
        )

    def test_fairness_requires_overlapping_decode_progress(self) -> None:
        long = request_result(
            case_id="long",
            completion_tokens=4,
            output_chunks=4,
            finish_reason="stop",
            output_event_times=[1.0, 2.0, 3.0, 4.0],
        )
        short = request_result(
            case_id="short",
            completion_tokens=3,
            output_chunks=3,
            finish_reason="stop",
            output_event_times=[2.0, 2.5, 3.5],
        )

        evidence = soak.concurrent_decode_overlap_evidence(long, short)

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["overlap_seconds"], 1.5)
        short.output_event_times = [5.0, 6.0]
        self.assertFalse(
            soak.concurrent_decode_overlap_evidence(long, short)["passed"]
        )

    def test_overlap_output_gap_includes_window_boundaries(self) -> None:
        result = request_result(
            completion_tokens=5,
            output_chunks=5,
            finish_reason="stop",
            output_event_times=[10.1, 10.2, 10.4, 10.6, 10.8],
        )

        healthy = soak.output_event_gap_evidence([result], 10.0, 11.0, 0.25)
        stalled = soak.output_event_gap_evidence([result], 10.0, 11.2, 0.25)

        self.assertTrue(healthy["passed"])
        self.assertAlmostEqual(healthy["maximum_observed_gap_seconds"], 0.2)
        self.assertFalse(stalled["passed"])
        self.assertAlmostEqual(stalled["maximum_observed_gap_seconds"], 0.4)

    def test_correctness_order_keeps_c3_remainder(self) -> None:
        specs = self.specs()

        normal = soak.correctness_order_specs(specs, False)
        reverse = soak.correctness_order_specs(specs, True)

        self.assertEqual(len(normal), 16)
        self.assertEqual(len(reverse), 16)
        self.assertEqual([spec.seed for spec in normal], list(range(16)))
        self.assertEqual([spec.seed for spec in reverse], list(reversed(range(16))))

    def test_cross_concurrency_measurement_reports_common_cohort(self) -> None:
        specs = self.specs()
        concurrencies = [1, 3, 8, 16]

        common = soak.common_full_batch_specs(specs, concurrencies)
        evidence = soak.common_full_batch_cohort_evidence(
            specs,
            common,
            concurrencies,
        )

        self.assertEqual(len(common), 15)
        self.assertTrue(evidence["complete"])
        self.assertEqual(evidence["requested_cases"], 16)
        self.assertEqual(evidence["common_cases"], 15)
        self.assertEqual(
            evidence["measurement_cases_by_concurrency"],
            {"1": 16, "3": 15, "8": 16, "16": 16},
        )
        self.assertEqual(
            evidence["excluded_cases"],
            [{"case_id": "case-15", "seed": 15}],
        )

    def test_multi_wave_measurement_is_not_reused_for_exact_replay(self) -> None:
        specs = self.specs()
        for concurrency in (3, 8):
            with self.subTest(concurrency=concurrency):
                measured = soak.full_batch_specs(specs, concurrency)
                exact, evidence = soak.resident_exact_replay_cohort(
                    measured,
                    concurrency,
                )

                self.assertGreater(evidence["measurement_waves"], 1)
                self.assertEqual(len(exact), concurrency)
                self.assertEqual(evidence["exact_cases"], concurrency)
                self.assertFalse(evidence["normal_phase_reusable"])
                self.assertEqual(
                    evidence["normal_reuse_reason"],
                    "dedicated_single_full_batch_required",
                )

    def test_partial_drain_correctness_uses_dedicated_exact_wave(self) -> None:
        specs = self.specs(5)

        exact, evidence = soak.fixed_seed_exact_replay_cohort(specs, 3)

        self.assertEqual(len(exact), 3)
        self.assertEqual(evidence["measurement_cases"], 5)
        self.assertEqual(evidence["measurement_waves"], 2)
        self.assertFalse(evidence["normal_phase_reusable"])
        self.assertEqual(
            evidence["normal_reuse_reason"],
            "dedicated_single_full_batch_required",
        )

    def test_exact_replay_reuses_only_single_batch_or_c1_serial(self) -> None:
        specs = self.specs()
        exact_c16, c16 = soak.resident_exact_replay_cohort(specs, 16)
        exact_c1, c1 = soak.resident_exact_replay_cohort(specs, 1)

        self.assertEqual(len(exact_c16), 16)
        self.assertTrue(c16["normal_phase_reusable"])
        self.assertEqual(c16["normal_reuse_reason"], "single_full_batch")
        self.assertEqual(len(exact_c1), 16)
        self.assertEqual(c1["measurement_waves"], 16)
        self.assertTrue(c1["normal_phase_reusable"])
        self.assertEqual(c1["cohort_kind"], "c1_serial")
        self.assertEqual(c1["normal_reuse_reason"], "c1_batch_shape_constant")

    def test_c3_context_measurement_is_balanced(self) -> None:
        contexts = (1_024, 8_192, 32_768, 100_000)
        specs = [
            soak.RequestSpec(
                case_id=f"case-{index}",
                messages=[],
                max_tokens=1,
                seed=index,
                context_tokens=contexts[index % len(contexts)],
            )
            for index in range(16)
        ]

        measured = soak.balanced_context_full_batch_specs(specs, 3)

        self.assertEqual(len(measured), 12)
        self.assertEqual(len(measured) % 3, 0)
        self.assertEqual(
            {context: sum(spec.context_tokens == context for spec in measured) for context in contexts},
            {context: 3 for context in contexts},
        )

    def test_common_wall_failure_cannot_hide_behind_active_decode(self) -> None:
        summaries = [
            {
                "concurrency": 1,
                "output_tok_s_common_wall": 50.0,
                "decode_tok_s_active": 200.0,
            },
            {
                "concurrency": 3,
                "output_tok_s_common_wall": 80.0,
                "decode_tok_s_active": 600.0,
            },
        ]

        evidence = soak.serving_throughput_evidence(
            summaries,
            {1: 100.0, 3: 250.0},
            0.8,
        )

        self.assertFalse(evidence["common_wall"]["passed"])
        self.assertTrue(evidence["decode_active"]["passed"])
        self.assertFalse(evidence["passed"])


class FixedSeedGateTests(unittest.TestCase):
    @staticmethod
    def comparison(
        exact_passed: bool = True,
        *,
        exact_gated: bool = True,
        coverage_complete: bool = True,
        statistical_passed: bool = True,
        semantic_passed: bool = True,
    ) -> dict:
        return soak.fixed_seed_comparison_evidence(
            {
                "passed": exact_passed and coverage_complete,
                "coverage_complete": coverage_complete,
                "mismatches": [] if exact_passed else [{}],
            },
            {"passed": statistical_passed},
            semantic_passed,
            exact_gated=exact_gated,
        )

    def test_same_shape_replay_exact_gates_every_concurrency(self) -> None:
        for concurrency in (1, 3, 8, 16):
            with self.subTest(concurrency=concurrency):
                evidence = {
                    "concurrency": concurrency,
                    **self.comparison(False),
                }

                self.assertTrue(evidence["statistical_comparison"]["passed"])
                self.assertTrue(evidence["semantic_passed"])
                self.assertTrue(evidence["exact_diagnostics_gated"])
                self.assertFalse(evidence["passed"])

    def test_ordering_and_cross_concurrency_do_not_exact_gate(self) -> None:
        evidence = self.comparison(False, exact_gated=False)

        self.assertFalse(evidence["exact_diagnostics"]["passed"])
        self.assertFalse(evidence["exact_diagnostics_gated"])
        self.assertTrue(evidence["passed"])

    def test_ungated_exact_diagnostics_still_require_complete_coverage(self) -> None:
        evidence = self.comparison(
            False,
            exact_gated=False,
            coverage_complete=False,
        )

        self.assertFalse(evidence["case_seed_coverage_complete"])
        self.assertFalse(evidence["passed"])

    def test_invariance_requires_every_replay_ordering_and_cross_phase(self) -> None:
        evidence = soak.fixed_seed_invariance_evidence(
            [self.comparison() for _ in range(3)],
            [self.comparison(exact_gated=False) for _ in range(3)],
            [self.comparison(exact_gated=False) for _ in range(2)],
            3,
        )

        self.assertTrue(evidence["passed"])
        self.assertFalse(
            soak.fixed_seed_invariance_evidence(
                [self.comparison() for _ in range(3)],
                [self.comparison(exact_gated=False) for _ in range(3)],
                [self.comparison(exact_gated=False)],
                3,
            )["passed"]
        )

    def test_invariance_rejects_wrong_gate_policy(self) -> None:
        self.assertFalse(
            soak.fixed_seed_invariance_evidence(
                [self.comparison(exact_gated=False)],
                [self.comparison(exact_gated=False)],
                [],
                1,
            )["passed"]
        )
        self.assertFalse(
            soak.fixed_seed_invariance_evidence(
                [self.comparison()],
                [self.comparison()],
                [],
                1,
            )["passed"]
        )

    def test_exact_diagnostics_gate_expected_case_seed_coverage(self) -> None:
        specs = [
            soak.RequestSpec(
                case_id=f"case-{index}",
                messages=[],
                max_tokens=1,
                seed=index,
            )
            for index in range(2)
        ]
        observed = [
            request_result(
                completion_tokens=1,
                output_chunks=1,
                finish_reason="length",
                case_id="case-0",
                seed=0,
            )
        ]

        evidence = soak.exact_output_diagnostics(
            observed,
            observed,
            "reference",
            "candidate",
            specs,
        )

        self.assertFalse(evidence["coverage_complete"])
        self.assertFalse(evidence["passed"])
        self.assertEqual(
            evidence["reference_missing_expected"],
            [{"case_id": "case-1", "seed": 1}],
        )
        self.assertEqual(
            evidence["candidate_missing_expected"],
            [{"case_id": "case-1", "seed": 1}],
        )


class DFlashCheckpointRetentionGateTests(unittest.TestCase):
    @staticmethod
    def snapshot(used: float, total: float = 17.0) -> dict[str, float]:
        return {
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE: used,
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE: total,
        }

    @classmethod
    def snapshots(cls) -> dict[str, dict[str, float]]:
        return {
            "before_cold": cls.snapshot(2.0),
            "after_cold": cls.snapshot(3.0),
            "after_hit": cls.snapshot(3.0),
            "after_pressure": cls.snapshot(13.0),
            "after_retry": cls.snapshot(13.0),
            "quiescent": cls.snapshot(13.0),
        }

    def test_bounded_prefix_retention_passes(self) -> None:
        evidence = soak.dflash_checkpoint_retention_evidence(
            self.snapshots(),
            distinct_successful_prefixes=11,
        )

        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["instrumentation_complete"])
        self.assertEqual(evidence["retained_capacity"], 16.0)
        self.assertEqual(
            evidence["checks"]["total_growth_bounded"]["observed_growth"],
            11.0,
        )

    def test_missing_optional_instrumentation_passes(self) -> None:
        snapshots = {stage: {} for stage in self.snapshots()}

        evidence = soak.dflash_checkpoint_retention_evidence(
            snapshots,
            distinct_successful_prefixes=11,
        )

        self.assertTrue(evidence["passed"])
        self.assertFalse(evidence["available"])
        self.assertFalse(evidence["instrumentation_complete"])

    def test_partial_instrumentation_is_rejected(self) -> None:
        snapshots = self.snapshots()
        snapshots["after_hit"].pop(
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE
        )

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 11)

        self.assertFalse(evidence["passed"])
        self.assertTrue(evidence["available"])
        self.assertFalse(evidence["instrumentation_complete"])

    def test_capacity_must_remain_stable(self) -> None:
        snapshots = self.snapshots()
        snapshots["after_pressure"][
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_TOTAL_GAUGE
        ] = 18.0

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 11)

        self.assertFalse(evidence["checks"]["stable_total"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_quiescent_state_must_leave_the_staging_slot_free(self) -> None:
        snapshots = self.snapshots()
        snapshots["quiescent"][
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE
        ] = 17.0

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 15)

        self.assertFalse(
            evidence["checks"]["quiescent_within_retained_capacity"]["passed"]
        )
        self.assertFalse(evidence["passed"])

    def test_cache_hit_must_not_grow_checkpoint_occupancy(self) -> None:
        snapshots = self.snapshots()
        snapshots["after_hit"][
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE
        ] = 4.0

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 11)

        self.assertFalse(evidence["checks"]["hit_no_growth"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_same_key_retry_must_not_grow_checkpoint_occupancy(self) -> None:
        snapshots = self.snapshots()
        snapshots["after_retry"][
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE
        ] = 14.0
        snapshots["quiescent"][
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE
        ] = 14.0

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 12)

        self.assertFalse(
            evidence["checks"]["same_key_retry_no_growth"]["passed"]
        )
        self.assertFalse(evidence["passed"])

    def test_growth_cannot_exceed_successful_distinct_prefixes(self) -> None:
        snapshots = self.snapshots()
        for stage in ("after_pressure", "after_retry", "quiescent"):
            snapshots[stage][
                soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE
            ] = 14.0

        evidence = soak.dflash_checkpoint_retention_evidence(snapshots, 11)

        self.assertFalse(evidence["checks"]["total_growth_bounded"]["passed"])
        self.assertFalse(evidence["passed"])

    def test_abort_cleanup_keeps_checkpoint_non_growth_gate(self) -> None:
        self.assertIn(
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
            soak.DFLASH_ABORT_CLEANUP_GAUGES,
        )
        self.assertNotIn(
            soak.DFLASH_WINDOWED_KV_CHECKPOINT_SLOTS_USED_GAUGE,
            soak.OPTIONAL_CLEANUP_GAUGES,
        )


class PrefixOwnershipGateTests(unittest.TestCase):
    def test_aborts_must_not_cache_and_successful_retry_must_take_ownership(self) -> None:
        evidence = soak.prefix_cached_ownership_evidence(0.0, 128.0, 0.0)

        self.assertTrue(evidence["passed"])
        self.assertFalse(soak.prefix_cached_ownership_evidence(1.0, 128.0, 0.0)["passed"])
        self.assertFalse(soak.prefix_cached_ownership_evidence(0.0, 0.0, 0.0)["passed"])
        self.assertFalse(soak.prefix_cached_ownership_evidence(0.0, 128.0, 1.0)["passed"])
        self.assertFalse(soak.prefix_cached_ownership_evidence(None, 128.0, 0.0)["passed"])


class ChurnCapacityGateTests(unittest.TestCase):
    @staticmethod
    def sample(running: float, waiting: float = 0.0, pending: float = 0.0) -> dict:
        return {
            "mistralrs_sequences_capacity": 16.0,
            "mistralrs_sequences_running": running,
            "mistralrs_sequences_waiting": waiting,
            "mistralrs_requests_pending_admission": pending,
        }

    def test_near_capacity_must_be_reached_and_sustained(self) -> None:
        sustained = soak.churn_capacity_evidence(
            [self.sample(15.0, waiting=1.0) for _ in range(4)],
            17,
            16,
            16.0,
            True,
            0,
        )
        serialized = soak.churn_capacity_evidence(
            [self.sample(1.0, waiting=1.0) for _ in range(4)],
            17,
            16,
            16.0,
            True,
            0,
        )
        transient = soak.churn_capacity_evidence(
            [self.sample(15.0, waiting=1.0), *[self.sample(1.0, waiting=1.0) for _ in range(9)]],
            17,
            16,
            16.0,
            True,
            0,
        )

        self.assertTrue(sustained["near_capacity_sustained"])
        self.assertTrue(sustained["passed"])
        self.assertFalse(serialized["near_capacity_sustained"])
        self.assertFalse(serialized["passed"])
        self.assertEqual(transient["peak_running"], 15.0)
        self.assertFalse(transient["near_capacity_sustained"])
        self.assertFalse(transient["passed"])

    def test_long_drain_does_not_erase_sustained_boundary_window(self) -> None:
        samples = [self.sample(15.0, waiting=1.0) for _ in range(18)]
        samples.extend(self.sample(1.0) for _ in range(8_543))

        evidence = soak.churn_capacity_evidence(
            samples,
            15,
            16,
            16.0,
            True,
            0,
        )

        self.assertTrue(evidence["near_capacity_sustained"])
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["near_capacity_consecutive_samples"], 18)
        self.assertLess(evidence["near_capacity_sample_fraction"], 0.01)
        self.assertFalse(evidence["near_capacity_sample_fraction_gated"])


class ArgumentValidationTests(unittest.TestCase):
    @staticmethod
    def adversarial_args(cancel_requests: int):
        return soak.build_parser().parse_args(
            [
                "adversarial",
                "--tokenizer",
                "tokenizer.json",
                "--server-pid",
                "1",
                "--min-output-tok-s-by-concurrency",
                "1:1,3:1,8:1,16:1",
                "--min-scaling-efficiency",
                "0.1",
                "--min-overlap-decode-events-per-second",
                "1",
                "--min-overlap-decode-throughput-ratio",
                "0.1",
                "--cancel-requests",
                str(cancel_requests),
            ]
        )

    @staticmethod
    def production_args(*extra: str):
        return soak.build_parser().parse_args(
            [
                "production",
                "--tokenizer",
                "tokenizer.json",
                "--min-output-tok-s-by-concurrency",
                "8:1,16:1",
                "--min-scaling-efficiency",
                "0.1",
                "--max-probe-ttft-seconds",
                "1",
                "--max-probe-tpot-seconds",
                "1",
                "--server-pid",
                "1",
                *extra,
            ]
        )

    def test_post_admission_cancellations_fit_sequence_capacity(self) -> None:
        valid = self.adversarial_args(16)
        soak.validate_args(valid)
        self.assertTrue(valid.acceptance_grade)
        self.assertTrue(valid.require_mtp)
        self.assertEqual(set(valid.expected_graph_components), {"target", "dflash"})
        with self.assertRaisesRegex(ValueError, "cannot exceed --max-seqs"):
            soak.validate_args(self.adversarial_args(17))

    def test_acceptance_adversarial_requires_server_pid(self) -> None:
        args = self.adversarial_args(16)
        args.server_pid = None

        with self.assertRaisesRegex(ValueError, "requires --server-pid"):
            soak.validate_args(args)

    def test_acceptance_adversarial_requires_sustained_churn(self) -> None:
        args = self.adversarial_args(16)
        args.churn_rounds = soak.DEFAULT_ADVERSARIAL_CHURN_ROUNDS - 1

        with self.assertRaisesRegex(ValueError, "--churn-rounds must be at least"):
            soak.validate_args(args)

        args.acceptance_grade = False
        args.require_mtp = False
        soak.validate_args(args)

    def test_acceptance_adversarial_cannot_weaken_overlap_stall_limits(self) -> None:
        decode_gap = self.adversarial_args(16)
        decode_gap.max_overlap_decode_gap_seconds = (
            soak.DEFAULT_MAX_OVERLAP_DECODE_GAP_SECONDS + 0.01
        )
        prefill_ttft = self.adversarial_args(16)
        prefill_ttft.max_overlap_prefill_ttft_seconds = (
            soak.DEFAULT_MAX_OVERLAP_PREFILL_TTFT_SECONDS + 1.0
        )

        with self.assertRaisesRegex(ValueError, "decode-gap-seconds must be at most"):
            soak.validate_args(decode_gap)
        with self.assertRaisesRegex(ValueError, "prefill-ttft-seconds must be at most"):
            soak.validate_args(prefill_ttft)

    def test_diagnostic_adversarial_accepts_focused_c4(self) -> None:
        args = self.adversarial_args(16)
        args.acceptance_grade = False
        args.require_mtp = False
        args.expected_graph_components = ("target",)
        args.server_pid = None
        args.throughput_concurrencies = (4,)
        args.min_output_tok_s_by_concurrency = {4: 1.0}

        soak.validate_args(args)

    def test_acceptance_adversarial_rejects_focused_c4(self) -> None:
        args = self.adversarial_args(16)
        args.throughput_concurrencies = (4,)
        args.min_output_tok_s_by_concurrency = {4: 1.0}

        with self.assertRaisesRegex(ValueError, "must include 1,3,8,16"):
            soak.validate_args(args)

    def test_long_correctness_rejects_concurrency_above_case_count(self) -> None:
        args = self.adversarial_args(16)
        args.long_correctness_concurrencies = (3, 8)

        with self.assertRaisesRegex(ValueError, "cannot exceed the number"):
            soak.validate_args(args)

    def test_acceptance_requires_every_long_context_boundary(self) -> None:
        args = self.adversarial_args(16)
        args.long_correctness_context_lengths = (60_000, 65_536, 100_000)

        with self.assertRaisesRegex(
            ValueError,
            "60000,65535,65536,65537,100000",
        ):
            soak.validate_args(args)

        args.acceptance_grade = False
        soak.validate_args(args)

    def test_long_resident_thresholds_require_exact_c1_c3_cohort(self) -> None:
        args = self.adversarial_args(16)
        args.min_long_resident_decode_tok_s_by_concurrency = {1: 190.0}

        with self.assertRaisesRegex(ValueError, "concurrencies 1 and 3 exactly"):
            soak.validate_args(args)

    def test_acceptance_long_resident_requires_64_tokens(self) -> None:
        args = self.adversarial_args(16)
        args.long_resident_max_tokens = 63

        with self.assertRaisesRegex(ValueError, "must be at least 64"):
            soak.validate_args(args)

        args.acceptance_grade = False
        soak.validate_args(args)

    def test_resident_decode_concurrencies_must_be_unique(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "resident-decode",
                "--tokenizer",
                "tokenizer.json",
                "--concurrencies",
                "1,1,8",
                "--min-output-tok-s-by-concurrency",
                "1:1,8:1",
                "--min-scaling-efficiency",
                "0.1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "--concurrencies must be unique"):
            soak.validate_args(args)

    def test_production_concurrencies_must_be_unique(self) -> None:
        args = self.production_args("--concurrencies", "8,8,16")

        with self.assertRaisesRegex(ValueError, "--concurrencies must be unique"):
            soak.validate_args(args)

    def test_quality_replay_stability_bound_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.touch()
            args = soak.build_parser().parse_args(
                [
                    "quality-replay",
                    "--tokenizer",
                    "tokenizer.json",
                    "--server-pid",
                    "1",
                    "--source-production-artifact",
                    str(source),
                    "--case-id",
                    "prod-c16-w0-r0",
                    "--max-stability-passes",
                    "0",
                ]
            )

            with self.assertRaisesRegex(ValueError, "--max-stability-passes"):
                soak.validate_args(args)

    def test_prefix_pressure_headroom_default_validates(self) -> None:
        args = self.adversarial_args(16)

        soak.validate_args(args)
        self.assertEqual(
            args.prefix_pressure_kv_headroom_fraction,
            soak.DEFAULT_PREFIX_PRESSURE_KV_HEADROOM_FRACTION,
        )

    def test_focused_prefix_pressure_requires_pid_and_validates(self) -> None:
        parser = soak.build_parser()
        missing_pid = parser.parse_args(
            ["prefix-pressure", "--tokenizer", "tokenizer.json"]
        )
        with self.assertRaisesRegex(ValueError, "requires --server-pid"):
            soak.validate_args(missing_pid)

        args = parser.parse_args(
            [
                "prefix-pressure",
                "--tokenizer",
                "tokenizer.json",
                "--server-pid",
                "1",
            ]
        )
        soak.validate_args(args)
        self.assertEqual(args.max_seqs, 16)
        self.assertEqual(args.prefix_pressure_context_tokens, 100_000)

    def test_invalid_prefix_pressure_headroom_is_rejected(self) -> None:
        args = self.adversarial_args(16)
        args.prefix_pressure_kv_headroom_fraction = 1.0

        with self.assertRaisesRegex(ValueError, "kv-headroom-fraction"):
            soak.validate_args(args)

    def test_speculative_gate_defaults_are_non_vacuous(self) -> None:
        args = soak.build_parser().parse_args(["canary"])

        self.assertEqual(
            args.min_mtp_acceptance_rate,
            soak.DEFAULT_MIN_MTP_ACCEPTANCE_RATE,
        )
        self.assertGreater(args.min_mtp_mean_advance, 1.0)
        self.assertGreater(args.min_mtp_proposal_depth, 0.0)
        self.assertLess(args.max_sparse_verifier_fallback_ratio, 1.0)
        self.assertEqual(
            args.min_sparse_verifier_accounting_coverage,
            soak.DEFAULT_MIN_SPARSE_VERIFIER_ACCOUNTING_COVERAGE,
        )
        soak.validate_args(args)

    def test_named_production_sampling_policy_is_exact(self) -> None:
        args = soak.build_parser().parse_args(["canary"])

        soak.validate_args(args)
        self.assertEqual(args.sampling_policy, soak.PART1_PRODUCTION_SAMPLING_POLICY)
        self.assertTrue(
            soak.production_sampling_policy_evidence(
                args.sampling_policy,
                soak.SamplingPolicy(
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.min_p,
                    args.repetition_penalty,
                ),
            )["passed"]
        )

        mismatched = soak.build_parser().parse_args(
            ["canary", "--temperature", "0.7"]
        )
        with self.assertRaisesRegex(ValueError, "production sampling values"):
            soak.validate_args(mismatched)

    def test_custom_sampling_cannot_claim_part1_complete(self) -> None:
        custom = soak.build_parser().parse_args(
            [
                "canary",
                "--sampling-policy",
                "custom",
                "--temperature",
                "0.7",
            ]
        )
        soak.validate_args(custom)

        complete = soak.build_parser().parse_args(
            [
                "canary",
                "--sampling-policy",
                "custom",
                "--temperature",
                "0.7",
                "--require-part1-complete",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --sampling-policy production"):
            soak.validate_args(complete)

    def test_invalid_sparse_fallback_ratio_is_rejected(self) -> None:
        args = soak.build_parser().parse_args(
            ["canary", "--max-sparse-verifier-fallback-ratio", "1.1"]
        )

        with self.assertRaisesRegex(ValueError, "fallback-ratio"):
            soak.validate_args(args)

    def test_invalid_sparse_accounting_coverage_is_rejected(self) -> None:
        args = soak.build_parser().parse_args(
            ["canary", "--min-sparse-verifier-accounting-coverage", "1.1"]
        )

        with self.assertRaisesRegex(ValueError, "accounting-coverage"):
            soak.validate_args(args)

    def test_production_memory_gate_defaults_validate(self) -> None:
        args = self.production_args()

        soak.validate_args(args)
        self.assertTrue(args.acceptance_grade)
        self.assertTrue(args.require_mtp)
        self.assertEqual(set(args.expected_graph_components), {"target", "dflash"})
        self.assertEqual(
            args.max_process_rss_drift_mib,
            soak.DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB,
        )
        self.assertEqual(
            args.max_gpu_memory_high_water_mib,
            soak.DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB,
        )
        self.assertEqual(args.max_recurrent_slot_utilization, 1.0)
        self.assertEqual(
            args.min_comparison_window_samples,
            soak.DEFAULT_MIN_COMPARISON_WINDOW_SAMPLES,
        )
        self.assertEqual(
            args.diagnostic_concurrency,
            soak.DEFAULT_PRODUCTION_DIAGNOSTIC_CONCURRENCY,
        )

    def test_production_requires_positive_diagnostic_concurrency(self) -> None:
        args = self.production_args("--diagnostic-concurrency", "0")

        with self.assertRaisesRegex(ValueError, "diagnostic-concurrency"):
            soak.validate_args(args)

    def test_production_throughput_floors_default_to_c8_and_c16_targets(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "production",
                "--tokenizer",
                "tokenizer.json",
                "--min-scaling-efficiency",
                "0.1",
                "--max-probe-ttft-seconds",
                "1",
                "--max-probe-tpot-seconds",
                "1",
                "--server-pid",
                "1",
            ]
        )

        soak.validate_args(args)
        self.assertEqual(
            args.min_output_tok_s_by_concurrency,
            {8: 450.0, 16: 500.0},
        )

    def test_acceptance_production_rejects_weakened_certification_gates(self) -> None:
        with self.assertRaisesRegex(ValueError, "min-telemetry-coverage"):
            soak.validate_args(
                self.production_args("--min-telemetry-coverage", "0.9")
            )
        with self.assertRaisesRegex(ValueError, "comparison-window samples"):
            soak.validate_args(
                self.production_args("--min-comparison-window-samples", "31")
            )
        with self.assertRaisesRegex(ValueError, "fixed-output-length"):
            soak.validate_args(self.production_args("--no-fixed-output-length"))
        with self.assertRaisesRegex(ValueError, "throughput-degradation"):
            soak.validate_args(
                self.production_args(
                    "--max-throughput-degradation-fraction",
                    "0.051",
                )
            )
        with self.assertRaisesRegex(ValueError, "latency-degradation"):
            soak.validate_args(
                self.production_args(
                    "--max-latency-degradation-fraction",
                    "0.201",
                )
            )

    def test_invalid_resource_utilization_is_rejected(self) -> None:
        args = self.production_args("--max-kv-block-utilization", "1.1")

        with self.assertRaisesRegex(ValueError, "max-kv-block-utilization"):
            soak.validate_args(args)

    def test_acceptance_grade_requires_mtp_and_both_graph_components(self) -> None:
        production_without_mtp = self.production_args("--no-require-mtp")
        adversarial_target_only = self.adversarial_args(16)
        adversarial_target_only.expected_graph_components = ("target",)

        with self.assertRaisesRegex(ValueError, "acceptance-grade"):
            soak.validate_args(production_without_mtp)
        with self.assertRaisesRegex(ValueError, "target,dflash"):
            soak.validate_args(adversarial_target_only)

    def test_non_acceptance_calibration_can_disable_mtp_and_dflash(self) -> None:
        production = self.production_args(
            "--no-acceptance-grade",
            "--no-require-mtp",
            "--expected-graph-components",
            "target",
        )
        adversarial = self.adversarial_args(16)
        adversarial.acceptance_grade = False
        adversarial.require_mtp = False
        adversarial.expected_graph_components = ("target",)

        soak.validate_args(production)
        soak.validate_args(adversarial)

        self.assertFalse(production.acceptance_grade)
        self.assertFalse(production.require_mtp)
        self.assertFalse(
            soak.acceptance_grade_evidence(False, False, ("target",))[
                "certification_complete"
            ]
        )

    def test_production_requires_complete_first_and_final_windows(self) -> None:
        args = self.production_args(
            "--duration-seconds",
            "3600",
            "--comparison-window-seconds",
            "3600",
        )

        with self.assertRaisesRegex(ValueError, "comparison windows"):
            soak.validate_args(args)


class Part1EdgeCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_max_length_remains_a_distinct_ignore_eos_case(self) -> None:
        args = soak.build_parser().parse_args(["canary"])

        class Client:
            tokenizer = None

        client = Client()
        client.request_json = AsyncMock(return_value=({}, {"ok": True}))
        writer = AsyncMock()
        validators = {
            name: (lambda *unused: (True, "valid"))
            for name in (
                "validate_stop",
                "validate_max_length",
                "validate_regex",
                "validate_json_schema",
                "validate_tool",
                "validate_eos",
            )
        }
        with patch.multiple(soak, **validators):
            evidence = await soak.run_edge_cases(client, args, writer)

        specs = [call.args[0] for call in client.request_json.await_args_list]
        max_length_specs = [
            spec for spec in specs if spec.case_id == "edge-max-length"
        ]
        self.assertEqual(len(max_length_specs), 1)
        self.assertTrue(max_length_specs[0].extra["ignore_eos"])
        self.assertEqual(max_length_specs[0].max_tokens, args.max_length_tokens)
        self.assertIn("max_length", evidence["edges"])


class ProductionGateTests(unittest.TestCase):
    @staticmethod
    def retrieval_result(
        case_id: str,
        *,
        stage: str | None = None,
        transcript: str = "stable-transcript",
        ttft_seconds: float = 0.1,
        tpot_seconds: float = 0.01,
    ) -> soak.RequestResult:
        expected = "BEGIN-AAAAAAAAAAAA|MIDDLE-BBBBBBBBBBBB|END-CCCCCCCCCCCC"
        tags = {"expected_answer": expected}
        if stage is not None:
            tags["sentinel_stage"] = stage
        return request_result(
            case_id=case_id,
            completion_tokens=8,
            output_chunks=8,
            finish_reason="length",
            output_text=expected,
            output_transcript=transcript,
            ttft_seconds=ttft_seconds,
            tpot_seconds=tpot_seconds,
            prompt_tokens=1_024,
            context_tokens=1_024,
            tags=tags,
        )

    def test_loaded_probe_gates_exactness_semantics_and_latency(self) -> None:
        baseline = self.retrieval_result("baseline")
        loaded = self.retrieval_result(
            "loaded",
            ttft_seconds=0.2,
            tpot_seconds=0.02,
        )

        evidence = soak.production_probe_evidence(
            [loaded],
            baseline,
            None,
            0.20,
            0.98,
            1,
            1.0,
            0.1,
            2.0,
            0.5,
        )

        self.assertTrue(evidence["passed"])
        loaded.output_transcript = "different-transcript"
        self.assertFalse(
            soak.production_probe_evidence(
                [loaded],
                baseline,
                None,
                0.20,
                0.98,
                1,
                1.0,
                0.1,
                2.0,
                0.5,
            )["passed"]
        )
        loaded.output_transcript = baseline.output_transcript
        loaded.tpot_seconds = 0.04
        self.assertFalse(
            soak.production_probe_evidence(
                [loaded],
                baseline,
                None,
                0.20,
                0.98,
                1,
                1.0,
                0.1,
                2.0,
                0.5,
            )["passed"]
        )

    def test_traffic_requires_exact_fixed_length_completion(self) -> None:
        complete = [self.retrieval_result(f"traffic-{index}") for index in range(4)]

        evidence = soak.fixed_length_completion_evidence(complete, 8)

        self.assertTrue(evidence["passed"])
        complete[1].completion_tokens = 7
        complete[2].finish_reason = "stop"
        evidence = soak.fixed_length_completion_evidence(complete, 8)
        self.assertFalse(evidence["passed"])
        self.assertEqual(len(evidence["failures"]), 2)

    def test_fixed_length_traffic_is_not_production_quality_evidence(self) -> None:
        fixed_traffic = request_result(
            case_id="fixed-throughput",
            completion_tokens=32,
            output_chunks=32,
            finish_reason="length",
            output_text="alpha beta gamma delta " * 12,
            tags={"role": "traffic"},
        )
        normal_eos = request_result(
            case_id="normal-eos-sentinel",
            completion_tokens=4,
            output_chunks=4,
            finish_reason="stop",
            output_text="A concise normal response.",
            tags={"role": "semantic_sentinel"},
        )

        fixed_evidence = soak.production_output_quality_evidence(
            [fixed_traffic, normal_eos],
            None,
            0.20,
            True,
        )
        normal_evidence = soak.production_output_quality_evidence(
            [fixed_traffic, normal_eos],
            None,
            0.20,
            False,
        )

        self.assertTrue(fixed_evidence["passed"])
        self.assertEqual(fixed_evidence["quality_output_contract"], "normal_eos")
        self.assertEqual(fixed_evidence["checked_requests"], 1)
        self.assertEqual(
            fixed_evidence["excluded_fixed_length_traffic_requests"],
            1,
        )
        self.assertFalse(normal_evidence["passed"])
        self.assertEqual(normal_evidence["checked_requests"], 2)

    def test_loaded_probe_gates_completion_length_and_finish_reason(self) -> None:
        baseline = self.retrieval_result("baseline")
        loaded = self.retrieval_result("loaded")
        loaded.completion_tokens += 1

        evidence = soak.production_probe_evidence(
            [loaded],
            baseline,
            None,
            0.20,
            0.98,
            1,
            1.0,
            0.1,
            2.0,
            0.5,
        )

        self.assertTrue(evidence["exact_transcripts"])
        self.assertFalse(evidence["exact_results"])
        self.assertFalse(evidence["passed"])

        loaded.completion_tokens = baseline.completion_tokens
        loaded.finish_reason = "stop"
        evidence = soak.production_probe_evidence(
            [loaded],
            baseline,
            None,
            0.20,
            0.98,
            1,
            1.0,
            0.1,
            2.0,
            0.5,
        )
        self.assertFalse(evidence["exact_results"])
        self.assertFalse(evidence["passed"])

    def test_loaded_probe_gates_schedule_lateness(self) -> None:
        baseline = self.retrieval_result("baseline")
        loaded = self.retrieval_result("loaded")
        loaded.client_queue_seconds = 0.75

        evidence = soak.production_probe_evidence(
            [loaded],
            baseline,
            None,
            0.20,
            0.98,
            1,
            1.0,
            0.1,
            2.0,
            0.5,
        )

        self.assertFalse(evidence["checks"][0]["schedule_lateness_ok"])
        self.assertFalse(evidence["passed"])

    def test_semantic_sentinels_require_every_stage(self) -> None:
        results = [
            self.retrieval_result(f"sentinel-{stage}", stage=stage)
            for stage in ("early", "middle", "late")
        ]
        evidence = soak.production_semantic_sentinel_evidence(
            results,
            [1_024],
            ["early", "middle", "late"],
            None,
            0.20,
            0.98,
            0.5,
        )

        self.assertTrue(evidence["passed"])
        self.assertFalse(
            soak.production_semantic_sentinel_evidence(
                results[:-1],
                [1_024],
                ["early", "middle", "late"],
                None,
                0.20,
                0.98,
                0.5,
            )["passed"]
        )

    def test_semantic_sentinels_gate_schedule_lateness(self) -> None:
        results = [
            self.retrieval_result(f"sentinel-{stage}", stage=stage)
            for stage in ("early", "middle", "late")
        ]
        results[-1].client_queue_seconds = 0.75

        evidence = soak.production_semantic_sentinel_evidence(
            results,
            [1_024],
            ["early", "middle", "late"],
            None,
            0.20,
            0.98,
            0.5,
        )

        self.assertFalse(evidence["semantic_checks"][-1]["schedule_lateness_ok"])
        self.assertFalse(evidence["passed"])

    def test_semantic_sentinels_gate_full_result_signature(self) -> None:
        results = [
            self.retrieval_result(f"sentinel-{stage}", stage=stage)
            for stage in ("early", "middle", "late")
        ]
        results[-1].completion_tokens += 1

        evidence = soak.production_semantic_sentinel_evidence(
            results,
            [1_024],
            ["early", "middle", "late"],
            None,
            0.20,
            0.98,
            0.5,
        )

        self.assertFalse(evidence["fixed_seed_exact"])
        self.assertFalse(evidence["passed"])


class OfflineComparisonProvenanceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def metadata(provenance: dict | None, include_provenance: bool) -> dict:
        arguments = {
            "sampling_policy": "production",
            "seed": soak.DEFAULT_SEED,
            "temperature": soak.DEFAULT_TEMPERATURE,
            "top_p": soak.DEFAULT_TOP_P,
            "top_k": soak.DEFAULT_TOP_K,
            "min_p": soak.DEFAULT_MIN_P,
            "repetition_penalty": soak.DEFAULT_REPETITION_PENALTY,
            "requests": 1,
            "max_tokens": 8,
            "concurrencies": [1],
        }
        summary = {"passed": True}
        if include_provenance:
            summary["server_provenance"] = provenance
        return {
            "run_start": {
                "arguments": arguments,
                "policy": {
                    "temperature": soak.DEFAULT_TEMPERATURE,
                    "top_p": soak.DEFAULT_TOP_P,
                    "top_k": soak.DEFAULT_TOP_K,
                    "min_p": soak.DEFAULT_MIN_P,
                    "repetition_penalty": soak.DEFAULT_REPETITION_PENALTY,
                },
            },
            "run_summary": summary,
        }

    async def compare(
        self,
        candidate_metadata: dict,
        reference_metadata: dict,
    ) -> tuple[dict, AsyncMock]:
        sample = request_result(
            completion_tokens=8,
            output_chunks=8,
            finish_reason="length",
            output_text="stable output",
            output_transcript="stable output",
        )
        args = SimpleNamespace(
            candidate=Path("candidate.jsonl"),
            reference=Path("reference.jsonl"),
            candidate_phase="canary-c1-normal",
            reference_phase="canary-c1-normal",
            stat_max_ks=0.35,
            stat_max_js=0.20,
            require_part1_complete=False,
        )
        writer = SimpleNamespace(emit=AsyncMock())
        with patch.object(
            soak,
            "load_canary_artifact",
            side_effect=[
                ([sample], candidate_metadata),
                ([sample], reference_metadata),
            ],
        ):
            result = await soak.compare_mode(args, None, writer)
        return result, writer.emit

    async def test_compare_propagates_candidate_provenance_verbatim(self) -> None:
        candidate = {"source": "candidate", "nested": {"value": 1}}
        reference = {"source": "reference", "nested": {"value": 2}}

        result, emit = await self.compare(
            self.metadata(candidate, True),
            self.metadata(reference, True),
        )

        self.assertIs(result["server_provenance"], candidate)
        self.assertIs(emit.await_args.kwargs["server_provenance"], candidate)

    async def test_compare_omits_missing_candidate_provenance(self) -> None:
        result, emit = await self.compare(
            self.metadata(None, False),
            self.metadata(None, False),
        )

        self.assertNotIn("server_provenance", result)
        self.assertNotIn("server_provenance", emit.await_args.kwargs)

    async def test_reference_provenance_cannot_replace_candidate(self) -> None:
        reference = {"source": "reference"}

        result, emit = await self.compare(
            self.metadata(None, False),
            self.metadata(reference, True),
        )

        self.assertNotIn("server_provenance", result)
        self.assertNotIn("server_provenance", emit.await_args.kwargs)


class MultimodalOracleTests(unittest.TestCase):
    @staticmethod
    def provenance() -> dict:
        return {
            "system_info": {"build": {"git_revision": "a" * 40}},
            "process": {
                "executable_sha256": "b" * 64,
                "serve_configuration": {
                    "subcommand": "serve",
                    "model": "model",
                    "paged_attn": "auto",
                    "pa_context_len": None,
                    "pa_memory_mb": None,
                    "pa_memory_fraction": None,
                    "pa_block_size": None,
                    "pa_cache_type": "f8e4m3",
                    "max_seqs": "16",
                    "prefix_cache_n": "20",
                    "max_num_batched_tokens": None,
                    "max_prefill_chunk_tokens": None,
                    "max_decode_steps_before_prefill": None,
                    "mtp_model": "draft",
                    "mtp_n_predict": "7",
                    "mtp_draft_sampling": "auto",
                },
            },
            "gpu_driver": {
                "gpus": [
                    {
                        "uuid": "GPU-1",
                        "name": "GH200",
                        "driver_version": "580.0",
                        "memory_total_mib": 97871.0,
                    }
                ]
            },
            "realized_kv_configuration": {
                "blocks_total": 1000.0,
                "sequence_capacity": 16.0,
                "recurrent_slots_total": 17.0,
            },
            "evidence": {"complete": True},
        }

    def test_provenance_rejects_meaningful_serving_knob_mismatches(self) -> None:
        expected = self.provenance()
        mismatches = {
            "prefix_cache_n": "16",
            "mtp_n_predict": "5",
            "mtp_draft_sampling": "greedy",
        }

        for field, value in mismatches.items():
            with self.subTest(field=field):
                actual = self.provenance()
                actual["process"]["serve_configuration"][field] = value
                evidence = soak.server_provenance_match_evidence(expected, actual)
                self.assertFalse(evidence["passed"])
                self.assertFalse(evidence["fields"]["serve_configuration"]["passed"])

    def test_provenance_rejects_missing_meaningful_serving_knobs(self) -> None:
        expected = self.provenance()

        for field in (
            "prefix_cache_n",
            "mtp_n_predict",
            "mtp_draft_sampling",
        ):
            with self.subTest(field=field):
                actual = self.provenance()
                del actual["process"]["serve_configuration"][field]
                evidence = soak.server_provenance_match_evidence(expected, actual)
                self.assertFalse(evidence["passed"])
                self.assertFalse(evidence["fields"]["serve_configuration"]["passed"])

    def test_text_prerequisites_require_acceptance_grade_soak_artifacts(self) -> None:
        provenance = self.provenance()
        canary = {
            "mode": "canary",
            "passed": True,
            "coverage_complete": True,
            "server_provenance": provenance,
        }
        adversarial = {
            "mode": "adversarial",
            "passed": True,
            "max_seqs_queue_evidence": {"passed": True},
            "server_provenance": provenance,
        }
        production = {
            "mode": "production",
            "passed": True,
            "telemetry_evidence": {"passed": True},
            "acceptance_grade": True,
            "acceptance_grade_evidence": {"certification_complete": True},
            "server_provenance": provenance,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, summary in (
                ("canary", canary),
                ("adversarial", adversarial),
                ("production", production),
            ):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                paths.append(path)

            self.assertFalse(
                soak.text_prerequisite_evidence(paths, provenance)["passed"]
            )
            adversarial["acceptance_grade"] = True
            adversarial["acceptance_grade_evidence"] = {
                "certification_complete": True
            }
            paths[1].write_text(json.dumps(adversarial), encoding="utf-8")
            self.assertTrue(
                soak.text_prerequisite_evidence(paths, provenance)["passed"]
            )

            mismatched = self.provenance()
            mismatched["process"]["serve_configuration"]["max_seqs"] = "8"
            evidence = soak.text_prerequisite_evidence(paths, mismatched)
            self.assertFalse(evidence["passed"])
            self.assertFalse(
                evidence["artifacts"][0]["server_provenance_match"]["passed"]
            )

    def test_multimodal_messages_put_unique_text_before_identical_image(self) -> None:
        messages = soak.multimodal_messages(
            "data:image/png;base64,AA==",
            "unique nonce",
            "read the image",
        )

        content = messages[0]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "text"])
        self.assertEqual(content[0]["text"], "unique nonce")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,AA==")

    def test_fresh_server_requires_zero_state_and_encoder_instrumentation(self) -> None:
        snapshot = {
            **{name: 0.0 for name in soak.MULTIMODAL_FRESH_GAUGES},
            "mistralrs_encoder_cache_hits_total": 0.0,
            "mistralrs_encoder_cache_misses_total": 0.0,
        }
        evidence = soak.fresh_multimodal_server_evidence(snapshot)
        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["encoder_instrumentation_complete"])

        snapshot[soak.KV_CACHE_PREFIX_CACHED_GAUGE] = 1.0
        self.assertFalse(soak.fresh_multimodal_server_evidence(snapshot)["passed"])
        snapshot[soak.KV_CACHE_PREFIX_CACHED_GAUGE] = 0.0
        snapshot["mistralrs_encoder_cache_hits_total"] = 1.0
        self.assertFalse(soak.fresh_multimodal_server_evidence(snapshot)["passed"])

    def test_vision_capability_requires_c8_capacity(self) -> None:
        models = [
            {
                "name": "model",
                "kind": "multimodal",
                "input_modalities": ["text", "vision"],
            }
        ]
        evidence = soak.multimodal_capability_evidence(
            models,
            "model",
            {"mistralrs_sequences_capacity": 8.0},
        )
        self.assertTrue(evidence["passed"])
        self.assertFalse(
            soak.multimodal_capability_evidence(
                models,
                "model",
                {"mistralrs_sequences_capacity": 7.0},
            )["passed"]
        )

    def test_encoder_cache_transitions_exclude_paged_attention_reuse(self) -> None:
        before = {
            "mistralrs_encoder_cache_hits_total": 0.0,
            "mistralrs_encoder_cache_misses_total": 0.0,
            "mistralrs_prefix_cache_tokens_reused_total": 0.0,
        }
        cold = {
            **before,
            "mistralrs_encoder_cache_misses_total": 1.0,
        }
        cold_evidence = soak.encoder_cache_transition_evidence(
            before,
            cold,
            "cold",
        )
        self.assertTrue(cold_evidence["passed"])
        hit = {
            **cold,
            "mistralrs_encoder_cache_hits_total": 1.0,
        }
        self.assertTrue(
            soak.encoder_cache_transition_evidence(cold, hit, "hit")["passed"]
        )
        hit["mistralrs_prefix_cache_tokens_reused_total"] = 32.0
        self.assertFalse(
            soak.encoder_cache_transition_evidence(cold, hit, "hit")["passed"]
        )

    def test_image_oracle_accepts_phrase_and_attribute_alternative(self) -> None:
        valid, evidence = soak.validate_image_output(
            request_result(
                completion_tokens=12,
                output_chunks=12,
                finish_reason="length",
                output_text=(
                    "The image says MISTRAL.RS in white and orange lettering on a "
                    "black background."
                ),
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
            required_phrases=["mistral.rs"],
            expected_attributes=[("dark background", "black background")],
        )

        self.assertTrue(valid)
        self.assertTrue(evidence["sampled_output_valid"])
        self.assertTrue(evidence["semantic_oracle_valid"])
        self.assertEqual(
            evidence["expected_attribute_checks"][0]["matches"],
            ["black background"],
        )

    def test_image_oracle_rejects_missing_required_phrase(self) -> None:
        valid, evidence = soak.validate_image_output(
            request_result(
                completion_tokens=8,
                output_chunks=8,
                finish_reason="length",
                output_text="White and orange lettering appears on a black background.",
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
            required_phrases=["mistral.rs"],
            expected_attributes=[],
        )

        self.assertFalse(valid)
        self.assertFalse(evidence["required_phrase_checks"][0]["matched"])

    def test_image_oracle_retains_repetition_gate(self) -> None:
        valid, evidence = soak.validate_image_output(
            request_result(
                completion_tokens=12,
                output_chunks=12,
                finish_reason="length",
                output_text="mistral.rs loop loop loop loop loop loop loop loop",
            ),
            tokenizer=None,
            max_repeated_ngram_ratio=0.20,
            required_phrases=["mistral.rs"],
            expected_attributes=[],
        )

        self.assertFalse(valid)
        self.assertFalse(evidence["sampled_output_valid"])
        self.assertTrue(evidence["semantic_oracle_valid"])

    def test_cli_parses_attribute_alternatives_and_serializes_paths(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "multimodal",
                "--server-pid",
                "123",
                "--tokenizer",
                "tokenizer.json",
                "--image-required-phrase",
                "mistral.rs",
                "--image-expected-attribute",
                "dark background|black background",
                "--text-prerequisite-artifacts",
                "canary.json",
                "adversarial.json",
                "production.json",
            ]
        )
        soak.validate_args(args)

        self.assertIn("mistral.rs", args.image_required_phrases)
        self.assertIn(
            ("dark background", "black background"),
            args.image_expected_attributes,
        )
        serialized = soak.serialized_arguments(args)
        json.dumps(serialized)
        self.assertEqual(
            serialized["text_prerequisite_artifacts"],
            ["canary.json", "adversarial.json", "production.json"],
        )
        self.assertNotIsInstance(
            serialized["text_prerequisite_artifacts"][0],
            Path,
        )

    def test_multimodal_cli_requires_a_semantic_oracle(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "multimodal",
                "--server-pid",
                "123",
                "--tokenizer",
                "tokenizer.json",
                "--text-prerequisite-artifacts",
                "canary.json",
                "adversarial.json",
                "production.json",
            ]
        )
        args.image_required_phrases = []
        args.image_expected_attributes = []

        with self.assertRaisesRegex(ValueError, "requires --image-required-phrase"):
            soak.validate_args(args)

    def test_acceptance_multimodal_requires_tokenizer_accounting(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "multimodal",
                "--server-pid",
                "123",
                "--text-prerequisite-artifacts",
                "canary.json",
                "adversarial.json",
                "production.json",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --tokenizer"):
            soak.validate_args(args)

    def test_acceptance_multimodal_gates_cannot_be_weakened(self) -> None:
        def arguments(*extra: str):
            return soak.build_parser().parse_args(
                [
                    "multimodal",
                    "--server-pid",
                    "123",
                    "--tokenizer",
                    "tokenizer.json",
                    "--text-prerequisite-artifacts",
                    "canary.json",
                    "adversarial.json",
                    "production.json",
                    *extra,
                ]
            )

        soak.validate_args(arguments())
        cases = (
            ("--phase-duration-seconds", "299"),
            ("--concurrency", "7"),
            ("--image-interval-seconds", "31"),
            ("--min-text-requests-per-phase", "31"),
            ("--min-image-requests", "9"),
            ("--min-mixed-throughput-ratio", "0.89"),
            ("--max-mixed-tpot-ratio", "1.21"),
            ("--max-mixed-ttft-p99-seconds", "2.01"),
            ("--min-recovery-throughput-ratio", "0.94"),
            ("--min-output-event-coverage", "0.97"),
            ("--min-telemetry-coverage", "0.94"),
            ("--min-cuda-graph-replay-ratio", "0.97"),
            ("--max-kv-block-utilization", "0.96"),
            ("--telemetry-interval-seconds", "5.01"),
            ("--cleanup-timeout-seconds", "30.01"),
            ("--cleanup-poll-seconds", "0.251"),
            ("--min-mtp-acceptance-rate", "0.04"),
            ("--min-mtp-mean-advance", "1.09"),
            ("--min-mtp-proposal-depth", "1.9"),
            ("--max-sparse-verifier-fallback-ratio", "0.02"),
            ("--min-sparse-verifier-accounting-coverage", "0.97"),
        )
        for flag, value in cases:
            with self.subTest(flag=flag), self.assertRaisesRegex(
                ValueError,
                "acceptance-grade multimodal",
            ):
                soak.validate_args(arguments(flag, value))

        non_acceptance = arguments(
            "--no-acceptance-grade",
            "--no-require-mtp",
            "--phase-duration-seconds",
            "1",
            "--concurrency",
            "1",
            "--min-text-requests-per-phase",
            "1",
            "--min-image-requests",
            "1",
        )
        soak.validate_args(non_acceptance)

    def test_multimodal_phase_performance_gates_all_ratios(self) -> None:
        args = soak.build_parser().parse_args(
            [
                "multimodal",
                "--server-pid",
                "123",
                "--tokenizer",
                "tokenizer.json",
                "--text-prerequisite-artifacts",
                "canary.json",
                "adversarial.json",
                "production.json",
            ]
        )
        baseline = {
            "nominal_output_tok_s": 100.0,
            "tpot_seconds": {"p95": 0.01},
        }
        mixed_text = {
            "nominal_output_tok_s": 91.0,
            "tpot_seconds": {"p95": 0.011},
        }
        mixed_all = {"offered_ttft_seconds": {"p99": 1.9}}
        recovery = {"nominal_output_tok_s": 96.0}

        evidence = soak.multimodal_phase_performance_evidence(
            baseline,
            mixed_text,
            mixed_all,
            recovery,
            args,
        )
        self.assertTrue(evidence["passed"])
        mixed_text["nominal_output_tok_s"] = 89.0
        self.assertFalse(
            soak.multimodal_phase_performance_evidence(
                baseline,
                mixed_text,
                mixed_all,
                recovery,
                args,
            )["passed"]
        )

    def test_multimodal_nominal_window_excludes_drain_tokens(self) -> None:
        results = [
            request_result(
                case_id="inside",
                completion_tokens=128,
                output_chunks=128,
                finish_reason="length",
                started=1.0,
                ended=11.0,
                ttft_seconds=0.1,
                client_queue_seconds=0.4,
                output_event_window_counts=[90],
                output_token_window_counts=[90],
            ),
            request_result(
                case_id="queued-after-window",
                completion_tokens=128,
                output_chunks=128,
                finish_reason="length",
                started=10.1,
                ended=12.0,
                ttft_seconds=0.1,
                output_event_window_counts=[0],
                output_token_window_counts=[0],
            ),
        ]

        summary = soak.nominal_multimodal_phase_summary(
            results,
            0.0,
            10.0,
            12.1,
            8,
            "mixed",
        )

        self.assertEqual(summary["submitted_requests"], 2)
        self.assertEqual(summary["offered_in_nominal_window_requests"], 1)
        self.assertEqual(summary["first_token_in_nominal_window_requests"], 1)
        self.assertEqual(summary["nominal_output_tokens"], 90)
        self.assertEqual(summary["nominal_output_tok_s"], 9.0)
        self.assertEqual(summary["offered_ttft_seconds"]["p99"], 0.5)
        self.assertEqual(summary["requests_ending_after_nominal_window"], 2)
        self.assertTrue(summary["drain_complete"])


if __name__ == "__main__":
    unittest.main()
