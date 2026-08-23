import json
import unittest
from pathlib import Path

from scripts import production_soak as soak


def request_result(
    *,
    completion_tokens: int,
    output_chunks: int,
    finish_reason: str,
    ok: bool = True,
    output_text: str = "output",
    case_id: str = "coverage",
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
        seed=1,
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
        reasoning_text="",
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

    def test_whole_device_memory_is_used_when_pid_metric_is_unavailable(self) -> None:
        snapshots = self.healthy_snapshots()
        for _, _, process in snapshots:
            process["process_gpu_memory_used_mib"] = None
        snapshots[1][2]["gpus"][0]["memory_used_mib"] = 91_500.0

        evidence = self.evidence(snapshots)

        self.assertTrue(evidence["gpu_memory"]["passed"])
        self.assertEqual(evidence["gpu_memory"]["source"], "whole_device")

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
    def test_c3_measurement_excludes_partial_drain_batch(self) -> None:
        specs = [
            soak.RequestSpec(
                case_id=f"case-{index}",
                messages=[],
                max_tokens=1,
                seed=index,
            )
            for index in range(16)
        ]

        measured = soak.full_batch_specs(specs, 3)

        self.assertEqual(len(measured), 15)
        self.assertEqual(len(measured) % 3, 0)

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


class ArgumentValidationTests(unittest.TestCase):
    @staticmethod
    def adversarial_args(cancel_requests: int):
        return soak.build_parser().parse_args(
            [
                "adversarial",
                "--tokenizer",
                "tokenizer.json",
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
        soak.validate_args(self.adversarial_args(16))
        with self.assertRaisesRegex(ValueError, "cannot exceed --max-seqs"):
            soak.validate_args(self.adversarial_args(17))

    def test_prefix_pressure_headroom_default_validates(self) -> None:
        args = self.adversarial_args(16)

        soak.validate_args(args)
        self.assertEqual(
            args.prefix_pressure_kv_headroom_fraction,
            soak.DEFAULT_PREFIX_PRESSURE_KV_HEADROOM_FRACTION,
        )

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
        self.assertEqual(
            args.max_process_rss_drift_mib,
            soak.DEFAULT_MAX_PROCESS_RSS_DRIFT_MIB,
        )
        self.assertEqual(
            args.max_gpu_memory_high_water_mib,
            soak.DEFAULT_MAX_GPU_MEMORY_HIGH_WATER_MIB,
        )
        self.assertEqual(args.max_recurrent_slot_utilization, 1.0)

    def test_invalid_resource_utilization_is_rejected(self) -> None:
        args = self.production_args("--max-kv-block-utilization", "1.1")

        with self.assertRaisesRegex(ValueError, "max-kv-block-utilization"):
            soak.validate_args(args)


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


class MultimodalOracleTests(unittest.TestCase):
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
                "--image",
                "website/public/og.png",
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

        self.assertEqual(args.image_required_phrases, ["mistral.rs"])
        self.assertEqual(
            args.image_expected_attributes,
            [("dark background", "black background")],
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
                "--image",
                "website/public/og.png",
                "--text-prerequisite-artifacts",
                "canary.json",
                "adversarial.json",
                "production.json",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --image-required-phrase"):
            soak.validate_args(args)


if __name__ == "__main__":
    unittest.main()
