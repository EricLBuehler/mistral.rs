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
    tags: dict | None = None,
) -> soak.RequestResult:
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
        output_event_times=output_event_times or [],
    )


class OutputEventCoverageTests(unittest.TestCase):
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

    def test_post_admission_cancellations_fit_sequence_capacity(self) -> None:
        soak.validate_args(self.adversarial_args(16))
        with self.assertRaisesRegex(ValueError, "cannot exceed --max-seqs"):
            soak.validate_args(self.adversarial_args(17))


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
