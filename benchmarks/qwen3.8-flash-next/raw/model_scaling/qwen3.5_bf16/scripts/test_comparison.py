"""CPU-only synthetic regressions for comparison gates and archive sanitization."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import compare_bf16 as comparison
import plan_comparison_archive as archive


def fixture(concurrency=1):
    count = 8 if concurrency == 1 else 24
    prompts = {f"p{i}": f"Prompt {i}" for i in range(8)}
    counts = {key: 2 for key in prompts}
    runs = []
    for trial in range(7):
        duration = 1000.0 if trial < 2 else float(trial - 1) * 3
        requests = []
        elapsed = count / concurrency * duration
        for index in range(count):
            name = f"p{index % 8}"
            prompt = prompts[name]
            start = index // concurrency * duration
            requests.append(
                {
                    "request_index": index,
                    "error": None,
                    "prompt_name": name,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "request": {
                        "model": "default",
                        "prompt": prompt,
                        "max_tokens": 128,
                        "temperature": 0.0,
                        "seed": comparison.SEED,
                        "ignore_eos": True,
                        "cache_prompt": False,
                    },
                    "response": {
                        "usage": {"completion_tokens": 128, "prompt_tokens": 2},
                        "choices": [{"text": "Synthetic output"}],
                    },
                    "started_seconds": start,
                    "finished_seconds": start + duration,
                    "wall_seconds": duration,
                    "output_tokens_per_second": 128 / duration,
                }
            )
        runs.append(
            {
                "complete": True,
                "trial": trial,
                "warmup": trial < 2,
                "requests": requests,
                "wall_seconds": elapsed,
                "completed_output_tokens": count * 128,
                "aggregate_output_tokens_per_second": concurrency * 128 / duration,
                "latency_weighted_request_output_tokens_per_second": 128 / duration,
                "mean_active_requests": float(concurrency),
                "mean_request_wall_seconds": duration,
                "mean_request_output_tokens_per_second": 128 / duration,
            }
        )
    return {
        "complete": True,
        "settings": {
            "concurrencies": [concurrency],
            "modes": ["closed-loop"],
            "trials": 5,
            "warmup": 2,
            "requests": count,
            "max_tokens": 128,
            "expected_prompt_tokens": counts,
        },
        "cases": {
            f"closed-loop_c{concurrency}": {
                "mode": "closed-loop",
                "concurrency": concurrency,
                "runs": runs,
            }
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_phase_conditions_keep_global_swap_and_missing_clocks_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stamp = datetime(2026, 9, 30, 18, 0, 1, tzinfo=timezone.utc).astimezone()
            (root / "gpu.csv").write_text(
                "timestamp, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W]\n"
                + stamp.strftime("%Y/%m/%d %H:%M:%S.%f")
                + ", 2400, [N/A], 60.5\n"
            )
            before = {
                "page_size_bytes": 4096,
                "pswpin_pages": 100,
                "pswpout_pages": 20,
                "model": {"VmSwap_KiB": 204800},
                "system": {"MemAvailable_KiB": 123456},
            }
            after = deepcopy(before)
            after["pswpin_pages"] += 3
            for point, value in (("before", before), ("after", after)):
                (root / f"c1.memory.{point}.json").write_text(json.dumps(value))
            metadata = {
                "phases": [
                    {
                        "name": "c1",
                        "started": {"utc": "2026-09-30T18:00:00+00:00"},
                        "finished": {"utc": "2026-09-30T18:00:02+00:00"},
                    }
                ]
            }
            result = comparison.operating_conditions(root, metadata, "vllm")["phases"][
                "c1"
            ]
            self.assertEqual(
                result["global_swap"]["pswpin_pages"]["delta_bytes"], 12288
            )
            self.assertEqual(result["global_swap"]["pswpout_pages"]["delta_bytes"], 0)
            self.assertEqual(
                result["process_VmSwap"], {"before_KiB": 204800, "after_KiB": 204800}
            )
            self.assertNotIn("clocks.current.memory [MHz]", result["gpu_ranges"])
            self.assertEqual(
                result["gpu_ranges"]["clocks.current.sm [MHz]"]["min"], 2400
            )
            self.assertIn("child processes", result["process_scope"])

    def test_warmups_excluded_and_all_five_rates_retained(self):
        for concurrency in (1, 6, 8):
            result, protocol, outputs = comparison.validate_phase(
                fixture(concurrency), concurrency
            )
            expected = [concurrency * 128 / duration for duration in (3, 6, 9, 12, 15)]
            actual = result["statistics"]["aggregate_output_tokens_per_second"]
            self.assertEqual(actual["samples"], expected)
            self.assertAlmostEqual(actual["mean"], statistics.mean(expected))
            self.assertAlmostEqual(actual["sample_stddev"], statistics.stdev(expected))
            self.assertEqual(len(outputs), (8 if concurrency == 1 else 24) * 5)
            self.assertGreater(len(protocol), len(outputs))

    def test_incomplete_extra_and_reordered_trials_rejected(self):
        for mutation in (
            lambda x: x.update(complete=False),
            lambda x: x["cases"]["closed-loop_c1"]["runs"].pop(),
            lambda x: x["cases"]["closed-loop_c1"]["runs"].append(
                deepcopy(x["cases"]["closed-loop_c1"]["runs"][-1])
            ),
            lambda x: x["cases"]["closed-loop_c1"]["runs"][2].update(trial=3),
        ):
            data = fixture()
            mutation(data)
            with self.assertRaises(ValueError):
                comparison.validate_phase(data, 1)

    def test_protocol_response_and_arithmetic_errors_rejected(self):
        mutations = [
            lambda r: r["request"].update(temperature=0.7),
            lambda r: r["request"].update(logprobs=1),
            lambda r: r["response"]["usage"].update(completion_tokens=127),
            lambda r: r["response"]["usage"].update(prompt_tokens=3),
            lambda r: r["response"]["usage"].update(
                prompt_tokens_details={"cached_tokens": 1}
            ),
            lambda r: r.update(prompt_sha256="wrong"),
            lambda r: r.update(wall_seconds=4.0),
            lambda r: r.update(error={"message": "failure"}),
        ]
        for mutation in mutations:
            data = fixture()
            mutation(data["cases"]["closed-loop_c1"]["runs"][2]["requests"][0])
            with self.assertRaises(ValueError):
                comparison.validate_phase(data, 1)
        data = fixture()
        data["cases"]["closed-loop_c1"]["runs"][2][
            "aggregate_output_tokens_per_second"
        ] += 1
        with self.assertRaises(ValueError):
            comparison.validate_phase(data, 1)

    def test_lifecycle_gate_precedes_reading_partial_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text(json.dumps({"complete": False}))
            with self.assertRaisesRegex(ValueError, "lifecycle incomplete"):
                comparison.read_engine(path, "mistralrs")

    def test_redaction_preserves_token_counts_and_hashes(self):
        data = {
            "api_key": "test-value",
            "prompt_tokens": 17,
            "tokenizer_sha256": "a" * 64,
            "env": ["HF_TOKEN" + "=not-a-real-token"],
            "command": ["serve", "--api-key", "fake-value"],
        }
        redacted = archive.sanitize(data)
        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertEqual(redacted["env"], ["[REDACTED]"])
        self.assertEqual(redacted["command"][-1], "[REDACTED]")
        self.assertEqual(redacted["prompt_tokens"], 17)
        self.assertEqual(redacted["tokenizer_sha256"], "a" * 64)

    def test_cache_and_tokenizer_are_external_even_when_small(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tokenizer.json"
            path.write_text("{}")
            self.assertIsNotNone(
                archive.excluded_reason(path, Path("checkpoint/tokenizer.json"))
            )
            self.assertIsNotNone(archive.excluded_reason(path, Path("cache/tiny.json")))


if __name__ == "__main__":
    unittest.main()
