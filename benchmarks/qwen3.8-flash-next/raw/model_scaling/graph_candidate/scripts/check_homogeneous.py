"""Exercise the homogeneous harness with local mocked HTTP responses only."""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "harnesses"))
import bench_homogeneous as diagnostic


class MockTokenizer:
    @classmethod
    def from_file(cls, _path):
        return cls()

    def encode(self, prompt, add_special_tokens):
        assert not add_special_tokens
        count = len(prompt.split())
        return argparse.Namespace(ids=list(range(count)))


def main():
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "requests": 0, "fail": False}

    def respond(request, timeout):
        assert timeout > 0
        body = json.loads(request.data)
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            index = state["requests"]
            state["requests"] += 1
        time.sleep(0.004)
        with lock:
            state["active"] -= 1
        text = "constant" if "Python" in body["prompt"] else f"variation {index % 2}"
        return io.BytesIO(
            json.dumps(
                {
                    "usage": {
                        "prompt_tokens": len(body["prompt"].split()),
                        "completion_tokens": 127 if state["fail"] else 128,
                    },
                    "choices": [{"text": text}],
                }
            ).encode()
        )

    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        tokenizer = folder / "tokenizer.json"
        tokenizer.write_text("mock tokenizer\n")
        paths = []
        with (
            patch.object(diagnostic.bench_serving, "Tokenizer", MockTokenizer),
            patch.object(diagnostic.shared.urllib.request, "urlopen", respond),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for name in diagnostic.PROMPT_NAMES:
                for concurrency in diagnostic.CONCURRENCIES:
                    state["peak"] = 0
                    args = argparse.Namespace(
                        base_url="http://mock.invalid",
                        label="mock",
                        tokenizer=tokenizer,
                        output=folder / f"{name}_c{concurrency}.json",
                        prompt_name=name,
                        concurrency=concurrency,
                        timeout=2,
                    )
                    diagnostic.measure(args)
                    data = json.loads(args.output.read_text())
                    summary = diagnostic.validate(data)
                    assert state["active"] == 0 and state["peak"] <= concurrency
                    assert state["peak"] == concurrency
                    assert len(data["runs"]) == 5
                    assert all(
                        len(run["requests"]) == diagnostic.REQUEST_COUNTS[concurrency]
                        for run in data["runs"]
                    )
                    assert len(summary["mean_active_requests"]["samples"]) == 3
                    assert data["runs"][0]["waves"][0]["agreement"][
                        "all_outputs_identical"
                    ] is (name == "python" or concurrency == 1)
                    paths.append(args.output)
            output = folder / "summary.json"
            diagnostic.summarize(argparse.Namespace(inputs=paths, output=output))
            summary = json.loads(output.read_text())
            assert len(summary["rows"]) == 2
            python_row, math_row = summary["rows"]
            assert python_row["c1_c8_equal_output_pair_fraction"] == 1
            assert math_row["c1_c8_equal_output_pair_fraction"] == 0.5
            for row in summary["rows"]:
                diagnostic.equal_number(
                    row["aggregate_c8_over_c1"],
                    row["c8"]["aggregate_output_tokens_per_second"]["mean"]
                    / row["c1"]["aggregate_output_tokens_per_second"]["mean"],
                )
            checks.extend(
                [
                    "All four cases complete with exact 2 warmups/3 trials and 8/24 requests",
                    "Mock peak request concurrency is exactly 1/8; no request overlaps phases",
                    "Recomputed token/latency/active-integral metrics and nonoverlapping finite waves pass",
                    "All output hashes preserved; constant outputs give agreement 1, alternating outputs give cross-C agreement 0.5",
                    "Same-prompt C8/C1 summary ratios independently checked",
                ]
            )
            for field in ("output_text_sha256", "response_sha256"):
                corrupt = copy.deepcopy(data)
                corrupt["runs"][0]["requests"][0][field] = "bad"
                try:
                    diagnostic.validate(corrupt)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(f"Accepted corrupt {field}")
            state["fail"] = True
            args.output = folder / "failure.json"
            try:
                diagnostic.measure(args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Accepted short output")
            failed = json.loads(args.output.read_text())
            assert failed["complete"] is False
            assert len(failed["runs"]) == 1
            assert len(failed["runs"][0]["requests"]) == 8
            assert all(item["error"] for item in failed["runs"][0]["requests"])
            assert diagnostic.bench_serving.PROMPTS.keys() != {args.prompt_name}
            checks.append(
                "Hash corruption rejected; short output persists failed raw wave, stops next wave, restores original prompts"
            )
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(
        json.dumps(
            {
                "complete": True,
                "scope": "CPU mock only; no server/GPU/build",
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"complete": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
