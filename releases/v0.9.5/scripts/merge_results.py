#!/usr/bin/env python3
"""Validate retained captures and regenerate the release's tables without inference."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"
LENGTHS = (512, 2048, 8192)
OUTPUT_TOKENS = 128
GGUF_REPETITIONS = 3
FP4_REPETITIONS = 3
FP4_REQUESTS_PER_ROUND = 6
RATE_KEYS = ("prefill_tokens_s", "decode_tokens_s")
CONCURRENCIES = (1, 2, 4, 8)
CONCURRENCY_METRICS = ("output_tokens_s", "ttft_ms", "decode_tokens_s")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
        actual,
        expected,
    )


def gains(candidate, baseline):
    return {key: 100 * (candidate[key] / baseline[key] - 1) for key in RATE_KEYS}


def normalized(comparison, engine, run, source, index, row):
    return {
        "comparison": comparison,
        "engine": engine,
        "run": run,
        "source": str(source),
        "source_row": index,
        **row,
    }


def gguf(manifest, results):
    runs = []
    fixture_hash = sha256(RAW / manifest["workload"]["prompts"])
    for run in manifest["runs"]:
        source = Path(run["result"])
        capture = read_json(RAW / source)
        assert capture.get("complete", "median" in capture)
        assert capture["prompts_sha256"] == fixture_hash
        assert len(capture["results"]) == len(LENGTHS) * (GGUF_REPETITIONS + 1)
        for index, row in enumerate(capture["results"]):
            times = row["token_timestamps_seconds"]
            assert (
                row["output_tokens"]
                == row["text_chunks"]
                == len(times)
                == OUTPUT_TOKENS
            )
            assert times == sorted(times)
            close(row["ttft_seconds"], times[0])
            close(row["tpot_seconds"], (times[-1] - times[0]) / (OUTPUT_TOKENS - 1))
            close(row["prompt_tokens_per_second"], row["prompt_tokens"] / times[0])
            close(row["decode_tokens_per_second"], 1 / row["tpot_seconds"])
            results.append(
                normalized(
                    "gguf",
                    run["engine"],
                    source.stem,
                    source,
                    index,
                    {
                        "phase": "warmup" if row["warmup"] else "measured",
                        "request_id": str(row["index"]),
                        "prompt_tokens": row["prompt_tokens"],
                        "output_tokens": row["output_tokens"],
                        "ttft_ms": row["ttft_seconds"] * 1000,
                        "tpot_ms": row["tpot_seconds"] * 1000,
                        "prefill_tokens_s": row["prompt_tokens_per_second"],
                        "decode_tokens_s": row["decode_tokens_per_second"],
                    },
                )
            )
        cells = {}
        for length in LENGTHS:
            all_rows = [r for r in capture["results"] if r["prompt_tokens"] == length]
            rows = [r for r in all_rows if not r["warmup"]]
            assert (
                len(rows) == GGUF_REPETITIONS and len(all_rows) == GGUF_REPETITIONS + 1
            )
            values = {
                key: statistics.median(r[key] for r in rows)
                for key in capture["median"][str(length)]
            }
            assert values == capture["median"][str(length)]
            cells[length] = {
                "requests": len(rows),
                "ttft_ms": values["ttft_seconds"] * 1000,
                "tpot_ms": values["tpot_seconds"] * 1000,
                "prefill_tokens_s": values["prompt_tokens_per_second"],
                "decode_tokens_s": values["decode_tokens_per_second"],
            }
        runs.append({"source": str(source), "engine": run["engine"], "cells": cells})
    (candidate,) = [run for run in runs if run["engine"] == "mistral.rs"]
    controls = [run for run in runs if run["engine"] == "llama.cpp"]
    assert len(controls) == 4
    cells = []
    for length in LENGTHS:
        winners = {
            key: max(controls, key=lambda r: r["cells"][length][key])
            for key in RATE_KEYS
        }
        baseline = {key: run["cells"][length][key] for key, run in winners.items()}
        baseline["sources"] = {key: run["source"] for key, run in winners.items()}
        ours = candidate["cells"][length]
        cells.append(
            {
                "prompt_tokens": length,
                "mistralrs": ours,
                "baseline": baseline,
                "gain_percent": gains(ours, baseline),
            }
        )
    return {
        "aggregation": manifest["aggregation"],
        "candidate_source": candidate["source"],
        "cells": cells,
        "runs": runs,
    }


def fp4_environment(root, summary):
    start, end = (
        datetime.fromisoformat(summary[key]) for key in ("started_at", "completed_at")
    )
    samples = read_jsonl(root / "memory.jsonl")
    inside = [r for r in samples if start <= datetime.fromisoformat(r["time"]) <= end]
    before = [r for r in samples if datetime.fromisoformat(r["time"]) <= start][-1]
    after = [r for r in samples if datetime.fromisoformat(r["time"]) >= end][0]
    gpu = [list(map(float, r["gpu"].split(","))) for r in inside]
    return {
        "started_at": summary["started_at"],
        "completed_at": summary["completed_at"],
        "memory": {
            "from": before["time"],
            "to": after["time"],
            "swap_in_mib": (after["pswpin"] - before["pswpin"])
            * before["page_size"]
            / 2**20,
            "swap_out_mib": (after["pswpout"] - before["pswpout"])
            * before["page_size"]
            / 2**20,
            "minimum_available_gib": min(r["mem_available_kib"] for r in inside)
            / 2**20,
        },
        "gpu": {
            "temperature_c_mean": statistics.mean(r[0] for r in gpu),
            "sm_clock_mhz_mean": statistics.mean(r[1] for r in gpu),
            "power_w_mean": statistics.mean(r[2] for r in gpu),
        },
    }


def fp4_correctness(root, reference):
    answers = read_json(root / "reference.json")
    assert len(answers) == len(reference) == 4
    for answer, expected in zip(answers, reference):
        assert answer["id"] == expected["id"]
        assert answer["request"]["messages"] == expected["request"]["messages"]
        assert (
            answer["response"]["choices"][0]["message"]["content"]
            == expected["response"]["choices"][0]["message"]["content"]
        )
    repeats = read_jsonl(root / "repeat-logprobs/responses.jsonl")
    assert len(repeats) == len(LENGTHS) * FP4_REPETITIONS
    for length in LENGTHS:
        rows = [r for r in repeats if r["input_tokens"] == length]
        assert sorted(r["repetition"] for r in rows) == list(range(FP4_REPETITIONS))
        first = rows[0]["response"]["choices"][0]
        assert len(first["logprobs"]["content"]) == OUTPUT_TOKENS
        for row in rows[1:]:
            assert row["prompt_sha256"] == rows[0]["prompt_sha256"]
            choice = row["response"]["choices"][0]
            assert choice["message"]["content"] == first["message"]["content"]
            assert choice["logprobs"] == first["logprobs"]
    return {
        "reference_answers_matching": 4,
        "repeat_requests": len(repeats),
        "identical_text_and_logprob_comparisons": 6,
    }


def fp4(manifest, results):
    fixture_path = RAW / manifest["workload"]["prompts"]
    fixture = read_json(fixture_path)
    prompts = {p["id"]: p for r in fixture["rounds"] for p in r["prompts"]}
    reference = read_json(RAW / manifest["runs"]["vllm"] / "reference.json")
    runs = {}
    for engine, relative_root in manifest["runs"].items():
        root = RAW / relative_root
        summary = read_json(root / "measured/summary.json")
        assert summary["success"] and summary["workload_sha256"] == sha256(fixture_path)
        conditioning = {
            r["request_id"]: r for r in read_jsonl(root / "conditioning/requests.jsonl")
        }
        rows = read_jsonl(root / "measured/requests.jsonl")
        assert len(conditioning) == len(rows) == len(prompts) == 57
        assert len({r["request_id"] for r in rows}) == len(rows)
        assert len(read_jsonl(root / "independent-prewarm/requests.jsonl")) == 5
        repeat_count = 0
        for index, row in enumerate(rows):
            assert (
                row["success"]
                and row["http_status"] == 200
                and row["finish_reason"] == "length"
            )
            assert row["concurrency"] == 1
            assert (
                row["usage"]["completion_tokens"]
                == row["output_tokens"]
                == OUTPUT_TOKENS
            )
            assert row["usage"]["prompt_tokens"] == row["input_tokens"]
            prompt = prompts[row["request_id"]]
            assert prompt["input_tokens"] == row["input_tokens"]
            assert prompt["rendered_sha256"] == row["prompt_sha256"]
            close(
                row["tpot_ms"], (row["e2e_ms"] - row["ttft_ms"]) / (OUTPUT_TOKENS - 1)
            )
            previous = conditioning[row["request_id"]]
            assert (
                previous["success"]
                and previous["prompt_sha256"] == row["prompt_sha256"]
            )
            if row["phase"] == "measured":
                assert previous["text"] == row["text"]
                repeat_count += 1
            results.append(
                normalized(
                    "fp4",
                    engine,
                    engine,
                    Path(relative_root) / "measured/requests.jsonl",
                    index,
                    {
                        "phase": row["phase"],
                        "request_id": row["request_id"],
                        "repetition": row["repetition"],
                        "prompt_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "prompt_sha256": row["prompt_sha256"],
                        "ttft_ms": row["ttft_ms"],
                        "tpot_ms": row["tpot_ms"],
                        "prefill_tokens_s": row["input_tokens"] * 1000 / row["ttft_ms"],
                        "decode_tokens_s": 1000 / row["tpot_ms"],
                    },
                )
            )
        cells = {}
        for length in LENGTHS:
            measured = [
                r
                for r in rows
                if r["phase"] == "measured" and r["input_tokens"] == length
            ]
            assert len(measured) == FP4_REPETITIONS * FP4_REQUESTS_PER_ROUND
            ttft = statistics.mean(r["ttft_ms"] for r in measured)
            tpot = statistics.mean(r["tpot_ms"] for r in measured)
            rounds = []
            for repetition in range(FP4_REPETITIONS):
                subset = [r for r in measured if r["repetition"] == repetition]
                assert len(subset) == FP4_REQUESTS_PER_ROUND
                rounds.append(
                    {
                        "repetition": repetition,
                        "prefill_tokens_s": length
                        * 1000
                        / statistics.mean(r["ttft_ms"] for r in subset),
                        "decode_tokens_s": 1000
                        / statistics.mean(r["tpot_ms"] for r in subset),
                    }
                )
            cells[length] = {
                "requests": len(measured),
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "prefill_tokens_s": length * 1000 / ttft,
                "decode_tokens_s": 1000 / tpot,
                "rounds": rounds,
            }
        runs[engine] = {
            "source": relative_root,
            "cells": cells,
            "validation": {
                **fp4_correctness(root, reference),
                "conditioning_outputs_repeated": repeat_count,
            },
            "environment": fp4_environment(root, summary),
        }
    cells = []
    for length in LENGTHS:
        ours = runs["mistralrs"]["cells"][length]
        baseline = runs["vllm"]["cells"][length]
        cells.append(
            {
                "prompt_tokens": length,
                "mistralrs": ours,
                "baseline": baseline,
                "gain_percent": gains(ours, baseline),
                "round_gain_percent": [
                    gains(a, b) for a, b in zip(ours["rounds"], baseline["rounds"])
                ],
            }
        )
    return {
        "aggregation": manifest["aggregation"],
        "precision_note": manifest["precision_note"],
        "cells": cells,
        "runs": runs,
    }


def fp4_concurrency(manifest, results):
    fixture_path = RAW / manifest["workload"]["prompts"]
    fixture = read_json(fixture_path)
    prompts = {p["id"]: p for r in fixture["rounds"] for p in r["prompts"]}
    runs = {}
    references = {}
    for engine, relative_root in manifest["runs"].items():
        root = RAW / relative_root
        summary = read_json(root / "measured/summary.json")
        assert summary["success"] and summary["workload_sha256"] == sha256(fixture_path)
        rows = read_jsonl(root / "measured/requests.jsonl")
        rounds = read_jsonl(root / "measured/rounds.jsonl")
        conditioning = read_jsonl(root / "conditioning/requests.jsonl")
        assert len(rows) == len(conditioning) == len(prompts) == 159
        assert len({r["request_id"] for r in rows}) == len(rows)
        assert all(r["success"] for r in conditioning)
        for index, row in enumerate(rows):
            assert row["success"] and row["http_status"] == 200
            assert row["finish_reason"] == "length"
            assert row["usage"]["prompt_tokens"] == row["input_tokens"] == 2048
            assert (
                row["usage"]["completion_tokens"]
                == row["output_tokens"]
                == OUTPUT_TOKENS
            )
            assert row["prompt_sha256"] == prompts[row["request_id"]]["rendered_sha256"]
            close(
                row["tpot_ms"], (row["e2e_ms"] - row["ttft_ms"]) / (OUTPUT_TOKENS - 1)
            )
            times = row["content_chunk_times_ms"]
            assert times and times == sorted(times)
            close(times[0], row["ttft_ms"])
            assert times[-1] <= row["e2e_ms"]
            results.append(
                normalized(
                    "fp4_concurrency",
                    engine,
                    engine,
                    Path(relative_root) / "measured/requests.jsonl",
                    index,
                    {
                        "phase": row["phase"],
                        "request_id": row["request_id"],
                        "repetition": row["repetition"],
                        "concurrency": row["concurrency"],
                        "prompt_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "prompt_sha256": row["prompt_sha256"],
                        "ttft_ms": row["ttft_ms"],
                        "tpot_ms": row["tpot_ms"],
                        "decode_tokens_s": 1000 / row["tpot_ms"],
                    },
                )
            )
        cells = {}
        for concurrency in CONCURRENCIES:
            measured = [
                r
                for r in rows
                if r["phase"] == "measured" and r["concurrency"] == concurrency
            ]
            measured_rounds = [
                r
                for r in rounds
                if r["phase"] == "measured" and r["concurrency"] == concurrency
            ]
            count = max(FP4_REQUESTS_PER_ROUND, FP4_REPETITIONS * concurrency)
            assert len(measured) == FP4_REPETITIONS * count
            assert sorted(r["repetition"] for r in measured_rounds) == list(
                range(FP4_REPETITIONS)
            )
            for round_ in measured_rounds:
                assert round_["success"] and round_["failed"] == 0
                assert round_["requested"] == round_["completed"] == count
                subset = [r for r in measured if r["round_id"] == round_["id"]]
                assert len(subset) == count
                close(
                    round_["output_tokens_per_second"],
                    sum(r["usage"]["completion_tokens"] for r in subset)
                    / round_["duration_seconds"],
                )
            ttft = statistics.mean(r["ttft_ms"] for r in measured)
            tpot = statistics.mean(r["tpot_ms"] for r in measured)
            cells[concurrency] = {
                "requests": len(measured),
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "output_tokens_s": statistics.median(
                    r["output_tokens_per_second"] for r in measured_rounds
                ),
                "decode_tokens_s": 1000 / tpot,
                "rounds": [
                    {
                        "repetition": r["repetition"],
                        "output_tokens_s": r["output_tokens_per_second"],
                    }
                    for r in measured_rounds
                ],
            }
        references[engine] = {}
        for filename, count in (("reference.json", 4), ("reference-c8.json", 8)):
            answers = read_json(root / filename)
            assert len(answers) == count
            references[engine][filename] = [
                (
                    r["id"],
                    r["request"]["messages"],
                    r["response"]["choices"][0]["message"]["content"],
                )
                for r in answers
            ]
        runs[engine] = {"source": relative_root, "cells": cells}
    assert references["mistralrs"] == references["vllm"]
    cells = []
    for concurrency in CONCURRENCIES:
        ours = runs["mistralrs"]["cells"][concurrency]
        baseline = runs["vllm"]["cells"][concurrency]
        gains = {
            key: 100 * (ours[key] / baseline[key] - 1) for key in CONCURRENCY_METRICS
        }
        gains["ttft_ms"] *= -1
        cells.append(
            {
                "concurrency": concurrency,
                "prompt_tokens": 2048,
                "mistralrs": ours,
                "baseline": baseline,
                "gain_percent": gains,
            }
        )
    return {
        "aggregation": manifest["aggregation"],
        "precision_note": manifest["precision_note"],
        "cells": cells,
        "runs": runs,
        "validation": {
            "serial_reference_matches": 4,
            "concurrent_reference_matches": 8,
            "measured_requests_per_engine": 144,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generated artifacts without writing them.",
    )
    args = parser.parse_args()
    manifest = read_json(RAW / "run_manifest.json")
    for name, expected in manifest["sha256"].items():
        assert sha256(RAW / name) == expected, f"Source checksum mismatch: {name}"
    results = []
    summary = {
        "schema_version": 1,
        "manifest_sha256": sha256(RAW / "run_manifest.json"),
        "comparisons": {
            "gguf": gguf(manifest["comparisons"]["gguf"], results),
            "fp4": fp4(manifest["comparisons"]["fp4"], results),
            "fp4_concurrency": fp4_concurrency(
                manifest["comparisons"]["fp4_concurrency"], results
            ),
        },
    }
    csv_file = io.StringIO(newline="")
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(
        [
            "comparison",
            "prompt_tokens",
            "concurrency",
            "metric",
            "unit",
            "mistralrs",
            "baseline",
            "gain_percent",
        ]
    )
    for name, comparison in summary["comparisons"].items():
        for cell in comparison["cells"]:
            for key in CONCURRENCY_METRICS if name == "fp4_concurrency" else RATE_KEYS:
                writer.writerow(
                    [
                        name,
                        cell["prompt_tokens"],
                        cell.get("concurrency", 1),
                        key,
                        "ms" if key == "ttft_ms" else "tokens/s",
                        cell["mistralrs"][key],
                        cell["baseline"][key],
                        cell["gain_percent"][key],
                    ]
                )
    outputs = {
        "summary.json": json.dumps(summary, indent=2) + "\n",
        "summary.csv": csv_file.getvalue(),
        "results.jsonl": "".join(
            json.dumps(r, separators=(",", ":")) + "\n" for r in results
        ),
    }
    for name, content in outputs.items():
        if args.check:
            assert (RAW / name).read_text() == content, f"Stale generated file: {name}"
        else:
            (RAW / name).write_text(content)
    print(
        f"Validated {len(manifest['sha256'])} source files and {len(results)} request records; {'checked' if args.check else 'wrote'} summaries."
    )


if __name__ == "__main__":
    main()
