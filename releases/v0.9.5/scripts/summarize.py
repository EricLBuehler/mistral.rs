#!/usr/bin/env python3
"""Validate stored C1 measurements and regenerate their canonical summary."""

import hashlib
import json
from pathlib import Path
import statistics


RAW = Path(__file__).resolve().parents[1] / "raw"
METRICS = (
    "ttft_seconds",
    "prompt_tokens_per_second",
    "tpot_seconds",
    "decode_tokens_per_second",
)


def main():
    manifest = json.loads((RAW / "run-manifest.json").read_text())
    fixture_hash = hashlib.sha256((RAW / "prompts.json").read_bytes()).hexdigest()
    assert fixture_hash == manifest["workload"]["fixture_sha256"]
    cells = []
    for cell in manifest["cells"]:
        result = json.loads((RAW / cell["result"]).read_text())
        assert result.get("complete", "median" in result)
        assert result["prompts_sha256"] == fixture_hash
        medians = {}
        for length in manifest["workload"]["prompt_lengths"]:
            all_rows = [r for r in result["results"] if r["prompt_tokens"] == length]
            rows = [r for r in all_rows if not r["warmup"]]
            assert len(rows) == manifest["workload"]["measured_requests_per_length"]
            assert (
                len(all_rows) - len(rows)
                == manifest["workload"]["warmup_requests_per_length"]
            )
            for row in all_rows:
                assert row["output_tokens"] == manifest["workload"]["output_tokens"]
                assert row["text_chunks"] == row["output_tokens"]
                assert len(row["token_timestamps_seconds"]) == row["output_tokens"]
                assert row["token_timestamps_seconds"] == sorted(
                    row["token_timestamps_seconds"]
                )
            medians[str(length)] = {
                metric: statistics.median(row[metric] for row in rows)
                for metric in METRICS
            }
        assert medians == result["median"]
        cells.append({**cell, "median": medians})
    comparisons = []
    different_quantization = []
    for candidate in cells:
        if candidate["engine"] != "mistral.rs":
            continue
        for baseline in cells:
            if (
                baseline["engine"] != "llama.cpp"
                or baseline["chunk_tokens"] != candidate["chunk_tokens"]
            ):
                continue
            comparison = {
                "candidate": candidate["result"],
                "baseline": baseline["result"],
                "chunk_tokens": candidate["chunk_tokens"],
                "candidate_quantization": candidate["quantization"],
                "baseline_quantization": baseline["quantization"],
                "percent_difference": {
                    length: {
                        metric: 100
                        * (values[metric] / baseline["median"][length][metric] - 1)
                        for metric in (
                            "prompt_tokens_per_second",
                            "decode_tokens_per_second",
                        )
                    }
                    for length, values in candidate["median"].items()
                },
            }
            if candidate["quantization"] == baseline["quantization"]:
                comparisons.append(comparison)
            else:
                comparison["caveat"] = (
                    "Different weight quantization; does not establish equal model quality "
                    "or isolate engine/kernel performance."
                )
                different_quantization.append(comparison)
    current = [
        cell
        for cell in cells
        if cell["engine"] == "mistral.rs"
        and cell.get("measurement_generation") == "current"
    ]
    best_control_comparisons = []
    for candidate in current:
        controls = [
            cell
            for cell in cells
            if cell["engine"] == "llama.cpp"
            and cell["quantization"] == candidate["quantization"]
        ]
        for length, medians in candidate["median"].items():
            metrics = {}
            for metric in ("prompt_tokens_per_second", "decode_tokens_per_second"):
                control = max(controls, key=lambda cell: cell["median"][length][metric])
                baseline = control["median"][length][metric]
                metrics[metric] = {
                    "candidate": medians[metric],
                    "best_llama": baseline,
                    "percent_difference": 100 * (medians[metric] / baseline - 1),
                    "control_result": control["result"],
                    "control_chunk_tokens": control["chunk_tokens"],
                }
            best_control_comparisons.append(
                {
                    "candidate_result": candidate["result"],
                    "prompt_tokens": int(length),
                    "metrics": metrics,
                }
            )
    summary = {
        "fixture_sha256": fixture_hash,
        "cells": cells,
        "matched_chunk_comparisons": comparisons,
        "different_quantization_comparisons": different_quantization,
        "best_recorded_llama_comparisons": best_control_comparisons,
        "best_control_method": "For each prompt length and metric, select the highest recorded llama median independently across all same-quantization controls.",
        "method": "Median of three measured requests per prompt length; one per-length warmup excluded.",
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(cells)} cells; wrote {RAW / 'summary.json'}")


if __name__ == "__main__":
    main()
