#!/usr/bin/env python3
"""Build the canonical v0.9.3 data and figures from the measured source runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from bench_decode_serving import (
    ENGINES,
    MISTRALRS_SOURCE_SHA256,
    MODES,
    VLLM_SERVER_PACKAGES,
    plot_summary,
)

RELEASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = RELEASE_DIR / "raw"
SOURCE_DIR = RAW_DIR / "source"
SOURCES = {
    "primary": SOURCE_DIR / "primary_manifest.json",
    "sglang_headroom": SOURCE_DIR / "sglang_headroom_manifest.json",
    "mistralrs_c128": SOURCE_DIR / "mistralrs_c128_manifest.json",
}
OVERRIDES = {
    "sglang:target:c64": "sglang_headroom",
    "sglang:target:c96": "sglang_headroom",
    "sglang:target:c128": "sglang_headroom",
    "mistralrs:dflash:c128": "mistralrs_c128",
}
CONCURRENCIES = (1, 8, 16, 32, 64, 96, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def key_parts(key: str) -> tuple[str, str, int]:
    engine, mode, concurrency = key.split(":")
    return engine, mode, int(concurrency.removeprefix("c"))


def portable_repetition(row: dict, source_run: str) -> dict:
    result = {
        key: value
        for key, value in row.items()
        if key not in {"raw_path", "server_log", "warmup_raw"}
    }
    result["source_run"] = source_run
    return result


def canonical_results(manifests: dict[str, dict]) -> tuple[dict, dict[str, str]]:
    selected = dict(manifests["primary"]["results"])
    sources = {key: "primary" for key in selected}
    for key, source in OVERRIDES.items():
        selected[key] = manifests[source]["results"][key]
        sources[key] = source
    return selected, sources


def build_summary(selected: dict, sources: dict[str, str], source_hashes: dict) -> dict:
    cells = {}
    for key, result in sorted(selected.items()):
        engine, mode, concurrency = key_parts(key)
        cell = {
            "concurrency": concurrency,
            "engine": engine,
            "mode": mode,
            "source_run": sources[key],
            "status": result["status"],
        }
        if result["status"] == "completed":
            cell.update({"num_prompts": result["num_prompts"], **result["medians"]})
        else:
            cell.update(
                {
                    "classification": result.get("classification"),
                    "stage": result.get("stage"),
                }
            )
        cells[key] = cell

    comparisons = {}
    for mode in MODES:
        for concurrency in CONCURRENCIES:
            throughputs = {}
            for engine in ENGINES:
                cell = cells[f"{engine}:{mode}:c{concurrency}"]
                if cell["status"] == "completed":
                    throughputs[engine] = cell["output_throughput"]
            comparison = {"output_throughput": throughputs}
            for engine in ("vllm", "sglang"):
                if engine in throughputs:
                    comparison[f"mistralrs_vs_{engine}_percent"] = 100 * (
                        throughputs["mistralrs"] / throughputs[engine] - 1
                    )
            comparisons[f"{mode}:c{concurrency}"] = comparison

    return {
        "schema_version": 2,
        "source_manifests": source_hashes,
        "canonical_overrides": OVERRIDES,
        "cells": cells,
        "comparisons": comparisons,
    }


def build_manifest(manifests: dict[str, dict], source_hashes: dict) -> dict:
    primary = manifests["primary"]
    provenance_keys = (
        "gpu",
        "mistralrs_binary_sha256",
        "mistralrs_commit",
        "sglang_commit",
        "sglang_packages",
        "vllm_server_commit",
        "vllm_server_source_sha256",
        "vllm_server_wheel_sha256",
        "vllm_server_wheel_url",
    )
    model_keys = (
        "draft_model",
        "draft_revision",
        "model",
        "model_revision",
        "original_config_sha256",
        "view_policy",
    )
    models = {key: primary["models"][key] for key in model_keys}
    models["view_policy"] = (
        "snapshot entries symlinked except config variants; verified config copied read-only "
        "as config.json"
    )
    serving = dict(primary["serving"])
    serving.update(
        {
            "sglang_sequence_headroom": 1,
            "sglang_headroom_min_concurrency": 64,
            "sglang_headroom_note": (
                "Target-only C64+ reserves one internal scheduler slot; client concurrency "
                "and CUDA decode graph size remain exactly C."
            ),
        }
    )
    provenance = {
        key: primary["provenance"][key]
        for key in provenance_keys
        if key in primary["provenance"]
    }
    provenance["vllm_benchmark_packages"] = primary["provenance"]["vllm_packages"]
    provenance["vllm_server_packages"] = VLLM_SERVER_PACKAGES
    provenance["mistralrs_source_sha256"] = MISTRALRS_SOURCE_SHA256
    return {
        "schema_version": 2,
        "canonical_overrides": OVERRIDES,
        "models": models,
        "provenance": provenance,
        "serving": serving,
        "source_manifests": source_hashes,
        "source_run_signatures": {
            name: manifest["run_signature_sha256"]
            for name, manifest in manifests.items()
        },
        "workload": primary["workload"],
    }


def write_results(selected: dict, sources: dict[str, str]) -> None:
    rows = []
    for key, result in sorted(selected.items()):
        if result["status"] != "completed":
            continue
        rows.extend(
            portable_repetition(row, sources[key]) for row in result["repetitions"]
        )
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (RAW_DIR / "results.jsonl").write_text(text, encoding="utf-8")


def write_csv(summary: dict) -> None:
    fields = (
        "mode",
        "concurrency",
        "engine",
        "status",
        "output_throughput",
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "source_run",
    )
    with (RAW_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for mode in MODES:
            for concurrency in CONCURRENCIES:
                for engine in ENGINES:
                    cell = summary["cells"][f"{engine}:{mode}:c{concurrency}"]
                    writer.writerow({field: cell.get(field) for field in fields})


def main() -> None:
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in SOURCES.items()
    }
    source_hashes = {name: sha256_file(path) for name, path in SOURCES.items()}
    selected, sources = canonical_results(manifests)
    summary = build_summary(selected, sources, source_hashes)
    write_json(RAW_DIR / "summary.json", summary)
    write_json(RAW_DIR / "run_manifest.json", build_manifest(manifests, source_hashes))
    write_results(selected, sources)
    write_csv(summary)
    plot_summary(RAW_DIR / "summary.json", RELEASE_DIR / "figures")


if __name__ == "__main__":
    main()
