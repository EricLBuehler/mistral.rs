#!/usr/bin/env python3
"""Validate the completed dense FP8/Q4K pair and recompute rates from raw requests."""

import argparse
import datetime
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REPO = Path("/home/ericbuehler/mistral.rs")
HARNESS = (
    ROOT / "harness"
    if (ROOT / "harness").is_dir()
    else REPO / "benchmarks/qwen3.8-flash-next"
)
PROMPTS_PATH = (
    ROOT / "canonical_prompts.jsonl"
    if (ROOT / "canonical_prompts.jsonl").is_file()
    else REPO / "releases/v0.9.3/raw/prompts.jsonl"
)
sys.path.insert(0, str(HARNESS))
bench_serving = importlib.import_module("bench_serving")
validation = importlib.import_module("summarize_concurrency")

CONCURRENCIES = (1, 8)
REQUESTS = 8
OUTPUT_TOKENS = 128
WARMUPS = 1
TRIALS = 3
WINDOW_METRIC = "sustained_completion_boundary_output_tokens_per_second"
MATCHED_METADATA = (
    "binary_sha256",
    "loaded_executable_sha256",
    "model",
    "model_revision",
    "model_config_sha256",
    "tokenizer_sha256",
    "weight_index_sha256",
    "source_repo",
    "source_head_at_launch",
    "source_diff_sha256_at_launch",
    "gpu",
    "component_hashes",
    "environment",
    "kv_cache",
    "target_only",
    "profiler",
)
require = validation.require


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_summary(actual, expected, name):
    require(set(actual) == set(expected), f"{name}: different metric fields")
    for metric, reference in expected.items():
        if reference is None:
            require(actual[metric] is None, f"{name}: unexpected {metric}")
        else:
            validation.check_stats(actual[metric], reference, f"{name}/{metric}")


def validate_arm(directory, expected_isq, shared_prompt_counts):
    metadata = read(directory / "metadata.json")
    require(metadata.get("complete") is True, f"{directory}: controller incomplete")
    require("error" not in metadata, f"{directory}: controller error")
    require("server_exit_code" in metadata, f"{directory}: server shutdown missing")
    require(
        not Path(f"/proc/{metadata['server_pid']}").exists(),
        f"{directory}: recorded server PID still exists",
    )
    raw = read(directory / "raw.json")
    summary = read(directory / "summary.json")
    require(raw["complete"] is True and summary["complete"] is True, "Incomplete data")
    require(metadata["isq"] == expected_isq, f"{directory}: wrong ISQ arm")
    require(raw["label"] == summary["label"], f"{directory}: label mismatch")
    require(metadata["target_only"] is True, "Speculation must be disabled")
    require(metadata["kv_cache"] == "BF16", "Expected BF16 KV cache")
    require(metadata["profiler"] is None, "Expected unprofiled run")
    require(
        "MISTRALRS_GGUF_AFFINE_BACKEND" not in metadata["environment"],
        "Expected default production GGUF backend without an affine override",
    )
    command = metadata["command"]
    for flag, expected in {
        "--dtype": "bf16",
        "--max-seqs": "8",
        "--max-model-len": "16384",
        "--pa-context-len": "16384",
        "--pa-cache-type": "auto",
        "--max-num-batched-tokens": "4096",
        "--max-prefill-chunk-tokens": "512",
        "--prefix-cache-n": "0",
    }.items():
        require(command.count(flag) == 1, f"Missing or duplicate {flag}")
        require(command[command.index(flag) + 1] == expected, f"Wrong {flag}")
    require(
        metadata["loaded_executable_sha256"] == metadata["binary_sha256"],
        "Loaded executable hash mismatch",
    )
    raw_hash = sha256(directory / "raw.json")
    require(
        metadata["raw_sha256"] == raw_hash == summary["raw_sha256"], "Raw hash mismatch"
    )
    require(
        raw["component_hashes"] == metadata["component_hashes"],
        "Component hash mismatch",
    )
    sources = {
        "control_script": ROOT / "run_dense_backend_control.py",
        "bench_concurrency": HARNESS / "bench_concurrency.py",
        "bench_serving": HARNESS / "bench_serving.py",
    }
    for name, path in sources.items():
        require(
            metadata["component_hashes"][name] == sha256(path),
            f"{name}: source hash mismatch",
        )
    require(
        raw["harness_sha256"] == metadata["component_hashes"]["control_script"],
        "Harness hash mismatch",
    )
    require(
        raw["tokenizer_sha256"] == metadata["tokenizer_sha256"],
        "Tokenizer hash mismatch",
    )
    require(
        raw["prompts_source_sha256"] == sha256(PROMPTS_PATH),
        "Canonical source hash mismatch",
    )
    require(raw["prompts"] == bench_serving.PROMPTS, "Canonical prompts differ")
    require(
        raw["prompts_sha256"]
        == hashlib.sha256(
            json.dumps(raw["prompts"], sort_keys=True).encode()
        ).hexdigest(),
        "Prompts hash mismatch",
    )
    settings = raw["settings"]
    required_settings = {
        "requests": REQUESTS,
        "max_tokens": OUTPUT_TOKENS,
        "concurrencies": list(CONCURRENCIES),
        "warmup_trials": WARMUPS,
        "measured_trials": TRIALS,
        "sampling_seed": bench_serving.SEED,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
        "logprobs_requested": False,
    }
    for name, expected in required_settings.items():
        require(settings[name] == expected, f"{directory}: wrong {name}")
    require(
        settings["expected_prompt_tokens"] == shared_prompt_counts,
        "Tokenizer-derived prompt counts differ",
    )
    adapted = settings | {"trials": TRIALS, "warmup": WARMUPS, "modes": ["closed-loop"]}
    validation.validate_settings(adapted)
    require(set(raw["cases"]) == {"1", "8"}, "Incomplete or extra concurrency cases")
    metrics = {}
    for concurrency in CONCURRENCIES:
        case = raw["cases"][str(concurrency)]
        require(case["concurrency"] == concurrency, "Wrong case concurrency")
        require(len(case["runs"]) == WARMUPS + TRIALS, "Wrong trial count")
        reconstructed = [
            validation.validate_trial(
                run, index, adapted, "closed-loop", concurrency, shared_prompt_counts
            )
            for index, run in enumerate(case["runs"])
        ][WARMUPS:]
        computed = {
            metric: validation.stats([trial[metric] for trial in reconstructed])
            for metric in validation.METRICS
        }
        windows = [
            trial["completion_boundary_window"]
            for trial in reconstructed
            if trial["completion_boundary_window"] is not None
        ]
        computed[WINDOW_METRIC] = (
            validation.stats(
                [
                    window["completion_boundary_output_tokens_per_second"]
                    for window in windows
                ]
            )
            if windows
            else None
        )
        check_summary(case["summary"], computed, f"{directory}/C{concurrency}/raw")
        check_summary(
            summary["metrics_by_concurrency"][str(concurrency)],
            computed,
            f"{directory}/C{concurrency}/summary",
        )
        metrics[str(concurrency)] = computed
    aggregate_ratio = (
        metrics["8"]["aggregate_output_tokens_per_second"]["mean"]
        / metrics["1"]["aggregate_output_tokens_per_second"]["mean"]
    )
    per_active_ratio = (
        metrics["8"]["latency_weighted_request_output_tokens_per_second"]["mean"]
        / metrics["1"]["latency_weighted_request_output_tokens_per_second"]["mean"]
    )
    validation.same_number(
        summary["c8_over_c1_aggregate"], aggregate_ratio, "C8/C1 aggregate"
    )
    validation.same_number(
        summary["c8_over_c1_per_active_request"], per_active_ratio, "C8/C1 per active"
    )
    return (
        {
            "directory": str(directory),
            "label": raw["label"],
            "isq": expected_isq,
            "metrics_by_concurrency": metrics,
            "c8_over_c1_aggregate": aggregate_ratio,
            "c8_over_c1_per_active_request": per_active_ratio,
            "validated_warmup_trials": 2,
            "validated_measured_trials": 6,
            "validated_requests_including_warmups": 64,
            "file_sha256": {
                name: sha256(directory / name)
                for name in ("raw.json", "summary.json", "metadata.json", "server.log")
            },
        },
        metadata,
        raw,
    )


def output_comparisons(fp8, q4k):
    comparisons = {}
    for concurrency in CONCURRENCIES:
        pairs = [
            (
                left["response"]["choices"][0]["text"],
                right["response"]["choices"][0]["text"],
            )
            for left_run, right_run in zip(
                fp8["cases"][str(concurrency)]["runs"][WARMUPS:],
                q4k["cases"][str(concurrency)]["runs"][WARMUPS:],
            )
            for left, right in zip(left_run["requests"], right_run["requests"])
        ]
        comparisons[str(concurrency)] = {
            "corresponding_measured_outputs": len(pairs),
            "exact_text_matches": sum(left == right for left, right in pairs),
        }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8", type=Path, default=ROOT / "dense_fp8")
    parser.add_argument("--q4k", type=Path, default=ROOT / "dense_fp8_to_q4k")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison.json")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Directory containing the pinned config.json, tokenizer.json and model.safetensors.index.json; no tensor shards are read.",
    )
    args = parser.parse_args()
    for directory in (args.fp8, args.q4k):
        require(
            read(directory / "metadata.json").get("complete") is True,
            f"{directory}: incomplete; refusing comparison",
        )
    inventory = read(ROOT / "checkpoint_inventory.json")["models"]["Qwen3.8-27B-FP8"]
    snapshot = args.snapshot or Path(inventory["snapshot"])
    for name, field in (
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("model.safetensors.index.json", "weight_index_sha256"),
    ):
        require(
            sha256(snapshot / name) == inventory[field],
            f"Cached {name} fingerprint differs",
        )
    tokenizer = bench_serving.Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    canonical = [json.loads(line) for line in PROMPTS_PATH.read_text().splitlines()]
    bench_serving.PROMPTS = {
        f"canonical_{i:02}": row["prompt"] for i, row in enumerate(canonical[:REQUESTS])
    }
    prompt_counts = {
        name: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        for name, prompt in bench_serving.PROMPTS.items()
    }
    fp8, fp8_meta, fp8_raw = validate_arm(args.fp8, None, prompt_counts)
    q4k, q4k_meta, q4k_raw = validate_arm(args.q4k, "q4k", prompt_counts)
    require(fp8["label"] != q4k["label"], "Arm labels must differ")
    for field in MATCHED_METADATA:
        require(fp8_meta[field] == q4k_meta[field], f"Unmatched metadata: {field}")
    for field, inventory_field in (
        ("model", "model"),
        ("model_revision", "revision"),
        ("model_config_sha256", "config_sha256"),
        ("tokenizer_sha256", "tokenizer_sha256"),
        ("weight_index_sha256", "weight_index_sha256"),
    ):
        require(
            fp8_meta[field] == inventory[inventory_field], f"Wrong checkpoint {field}"
        )
    require("--isq" not in fp8_meta["command"], "Baseline unexpectedly requests ISQ")
    require(
        q4k_meta["command"] == fp8_meta["command"] + ["--isq", "q4k"],
        "Server commands differ beyond ISQ",
    )
    require(fp8_raw["settings"] == q4k_raw["settings"], "Request settings differ")
    require(
        fp8_raw["prompts_sha256"] == q4k_raw["prompts_sha256"], "Prompt hashes differ"
    )
    comparison = {
        "complete": True,
        "validated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "validation": {
            "matched_metadata_fields": list(MATCHED_METADATA),
            "only_launch_argument_difference": ["--isq", "q4k"],
            "prompt_tokens": prompt_counts,
            "warmups_per_concurrency": WARMUPS,
            "measured_trials_per_concurrency": TRIALS,
            "requests_per_trial": REQUESTS,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "recomputation": "Per-request elapsed time from finish minus start, output counts from raw API usage, whole-trial elapsed time from client trial timer. Recomputed trial metrics and means/sample SD agree with both stored summaries.",
            "source_hashes": {
                "comparator": sha256(__file__),
                "independent_validator": sha256(Path(validation.__file__)),
                "checkpoint_inventory": sha256(ROOT / "checkpoint_inventory.json"),
            },
            "executable_hash_note": "Matched launch-time binary and /proc/PID/exe fingerprints; executable payload was not rehashed by this comparator.",
        },
        "arms": {"fp8": fp8, "q4k": q4k},
        "q4k_over_fp8": {
            str(concurrency): {
                metric: q4k["metrics_by_concurrency"][str(concurrency)][metric]["mean"]
                / fp8["metrics_by_concurrency"][str(concurrency)][metric]["mean"]
                for metric in validation.METRICS
            }
            for concurrency in CONCURRENCIES
        },
        "q4k_over_fp8_c8_c1_scaling_ratio": q4k["c8_over_c1_aggregate"]
        / fp8["c8_over_c1_aggregate"],
        "output_comparisons": output_comparisons(fp8_raw, q4k_raw),
        "caveats": [
            "Same dense architecture, FP8 checkpoint, binary, hardware and request settings. Q4K dequantizes published FP8 weights before requantization; this is not BF16-source Q4K and establishes no quality equivalence.",
            "ISQ changes eligible weight precision, activation quantization, sensitive lm_head policy, and packed projection layout. Immediate ISQ uses separate gate/up weight tensors, but their small-batch MMVQ computation is fused. The production packed-affine Marlin backend defaults off and neither recorded environment opts in. This is a full execution-path comparison, not a single-kernel substitution.",
            "Both runs are target-only with BF16 KV and max_seqs=8. C8 has only eight requests, so it is one finite wave rather than sustained traffic.",
            "Rates include prefill and HTTP overhead. Statistics are mean and sample SD across three measured trials; one warmup per concurrency is excluded.",
            "Dense FFN has no expert routing and much larger inner dimensions than Flash-Next experts. These data cannot isolate sparse route occupancy, prove a hardware ceiling, or establish an old-versus-new engine regression.",
            "Output equality is diagnostic only. Greedy output can change with quantization and batching; token counts are validated, but semantic quality and numerical equivalence are not asserted.",
        ],
    }
    args.output.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print("Arm       C  Aggregate tok/s     Per-active tok/s    Mean active")
    for name, arm in comparison["arms"].items():
        for concurrency in CONCURRENCIES:
            row = arm["metrics_by_concurrency"][str(concurrency)]
            total = row["aggregate_output_tokens_per_second"]
            active = row["latency_weighted_request_output_tokens_per_second"]
            print(
                f"{name:9} {concurrency}  {total['mean']:8.3f} +/- {total['sample_stddev']:.3f}  {active['mean']:8.3f} +/- {active['sample_stddev']:.3f}  {row['mean_active_requests']['mean']:.3f}"
            )
        print(
            f"  C8/C1 aggregate: {arm['c8_over_c1_aggregate']:.4f}x; per-active retention: {100 * arm['c8_over_c1_per_active_request']:.2f}%"
        )
    print(args.output)


if __name__ == "__main__":
    main()
