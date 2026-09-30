#!/usr/bin/env python3
"""Validate and package the completed dense control without running inference."""

import argparse
import datetime
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = Path("/home/ericbuehler/mistral.rs")
OUTPUT = Path("benchmarks/qwen3.8-flash-next")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(source, repo):
    metadata = read(source / "metadata.json")
    require(metadata["complete"] is True, "Dense control has not completed")
    require("server_exit_code" in metadata, "Dense control has not finalized")
    raw, summary = read(source / "raw.json"), read(source / "summary.json")
    require(
        raw["complete"] is True and summary["complete"] is True, "Incomplete results"
    )
    raw_hash = sha256(source / "raw.json")
    require(
        metadata["raw_sha256"] == summary["raw_sha256"] == raw_hash,
        "Raw result hashes disagree",
    )
    require(
        metadata["loaded_executable_sha256"] == metadata["binary_sha256"],
        "Loaded binary hash differs",
    )
    require(metadata["target_only"] is True, "Expected target-only control")
    require(metadata["kv_cache"] == "BF16", "Report expects BF16 KV")
    require(
        raw["label"] == summary["label"] and set(raw["cases"]) == {"1", "8"},
        "Unexpected control cases",
    )
    require(
        raw["component_hashes"] == metadata["component_hashes"],
        "Component hashes differ",
    )
    components = {
        "control_script": WORK / "run_dense_control.py",
        "bench_concurrency": repo / OUTPUT / "bench_concurrency.py",
        "bench_serving": repo / OUTPUT / "bench_serving.py",
    }
    for name, path in components.items():
        require(
            sha256(path) == metadata["component_hashes"][name],
            f"Changed component: {name}",
        )
    require(
        raw["harness_sha256"] == metadata["component_hashes"]["control_script"],
        "Wrong harness hash",
    )
    require(
        raw["tokenizer_sha256"] == metadata["tokenizer_sha256"],
        "Tokenizer hashes differ",
    )
    require(
        raw["prompts_sha256"]
        == hashlib.sha256(
            json.dumps(raw["prompts"], sort_keys=True).encode()
        ).hexdigest(),
        "Prompt hash differs",
    )
    sys.path.insert(0, str(repo / OUTPUT))
    validator = importlib.import_module("summarize_concurrency")
    validator.bench_serving.PROMPTS = raw["prompts"]
    settings = raw["settings"] | {"warmup": 1, "trials": 3}
    require(
        settings["requests"] == 8
        and settings["max_tokens"] == 128
        and settings["warmup_trials"] == 1
        and settings["measured_trials"] == 3
        and settings["concurrencies"] == [1, 8],
        "Unexpected dense control workload",
    )
    counts = {}
    for concurrency in (1, 8):
        case = raw["cases"][str(concurrency)]
        require(
            case["concurrency"] == concurrency and len(case["runs"]) == 4,
            "Incomplete trials",
        )
        measured = []
        for index, run in enumerate(case["runs"]):
            reconstructed = validator.validate_trial(
                run, index, settings, "closed-loop", concurrency, counts
            )
            if not run["warmup"]:
                measured.append(reconstructed)
        for metric in validator.METRICS:
            expected = validator.stats([run[metric] for run in measured])
            validator.check_stats(case["summary"][metric], expected, metric)
            validator.check_stats(
                summary["metrics_by_concurrency"][str(concurrency)][metric],
                expected,
                metric,
            )
    metrics = summary["metrics_by_concurrency"]
    for ratio, metric in (
        ("c8_over_c1_aggregate", "aggregate_output_tokens_per_second"),
        (
            "c8_over_c1_per_active_request",
            "latency_weighted_request_output_tokens_per_second",
        ),
    ):
        validator.same_number(
            summary[ratio],
            metrics["8"][metric]["mean"] / metrics["1"][metric]["mean"],
            ratio,
        )
    return summary, metadata


def render(summary, comparison, probe):
    prefix = "raw/scaling_investigation"
    old_rows = []
    labels = (
        "v0.9.3 dense target-only",
        "v0.9.3 dense + DFlash2",
        "Flash-Next MTP before tuner exploration fix",
    )
    for label, series in zip(labels, comparison["series"], strict=True):
        c1, c8 = series["concurrency"]["1"], series["concurrency"]["8"]
        old_rows.append(
            f"| {label} | {c1['aggregate_tokens_per_second']:.2f} | "
            f"{c8['aggregate_tokens_per_second']:,.2f} | {series['c8_over_c1_aggregate']:.2f}x | "
            f"{c8['latency_weighted_per_active_request_tokens_per_second']:.2f} |"
        )
    dense_rows = []
    for concurrency in (1, 8):
        metrics = summary["metrics_by_concurrency"][str(concurrency)]
        aggregate = metrics["aggregate_output_tokens_per_second"]
        per_active = metrics["latency_weighted_request_output_tokens_per_second"]
        dense_rows.append(
            f"| {concurrency} | {aggregate['mean']:.2f} +/- {aggregate['sample_stddev']:.2f} | "
            f"{per_active['mean']:.2f} +/- {per_active['sample_stddev']:.2f} | "
            f"{metrics['mean_active_requests']['mean']:.2f} | "
            f"{metrics['mean_request_wall_seconds']['mean']:.3f} |"
        )
    probe_rows = []
    for result in probe["results"]:
        if result["rows"] == 56:
            routing = (
                "spread" if result["routing"] == "independent" else "groups of seven"
            )
            probe_rows.append(
                f"| {result['scheme'].upper()} | {routing} | "
                f"{result['normal_median_ms']:.3f} | {result['tight_median_ms']:.3f} | "
                f"{result['median_paired_speedup']:.2f}x |"
            )
    return f"""# Concurrency scaling investigation

The recorded Flash-Next workload scales less strongly than the v0.9.3 workloads. This comparison changes hardware, model, quantization, sampling, and request lengths together; it does not isolate an engine regression.

| Workload | C1 aggregate tok/s | C8 aggregate tok/s | C8/C1 | C8 tok/s per active request |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(old_rows)}

The [release measurements](../../releases/v0.9.3/report.md) used Qwen3.8-27B-FP8 on GH200, FP8 KV, 64 requests with 512 outputs, stochastic sampling, and optionally seven external DFlash2 proposals. Flash-Next used ISQ q4k on GB10, BF16 KV, 24 requests at C8 with 128 outputs, greedy sampling, and adaptive built-in MTP. Release values are medians of three trials; these historical Flash-Next values are means of five. The [comparison JSON]({prefix}/release_comparison.json) preserves unrounded values, source hashes/selectors, and the old per-active-request reconstruction from mean TTFT/TPOT. Both suites include prefill and scheduling in client wall time. Mean C8 activity was approximately 8.00, 7.77, and 7.56 respectively, so startup/drain occupancy alone does not explain the gap.

The release client also applied the model's chat template: an xhigh reasoning system instruction, role markers, and an assistant `<think>` prefix added exactly 52 tokens to every canonical prompt. CPU rendering reconciles all 64 prompts to the recorded 128-139 input-token range and 8,592-token total. The comparison JSON contains the pinned vLLM source commit and lines, all count pairs, and one rendered sample.

## Same-GB10 dense control

The current binary's target-only dense FP8 control used the same pinned 27B checkpoint as the release, BF16 KV, greedy sampling, the first eight raw canonical prompts (77-85 input tokens), and 128 outputs. Server capacity remained eight sequences for C1 and C8. Each case had one warmup and three measured trials. C8 is one finite wave of eight requests, not sustained traffic. [Raw requests/responses]({prefix}/dense_control/raw.json), [validated summary]({prefix}/dense_control/summary.json), and [binary/model provenance]({prefix}/dense_control/metadata.json) are preserved.

Only the clean rerun, performed after pausing the identified rust-analyzer compiler processes, supplies the dense figures below. An earlier run overlapped a background debug CUDA build and is [archived separately]({prefix}/dense_control_background_build/summary.json); its [interference note]({prefix}/dense_control_background_build/interference.json) and [process identities]({prefix}/dense_control/paused_editor_processes.json) preserve that limitation. Stable trial rates in the earlier run did not establish CPU isolation.

| Concurrency | Aggregate tok/s | Tok/s per active request | Mean active requests | Mean request latency (s) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(dense_rows)}

Values are means +/- sample SD where shown. Aggregate throughput divides outputs by whole-trial wall time; per-active-request throughput divides outputs by summed request latency. C8/C1 aggregate scaling is {summary["c8_over_c1_aggregate"]:.2f}x, retaining {summary["c8_over_c1_per_active_request"] * 100:.1f}% of the C1 per-active-request rate. This measures current dense-model behavior on GB10. The model, prompts, speculation, and finite-wave workload differ from Flash-Next; the old release also differs in hardware, sampling, formatting, output length, and KV precision. A version regression requires matched runs under both binaries. The [control script]({prefix}/run_dense_control.py) records binary hashes and exact server/request settings.

## Expert dispatch probe

A [single-layer probe]({prefix}/expert_dispatch/summary.json) used actual Flash-Next layer 8 weights, BF16 synthetic activations, 512 experts, top-10 routing, hidden width 2,560, and intermediate width 640. It compared forced GEMV with grouped MMQ through the full expert pipeline. This original probe also overlapped the background compiler; its timings are diagnostic evidence with possible CPU-launch interference.

A separate [clean known-occupancy probe]({prefix}/expert_dispatch/known_occupancy_clean/summary.json) ran with the compiler processes paused, verified before and after each run. It compared the same grouped projection pipeline with the normal column bound versus a bound of eight, after verifying every synthetic expert receives at most eight rows. Representative 56-row median times and median paired speedups:

| Weights | Synthetic routing | Normal bound (ms) | Known bound 8 (ms) | Paired speedup |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(probe_rows)}

These routes are deterministic spread controls or groups of seven sharing experts, not captured model routes. Timings cover seven paired alternating rounds of ten invocations after warmup; CUDA events include host launch gaps. Repeated single-layer weight reuse differs from a full 48-layer model. All clean-probe BF16 outputs matched both the normal bound and packed reference exactly. Observed gains were roughly 0-11%, with small regressions in some cases. This is a modest ideal-occupancy opportunity: applying a bound of eight generally can omit expert rows. The measured projection pipeline also quantizes gate/up activations separately, while the packed production pipeline shares that quantization. Neither probe establishes a memory-bandwidth ceiling or predicts serving throughput. [Original provenance]({prefix}/expert_dispatch/provenance.json), [post-measurement review]({prefix}/expert_dispatch/post_measurement_review.json), and [clean-probe metadata]({prefix}/expert_dispatch/known_occupancy_clean/metadata.json) preserve the scope and implementations.

## Tuner exploration bug

The [automatic-depth search](../../mistralrs-core/src/speculative/autotuner.rs) previously explored only neighboring depths. Costs need not be monotonic across the supported depths 2, 3, 4, and 6: starting at 6, a slow depth 4 could prevent discovery of a faster depth 3. Warmup now samples every undersampled candidate, excluding the currently preferred depth; steady-state refresh remains local. The regression test models this slower-neighbor barrier and requires discovery of depth 3. The historical Flash-Next row above predates this change. Full-model measurements after this exploration fix are still pending; this search correction alone is not evidence of a measured serving speedup.
"""


def copy_verified(source, destination, records):
    expected = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(sha256(destination) == expected, f"Copy differs: {source}")
    records.append(
        {"source": str(source), "file": destination.name, "sha256": expected}
    )


def check_background_archive(investigation):
    destination = investigation / "dense_control_background_build"
    existing = destination if destination.exists() else investigation / "dense_control"
    require(existing.is_dir(), "Original packaged dense control is missing")
    require(
        sha256(existing / "raw.json") == sha256(WORK / "current_dense_fp8/raw.json"),
        "Existing dense archive is not the original background-build run",
    )
    for line in (existing / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(
            sha256(existing / name) == expected, f"Changed original artifact: {name}"
        )
    return existing, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--source", type=Path, default=WORK / "current_dense_fp8_clean")
    parser.add_argument(
        "--launcher-log",
        type=Path,
        default=WORK / "current_dense_fp8_clean.launcher.log",
    )
    args = parser.parse_args()
    summary, metadata = validate(args.source, args.repo)
    report_dir = args.repo / OUTPUT
    investigation = report_dir / "raw/scaling_investigation"
    destination = investigation / "dense_control"
    background_source, background_destination = check_background_archive(investigation)
    require(
        background_source == destination or not destination.exists(),
        f"Refusing to replace existing clean results: {destination}",
    )
    require(
        sha256(args.source / "raw.json") != sha256(background_source / "raw.json"),
        "The final report must use the separate clean rerun",
    )
    paused_path = WORK / "paused_editor_processes.json"
    paused = read(paused_path)
    require(
        datetime.datetime.fromisoformat(metadata["date"]).timestamp()
        > paused["paused_at"],
        "Control started before the editor compiler processes were paused",
    )
    comparison_path = WORK / "release_comparison.json"
    comparison = read(comparison_path)
    probe = read(investigation / "expert_dispatch/known_occupancy_clean/summary.json")
    report = render(summary, comparison, probe)
    require(args.launcher_log.is_file(), "Missing dense launcher log")
    for name in ("run_dense_control.py", "release_comparison.json"):
        existing = investigation / name
        require(
            not existing.exists() or sha256(existing) == sha256(WORK / name),
            f"Existing evidence differs: {existing}",
        )
    files = [
        (args.source / name, name)
        for name in ("raw.json", "summary.json", "metadata.json", "server.log")
    ]
    files.append((args.launcher_log, "launcher.log"))
    files.append((paused_path, "paused_editor_processes.json"))
    for path, _ in files:
        require(path.is_file(), f"Missing control artifact: {path}")
    investigation.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="dense-staging-", dir=investigation
    ) as temporary:
        stage = Path(temporary) / "dense_control"
        records = []
        for source, name in files:
            copy_verified(source, stage / name, records)
        manifest = {
            "packaged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "packaging_script_sha256": sha256(Path(__file__)),
            "validator_sha256": sha256(report_dir / "summarize_concurrency.py"),
            "validation": "Completed metadata, binary/harness/raw hashes, every response and request setting, prompt/output counts, concurrency limits, raw timestamps, whole-trial metrics and summary statistics verified before copying.",
            "loaded_binary_sha256": metadata["loaded_executable_sha256"],
            "interference_control": "Rerun started after the identified rust-analyzer/debug CUDA compiler process tree was paused. The previous run is preserved in dense_control_background_build and excluded from the final dense table.",
            "files": records,
        }
        (stage / "packaging.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "SHA256SUMS").write_text(
            "".join(
                f"{sha256(path)}  {path.name}\n" for path in sorted(stage.iterdir())
            )
        )
        if background_source != background_destination:
            background_source.rename(background_destination)
        interference = {
            "reason": "This original run overlapped rust-analyzer's background debug CUDA compilation, which started at approximately 2026-09-30 03:29 UTC. Its rates are not CPU-isolated measurements.",
            "report_use": "Archived diagnostic only; excluded from the final dense-control table and ratios.",
            "original_sha256sums_preserved": sha256(
                background_destination / "SHA256SUMS"
            ),
            "paused_processes_sha256": sha256(paused_path),
            "pause_record": "../dense_control/paused_editor_processes.json",
            "original_expert_probe": "The original expert_dispatch probe also overlapped this compiler activity; later ideal-tile probes ran after these processes were paused.",
        }
        (background_destination / "interference.json").write_text(
            json.dumps(interference, indent=2) + "\n"
        )
        stage.rename(destination)
    for name in ("run_dense_control.py", "release_comparison.json"):
        copy_verified(WORK / name, investigation / name, [])
    copy_verified(Path(__file__), investigation / "package_scaling.py", [])
    report_path = report_dir / "scaling.md"
    report_path.write_text(report)
    print(f"Packaged validated dense control: {destination}")
    print(f"Wrote scaling note: {report_path}")


if __name__ == "__main__":
    main()
