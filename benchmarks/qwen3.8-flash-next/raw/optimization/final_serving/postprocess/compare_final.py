"""Validate completed staged serving measurements without server/GPU access."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import counter_summary
import mixed_context_smoke
import serving_summary
import summarize_concurrency

ROOT = Path("/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next")
OLD_BINARY = "d245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588"
NEW_BINARY = "d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d"
PHASES = {
    "serving": "isq_mtp_tuner.serving",
    "text": "isq_mtp_tuner.text",
    "c1": "concurrency_c1",
    "c6_c8": "concurrency_c6_c8",
    "mixed_context": "mixed_context",
}
OLD_LABEL = "pre_moe_optimization"
NEW_LABEL = "optimized_candidate"
MIB = 1024 * 1024
LIMITS = [
    "Staged before/after measurements, not randomized causal A/B trials; percentages are ratios of means.",
    "Candidate includes masked-Y MMQ and the scoped small-group dispatch, plus documented intervening fixes; no isolated causal speedup is inferred.",
    "Whole-trial closed-loop rates include startup/drain. Per-active-request rates use tokens divided by summed request latency.",
    "Counters bracket whole subprocesses including warmups and boundary overhead. C6/C8 counters cannot be split by concurrency.",
    "Global swap counters do not identify model page-ins or their timing impact. Process VmSwap is separate.",
    "Graph dispatch counts are not kernel-time coverage. No bandwidth or hardware-ceiling claim follows.",
    "Output text differences are diagnostic; finite logprobs and nonempty output do not establish model quality.",
    "Earlier metadata did not record every inherited environment variable; only shared recorded values can be checked.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON literal in {path}: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalized_server_command(command):
    result = command[1:]
    result[result.index("-p") + 1] = "PORT"
    return result


def verify_manifest(folder, manifest):
    for relative, expected in manifest.items():
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts, "Invalid artifact path"
        )
        require(
            sha(folder / path) == expected, f"Artifact hash mismatch: {folder / path}"
        )


def relabel(report, label):
    report = copy.deepcopy(report)
    report["label"] = label
    report["settings"]["label"] = label
    return report


def change(before, after):
    require(
        math.isfinite(before) and math.isfinite(after) and before > 0 and after > 0,
        "Invalid rate",
    )
    return {
        "ratio_of_means": after / before,
        "percent_change": 100 * (after / before - 1),
    }


def memory_delta(before, after, page_size):
    require(
        before["model_pid"] == after["model_pid"],
        "Model PID changed across memory window",
    )
    elapsed = after.get("monotonic", after.get("captured_monotonic")) - before.get(
        "monotonic", before.get("captured_monotonic")
    )
    require(elapsed > 0 and math.isfinite(elapsed), "Invalid memory snapshot interval")
    result = {"snapshot_window_seconds": elapsed, "page_size_bytes": page_size}
    for field in ("pswpin_pages", "pswpout_pages"):
        delta = after[field] - before[field]
        require(delta >= 0, f"Global swap counter reset: {field}")
        result[field + "_delta"] = delta
        result[field.replace("_pages", "_MiB") + "_delta"] = delta * page_size / MIB
    result["before_process"] = before.get("model", before.get("process"))
    result["after_process"] = after.get("model", after.get("process"))
    result["before_system"] = before["system"]
    result["after_system"] = after["system"]
    return result


def counter_window(name, before, after):
    value = counter_summary.summarize(
        name, counter_summary.counter_deltas(before, after)
    )
    value["scope"] = (
        "Whole command including warmups; c6_c8 combines both concurrencies."
    )
    value["target_graph_replays"] = sum(
        item["dispatches"]
        for item in value["graph_dispatches"]
        if item["labels"].get("component") == "target"
        and item["labels"].get("mode") == "replay"
    )
    value["target_eager_dispatches"] = sum(
        item["dispatches"]
        for item in value["graph_dispatches"]
        if item["labels"].get("component") == "target"
        and item["labels"].get("mode") == "eager"
    )
    return value


def check_text_smokes(before, after):
    require(before["request"] == after["request"], "Chat smoke requests differ")
    require(
        before["response"]["usage"]["prompt_tokens"]
        == after["response"]["usage"]["prompt_tokens"],
        "Chat smoke prompt token counts differ",
    )
    result = {
        "request_sha256": text_sha(json.dumps(before["request"], sort_keys=True)),
        "semantic_review": "Human inspection required; no semantic pass is inferred from HTTP success.",
    }
    for label, report in ((OLD_LABEL, before), (NEW_LABEL, after)):
        response = report["response"]
        require(
            "error" not in response and len(response["choices"]) == 1,
            "Invalid chat smoke response",
        )
        choice = response["choices"][0]
        message = choice["message"]
        require(
            0
            < response["usage"]["completion_tokens"]
            <= report["request"]["max_tokens"],
            "Chat smoke output count is outside the requested limit",
        )
        require(
            isinstance(message.get("content"), str) and message["content"].strip(),
            "Empty chat smoke message",
        )
        result[label] = {
            "message": message,
            "finish_reason": choice["finish_reason"],
            "usage": response["usage"],
        }
    return result


def check_mixed_smokes(before, after):
    for key in ("harness_sha256", "shared_prompt_harness_sha256", "tokenizer_sha256"):
        require(before[key] == after[key], f"Mixed-context {key} differs")
    require(before["complete"] and after["complete"], "Mixed-context smoke incomplete")
    require(
        len(before["requests"]) == len(after["requests"]) == 8,
        "Mixed-context request count",
    )
    pairs = []
    for reference, candidate in zip(before["requests"], after["requests"]):
        for key in (
            "index",
            "request",
            "prompt_seed",
            "prompt_sha256",
            "expected_prompt_tokens",
            "expected_completion_tokens",
        ):
            require(
                reference[key] == candidate[key], f"Mixed-context request {key} differs"
            )
        for result in (reference, candidate):
            require(
                result["complete"]
                and not result.get("error")
                and result["http_status"] == 200,
                "Mixed-context request failed",
            )
            require(
                text_sha(result["request"]["prompt"]) == result["prompt_sha256"],
                "Mixed prompt hash mismatch",
            )
            mixed_context_smoke.validate_response(
                result["response"], result["expected_prompt_tokens"]
            )
            choices = result["response"]["choices"]
            require(
                len(choices) == 1
                and isinstance(choices[0].get("text"), str)
                and choices[0]["text"].strip(),
                "Empty or malformed mixed-context completion",
            )
        pairs.append((f"mixed/{reference['index']}", reference, candidate))
    return {
        "requests_validated": 8,
        "matching_requests": True,
        "finite_logprobs": True,
        "output_comparisons": serving_summary.compare_outputs(pairs),
    }


def collect_windows(raw, candidate, metadata):
    old_metrics = raw / "scaling_investigation/final_metrics"
    old_memory = raw / "scaling_investigation/final_memory"
    old_memory_summary = load(old_memory / "summary.json")
    metric_manifest = load(old_metrics / "manifest.json")
    verify_manifest(
        old_metrics,
        {name: value["sha256"] for name, value in metric_manifest["files"].items()},
    )
    old_commands = {item["name"]: item for item in load(raw / "isq_mtp.commands.json")}
    new_commands = {item["name"]: item for item in metadata["phases"]}
    metrics, memory = {}, {}
    for new_name, old_name in PHASES.items():
        require(
            new_commands[new_name]["complete"]
            and new_commands[new_name]["returncode"] == 0,
            f"Candidate phase incomplete: {new_name}",
        )
        require(
            old_commands[old_name]["returncode"] == 0,
            f"Baseline phase failed: {old_name}",
        )
        metrics[new_name], memory[new_name] = {}, {}
        for label, folder, name in (
            (OLD_LABEL, old_metrics, old_name),
            (NEW_LABEL, candidate, new_name),
        ):
            texts = [
                (folder / f"{name}.metrics.{point}.txt").read_text()
                for point in ("before", "after")
            ]
            metrics[new_name][label] = counter_window(new_name, *texts)
        old_snapshots = [
            load(old_memory / f"{old_name}.swap.{point}.json")
            for point in ("before", "after")
        ]
        for point in ("before", "after"):
            name = f"{old_name}.swap.{point}.json"
            require(
                sha(old_memory / name) == old_memory_summary["files"][name]["sha256"],
                f"Baseline memory snapshot hash mismatch: {name}",
            )
        new_snapshots = [
            load(candidate / f"{new_name}.memory.{point}.json")
            for point in ("before", "after")
        ]
        require(
            new_snapshots[0]["page_size_bytes"] == new_snapshots[1]["page_size_bytes"],
            "Page size changed",
        )
        memory[new_name][OLD_LABEL] = memory_delta(
            *old_snapshots, old_memory_summary["system_page_bytes"]
        )
        memory[new_name][NEW_LABEL] = memory_delta(
            *new_snapshots, new_snapshots[0]["page_size_bytes"]
        )
        for point in ("before", "after"):
            isolation = load(candidate / f"{new_name}.{point}.processes.json")
            require(
                not isolation["unexpected_active"],
                f"Active interference at {new_name}/{point}",
            )
    return metrics, memory


def render(result):
    lines = [
        "# Staged optimized-serving comparison",
        "",
        "Mean +/- sample standard deviation across five measured trials after two warmups. Changes are ratios of means; these runs were not randomized causal A/B trials.",
        "",
        "## Serving workloads",
        "",
        "| Workload | Before tok/s | Candidate tok/s | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["serving_changes"]:
        left, right = row[OLD_LABEL], row[NEW_LABEL]
        lines.append(
            f"| {row['workload']} | {left['mean_tokens_per_second']:.2f} +/- {left['sample_stddev']:.2f} | {right['mean_tokens_per_second']:.2f} +/- {right['sample_stddev']:.2f} | {row['percent_change']:+.2f}% |"
        )
    prompt = result["ordinary_prompt_mean"]
    lines += [
        "",
        f"Arithmetic mean across the eight ordinary prompt throughputs: {prompt[OLD_LABEL]:.2f} -> {prompt[NEW_LABEL]:.2f} tok/s ({prompt['percent_change']:+.2f}%). This is not a pooled request rate. C4/C8 serving rows are finite bursts.",
        "",
        "## Closed-loop concurrency",
        "",
        "| C | Requests/trial | Before aggregate tok/s | Candidate aggregate tok/s | Change | Before per-active tok/s | Candidate per-active tok/s | Candidate mean active | Candidate mean latency (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(value):
        return f"{value['mean']:.2f} +/- {value['sample_stddev']:.2f}"

    for row in result["concurrency_changes"]:
        left, right = row[OLD_LABEL]["metrics"], row[NEW_LABEL]["metrics"]
        lines.append(
            f"| {row['concurrency']} | {row[NEW_LABEL]['requests_per_trial']} | {cell(left['aggregate_output_tokens_per_second'])} | {cell(right['aggregate_output_tokens_per_second'])} | {row['aggregate_change']['percent_change']:+.2f}% | {cell(left['latency_weighted_request_output_tokens_per_second'])} | {cell(right['latency_weighted_request_output_tokens_per_second'])} | {cell(right['mean_active_requests'])} | {cell(right['mean_request_wall_seconds'])} |"
        )
    lines += [
        "",
        "Per-active-request throughput is output tokens divided by summed request latency. Whole-trial rates include startup/drain; completion-boundary estimates remain secondary in concurrency.summary.json.",
        "",
        "## Whole-command counters and swap",
        "",
        "| Phase | Stage | Acceptance | Mean draft depth | Target graph replays | Target eager | Swap-in MiB | Swap-out MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stages in result["counters"].items():
        for label, value in stages.items():
            memory = result["memory"][name][label]
            acceptance = value["draft_token_acceptance_fraction"]
            depth = value["mean_proposed_depth_per_sequence_proposal"]
            acceptance = f"{100 * acceptance:.1f}%" if acceptance is not None else "n/a"
            depth = f"{depth:.2f}" if depth is not None else "n/a"
            lines.append(
                f"| {name} | {label} | {acceptance} | {depth} | {value['target_graph_replays']} | {value['target_eager_dispatches']} | {memory['pswpin_MiB_delta']:.2f} | {memory['pswpout_MiB_delta']:.2f} |"
            )
    lines += ["", "## Chat smoke for human inspection", ""]
    for label in (OLD_LABEL, NEW_LABEL):
        sample = result["text_smoke"][label]
        lines += [
            f"{label}, finish_reason={sample['finish_reason']}:",
            "",
            "```json",
            json.dumps(sample["message"], indent=2, ensure_ascii=True),
            "```",
            "",
        ]
    lines += [
        "Mixed-context request counts and finite-logprob checks passed for both stages. Semantic quality is not inferred from these structural checks.",
        "",
        "## Limits",
        "",
    ]
    lines.extend(f"- {limit}" for limit in result["limitations"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", default=NEW_BINARY)
    args = parser.parse_args()
    candidate, raw = args.candidate.resolve(), args.baseline.resolve()
    metadata = load(candidate / "metadata.json")
    require(
        metadata["complete"] is True,
        "Candidate is not complete; wait for clean shutdown",
    )
    require(
        metadata["binary_sha256"]
        == metadata["binary_sha256_after"]
        == args.expected_binary_sha256,
        "Candidate binary mismatch",
    )
    require(
        not metadata["server_shutdown"]["forced_kill"]
        and metadata["server_shutdown"]["returncode"] in (0, -15),
        "Unclean server shutdown",
    )
    require(not metadata["profiled"], "Candidate was profiled")
    process_stat = Path("/proc") / str(metadata["model_pid"]) / "stat"
    try:
        current_start = int(process_stat.read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        pass
    else:
        original_start = load(candidate / "serving.memory.before.json")["model"][
            "starttime_ticks"
        ]
        require(
            current_start != original_start, "Candidate model process is still alive"
        )
    verify_manifest(candidate, load(candidate / "SHA256SUMS.json"))
    dependencies = load(Path(__file__).with_name("dependencies.json"))
    verify_manifest(
        Path(__file__).parent,
        {name: value["sha256"] for name, value in dependencies.items()},
    )
    old_metadata = load(raw / "isq_mtp.metadata.json")
    require(old_metadata["binary_sha256"] == OLD_BINARY, "Wrong pre-MoE baseline")
    require(load(raw / "isq_mtp.orchestration.json")["complete"], "Baseline incomplete")
    require(
        normalized_server_command(metadata["command"])
        == normalized_server_command(old_metadata["command"]),
        "Server settings/checkpoint differ",
    )
    for key, value in old_metadata["env"].items():
        require(
            metadata["env"].get(key) == value, f"Recorded environment differs: {key}"
        )
    archive = load(raw / "final_artifacts.provenance.json")
    required_baseline_files = {
        "isq_mtp.json",
        "isq_mtp.metadata.json",
        "isq_mtp.text.json",
        "isq_mtp.mixed_context.json",
        "concurrency.isq_mtp.c1.json",
        "concurrency.isq_mtp.c6_c8.json",
        "isq_mtp.commands.json",
        "isq_mtp.orchestration.json",
    }
    archived = {Path(item["destination"]).name: item["sha256"] for item in archive}
    verify_manifest(raw, {name: archived[name] for name in required_baseline_files})
    old_serving, new_serving = [
        relabel(load(path), label)
        for path, label in (
            (raw / "isq_mtp.json", OLD_LABEL),
            (candidate / "serving.json", NEW_LABEL),
        )
    ]
    serving = serving_summary.summarize([old_serving, new_serving])
    require(
        serving["validation"]["harness_sha256"]
        == sha(Path(__file__).with_name("bench_serving.py")),
        "Serving harness source mismatch",
    )
    reports = []
    for folder, names, label in (
        (
            raw,
            ("concurrency.isq_mtp.c1.json", "concurrency.isq_mtp.c6_c8.json"),
            OLD_LABEL,
        ),
        (candidate, ("concurrency.c1.json", "concurrency.c6_c8.json"), NEW_LABEL),
    ):
        for name in names:
            report = relabel(load(folder / name), label)
            require(
                report["settings"]["warmup"] == 2
                and report["settings"]["trials"] == 5
                and report["settings"]["max_tokens"] == 128,
                "Concurrency repeat/output settings changed",
            )
            require(
                report["settings"]["modes"] == ["closed-loop"],
                "Unexpected workload mode",
            )
            reports.append(report)
    concurrency = summarize_concurrency.summarize_reports(reports)
    concurrent_cases = {
        (report["label"], name): case
        for report in reports
        for name, case in report["cases"].items()
    }
    concurrent_output_pairs = []
    for name in ("closed-loop_c1", "closed-loop_c6", "closed-loop_c8"):
        before_runs = concurrent_cases[OLD_LABEL, name]["runs"]
        after_runs = concurrent_cases[NEW_LABEL, name]["runs"]
        for before_run, after_run in zip(before_runs, after_runs):
            for before, after in zip(before_run["requests"], after_run["requests"]):
                require(
                    before["request"] == after["request"],
                    "Concurrency request bodies differ",
                )
                serving_summary.same_input(before, after, name)
                if not before_run["warmup"]:
                    concurrent_output_pairs.append(
                        (
                            f"{name}/trial{before_run['trial']}/request{before['request_index']}",
                            before,
                            after,
                        )
                    )
    indexed = {(row["label"], row["concurrency"]): row for row in concurrency["rows"]}
    require(
        set(indexed)
        == {(label, count) for label in (OLD_LABEL, NEW_LABEL) for count in (1, 6, 8)},
        "Missing concurrency cases",
    )
    serving_changes = [
        {
            "workload": name,
            OLD_LABEL: serving["throughput"][OLD_LABEL][name],
            NEW_LABEL: serving["throughput"][NEW_LABEL][name],
            **change(
                serving["throughput"][OLD_LABEL][name]["mean_tokens_per_second"],
                serving["throughput"][NEW_LABEL][name]["mean_tokens_per_second"],
            ),
        }
        for name in old_serving["cases"]
    ]
    ordinary = {
        label: statistics.mean(
            serving["throughput"][label][name]["mean_tokens_per_second"]
            for name in serving_summary.PROMPT_NAMES
        )
        for label in (OLD_LABEL, NEW_LABEL)
    }
    ordinary.update(change(ordinary[OLD_LABEL], ordinary[NEW_LABEL]))
    concurrency_changes, scaling = [], {}
    for count in (1, 6, 8):
        left, right = indexed[OLD_LABEL, count], indexed[NEW_LABEL, count]
        require(
            left["requests_per_trial"]
            == right["requests_per_trial"]
            == (8 if count == 1 else 24),
            "Requests per concurrency differ",
        )
        row = {"concurrency": count, OLD_LABEL: left, NEW_LABEL: right}
        for name, key in (
            ("aggregate_change", "aggregate_output_tokens_per_second"),
            ("per_active_change", "latency_weighted_request_output_tokens_per_second"),
            ("latency_change", "mean_request_wall_seconds"),
        ):
            row[name] = change(
                left["metrics"][key]["mean"], right["metrics"][key]["mean"]
            )
        concurrency_changes.append(row)
    for label in (OLD_LABEL, NEW_LABEL):
        first, last = indexed[label, 1]["metrics"], indexed[label, 8]["metrics"]
        scaling[label] = {
            "c8_over_c1_aggregate": last["aggregate_output_tokens_per_second"]["mean"]
            / first["aggregate_output_tokens_per_second"]["mean"],
            "per_active_retention": last[
                "latency_weighted_request_output_tokens_per_second"
            ]["mean"]
            / first["latency_weighted_request_output_tokens_per_second"]["mean"],
        }
    output_pairs = []
    for name in old_serving["cases"]:
        for (location, before), (_, after) in zip(
            serving_summary.measured_requests(old_serving, name),
            serving_summary.measured_requests(new_serving, name),
        ):
            output_pairs.append((location, before, after))
    metrics, memory = collect_windows(raw, candidate, metadata)
    text_smoke = check_text_smokes(
        load(raw / "isq_mtp.text.json"), load(candidate / "text.json")
    )
    mixed_smoke = check_mixed_smokes(
        load(raw / "isq_mtp.mixed_context.json"), load(candidate / "mixed_context.json")
    )
    config_record = load(
        raw / "scaling_investigation/backend_comparison/architecture_comparison.json"
    )["models"]["flash_next"]
    require(
        config_record["config_sha256"] == sha(candidate / "checkpoint/config.json"),
        "Pinned config differs from archived architecture evidence",
    )
    require(
        serving["validation"]["tokenizer_sha256"]
        == sha(candidate / "checkpoint/tokenizer.json"),
        "Saved tokenizer mismatch",
    )
    result = {
        "validation": {
            "complete": True,
            "matching_server_settings": True,
            "matching_prompt_hashes_and_token_counts": True,
            "matching_harnesses_and_tokenizer": True,
            "config_sha256": config_record["config_sha256"],
            "serving": serving["validation"],
            "concurrency": concurrency["validation"],
        },
        "provenance": {
            "baseline_directory": str(raw),
            "candidate_directory": str(candidate),
            "baseline_binary_sha256": OLD_BINARY,
            "candidate_binary_sha256": metadata["binary_sha256"],
            "baseline_metadata_sha256": sha(raw / "isq_mtp.metadata.json"),
            "candidate_metadata_sha256": sha(candidate / "metadata.json"),
            "candidate_artifact_manifest_sha256": sha(candidate / "SHA256SUMS.json"),
            "script_sha256": sha(Path(__file__)),
            "dependencies": dependencies,
        },
        "serving_changes": serving_changes,
        "ordinary_prompt_mean": ordinary,
        "concurrency_changes": concurrency_changes,
        "scaling": scaling,
        "output_comparisons": serving_summary.compare_outputs(output_pairs),
        "concurrency_output_comparisons": serving_summary.compare_outputs(
            concurrent_output_pairs
        ),
        "counters": metrics,
        "memory": memory,
        "text_smoke": text_smoke,
        "mixed_context_smoke": mixed_smoke,
        "limitations": LIMITS,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "comparison.json", result)
    save(args.output / "serving.summary.json", serving)
    save(args.output / "concurrency.summary.json", concurrency)
    (args.output / "comparison.md").write_text(render(result))
    save(
        args.output / "SHA256SUMS.json",
        {path.name: sha(path) for path in args.output.iterdir() if path.is_file()},
    )
    print((args.output / "comparison.md").read_text())


if __name__ == "__main__":
    main()
