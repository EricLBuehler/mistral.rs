#!/usr/bin/env python3
"""Validate concurrency reports and summarize whole-trial serving rates.

Per-active-request throughput is output tokens divided by summed request latency.
Completion-boundary estimates are reported separately from whole-trial metrics.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import bench_serving

FLOAT_REL_TOL = 1e-7
FLOAT_ABS_TOL = 1e-7
MODES = ("burst", "closed-loop")
METRICS = (
    "aggregate_output_tokens_per_second",
    "mean_active_requests",
    "mean_request_wall_seconds",
    "mean_request_output_tokens_per_second",
    "latency_weighted_request_output_tokens_per_second",
)
HASH_FIELDS = (
    "harness_sha256",
    "shared_prompt_harness_sha256",
    "prompts_sha256",
    "tokenizer_sha256",
)
VARIABLE_SETTINGS = {
    "base_url",
    "label",
    "output",
    "tokenizer",
    "concurrencies",
    "modes",
    "requests",
    "trials",
    "warmup",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, f"Invalid {name}: {value}")
    return value


def finite(value, name, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"Invalid {name}: {value}",
    )
    return value


def same_number(actual, expected, name):
    finite(actual, name)
    require(
        math.isclose(actual, expected, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL),
        f"Incorrect {name}: {actual}, expected {expected}",
    )


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def check_stats(actual, expected, name):
    require(set(actual) == set(expected), f"Unexpected statistics fields for {name}")
    for field in ("mean", "sample_stddev"):
        same_number(actual[field], expected[field], f"{name}/{field}")
    require(
        len(actual["samples"]) == len(expected["samples"]),
        f"Wrong sample count for {name}",
    )
    for index, (value, reference) in enumerate(
        zip(actual["samples"], expected["samples"])
    ):
        same_number(value, reference, f"{name}/sample{index}")


def validate_request(result, index, settings, prompt_counts):
    require(
        result["request_index"] == index, "Request indices must be complete and ordered"
    )
    require(result["error"] is None, f"Request {index} failed: {result['error']}")
    name, prompt = list(bench_serving.PROMPTS.items())[
        index % len(bench_serving.PROMPTS)
    ]
    require(result["prompt_name"] == name, f"Wrong prompt cycle at request {index}")
    require(
        result["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(),
        f"Wrong prompt hash at request {index}",
    )
    expected_body = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": settings["max_tokens"],
        "temperature": 0.0,
        "seed": bench_serving.SEED,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    require(
        result["request"] == expected_body,
        f"Request settings differ at request {index}",
    )
    response = result["response"]
    require("error" not in response, f"API error at request {index}")
    require(
        len(response["choices"]) == 1
        and isinstance(response["choices"][0]["text"], str),
        f"Expected one text completion at request {index}",
    )
    usage = response["usage"]
    count = integer(usage["prompt_tokens"], "prompt tokens")
    tokens = integer(usage["completion_tokens"], "completion tokens")
    require(
        tokens == settings["max_tokens"], f"Wrong output token count at request {index}"
    )
    if "total_tokens" in usage:
        require(
            usage["total_tokens"] == count + tokens,
            f"Wrong total token count at request {index}",
        )
    require(
        not (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "Prefix reuse must be disabled",
    )
    if settings["expected_prompt_tokens"] is not None:
        require(
            count == settings["expected_prompt_tokens"][name],
            f"Tokenizer count differs for {name}",
        )
    require(
        count == prompt_counts.setdefault(name, count),
        f"Prompt token counts differ for {name}",
    )
    submitted = finite(result["submitted_seconds"], "submission timestamp")
    start = finite(result["started_seconds"], "start timestamp")
    finish = finite(result["finished_seconds"], "finish timestamp")
    require(submitted <= start < finish, f"Invalid timestamp order at request {index}")
    same_number(result["wall_seconds"], finish - start, f"request {index} duration")
    same_number(
        result["output_tokens_per_second"],
        tokens / (finish - start),
        f"request {index} throughput",
    )
    return tokens


def validate_concurrency(requests, mode, concurrency):
    events = [(request["submitted_seconds"], 1) for request in requests]
    events += [(request["finished_seconds"], -1) for request in requests]
    active = 0
    for _, delta in sorted(events):
        active += delta
        require(
            0 <= active <= concurrency,
            "Submitted request overlap exceeds configured concurrency",
        )
    require(active == 0, "Requests remain active after the trial")
    if mode == "burst":
        for first in range(concurrency, len(requests), concurrency):
            previous_finish = max(
                request["finished_seconds"]
                for request in requests[first - concurrency : first]
            )
            next_submission = min(
                request["submitted_seconds"]
                for request in requests[first : first + concurrency]
            )
            require(previous_finish <= next_submission, "Burst waves overlap")


def reconstruct_window(requests, mode, concurrency):
    if mode == "burst" or len(requests) <= concurrency:
        return None
    start = min(request["finished_seconds"] for request in requests)
    end = max(request["submitted_seconds"] for request in requests[concurrency:])
    completed = [
        request for request in requests if start < request["finished_seconds"] <= end
    ]
    if end <= start or len(completed) < concurrency:
        return None
    tokens = sum(
        request["response"]["usage"]["completion_tokens"] for request in completed
    )
    active_seconds = sum(
        max(
            0.0,
            min(end, request["finished_seconds"])
            - max(start, request["started_seconds"]),
        )
        for request in requests
    )
    return {
        "start_seconds": start,
        "end_seconds": end,
        "wall_seconds": end - start,
        "completed_request_indices": [
            request["request_index"] for request in completed
        ],
        "completed_requests": len(completed),
        "completed_output_tokens": tokens,
        "completion_boundary_output_tokens_per_second": tokens / (end - start),
        "mean_active_requests": active_seconds / (end - start),
    }


def validate_trial(run, index, settings, mode, concurrency, prompt_counts):
    require(run["complete"] is True, "Incomplete trial")
    require(run["trial"] == index, "Trials must be complete and ordered")
    require(run["warmup"] is (index < settings["warmup"]), "Incorrect warmup ordering")
    requests = run["requests"]
    require(len(requests) == settings["requests"], "Wrong request count in trial")
    tokens = sum(
        validate_request(request, i, settings, prompt_counts)
        for i, request in enumerate(requests)
    )
    wall = finite(run["wall_seconds"], "trial duration", positive=True)
    require(
        wall >= max(request["finished_seconds"] for request in requests),
        "Trial ends before requests finish",
    )
    validate_concurrency(requests, mode, concurrency)
    latencies = [
        request["finished_seconds"] - request["started_seconds"] for request in requests
    ]
    active_seconds = sum(latencies)
    reconstructed = {
        "aggregate_output_tokens_per_second": tokens / wall,
        "mean_active_requests": active_seconds / wall,
        "mean_request_wall_seconds": statistics.mean(latencies),
        "mean_request_output_tokens_per_second": statistics.mean(
            request["response"]["usage"]["completion_tokens"] / latency
            for request, latency in zip(requests, latencies)
        ),
        "latency_weighted_request_output_tokens_per_second": tokens / active_seconds,
    }
    require(
        run["completed_output_tokens"] == tokens, "Incorrect trial output token total"
    )
    for metric, expected in reconstructed.items():
        same_number(run[metric], expected, metric)
    window = reconstruct_window(requests, mode, concurrency)
    require(
        (run["sustained_window"] is None) == (window is None),
        "Unexpected sustained-window availability",
    )
    if window is not None:
        for field, expected in window.items():
            if field == "completed_request_indices":
                require(
                    run["sustained_window"][field] == expected,
                    "Wrong sustained-window requests",
                )
            else:
                same_number(
                    run["sustained_window"][field],
                    expected,
                    f"sustained window/{field}",
                )
    reconstructed["completion_boundary_window"] = window
    return reconstructed


def validate_settings(settings):
    for name in ("requests", "trials", "max_tokens"):
        integer(settings[name], name)
    integer(settings["warmup"], "warmup", minimum=0)
    finite(settings["timeout"], "timeout", positive=True)
    require(settings["concurrencies"] and settings["modes"], "No workloads configured")
    require(
        len(set(settings["concurrencies"])) == len(settings["concurrencies"]),
        "Duplicate configured concurrency",
    )
    require(
        len(set(settings["modes"])) == len(settings["modes"]),
        "Duplicate configured mode",
    )
    for concurrency in settings["concurrencies"]:
        integer(concurrency, "concurrency")
        require(
            concurrency <= settings["requests"], "Concurrency exceeds request count"
        )
    require(all(mode in MODES for mode in settings["modes"]), "Unknown workload mode")
    if settings["expected_prompt_tokens"] is not None:
        require(
            set(settings["expected_prompt_tokens"]) == set(bench_serving.PROMPTS),
            "Incomplete expected prompt token counts",
        )
        for count in settings["expected_prompt_tokens"].values():
            integer(count, "expected prompt tokens")


def summarize_reports(reports):
    require(bool(reports), "No reports supplied")
    reference = reports[0]
    rows, seen, prompt_counts = [], set(), {}
    expected_hashes = {
        "harness_sha256": hashlib.sha256(
            Path(__file__).with_name("bench_concurrency.py").read_bytes()
        ).hexdigest(),
        "shared_prompt_harness_sha256": hashlib.sha256(
            Path(bench_serving.__file__).read_bytes()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(
            json.dumps(bench_serving.PROMPTS, sort_keys=True).encode()
        ).hexdigest(),
    }
    for report in reports:
        require(report["complete"] is True, f"Incomplete report: {report['label']}")
        settings = report["settings"]
        validate_settings(settings)
        require(
            isinstance(report["label"], str) and report["label"], "Empty report label"
        )
        require(
            settings["label"] == report["label"], "Report and settings labels differ"
        )
        for field in HASH_FIELDS:
            value = report[field]
            require(
                (field == "tokenizer_sha256" and value is None)
                or isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{64}", value),
                f"Invalid {field}",
            )
            require(
                value == reference[field], f"Different {field} in {report['label']}"
            )
            if field in expected_hashes:
                require(
                    value == expected_hashes[field],
                    f"{field} does not match the checked-in harness/prompts",
                )
        require(
            (report["tokenizer_sha256"] is None)
            == (settings["expected_prompt_tokens"] is None),
            "Tokenizer hash and expected counts must accompany each other",
        )
        shared = {
            key: value
            for key, value in settings.items()
            if key not in VARIABLE_SETTINGS
        }
        expected_shared = {
            key: value
            for key, value in reference["settings"].items()
            if key not in VARIABLE_SETTINGS
        }
        require(
            shared == expected_shared, f"Request settings differ in {report['label']}"
        )
        expected_cases = {
            f"{mode}_c{concurrency}"
            for mode in settings["modes"]
            for concurrency in settings["concurrencies"]
        }
        require(
            set(report["cases"]) == expected_cases, "Incomplete or unexpected cases"
        )
        for name, case in report["cases"].items():
            mode, concurrency = case["mode"], case["concurrency"]
            require(name == f"{mode}_c{concurrency}", "Case name and settings differ")
            collision = (report["label"], mode, concurrency)
            require(
                collision not in seen, f"Duplicate label/mode/concurrency: {collision}"
            )
            seen.add(collision)
            require(
                len(case["runs"]) == settings["warmup"] + settings["trials"],
                f"Incomplete trials for {collision}",
            )
            runs = [
                validate_trial(run, i, settings, mode, concurrency, prompt_counts)
                for i, run in enumerate(case["runs"])
            ]
            measured = runs[settings["warmup"] :]
            metrics = {
                metric: stats([run[metric] for run in measured]) for metric in METRICS
            }
            windows = [
                run["completion_boundary_window"]
                for run in measured
                if run["completion_boundary_window"] is not None
            ]
            boundary_stats = (
                stats(
                    [
                        window["completion_boundary_output_tokens_per_second"]
                        for window in windows
                    ]
                )
                if windows
                else None
            )
            require(
                set(case["summary"])
                == set(METRICS)
                | {"sustained_completion_boundary_output_tokens_per_second"},
                "Unexpected stored summary fields",
            )
            for metric, expected in metrics.items():
                check_stats(case["summary"][metric], expected, f"{collision}/{metric}")
            actual_boundary = case["summary"][
                "sustained_completion_boundary_output_tokens_per_second"
            ]
            if boundary_stats is None:
                require(actual_boundary is None, "Unexpected stored boundary summary")
            else:
                check_stats(
                    actual_boundary, boundary_stats, f"{collision}/completion boundary"
                )
            rows.append(
                {
                    "label": report["label"],
                    "mode": mode,
                    "concurrency": concurrency,
                    "requests_per_trial": settings["requests"],
                    "output_tokens_per_request": settings["max_tokens"],
                    "measured_trials": settings["trials"],
                    "warmup_trials": settings["warmup"],
                    "metrics": metrics,
                    "completion_boundary_estimate_tokens_per_second": boundary_stats,
                    "measured_trials_recomputed": measured,
                }
            )
    return {
        "validation": {
            "complete": True,
            **{field: reference[field] for field in HASH_FIELDS},
            "prompt_token_counts": prompt_counts,
            "shared_request_settings": expected_shared,
            "metrics_recomputed_from_timestamps_and_token_counts": True,
        },
        "rows": rows,
    }


def cell(value, digits=1):
    return f"{value['mean']:.{digits}f} +/- {value['sample_stddev']:.{digits}f}"


def print_tables(result):
    print(
        "Whole-trial rates: mean +/- sample standard deviation across measured trials."
    )
    print()
    print(
        "| Label | Mode | C | Requests/trial | Trials | Aggregate tok/s | Per active request tok/s | Mean active | Mean latency (s) |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result["rows"]:
        metrics = row["metrics"]
        print(
            f"| {row['label']} | {row['mode']} | {row['concurrency']} | {row['requests_per_trial']} | {row['measured_trials']} | "
            f"{cell(metrics['aggregate_output_tokens_per_second'])} | {cell(metrics['latency_weighted_request_output_tokens_per_second'])} | "
            f"{cell(metrics['mean_active_requests'], 2)} | {cell(metrics['mean_request_wall_seconds'], 3)} |"
        )
    windows = [
        row
        for row in result["rows"]
        if row["completion_boundary_estimate_tokens_per_second"] is not None
    ]
    if windows:
        print(
            "\nSecondary completion-boundary estimates count whole responses, not streaming token arrivals.\n"
        )
        print(
            "| Label | Mode | C | Requests/trial | Completion-boundary estimate tok/s |"
        )
        print("| --- | --- | ---: | ---: | ---: |")
        for row in windows:
            print(
                f"| {row['label']} | {row['mode']} | {row['concurrency']} | {row['requests_per_trial']} | {cell(row['completion_boundary_estimate_tokens_per_second'])} |"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize_reports(
            [json.loads(path.read_text()) for path in args.results]
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print_tables(result)


if __name__ == "__main__":
    main()
