#!/usr/bin/env python3
"""Validate and summarize Flash-Next serving benchmarks using client wall time."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

PREFILL_LENGTHS = (512, 2048, 8192)
PROMPT_NAMES = (
    "python",
    "rust",
    "prose",
    "json",
    "primes",
    "math",
    "translation",
    "quicksort",
)
CONCURRENCIES = (4, 8)
LOCATION_SETTINGS = {"base_url", "tokenizer", "output", "label"}
REPETITIVE_OUTPUT_TOKENS = 64


def shared_settings(report):
    return {k: v for k, v in report["settings"].items() if k not in LOCATION_SETTINGS}


def requests(run):
    return run["requests"] if "requests" in run else [run]


def validate_request(run, output_tokens, logprobs=False):
    if not math.isfinite(run["wall_seconds"]) or run["wall_seconds"] <= 0:
        raise ValueError("Request duration must be finite and positive")
    response = run["response"]
    usage = response["usage"]
    if usage["completion_tokens"] != output_tokens or usage["prompt_tokens"] <= 0:
        raise ValueError(f"Unexpected token counts: {usage}")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("Prefix caching must be disabled")
    if len(response["choices"]) != 1 or not isinstance(
        response["choices"][0]["text"], str
    ):
        raise ValueError("Expected one text completion")
    if not logprobs:
        return
    probabilities = response["choices"][0].get("logprobs") or {}
    if "content" in probabilities:
        tokens = probabilities["content"]
        values = [token.get("logprob") for token in tokens]
        alternatives = [
            alternative.get("logprob")
            for token in tokens
            for alternative in token.get("top_logprobs", [])
        ]
    else:
        values = probabilities.get("token_logprobs", [])
        alternatives = [
            value
            for token in probabilities.get("top_logprobs", [])
            for value in token.values()
        ]
    if len(values) != output_tokens or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values + alternatives
    ):
        raise ValueError("Validation request has missing or non-finite token logprobs")


def validate_report(report):
    settings = report["settings"]
    output_tokens = settings["max_tokens"]
    warmup, iterations = settings["warmup"], settings["iterations"]
    if iterations < 1 or warmup < 0 or output_tokens < 1:
        raise ValueError("Invalid benchmark settings")
    expected = {f"pp{length}" for length in PREFILL_LENGTHS} | set(PROMPT_NAMES)
    expected.add(f"tg{output_tokens}_d16")
    if not settings["skip_concurrency"]:
        expected.update(f"concurrency{count}" for count in CONCURRENCIES)
    if set(report["cases"]) != expected:
        raise ValueError(
            f"Incomplete or unexpected workload cases for {report['label']}"
        )
    if set(report["correctness_checks"]) != set(PROMPT_NAMES):
        raise ValueError(f"Incomplete validation requests for {report['label']}")
    for name, runs in report["cases"].items():
        if len(runs) != warmup + iterations:
            raise ValueError(f"Incomplete samples for {report['label']}/{name}")
        for index, run in enumerate(runs):
            if run["warmup"] != (index < warmup):
                raise ValueError(f"Unexpected warmup ordering for {name}")
            count = (
                int(name.removeprefix("concurrency"))
                if name.startswith("concurrency")
                else 1
            )
            if len(requests(run)) != count:
                raise ValueError(f"Unexpected request count for {name}")
            if not math.isfinite(run["wall_seconds"]) or run["wall_seconds"] <= 0:
                raise ValueError(f"Invalid batch duration for {name}")
            for request in requests(run):
                validate_request(request, 1 if name.startswith("pp") else output_tokens)
    for name, run in report["correctness_checks"].items():
        validate_request(run, output_tokens, logprobs=True)
        same_input(run, report["cases"][name][0], name)
    if "repetitive_prompt" not in report:
        raise ValueError(
            f"Missing final repetitive-prompt validation for {report['label']}"
        )
    validate_request(
        report["repetitive_prompt"], REPETITIVE_OUTPUT_TOKENS, logprobs=True
    )


def same_input(run, reference, context):
    if run["prompt_sha256"] != reference["prompt_sha256"]:
        raise ValueError(f"Prompt hashes differ for {context}")
    for key in ("prompt_tokens", "completion_tokens"):
        if run["response"]["usage"][key] != reference["response"]["usage"][key]:
            raise ValueError(f"{key} differ for {context}")


def compare_outputs(pairs):
    count, identical, differences = 0, 0, []
    for location, reference, candidate in pairs:
        count += 1
        left = reference["response"]["choices"][0]["text"]
        right = candidate["response"]["choices"][0]["text"]
        if left == right:
            identical += 1
            continue
        prefix = next(
            (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
            min(len(left), len(right)),
        )
        differences.append(
            {
                "location": location,
                "common_prefix_characters": prefix,
                "reference_characters": len(left),
                "candidate_characters": len(right),
                "reference_text_sha256": hashlib.sha256(left.encode()).hexdigest(),
                "candidate_text_sha256": hashlib.sha256(right.encode()).hexdigest(),
            }
        )
    return {
        "comparisons": count,
        "identical_text": identical,
        "differences": differences,
    }


def measured_requests(report, name):
    for sample, run in enumerate(report["cases"][name]):
        if not run["warmup"]:
            for request_index, request in enumerate(requests(run)):
                yield f"{name}/sample{sample}/request{request_index}", request


def output_diagnostics(report):
    repeats, validation = [], []
    for name in report["cases"]:
        first = {}
        for location, request in measured_requests(report, name):
            prompt = request["prompt_sha256"]
            if prompt in first:
                repeats.append((location, first[prompt], request))
            else:
                first[prompt] = request
            if name in report["correctness_checks"]:
                validation.append(
                    (location, request, report["correctness_checks"][name])
                )
    return {
        "timed_repeats": compare_outputs(repeats),
        "validation_vs_timed": compare_outputs(validation),
    }


def summarize(reports):
    reference = reports[0]
    labels = [report["label"] for report in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("Report labels must be unique")
    throughput, diagnostics = {}, {}
    for report in reports:
        validate_report(report)
        for key in ("tokenizer_sha256", "harness_sha256"):
            if not report[key] or report[key] != reference[key]:
                raise ValueError(f"{key} differs for {report['label']}")
        if shared_settings(report) != shared_settings(reference):
            raise ValueError(f"Benchmark settings differ for {report['label']}")
        cases = {}
        for name, runs in report["cases"].items():
            values = []
            for run, original in zip(runs, reference["cases"][name]):
                for request, expected in zip(requests(run), requests(original)):
                    same_input(request, expected, name)
                count_key = (
                    "prompt_tokens" if name.startswith("pp") else "completion_tokens"
                )
                tokens = sum(
                    request["response"]["usage"][count_key] for request in requests(run)
                )
                if not run["warmup"]:
                    values.append(tokens / run["wall_seconds"])
            cases[name] = {
                "mean_tokens_per_second": statistics.mean(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "samples": values,
            }
        for name, check in report["correctness_checks"].items():
            same_input(check, reference["correctness_checks"][name], name)
        same_input(
            report["repetitive_prompt"],
            reference["repetitive_prompt"],
            "repetitive_prompt",
        )
        throughput[report["label"]] = cases
        diagnostics[report["label"]] = output_diagnostics(report)
    by_label = {report["label"]: report for report in reports}
    mtp_comparison = None
    if "isq" in by_label and "isq_mtp" in by_label:
        pairs = []
        for name in reference["cases"]:
            for (location, baseline), (_, mtp) in zip(
                measured_requests(by_label["isq"], name),
                measured_requests(by_label["isq_mtp"], name),
            ):
                pairs.append((location, baseline, mtp))
        for name, baseline in by_label["isq"]["correctness_checks"].items():
            pairs.append(
                (
                    f"validation/{name}",
                    baseline,
                    by_label["isq_mtp"]["correctness_checks"][name],
                )
            )
        pairs.append(
            (
                "repetitive_prompt",
                by_label["isq"]["repetitive_prompt"],
                by_label["isq_mtp"]["repetitive_prompt"],
            )
        )
        mtp_comparison = compare_outputs(pairs)
    return {
        "validation": {
            "complete": True,
            "tokenizer_sha256": reference["tokenizer_sha256"],
            "harness_sha256": reference["harness_sha256"],
            "shared_settings": shared_settings(reference),
            "matching_prompt_hashes_and_token_counts": True,
            "finite_validation_logprobs": True,
        },
        "throughput": throughput,
        "output_comparisons": {
            "within_variants": diagnostics,
            "isq_mtp_vs_isq": mtp_comparison,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.results]
    result = summarize(reports)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = result["throughput"]
    print("| Workload | " + " | ".join(summary) + " |")
    print("| --- | " + " | ".join("---:" for _ in summary) + " |")
    for name in reports[0]["cases"]:
        cells = []
        for label in summary:
            value = summary[label][name]
            cells.append(
                f"{value['mean_tokens_per_second']:.1f} +/- {value['sample_stddev']:.1f}"
            )
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
