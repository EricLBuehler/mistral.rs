#!/usr/bin/env python3
"""Compare synchronized request bursts with a replenished closed-loop workload.

Each trial submits the same ordered cycle of the eight serving benchmark prompts.
Burst mode waits for each group of C requests before submitting the next group.
Closed-loop mode starts C requests and replaces each completion until the request
count is exhausted. Its optional sustained estimate counts whole responses between
the first completion and last replenishment, not streaming token arrivals.
Latency-weighted request throughput is total output tokens divided by summed
request wall time, equivalent to aggregate throughput divided by mean active requests.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import platform
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import bench_serving

DEFAULT_CONCURRENCIES = (1, 2, 4, 6, 8)
DEFAULT_REQUESTS = 32
DEFAULT_TRIALS = 5
DEFAULT_WARMUP = 2
DEFAULT_OUTPUT_TOKENS = 128
MODES = ("burst", "closed-loop")


def validate_response(response, output_tokens, expected_prompt_tokens):
    if "error" in response:
        raise ValueError(str(response["error"]))
    usage = response["usage"]
    if usage["completion_tokens"] != output_tokens or usage["prompt_tokens"] <= 0:
        raise ValueError(f"Unexpected token counts: {usage}")
    if (
        expected_prompt_tokens is not None
        and usage["prompt_tokens"] != expected_prompt_tokens
    ):
        raise ValueError(
            f"Expected {expected_prompt_tokens} prompt tokens, got {usage['prompt_tokens']}"
        )
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("Prefix caching must be disabled")
    if len(response["choices"]) != 1 or not isinstance(
        response["choices"][0]["text"], str
    ):
        raise ValueError("Expected one text completion")


def request(args, index, submitted_seconds, trial_start):
    name, prompt = list(bench_serving.PROMPTS.items())[
        index % len(bench_serving.PROMPTS)
    ]
    body = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": bench_serving.SEED,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    response, error = None, None
    start_ns = time.perf_counter_ns()
    start = start_ns / 1e9
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as result:
            response = json.load(result)
    except urllib.error.HTTPError as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "status": exc.code}
        response = {"http_error_body": exc.read().decode(errors="replace")}
    except (OSError, ValueError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finish_ns = time.perf_counter_ns()
    finish = finish_ns / 1e9
    if error is None:
        try:
            expected_prompt_tokens = (
                args.expected_prompt_tokens.get(name)
                if args.expected_prompt_tokens
                else None
            )
            validate_response(response, args.max_tokens, expected_prompt_tokens)
        except (KeyError, TypeError, ValueError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = finish - start
    return {
        "request_index": index,
        "prompt_name": name,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "request": body,
        "submitted_seconds": submitted_seconds,
        "started_perf_counter_ns": start_ns,
        "finished_perf_counter_ns": finish_ns,
        "started_seconds": start - trial_start,
        "finished_seconds": finish - trial_start,
        "wall_seconds": elapsed,
        "output_tokens_per_second": args.max_tokens / elapsed
        if error is None
        else None,
        "response": response,
        "error": error,
    }


def sustained_window(results, concurrency, last_replenishment):
    if last_replenishment is None:
        return None
    start = min(result["finished_seconds"] for result in results)
    end = last_replenishment
    completed = [
        result for result in results if start < result["finished_seconds"] <= end
    ]
    if end <= start or len(completed) < concurrency:
        return None
    active_seconds = sum(
        max(
            0.0,
            min(end, result["finished_seconds"])
            - max(start, result["started_seconds"]),
        )
        for result in results
    )
    output_tokens = sum(
        result["response"]["usage"]["completion_tokens"] for result in completed
    )
    return {
        "definition": "whole responses completed after first completion, through last replenishment",
        "start_seconds": start,
        "end_seconds": end,
        "wall_seconds": end - start,
        "completed_request_indices": [result["request_index"] for result in completed],
        "completed_requests": len(completed),
        "completed_output_tokens": output_tokens,
        "completion_boundary_output_tokens_per_second": output_tokens / (end - start),
        "mean_active_requests": active_seconds / (end - start),
    }


def run_trial(args, mode, concurrency):
    results = []
    last_replenishment = None
    trial_start_ns = time.perf_counter_ns()
    trial_start = trial_start_ns / 1e9

    def submit(pool, index):
        submitted = time.perf_counter() - trial_start
        return pool.submit(request, args, index, submitted, trial_start), submitted

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        if mode == "burst":
            for first in range(0, args.requests, concurrency):
                futures = [
                    submit(pool, index)[0]
                    for index in range(first, min(first + concurrency, args.requests))
                ]
                results.extend(
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                )
                if any(result["error"] for result in results):
                    break
        else:
            pending = {submit(pool, index)[0] for index in range(concurrency)}
            next_index = concurrency
            failed = False
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                completed = sorted(
                    (future.result() for future in done),
                    key=lambda result: result["finished_seconds"],
                )
                for result in completed:
                    results.append(result)
                    failed |= result["error"] is not None
                    if next_index < args.requests and not failed:
                        future, last_replenishment = submit(pool, next_index)
                        pending.add(future)
                        next_index += 1
    elapsed = time.perf_counter() - trial_start
    results.sort(key=lambda result: result["request_index"])
    complete = len(results) == args.requests and all(
        result["error"] is None for result in results
    )
    output_tokens = sum(
        result["response"]["usage"]["completion_tokens"]
        for result in results
        if result["error"] is None
    )
    active_seconds = sum(result["wall_seconds"] for result in results)
    window = (
        sustained_window(results, concurrency, last_replenishment) if complete else None
    )
    return {
        "trial_start_perf_counter_ns": trial_start_ns,
        "complete": complete,
        "wall_seconds": elapsed,
        "completed_output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": output_tokens / elapsed,
        "mean_active_requests": active_seconds / elapsed,
        "mean_request_wall_seconds": statistics.mean(
            result["wall_seconds"] for result in results
        ),
        "mean_request_output_tokens_per_second": (
            statistics.mean(result["output_tokens_per_second"] for result in results)
            if complete
            else None
        ),
        "latency_weighted_request_output_tokens_per_second": (
            output_tokens / active_seconds if complete else None
        ),
        "sustained_window": window,
        "sustained_window_unavailable_reason": (
            None
            if window is not None
            else "burst traffic, failed trial, or fewer than C completions between first completion and last replenishment"
        ),
        "requests": results,
    }


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def summarize(runs):
    measured = [run for run in runs if not run["warmup"] and run["complete"]]
    if not measured:
        return None
    summary = {
        field: stats([run[field] for run in measured])
        for field in (
            "aggregate_output_tokens_per_second",
            "mean_active_requests",
            "mean_request_wall_seconds",
            "mean_request_output_tokens_per_second",
            "latency_weighted_request_output_tokens_per_second",
        )
    }
    windows = [
        run["sustained_window"]
        for run in measured
        if run["sustained_window"] is not None
    ]
    summary["sustained_completion_boundary_output_tokens_per_second"] = (
        stats(
            [
                window["completion_boundary_output_tokens_per_second"]
                for window in windows
            ]
        )
        if windows
        else None
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Validate prompt token counts and record this tokenizer's SHA256.",
    )
    parser.add_argument(
        "--concurrencies", nargs="+", type=int, default=list(DEFAULT_CONCURRENCIES)
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Total requests per trial in each mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=float, default=bench_serving.TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.trials < 1 or args.warmup < 0 or args.requests < 1 or args.max_tokens < 1:
        parser.error(
            "trials, requests and max-tokens must be positive; warmup must be nonnegative"
        )
    if any(count < 1 or count > args.requests for count in args.concurrencies):
        parser.error("each concurrency must be positive and no larger than requests")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if len(set(args.concurrencies)) != len(args.concurrencies) or len(
        set(args.modes)
    ) != len(args.modes):
        parser.error("concurrencies and modes must not contain duplicates")

    args.expected_prompt_tokens = None
    if args.tokenizer:
        tokenizer = bench_serving.Tokenizer.from_file(str(args.tokenizer))
        args.expected_prompt_tokens = {
            name: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            for name, prompt in bench_serving.PROMPTS.items()
        }

    data = {
        "label": args.label,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_prompt_harness_sha256": hashlib.sha256(
            Path(bench_serving.__file__).read_bytes()
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        if args.tokenizer
        else None,
        "prompts_sha256": hashlib.sha256(
            json.dumps(bench_serving.PROMPTS, sort_keys=True).encode()
        ).hexdigest(),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "complete": False,
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    for concurrency in args.concurrencies:
        for mode in args.modes:
            data["cases"][f"{mode}_c{concurrency}"] = {
                "mode": mode,
                "concurrency": concurrency,
                "runs": [],
            }
        for trial in range(args.warmup + args.trials):
            for mode in args.modes:
                case = data["cases"][f"{mode}_c{concurrency}"]
                run = run_trial(args, mode, concurrency)
                run.update({"trial": trial, "warmup": trial < args.warmup})
                case["runs"].append(run)
                case["summary"] = summarize(case["runs"])
                save()
                print(
                    f"{mode} C={concurrency} trial={trial} warmup={run['warmup']} "
                    f"aggregate={run['aggregate_output_tokens_per_second']:.2f} tok/s "
                    f"active={run['mean_active_requests']:.2f} "
                    f"request_latency={run['mean_request_wall_seconds']:.3f}s",
                    flush=True,
                )
                if not run["complete"]:
                    raise SystemExit(
                        f"Trial failed; request details saved to {args.output}"
                    )
    data["complete"] = True
    save()


if __name__ == "__main__":
    main()
