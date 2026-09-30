"""Homogeneous-input finite-wave diagnostic; identical inputs do not imply identical routes."""

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

import bench_concurrency as shared
import bench_serving

PROMPT_NAMES = ("python", "math")
CONCURRENCIES = (1, 8)
REQUEST_COUNTS = {1: 8, 8: 24}
WARMUPS = 2
TRIALS = 3
OUTPUT_TOKENS = 128


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def agreement(hashes):
    counts = Counter(hashes)
    pairs = len(hashes) * (len(hashes) - 1) // 2
    return {
        "request_count": len(hashes),
        "unique_output_text_hashes": len(counts),
        "hash_counts": dict(sorted(counts.items())),
        "all_outputs_identical": len(counts) == 1,
        "equal_pair_fraction": (
            sum(count * (count - 1) // 2 for count in counts.values()) / pairs
            if pairs
            else None
        ),
    }


def wave_records(run, concurrency):
    requests = run["requests"]
    waves = []
    for first in range(0, len(requests), concurrency):
        wave = requests[first : first + concurrency]
        valid = [item for item in wave if item["error"] is None]
        waves.append(
            {
                "request_indices": [item["request_index"] for item in wave],
                "started_seconds": min(item["started_seconds"] for item in wave),
                "finished_seconds": max(item["finished_seconds"] for item in wave),
                "mean_active_requests": sum(item["wall_seconds"] for item in wave)
                / (
                    max(item["finished_seconds"] for item in wave)
                    - min(item["started_seconds"] for item in wave)
                ),
                "complete": len(valid) == len(wave),
                "agreement": agreement([item["output_text_sha256"] for item in valid]),
            }
        )
    return waves


def annotate(run, concurrency):
    for item in run["requests"]:
        item["response_sha256"] = json_sha(item["response"])
        item["output_text_sha256"] = (
            sha_bytes(item["response"]["choices"][0]["text"].encode())
            if item["error"] is None
            else None
        )
    run["waves"] = wave_records(run, concurrency)


def measure(args):
    prompt = bench_serving.PROMPTS[args.prompt_name]
    source_prompts_sha = json_sha(bench_serving.PROMPTS)
    tokenizer = bench_serving.Tokenizer.from_file(str(args.tokenizer))
    args.expected_prompt_tokens = {
        args.prompt_name: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    }
    args.requests = REQUEST_COUNTS[args.concurrency]
    args.max_tokens = OUTPUT_TOKENS
    data = {
        "complete": False,
        "diagnostic": "homogeneous input, synchronized finite waves; not a normal mixed-prompt benchmark",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "label": args.label,
        "source_prompts_sha256": source_prompts_sha,
        "source_files_sha256": {
            Path(path).name: sha_bytes(Path(path).read_bytes())
            for path in (__file__, shared.__file__, bench_serving.__file__)
        },
        "tokenizer_sha256": sha_bytes(args.tokenizer.read_bytes()),
        "prompt_name": args.prompt_name,
        "prompt": prompt,
        "prompt_sha256": sha_bytes(prompt.encode()),
        "expected_prompt_tokens": args.expected_prompt_tokens[args.prompt_name],
        "settings": {
            "mode": "burst",
            "concurrency": args.concurrency,
            "requests": args.requests,
            "warmup": WARMUPS,
            "trials": TRIALS,
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "seed": bench_serving.SEED,
            "ignore_eos": True,
            "cache_prompt": False,
            "logprobs": None,
            "base_url": args.base_url,
            "timeout": args.timeout,
        },
        "runs": [],
        "limits": [
            "Only this subprocess replaces the shared prompt cycle with the selected original prompt; shared scheduling and request code are unchanged.",
            "Every wave waits for all responses before the next wave; rates include prefill, startup, drain, and client overhead.",
            "Identical inputs and greedy sampling do not guarantee identical expert routes or outputs across batch shapes or adaptive MTP states.",
            "C1 has eight sequential requests; C8 has three waves of eight requests. This is a finite-wave diagnostic, not sustained traffic.",
        ],
    }
    original = bench_serving.PROMPTS
    bench_serving.PROMPTS = {args.prompt_name: prompt}
    save(args.output, data)
    try:
        for trial in range(WARMUPS + TRIALS):
            run = shared.run_trial(args, "burst", args.concurrency)
            run.update(trial=trial, warmup=trial < WARMUPS)
            annotate(run, args.concurrency)
            data["runs"].append(run)
            data["summary"] = shared.summarize(data["runs"])
            save(args.output, data)
            print(
                f"{args.prompt_name} C={args.concurrency} trial={trial} "
                f"warmup={run['warmup']} aggregate={run['aggregate_output_tokens_per_second']:.3f} "
                f"active={run['mean_active_requests']:.3f}",
                flush=True,
            )
            if not run["complete"]:
                raise RuntimeError(f"Incomplete diagnostic trial; see {args.output}")
        data["complete"] = True
        save(args.output, data)
    finally:
        bench_serving.PROMPTS = original


def equal_number(actual, expected):
    assert math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=1e-9, abs_tol=1e-9
    )


def validate(data):
    assert data["complete"] is True
    name = data["prompt_name"]
    prompt = bench_serving.PROMPTS[name]
    assert name in PROMPT_NAMES and data["prompt"] == prompt
    assert data["prompt_sha256"] == sha_bytes(prompt.encode())
    assert data["source_prompts_sha256"] == json_sha(bench_serving.PROMPTS)
    for path in (__file__, shared.__file__, bench_serving.__file__):
        assert data["source_files_sha256"][Path(path).name] == sha_bytes(
            Path(path).read_bytes()
        )
    settings = data["settings"]
    concurrency = settings["concurrency"]
    assert concurrency in CONCURRENCIES
    assert settings["requests"] == REQUEST_COUNTS[concurrency]
    assert settings["mode"] == "burst"
    assert settings["temperature"] == 0.0
    assert settings["seed"] == bench_serving.SEED
    assert settings["ignore_eos"] is True and settings["cache_prompt"] is False
    assert settings["logprobs"] is None
    assert data["expected_prompt_tokens"] > 0
    assert (settings["warmup"], settings["trials"], settings["max_tokens"]) == (
        WARMUPS,
        TRIALS,
        OUTPUT_TOKENS,
    )
    assert len(data["runs"]) == WARMUPS + TRIALS
    measured = []
    for index, run in enumerate(data["runs"]):
        assert (
            run["complete"]
            and run["trial"] == index
            and run["warmup"] == (index < WARMUPS)
        )
        assert math.isfinite(run["wall_seconds"]) and run["wall_seconds"] > 0
        requests = run["requests"]
        assert [item["request_index"] for item in requests] == list(
            range(settings["requests"])
        )
        events = []
        for item in requests:
            assert item["error"] is None and item["prompt_name"] == name
            assert item["prompt_sha256"] == data["prompt_sha256"]
            assert item["request"] == {
                "model": "default",
                "prompt": prompt,
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0.0,
                "seed": bench_serving.SEED,
                "ignore_eos": True,
                "cache_prompt": False,
            }
            shared.validate_response(
                item["response"], OUTPUT_TOKENS, data["expected_prompt_tokens"]
            )
            assert item["response_sha256"] == json_sha(item["response"])
            assert item["output_text_sha256"] == sha_bytes(
                item["response"]["choices"][0]["text"].encode()
            )
            start, end = item["started_seconds"], item["finished_seconds"]
            assert 0 <= item["submitted_seconds"] <= start < end <= run["wall_seconds"]
            equal_number(item["wall_seconds"], end - start)
            equal_number(
                item["output_tokens_per_second"], OUTPUT_TOKENS / item["wall_seconds"]
            )
            events.extend(((start, 1), (end, -1)))
        active = peak = 0
        for _, delta in sorted(events):
            active += delta
            peak = max(peak, active)
        assert active == 0 and peak <= concurrency
        waves = wave_records(run, concurrency)
        assert run["waves"] == waves
        assert all(
            left["finished_seconds"] <= right["started_seconds"]
            for left, right in zip(waves, waves[1:])
        )
        total_tokens = settings["requests"] * OUTPUT_TOKENS
        active_seconds = sum(item["wall_seconds"] for item in requests)
        derived = {
            "aggregate_output_tokens_per_second": total_tokens / run["wall_seconds"],
            "latency_weighted_request_output_tokens_per_second": total_tokens
            / active_seconds,
            "mean_active_requests": active_seconds / run["wall_seconds"],
            "mean_request_wall_seconds": active_seconds / len(requests),
        }
        assert run["completed_output_tokens"] == total_tokens
        for field, value in derived.items():
            equal_number(run[field], value)
        if not run["warmup"]:
            measured.append(derived)
    summary = {
        field: shared.stats([run[field] for run in measured]) for field in measured[0]
    }
    for field, stats in summary.items():
        for key in ("mean", "sample_stddev"):
            equal_number(data["summary"][field][key], stats[key])
    return summary


def summarize(args):
    cases = {}
    identities = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        key = (data["prompt_name"], data["settings"]["concurrency"])
        assert key not in cases, f"Duplicate homogeneous case: {key}"
        summary = validate(data)
        cases[key] = {
            "path": str(path),
            "sha256": sha_bytes(path.read_bytes()),
            "data": data,
            "summary": summary,
        }
        identities.append(
            {
                field: data[field]
                for field in (
                    "label",
                    "source_prompts_sha256",
                    "source_files_sha256",
                    "tokenizer_sha256",
                )
            }
        )
    assert all(identity == identities[0] for identity in identities)
    assert set(cases) == {(name, c) for name in PROMPT_NAMES for c in CONCURRENCIES}
    rows = []
    for name in PROMPT_NAMES:
        left, right = (cases[(name, c)] for c in CONCURRENCIES)
        assert (
            left["data"]["expected_prompt_tokens"]
            == right["data"]["expected_prompt_tokens"]
        )
        excluded = {"concurrency", "requests"}
        assert {
            k: v for k, v in left["data"]["settings"].items() if k not in excluded
        } == {k: v for k, v in right["data"]["settings"].items() if k not in excluded}
        output_hashes = {
            str(c): [
                item["output_text_sha256"]
                for run in cases[(name, c)]["data"]["runs"]
                if not run["warmup"]
                for item in run["requests"]
            ]
            for c in CONCURRENCIES
        }
        first_counts, second_counts = (
            Counter(output_hashes[str(c)]) for c in CONCURRENCIES
        )
        cross_equal = sum(
            count * second_counts[key] for key, count in first_counts.items()
        )
        rows.append(
            {
                "prompt_name": name,
                "prompt_sha256": left["data"]["prompt_sha256"],
                "c1": left["summary"],
                "c8": right["summary"],
                "aggregate_c8_over_c1": right["summary"][
                    "aggregate_output_tokens_per_second"
                ]["mean"]
                / left["summary"]["aggregate_output_tokens_per_second"]["mean"],
                "per_active_c8_over_c1": right["summary"][
                    "latency_weighted_request_output_tokens_per_second"
                ]["mean"]
                / left["summary"]["latency_weighted_request_output_tokens_per_second"][
                    "mean"
                ],
                "measured_output_agreement": {
                    str(c): agreement(output_hashes[str(c)]) for c in CONCURRENCIES
                },
                "c1_c8_equal_output_pair_fraction": cross_equal
                / (len(output_hashes["1"]) * len(output_hashes["8"])),
                "measured_waves": {
                    str(c): [
                        wave
                        for run in cases[(name, c)]["data"]["runs"]
                        if not run["warmup"]
                        for wave in run["waves"]
                    ]
                    for c in CONCURRENCIES
                },
            }
        )
    result = {
        "complete": True,
        "diagnostic": "same-prompt C1 versus C8 finite waves",
        "identity": identities[0],
        "rows": rows,
        "inputs": [
            {"path": value["path"], "sha256": value["sha256"]}
            for value in cases.values()
        ],
        "definition": "Mean +/- sample SD across three measured trials after two warmups; C1 has eight requests, C8 has three waves of eight. Whole-trial rates include prefill/client/startup/drain. Per-active rate is output tokens / summed request latency.",
        "limits": [
            "Homogeneous inputs do not prove identical expert routing or weight reuse.",
            "Batch-dependent numerics and adaptive MTP can change outputs, acceptance, and depth; output equality is diagnostic, not required.",
            "Sequential phases are not randomized causal trials. Metrics phase boundaries include warmups; these finite-wave results are separate from normal mixed-prompt serving and closed-loop results.",
        ],
    }
    save(args.output, result)
    print(result["definition"])
    print(
        "Prompt | C1 tok/s | C8 tok/s | C8/C1 | C1 per-active | C8 per-active | C8 active"
    )
    for row in rows:
        rate = "aggregate_output_tokens_per_second"
        per_active = "latency_weighted_request_output_tokens_per_second"
        print(
            f"{row['prompt_name']} | {row['c1'][rate]['mean']:.3f} +/- {row['c1'][rate]['sample_stddev']:.3f} | {row['c8'][rate]['mean']:.3f} +/- {row['c8'][rate]['sample_stddev']:.3f} | {row['aggregate_c8_over_c1']:.3f}x | {row['c1'][per_active]['mean']:.3f} | {row['c8'][per_active]['mean']:.3f} | {row['c8']['mean_active_requests']['mean']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--base-url", default="http://127.0.0.1:1234")
    run.add_argument("--label", required=True)
    run.add_argument("--tokenizer", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prompt-name", choices=PROMPT_NAMES, required=True)
    run.add_argument("--concurrency", type=int, choices=CONCURRENCIES, required=True)
    run.add_argument("--timeout", type=float, default=bench_serving.TIMEOUT_SECONDS)
    summary = commands.add_parser("summarize")
    summary.add_argument("inputs", nargs="+", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("timeout must be finite and positive")
        measure(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
