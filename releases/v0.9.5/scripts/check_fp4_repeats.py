#!/usr/bin/env python3
"""Repeat the fixed FP4 prompts and compare deterministic text and logprobs."""

import argparse
import datetime
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CASES = {
    512: "p512-c1-measured-0-r0",
    2048: "p2048-c1-measured-0-r1",
    8192: "p8192-c1-measured-0-r2",
}
REPETITIONS = 3
OUTPUT_TOKENS = 128
REQUEST_SEED = 20260904
TOP_LOGPROBS = 5
REQUEST_TIMEOUT_SECONDS = 600


def token_text(item):
    value = item.get("bytes")
    if isinstance(value, list):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(item.get("token"))


def normalize(response):
    choice = response["choices"][0]
    result = []
    for item in (choice.get("logprobs") or {}).get("content", []):
        top = [
            {"token": token_text(candidate), "logprob": candidate["logprob"]}
            for candidate in item.get("top_logprobs", [])
        ]
        top.sort(key=lambda candidate: candidate["logprob"], reverse=True)
        result.append(
            {"token": token_text(item), "logprob": item["logprob"], "top": top}
        )
    return result


def compare(first, later):
    a, b = normalize(first["response"]), normalize(later["response"])
    index = next(
        (i for i, pair in enumerate(zip(a, b)) if pair[0]["token"] != pair[1]["token"]),
        None,
    )
    if index is None and len(a) != len(b):
        index = min(len(a), len(b))
    prefix = min(len(a), len(b)) if index is None else index
    finite_deltas = [
        abs(a[i]["logprob"] - b[i]["logprob"])
        for i in range(prefix)
        if math.isfinite(a[i]["logprob"]) and math.isfinite(b[i]["logprob"])
    ]
    return {
        "input_tokens": first["input_tokens"],
        "first_repetition": first["repetition"],
        "later_repetition": later["repetition"],
        "text_equal": first["response"]["choices"][0]["message"]["content"]
        == later["response"]["choices"][0]["message"]["content"],
        "logprobs_present": bool(a) and bool(b),
        "token_counts": [len(a), len(b)],
        "first_token_difference": index,
        "max_common_prefix_logprob_delta": max(finite_deltas, default=None),
        "first_distribution": a[index]
        if index is not None and index < len(a)
        else None,
        "later_distribution": b[index]
        if index is not None and index < len(b)
        else None,
    }


def check_repeats(base_url, model, output_dir, workload_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    workload_bytes = workload_path.read_bytes()
    workload = json.loads(workload_bytes)
    prompts = {
        prompt["id"]: prompt
        for block in workload["rounds"]
        for prompt in block["prompts"]
    }
    rows = []
    with (output_dir / "responses.jsonl").open("w") as handle:
        for repetition in range(REPETITIONS):
            for length, request_id in CASES.items():
                prompt = prompts[request_id]
                payload = {
                    "model": model,
                    "messages": prompt["messages"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 0,
                    "seed": REQUEST_SEED,
                    "max_tokens": OUTPUT_TOKENS,
                    "ignore_eos": True,
                    "logprobs": True,
                    "top_logprobs": TOP_LOGPROBS,
                }
                row = {
                    "repetition": repetition,
                    "input_tokens": length,
                    "request_id": prompt["id"],
                    "prompt_sha256": prompt["rendered_sha256"],
                    "started_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "request": payload,
                }
                started = time.perf_counter()
                request = urllib.request.Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    row["response"] = json.load(response)
                row["elapsed_seconds"] = time.perf_counter() - started
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                usage = row["response"]["usage"]
                if (
                    usage["prompt_tokens"] != length
                    or usage["completion_tokens"] != OUTPUT_TOKENS
                ):
                    raise RuntimeError(f"incorrect token counts: {usage}")
                if row["response"]["choices"][0]["finish_reason"] != "length":
                    raise RuntimeError(
                        "repeat request did not finish at the token limit"
                    )
                if not normalize(row["response"]):
                    raise RuntimeError("response omitted requested logprobs")
                rows.append(row)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in ("repetition", "input_tokens", "elapsed_seconds")
                        }
                    ),
                    flush=True,
                )
    comparisons = []
    for length in CASES:
        group = [row for row in rows if row["input_tokens"] == length]
        comparisons.extend(compare(group[0], row) for row in group[1:])
    summary = {
        "purpose": "correctness, not performance; nonstreaming logprobs changes serving work",
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "alternating_contexts": list(CASES),
        "repetitions": REPETITIONS,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for comparison in comparisons:
        if not (
            comparison["text_equal"]
            and comparison["logprobs_present"]
            and comparison["first_token_difference"] is None
            and comparison["max_common_prefix_logprob_delta"] == 0.0
        ):
            raise RuntimeError(f"repeat check failed: {comparison}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workload", type=Path, default=RELEASE_ROOT / "raw/prompts/fp4.json"
    )
    args = parser.parse_args()
    check_repeats(args.base_url, args.model, args.output_dir, args.workload)


if __name__ == "__main__":
    main()
