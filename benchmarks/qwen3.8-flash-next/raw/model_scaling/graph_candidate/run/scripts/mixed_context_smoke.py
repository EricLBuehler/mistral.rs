#!/usr/bin/env python3
"""Check mixed dense/sparse QSA contexts concurrently; this is not a throughput benchmark."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import platform
import threading
import urllib.error
import urllib.request
from pathlib import Path

import bench_serving
from tokenizers import Tokenizer

PROMPT_LENGTHS = (2304, 1024)
PROMPTS_PER_LENGTH = 4
MAX_TOKENS = 128
CONCURRENCY = len(PROMPT_LENGTHS) * PROMPTS_PER_LENGTH


def reject_nonfinite_json(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def validate_response(response, expected_prompt_tokens):
    if "error" in response:
        raise ValueError(response["error"])
    usage = response["usage"]
    assert usage["prompt_tokens"] == expected_prompt_tokens, usage
    assert usage["completion_tokens"] == MAX_TOKENS, usage
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached == 0, f"Prefix caching must be disabled, reused {cached} tokens"
    probabilities = response["choices"][0].get("logprobs") or {}
    if "content" in probabilities:
        content = probabilities["content"]
        values = [token.get("logprob") for token in content]
        alternatives = [
            entry.get("logprob")
            for token in content
            for entry in token.get("top_logprobs", [])
        ]
    else:
        values = probabilities.get("token_logprobs", [])
        alternatives = [
            value
            for token in probabilities.get("top_logprobs", [])
            for value in token.values()
        ]
    assert len(values) == MAX_TOKENS, f"Expected {MAX_TOKENS} token logprobs"
    for value in values + alternatives:
        assert isinstance(value, (int, float)) and math.isfinite(value), (
            f"Non-finite or missing token logprob: {value}"
        )


def run_request(base_url, case, barrier):
    result = dict(case)
    result["complete"] = False
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(case["request"]).encode(),
            headers={"Content-Type": "application/json"},
        )
        barrier.wait()
        try:
            with urllib.request.urlopen(
                request, timeout=bench_serving.TIMEOUT_SECONDS
            ) as response:
                result["http_status"] = response.status
                result["response_body"] = response.read().decode()
        except urllib.error.HTTPError as error:
            result["http_status"] = error.code
            result["response_body"] = error.read().decode(errors="replace")
            raise
        result["response"] = json.loads(
            result["response_body"], parse_constant=reject_nonfinite_json
        )
        validate_response(result["response"], case["expected_prompt_tokens"])
        result["complete"] = True
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = []
    for index in range(CONCURRENCY):
        length = PROMPT_LENGTHS[index % len(PROMPT_LENGTHS)]
        seed = bench_serving.SEED + index
        prompt = bench_serving.synthetic_prompt(tokenizer, length, seed)
        cases.append(
            {
                "index": index,
                "prompt_seed": seed,
                "expected_prompt_tokens": length,
                "expected_completion_tokens": MAX_TOKENS,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request": {
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "seed": bench_serving.SEED,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "logprobs": 1,
                },
                "complete": False,
            }
        )
    assert len({case["prompt_sha256"] for case in cases}) == CONCURRENCY
    data = {
        "purpose": "Model integration correctness smoke; no throughput claims.",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_prompt_harness_sha256": hashlib.sha256(
            Path(bench_serving.__file__).read_bytes()
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "settings": {
            "base_url": args.base_url,
            "tokenizer": str(args.tokenizer),
            "concurrency": CONCURRENCY,
            "timeout_seconds": bench_serving.TIMEOUT_SECONDS,
            "total_expected_tokens": sum(
                case["expected_prompt_tokens"] + MAX_TOKENS for case in cases
            ),
        },
        "complete": False,
        "requests": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    barrier = threading.Barrier(CONCURRENCY)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(run_request, args.base_url, case, barrier) for case in cases
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cases[result["index"]] = result
            save()
    data["complete"] = all(case["complete"] for case in cases)
    save()
    for case in cases:
        if not case["complete"]:
            print(f"Request {case['index']} failed: {case['error']}", flush=True)
    if not data["complete"]:
        raise SystemExit(1)
    print(
        f"Mixed-context smoke passed: {CONCURRENCY} requests, "
        f"{data['settings']['total_expected_tokens']} total tokens. Saved {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
