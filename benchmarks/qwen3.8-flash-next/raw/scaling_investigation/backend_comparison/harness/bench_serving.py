#!/usr/bin/env python3
"""Measure a local OpenAI-compatible server with reproducible Flash-Next workloads."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import platform
import random
import statistics
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

SEED = 20260929
TIMEOUT_SECONDS = 1800
WORDS = "the river forest city mountain light red blue green moves reflects contains above below through morning evening paper pencil science history question answer explain compare create find write read walk build small large clear simple careful useful quiet bright dark warm cold".split()
PROMPTS = {
    "python": "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "rust": "Write a Rust struct for a 2D point with methods for distance and midpoint.",
    "prose": "Explain in a few paragraphs how photosynthesis works.",
    "json": "Produce a JSON array of five fictional users with id, name, email and age fields.",
    "primes": "List the first twenty prime numbers separated by commas:",
    "math": "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed in m/s?",
    "translation": "Translate to French: The weather is lovely today and we are going to the park after lunch.",
    "quicksort": "def quicksort(arr):\n",
}


def synthetic_prompt(tokenizer, length, seed):
    rng = random.Random(seed)
    text = " ".join(rng.choices(WORDS, k=length * 2))
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:length]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(tokenizer.encode(text, add_special_tokens=False).ids) == length
    return text


def request(base_url, prompt, max_tokens, logprobs=None):
    body = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": SEED,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    if "error" in result:
        raise RuntimeError(result["error"])
    usage = result["usage"]
    if usage["completion_tokens"] != max_tokens:
        raise RuntimeError(f"Expected {max_tokens} tokens, got {usage}")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if cached_tokens:
        raise RuntimeError(
            f"Prefix caching must be disabled, reused {cached_tokens} tokens"
        )
    if logprobs is not None:
        probabilities = result["choices"][0].get("logprobs") or {}
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
        if len(values) != max_tokens:
            raise RuntimeError(f"Expected {max_tokens} token logprobs, got {values}")
        for value in values + alternatives:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"Non-finite or missing token logprob: {value}")
    return {
        "wall_seconds": elapsed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response": result,
    }


def stats(values):
    return {
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--skip-concurrency", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.max_tokens < 1:
        parser.error(
            "iterations and max-tokens must be positive; warmup must be nonnegative"
        )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    data = {
        "label": args.label,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": vars(args)
        | {"tokenizer": str(args.tokenizer), "output": str(args.output)},
        "cases": {},
        "correctness_checks": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    cases = [(f"pp{n}", n, 1) for n in (512, 2048, 8192)]
    cases += [(f"tg{args.max_tokens}_d16", 16, args.max_tokens)]
    for name, length, output_len in cases:
        runs = []
        for i in range(args.warmup + args.iterations):
            prompt = synthetic_prompt(tokenizer, length, SEED + length + i)
            result = request(args.base_url, prompt, output_len)
            if result["response"]["usage"]["prompt_tokens"] != length:
                raise RuntimeError(
                    f"Tokenizer mismatch in {name}: {result['response']['usage']}"
                )
            result["warmup"] = i < args.warmup
            result["prompt_seed"] = SEED + length + i
            runs.append(result)
            data["cases"][name] = runs
            save()
        measured = [r for r in runs if not r["warmup"]]
        print(name, stats([r["wall_seconds"] for r in measured]), flush=True)

    for name, prompt in PROMPTS.items():
        runs = []
        for i in range(args.warmup + args.iterations):
            result = request(args.base_url, prompt, args.max_tokens)
            result["warmup"] = i < args.warmup
            runs.append(result)
            data["cases"][name] = runs
            save()
        print(
            name,
            stats(
                [
                    r["response"]["usage"]["completion_tokens"] / r["wall_seconds"]
                    for r in runs
                    if not r["warmup"]
                ]
            ),
            flush=True,
        )
        data["correctness_checks"][name] = request(
            args.base_url, prompt, args.max_tokens, logprobs=1
        )
        save()

    if not args.skip_concurrency:
        for concurrency in (4, 8):
            rounds = []
            for i in range(args.warmup + args.iterations):
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as pool:
                    futures = [
                        pool.submit(request, args.base_url, p, args.max_tokens)
                        for p in list(PROMPTS.values())[:concurrency]
                    ]
                    results = [future.result() for future in futures]
                elapsed = time.perf_counter() - start
                rounds.append(
                    {
                        "warmup": i < args.warmup,
                        "wall_seconds": elapsed,
                        "aggregate_tokens_per_second": sum(
                            r["response"]["usage"]["completion_tokens"] for r in results
                        )
                        / elapsed,
                        "requests": results,
                    }
                )
                data["cases"][f"concurrency{concurrency}"] = rounds
                save()
            print(
                f"concurrency{concurrency}",
                stats(
                    [
                        r["aggregate_tokens_per_second"]
                        for r in rounds
                        if not r["warmup"]
                    ]
                ),
                flush=True,
            )

    prompt = (
        "The following is a repeated phrase.\n"
        + "The quiet river flows past the green forest. " * 900
        + "\nWrite one sentence about the river."
    )
    data["repetitive_prompt"] = request(args.base_url, prompt, 64, logprobs=1)
    save()


if __name__ == "__main__":
    main()
