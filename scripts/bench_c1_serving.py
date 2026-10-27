#!/usr/bin/env python3
"""Measure C1 TTFT and decode through an OpenAI-compatible streaming endpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

DEFAULT_LENGTHS = (512, 2048, 8192)
DEFAULT_OUTPUT_TOKENS = 128
REQUEST_TIMEOUT_SECONDS = 1800
CORPUS_FILES = (
    "mistralrs-core/src/vision_models/qwen4_exp/text.rs",
    "mistralrs-core/src/vision_models/qwen4_exp/qsa.rs",
    "mistralrs-core/src/vision_models/qwen4_exp/ple.rs",
    "mistralrs-core/src/moe/experts/backends.rs",
)


def prepare_prompts(tokenizer_path, output, lengths, count):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    root = Path(__file__).resolve().parents[1]
    corpus = "\n\n".join((root / name).read_text() for name in CORPUS_FILES)
    tokens = tokenizer.encode(corpus, add_special_tokens=False).ids
    prompts = []
    for length in lengths:
        for index in range(count):
            prefix = tokenizer.encode(
                f"Code review request {index + 1}. Explain the behavior and performance "
                "of the following Rust inference implementation. Discuss concrete "
                "optimization opportunities and correctness constraints.\n\n",
                add_special_tokens=False,
            ).ids
            offset = (index * len(tokens) // count) % len(tokens)
            body = (tokens[offset:] + tokens[:offset]) * (length // len(tokens) + 1)
            prompt = (prefix + body)[:length]
            prompts.append({"length": length, "index": index, "tokens": prompt})
    output.write_text(json.dumps(prompts) + "\n")


def token_counters(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics") as response:
        lines = response.read().decode().splitlines()
    names = (
        "mistralrs_prefill_tokens_processed_total",
        "mistralrs_decode_tokens_processed_total",
    )
    return {
        name: int(
            float(
                next(
                    (line.split()[1] for line in lines if line.startswith(name + " ")),
                    "0",
                )
            )
        )
        for name in names
    }


def request(
    url, model, prompt, output_tokens, prompt_tokens=None, mistralrs_metrics=False
):
    if prompt_tokens is None:
        prompt_tokens = len(prompt)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    before = token_counters(url) if mistralrs_metrics else None
    start = time.perf_counter()
    first = last = None
    usage = None
    text = []
    token_times = []
    chunks = 0
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("text", "")
                if content:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    token_times.append(now - start)
                    text.append(content)
                    chunks += 1
    elapsed = time.perf_counter() - start
    usage_source = "stream"
    if usage is None and before is not None:
        after = token_counters(url)
        usage = {
            "prompt_tokens": after["mistralrs_prefill_tokens_processed_total"]
            - before["mistralrs_prefill_tokens_processed_total"],
            "completion_tokens": after["mistralrs_decode_tokens_processed_total"]
            - before["mistralrs_decode_tokens_processed_total"]
            + 1,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage_source = "isolated_server_counters"
    if first is None or usage is None:
        raise RuntimeError(f"Missing streamed tokens or usage: {usage=}, {chunks=}")
    completed = usage["completion_tokens"]
    if completed != output_tokens:
        raise RuntimeError(
            f"Expected {output_tokens} output tokens, received {completed}"
        )
    if usage["prompt_tokens"] != prompt_tokens:
        raise RuntimeError(
            f"Prompt token count differs: {usage['prompt_tokens']} vs {prompt_tokens}"
        )
    ttft = first - start
    decode = last - first
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": completed,
        "ttft_seconds": ttft,
        "prompt_tokens_per_second": prompt_tokens / ttft,
        "tpot_seconds": decode / (completed - 1),
        "decode_tokens_per_second": (completed - 1) / decode,
        "elapsed_seconds": elapsed,
        "text_chunks": chunks,
        "token_timestamps_seconds": token_times,
        "text": "".join(text),
        "usage": usage,
        "usage_source": usage_source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1235")
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prepare-tokenizer", type=Path)
    parser.add_argument("--text-tokenizer", type=Path)
    parser.add_argument("--mistralrs-metrics", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_tokenizer:
        prepare_prompts(
            args.prepare_tokenizer,
            args.prompts,
            args.lengths,
            args.repetitions + args.warmup,
        )
        return
    prompts = json.loads(args.prompts.read_text())
    tokenizer = None
    if args.text_tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.text_tokenizer))
    results = []
    report = {
        "url": args.url,
        "model": args.model,
        "prompt_transport": "text" if tokenizer else "token_ids",
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "complete": False,
        "results": results,
    }
    for length in args.lengths:
        cases = [p for p in prompts if p["length"] == length]
        for index, case in enumerate(cases[: args.repetitions + args.warmup]):
            prompt = case["tokens"]
            if tokenizer:
                prompt = tokenizer.decode(prompt, skip_special_tokens=False)
                if (
                    tokenizer.encode(prompt, add_special_tokens=True).ids
                    != case["tokens"]
                ):
                    raise RuntimeError("Text transport changes the fixture's token IDs")
            result = request(
                args.url,
                args.model,
                prompt,
                args.output_tokens,
                len(case["tokens"]),
                args.mistralrs_metrics,
            )
            result.update(index=index, warmup=index < args.warmup)
            results.append(result)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("text", "usage", "token_timestamps_seconds")
                    }
                ),
                flush=True,
            )
    summary = {}
    for length in args.lengths:
        rows = [r for r in results if r["prompt_tokens"] == length and not r["warmup"]]
        summary[str(length)] = {
            key: statistics.median(row[key] for row in rows)
            for key in (
                "ttft_seconds",
                "prompt_tokens_per_second",
                "tpot_seconds",
                "decode_tokens_per_second",
            )
        }
    report.update(complete=True, median=summary)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
