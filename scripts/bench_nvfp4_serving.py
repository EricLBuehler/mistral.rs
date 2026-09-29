#!/usr/bin/env python3
"""Compare fixed-concurrency serving with a shared, exact-token prompt corpus."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260904
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_INPUT_LENGTHS = (128, 1024)
DEFAULT_OUTPUT_LENGTH = 128
DEFAULT_REPETITIONS = 3
DEFAULT_REQUESTS_PER_WORKER = 3
MIN_REQUESTS = 6
REQUEST_TIMEOUT_SECONDS = 600
PROMPT_ADJUSTMENT_LIMIT = 20
WORDS = (
    "apple river garden blue stone cloud paper glass music train green water tree "
    "moon star bird road lake book farm wind gold red ship desk light house city "
    "field rain snow sun earth forest mountain ocean silver copper iron steel "
    "bridge school science math history art language design system network data "
    "memory model process energy power heat time space sound color shape number"
).split()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def append_json(handle, value: object) -> None:
    handle.write(json.dumps(value, separators=(",", ":")) + "\n")
    handle.flush()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def prepare(args: argparse.Namespace) -> None:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    tokenizer_path = args.tokenizer / "tokenizer.json"
    template_path = args.tokenizer / "chat_template.jinja"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    template_text = template_path.read_text()
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string(template_text)

    def render(messages: list[dict]) -> tuple[str, list[int]]:
        rendered = template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,
        )
        return rendered, tokenizer.encode(rendered, add_special_tokens=False).ids

    def prompt(request_id: str, input_tokens: int) -> dict:
        rng = random.Random(f"{args.seed}:{request_id}")
        words = [rng.choice(WORDS) for _ in range(input_tokens * 2)]
        word_count = input_tokens
        for _ in range(PROMPT_ADJUSTMENT_LIMIT):
            content = " ".join(words[:word_count])
            messages = [{"role": "user", "content": content}]
            rendered, token_ids = render(messages)
            difference = input_tokens - len(token_ids)
            if difference == 0:
                return {
                    "id": request_id,
                    "messages": messages,
                    "input_tokens": input_tokens,
                    "rendered_sha256": digest(rendered.encode()),
                    "token_ids_sha256": digest(json.dumps(token_ids).encode()),
                }
            word_count += difference
            if word_count <= 0:
                break
        raise ValueError(f"cannot construct {input_tokens}-token prompt {request_id}")

    rounds = []
    prompt_hashes = set()
    for input_tokens in args.input_lengths:
        for concurrency in args.concurrencies:
            for repetition in range(-1, args.repetitions):
                phase = "warmup" if repetition == -1 else "measured"
                count = (
                    concurrency
                    if phase == "warmup"
                    else max(MIN_REQUESTS, args.requests_per_worker * concurrency)
                )
                round_id = f"p{input_tokens}-c{concurrency}-{phase}-{repetition}"
                prompts = [
                    prompt(f"{round_id}-r{index}", input_tokens)
                    for index in range(count)
                ]
                for item in prompts:
                    if item["rendered_sha256"] in prompt_hashes:
                        raise ValueError("duplicate prompt in corpus")
                    prompt_hashes.add(item["rendered_sha256"])
                rounds.append(
                    {
                        "id": round_id,
                        "phase": phase,
                        "repetition": repetition,
                        "input_tokens": input_tokens,
                        "output_tokens": args.output_length,
                        "concurrency": concurrency,
                        "prompts": prompts,
                    }
                )
    value = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "seed": args.seed,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": digest(tokenizer_path.read_bytes()),
        "chat_template_sha256": digest(template_text.encode()),
        "prompt_distribution": "deterministic independent common-word sequences",
        "chat_template_kwargs": {"enable_thinking": False},
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, value)
    print(
        json.dumps(
            {
                "workload": str(args.output),
                "sha256": digest(args.output.read_bytes()),
                "rounds": len(rounds),
                "requests": len(prompt_hashes),
            }
        ),
        flush=True,
    )


async def stream_request(session, url: str, payload: dict, row: dict) -> dict:
    started = time.perf_counter()
    chunks = []
    text_parts = []
    usage = None
    finish_reason = None
    finished = None
    done = False
    first_token = None
    last_token = None
    row["started_at"] = utc_now()
    try:
        async with session.post(url, json=payload) as response:
            row["http_status"] = response.status
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
            event_data = []
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    event_data.append(line[5:].lstrip(" "))
                    continue
                if line or not event_data:
                    continue
                data = "\n".join(event_data)
                event_data.clear()
                received = time.perf_counter()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index") != 0:
                        raise ValueError(f"unexpected choice: {choice}")
                    delta = choice.get("delta", {})
                    text = "".join(
                        delta.get(key) or ""
                        for key in ("reasoning", "reasoning_content", "content")
                    )
                    if text:
                        if first_token is None:
                            first_token = received
                        last_token = received
                        chunks.append((received - started) * 1000)
                        text_parts.append(text)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                        finished = received
            if not done:
                raise ValueError("stream ended without [DONE]")
        if first_token is None or finished is None:
            raise ValueError("missing generated content or finish event")
        if finish_reason != "length":
            raise ValueError(f"expected length finish reason, got {finish_reason!r}")
        if usage is None:
            raise ValueError("missing final usage")
        expected = {
            "prompt_tokens": row["input_tokens"],
            "completion_tokens": row["output_tokens"],
            "total_tokens": row["input_tokens"] + row["output_tokens"],
        }
        for key, value in expected.items():
            if usage.get(key) != value:
                raise ValueError(f"expected {key}={value}, got usage={usage}")
        row.update(
            success=True,
            ttft_ms=(first_token - started) * 1000,
            e2e_ms=(finished - started) * 1000,
            tpot_ms=(finished - first_token) * 1000 / (row["output_tokens"] - 1),
            last_content_ms=(last_token - started) * 1000,
            finish_reason=finish_reason,
        )
    except Exception as error:
        row.update(success=False, error=f"{type(error).__name__}: {error}")
    row.update(
        transport_ms=(time.perf_counter() - started) * 1000,
        usage=usage,
        content_chunk_times_ms=chunks,
        text="".join(text_parts),
    )
    return row


async def run_round(session, args, round_data, request_handle) -> dict:
    queue = iter(round_data["prompts"])
    rows = []
    failed = False

    async def worker() -> None:
        nonlocal failed
        while not failed:
            prompt = next(queue, None)
            if prompt is None:
                return
            payload = {
                "model": args.model,
                "messages": prompt["messages"],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "seed": DEFAULT_SEED,
                "n": 1,
                "max_tokens": round_data["output_tokens"],
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            row = {
                "engine": args.engine,
                "round_id": round_data["id"],
                "request_id": prompt["id"],
                "phase": round_data["phase"],
                "repetition": round_data["repetition"],
                "concurrency": round_data["concurrency"],
                "input_tokens": round_data["input_tokens"],
                "output_tokens": round_data["output_tokens"],
                "prompt_sha256": prompt["rendered_sha256"],
            }
            row = await stream_request(
                session, args.base_url.rstrip("/") + "/chat/completions", payload, row
            )
            append_json(request_handle, row)
            rows.append(row)
            failed |= not row["success"]

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(round_data["concurrency"])))
    duration = time.perf_counter() - started
    successful = [row for row in rows if row["success"]]
    result = {key: value for key, value in round_data.items() if key != "prompts"}
    result.update(
        engine=args.engine,
        completed_at=utc_now(),
        requested=len(round_data["prompts"]),
        completed=len(successful),
        failed=len(rows) - len(successful),
        duration_seconds=duration,
        success=len(successful) == len(round_data["prompts"]),
    )
    if result["success"]:
        result.update(
            requests_per_second=len(rows) / duration,
            output_tokens_per_second=sum(
                row["usage"]["completion_tokens"] for row in rows
            )
            / duration,
            total_tokens_per_second=sum(row["usage"]["total_tokens"] for row in rows)
            / duration,
        )
        for metric in ("ttft_ms", "e2e_ms", "tpot_ms"):
            values = [row[metric] for row in rows]
            result[metric] = {
                "mean": statistics.mean(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
    else:
        result["errors"] = [row["error"] for row in rows if not row["success"]]
    return result


async def run(args: argparse.Namespace) -> None:
    import aiohttp

    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    if workload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported workload schema")
    rounds = [
        item
        for item in workload["rounds"]
        if (not args.concurrencies or item["concurrency"] in args.concurrencies)
        and (not args.input_lengths or item["input_tokens"] in args.input_lengths)
    ]
    if not rounds:
        raise ValueError("selected no benchmark rounds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "engine": args.engine,
        "model": args.model,
        "base_url": args.base_url,
        "started_at": utc_now(),
        "workload": str(args.workload.resolve()),
        "workload_sha256": digest(workload_bytes),
        "measurement": {
            "scheduling": "closed loop, shared queue, at most C requests in flight",
            "ttft": "request start to first nonempty content or reasoning SSE delta",
            "e2e": "request start to finish_reason SSE event",
            "tpot": "(e2e - ttft) / (server completion_tokens - 1)",
            "throughput": "sum of server token usage / entire round wall time",
            "prefix_cache": "must be disabled in each server launch",
            "temperature": 0,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "rounds": [],
    }
    write_json(summary_path, summary)
    connector = aiohttp.TCPConnector(
        limit=max(item["concurrency"] for item in rounds),
        ttl_dns_cache=REQUEST_TIMEOUT_SECONDS,
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with (
            (args.output_dir / "requests.jsonl").open("w") as request_handle,
            (args.output_dir / "rounds.jsonl").open("w") as round_handle,
        ):
            for item in rounds:
                result = await run_round(session, args, item, request_handle)
                append_json(round_handle, result)
                summary["rounds"].append(result)
                summary["updated_at"] = utc_now()
                write_json(summary_path, summary)
                print(json.dumps(result), flush=True)
                if not result["success"]:
                    raise RuntimeError(f"round {item['id']} failed: {result['errors']}")
    summary["completed_at"] = utc_now()
    summary["success"] = True
    write_json(summary_path, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--tokenizer", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(DEFAULT_CONCURRENCIES)
    )
    prepare_parser.add_argument(
        "--input-lengths", type=int, nargs="+", default=list(DEFAULT_INPUT_LENGTHS)
    )
    prepare_parser.add_argument(
        "--output-length", type=int, default=DEFAULT_OUTPUT_LENGTH
    )
    prepare_parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    prepare_parser.add_argument(
        "--requests-per-worker", type=int, default=DEFAULT_REQUESTS_PER_WORKER
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workload", type=Path, required=True)
    run_parser.add_argument("--base-url", required=True, help="API root ending in /v1")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--engine", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--concurrencies", type=int, nargs="+")
    run_parser.add_argument("--input-lengths", type=int, nargs="+")
    run_parser.add_argument(
        "--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS
    )
    args = parser.parse_args()
    for field in ("concurrencies", "input_lengths"):
        values = getattr(args, field, None)
        if values and (
            any(value <= 0 for value in values) or len(set(values)) != len(values)
        ):
            parser.error(f"{field} must contain unique positive integers")
    if args.command == "prepare" and (
        args.output_length <= 1
        or args.repetitions <= 0
        or args.requests_per_worker <= 0
    ):
        parser.error(
            "output-length must exceed one; repetitions and requests-per-worker must be positive"
        )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    else:
        asyncio.run(run(arguments))
