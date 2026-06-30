#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    threads: int
    cpu_list: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CPU dense-model bench runs between mistral.rs and llama.cpp."
    )
    parser.add_argument("--mistralrs-bin", default="target/release/mistralrs")
    parser.add_argument("--llamacpp-bench", default="../llama.cpp/build/bin/llama-bench")
    parser.add_argument("--mistralrs-model", required=True)
    parser.add_argument("--llamacpp-model", required=True)
    parser.add_argument("--mistralrs-format", default=None)
    parser.add_argument("--token-source", default="none")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--cpu-list", default=None)
    parser.add_argument("--cpu-mask", default=None)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="NAME:THREADS:CPU_LIST",
        help="Run an additional case, for example big5:5:15-19 or big10:10:5-9,15-19.",
    )
    parser.add_argument("--taskset", default="taskset")
    parser.add_argument("--llamacpp-ngl", type=int, default=0)
    parser.add_argument("--no-llamacpp-mask", action="store_true")
    parser.add_argument(
        "--allow-llamacpp-cuda",
        action="store_true",
        help="Leave CUDA_VISIBLE_DEVICES unchanged for llama.cpp. By default it is hidden.",
    )
    parser.add_argument("--mistralrs-extra-arg", action="append", default=[])
    parser.add_argument("--llamacpp-extra-arg", action="append", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_case(value: str) -> Case:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"case must be NAME:THREADS:CPU_LIST, got {value!r}")
    name, threads, cpu_list = parts
    if not name:
        raise ValueError("case name cannot be empty")
    if not threads.isdigit() or int(threads) < 1:
        raise ValueError(f"case threads must be positive, got {threads!r}")
    return Case(name=name, threads=int(threads), cpu_list=cpu_list or None)


def expand_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range {part!r}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    if any(cpu < 0 for cpu in cpus):
        raise ValueError(f"CPU list must be non-negative, got {cpu_list!r}")
    if not cpus:
        raise ValueError("CPU list cannot be empty")
    return sorted(set(cpus))


def cpu_list_to_mask(cpu_list: str) -> str:
    mask = 0
    for cpu in expand_cpu_list(cpu_list):
        mask |= 1 << cpu
    return f"0x{mask:x}"


def cpu_mask_to_list(cpu_mask: str) -> str:
    if "," in cpu_mask:
        raise ValueError("cannot derive taskset CPU list from a comma-separated CPU mask")
    mask = int(cpu_mask, 16)
    cpus = [cpu for cpu in range(mask.bit_length()) if mask & (1 << cpu)]
    if not cpus:
        raise ValueError("CPU mask cannot be zero")

    ranges: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def output_path(value: str | None) -> Path:
    if value:
        return Path(value)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("target") / f"cpu_dense_compare_{stamp}.json"


def build_cases(args: argparse.Namespace) -> list[Case]:
    if args.case:
        return [parse_case(value) for value in args.case]
    cpu_list = args.cpu_list
    if cpu_list is None and args.cpu_mask is not None:
        cpu_list = cpu_mask_to_list(args.cpu_mask)
    name = f"threads{args.threads}"
    if cpu_list:
        name += f"_cpus{cpu_list.replace(',', '_')}"
    return [Case(name=name, threads=args.threads, cpu_list=cpu_list)]


def prefixed_with_taskset(cmd: list[str], taskset: str, cpu_list: str | None) -> list[str]:
    if not cpu_list:
        return cmd
    return [taskset, "-c", cpu_list, *cmd]


def build_mistralrs_cmd(args: argparse.Namespace, case: Case) -> tuple[list[str], dict[str, str]]:
    cmd = [
        args.mistralrs_bin,
        "bench",
        "-m",
        args.mistralrs_model,
        "--cpu",
        "--token-source",
        args.token_source,
        "--dtype",
        args.dtype,
        "--prompt-len",
        str(args.prompt_len),
        "--gen-len",
        str(args.gen_len),
        "--depth",
        str(args.depth),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
    ]
    if args.mistralrs_format:
        cmd.extend(["--format", args.mistralrs_format])
    cmd.extend(args.mistralrs_extra_arg)
    env = {
        "CANDLE_NUM_THREADS": str(case.threads),
        "RAYON_NUM_THREADS": str(case.threads),
    }
    return prefixed_with_taskset(cmd, args.taskset, case.cpu_list), env


def build_llamacpp_cmd(
    args: argparse.Namespace, case: Case
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        args.llamacpp_bench,
        "-m",
        args.llamacpp_model,
        "-ngl",
        str(args.llamacpp_ngl),
        "-p",
        str(args.prompt_len),
        "-n",
        str(args.gen_len),
        "-d",
        str(args.depth),
        "-t",
        str(case.threads),
        "-r",
        str(args.iterations),
        "-o",
        "json",
    ]
    if args.warmup == 0:
        cmd.append("--no-warmup")
    if case.cpu_list and not args.no_llamacpp_mask:
        cmd.extend(["-C", cpu_list_to_mask(case.cpu_list), "--cpu-strict", "1"])
    cmd.extend(args.llamacpp_extra_arg)
    env = {} if args.allow_llamacpp_cuda else {"CUDA_VISIBLE_DEVICES": ""}
    return prefixed_with_taskset(cmd, args.taskset, case.cpu_list), env


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def parse_mistralrs_output(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in strip_ansi(stdout).splitlines():
        for label, key in (("Prefill", "prefill_tps"), ("Decode", "decode_tps")):
            if label not in line:
                continue
            cells = re.split(r"[|\u2502\u2506]", line)
            text = cells[2] if len(cells) > 2 else line.split(label, 1)[1]
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
            if numbers:
                parsed[key] = float(numbers[0])
    return parsed


def parse_llamacpp_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return parse_llamacpp_json(data)
    except json.JSONDecodeError:
        return parse_llamacpp_csv(text)


def parse_llamacpp_json(data: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("avg_ts", "stddev_ts", "test")):
                rows.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    parsed: dict[str, Any] = {"rows": rows}
    for row in rows:
        test = str(row.get("test", row.get("type", ""))).lower()
        avg = row.get("avg_ts", row.get("tokens_per_second"))
        if avg is None:
            continue
        try:
            avg_float = float(avg)
        except (TypeError, ValueError):
            continue
        if test.startswith("pp") or test == "prompt":
            parsed["prefill_tps"] = avg_float
        elif test.startswith("tg") or test == "generation":
            parsed["decode_tps"] = avg_float
        elif row.get("n_prompt", 0) and not row.get("n_gen", 0):
            parsed["prefill_tps"] = avg_float
        elif row.get("n_gen", 0):
            parsed["decode_tps"] = avg_float
    return parsed


def parse_llamacpp_csv(text: str) -> dict[str, Any]:
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    parsed: dict[str, Any] = {"rows": rows}
    for row in rows:
        test = row.get("test", row.get("type", "")).lower()
        avg = row.get("avg_ts", row.get("tokens_per_second", ""))
        if not avg:
            continue
        try:
            avg_float = float(avg)
        except ValueError:
            continue
        if test.startswith("pp") or test == "prompt":
            parsed["prefill_tps"] = avg_float
        elif test.startswith("tg") or test == "generation":
            parsed["decode_tps"] = avg_float
        elif row.get("n_prompt") and row.get("n_gen") in ("", "0", None):
            parsed["prefill_tps"] = avg_float
        elif row.get("n_gen") and row.get("n_gen") != "0":
            parsed["decode_tps"] = avg_float
    return parsed


def run_command(
    cmd: list[str],
    env_delta: dict[str, str] | None,
    dry_run: bool,
    parser,
) -> dict[str, Any]:
    printable = shlex.join(cmd)
    if env_delta:
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_delta.items())
        printable = f"{env_text} {printable}"
    print(printable)
    if dry_run:
        return {
            "cmd": cmd,
            "env": env_delta or {},
            "returncode": None,
            "elapsed_s": 0.0,
            "stdout": "",
            "stderr": "",
            "parsed": {},
        }

    env = os.environ.copy()
    if env_delta:
        env.update(env_delta)
    started = time.perf_counter()
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    parsed = parser(completed.stdout) if completed.returncode == 0 else {}
    return {
        "cmd": cmd,
        "env": env_delta or {},
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed": parsed,
    }


def print_result(name: str, result: dict[str, Any]) -> None:
    code = result["returncode"]
    elapsed = result["elapsed_s"]
    parsed = result.get("parsed") or {}
    parts = [f"{name}: returncode={code}", f"elapsed={elapsed:.2f}s"]
    if "prefill_tps" in parsed:
        parts.append(f"prefill={parsed['prefill_tps']:.2f} tok/s")
    if "decode_tps" in parsed:
        parts.append(f"decode={parsed['decode_tps']:.2f} tok/s")
    print(", ".join(parts))


def build_comparison(case_result: dict[str, Any]) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    mistral = case_result["mistralrs"].get("parsed") or {}
    llama = case_result["llamacpp"].get("parsed") or {}
    for name, key in (("prefill", "prefill_tps"), ("decode", "decode_tps")):
        mistral_tps = mistral.get(key)
        llama_tps = llama.get(key)
        if not mistral_tps or not llama_tps:
            continue
        comparison[name] = {
            "mistralrs_tps": mistral_tps,
            "llamacpp_tps": llama_tps,
            "mistralrs_over_llamacpp": mistral_tps / llama_tps,
        }
    return comparison


def print_comparison(comparison: dict[str, dict[str, float]]) -> None:
    parts = []
    for name in ("prefill", "decode"):
        values = comparison.get(name)
        if values:
            parts.append(f"{name}={values['mistralrs_over_llamacpp']:.2f}x")
    if parts:
        print("comparison: " + ", ".join(parts))


def main() -> int:
    args = parse_args()
    cases = build_cases(args)
    out = output_path(args.out)
    results: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "parameters": {
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "depth": args.depth,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "mistralrs_model": args.mistralrs_model,
            "llamacpp_model": args.llamacpp_model,
            "llamacpp_ngl": args.llamacpp_ngl,
            "allow_llamacpp_cuda": args.allow_llamacpp_cuda,
        },
        "cases": [],
    }

    any_failed = False
    for case in cases:
        print(f"== {case.name} ==")
        mistral_cmd, mistral_env = build_mistralrs_cmd(args, case)
        llama_cmd, llama_env = build_llamacpp_cmd(args, case)
        case_result: dict[str, Any] = {"case": asdict(case)}
        case_result["mistralrs"] = run_command(
            mistral_cmd, mistral_env, args.dry_run, parse_mistralrs_output
        )
        print_result("mistral.rs", case_result["mistralrs"])
        case_result["llamacpp"] = run_command(
            llama_cmd, llama_env, args.dry_run, parse_llamacpp_output
        )
        print_result("llama.cpp", case_result["llamacpp"])
        case_result["comparison"] = build_comparison(case_result)
        print_comparison(case_result["comparison"])
        results["cases"].append(case_result)
        any_failed |= any(
            backend["returncode"] not in (0, None)
            for backend in (case_result["mistralrs"], case_result["llamacpp"])
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
