#!/usr/bin/env python3
"""CPU bench sweep: mistral.rs vs llama.cpp on GB10 (aarch64, 10x Cortex-X925 + 10x Cortex-A725).

Phases:
  affinity  - small workload across affinity strategies to pick the best per engine
  full      - full model x quant sweep using --mrs-mode/--lcpp-mode

Results are appended to results.jsonl (one row per measurement) and raw engine
stdout is saved per run for auditability.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MISTRALRS = REPO / "target/release/mistralrs"
LLAMA_BENCH = Path.home() / "llama.cpp/build-cpu/bin/llama-bench"
MODELS_DIR = Path.home() / "hf_models/bench_v090"

BIG_CORES = "5-9,15-19"
BIG_CORES_HEX = "0xF83E0"
N_BIG = 10

# model -> quant -> per-engine spec
UQFF_DIR = Path.home() / "hf_models/bench_v090"
UQFF = {
    ("qwen3-4b", "q4k"): ("Qwen/Qwen3-4B", "qwen3-4b-uqff-q4k/q4k-0.uqff"),
    ("qwen3-4b", "q6k"): ("Qwen/Qwen3-4B", "qwen3-4b-uqff-q6k/q6k-0.uqff"),
    ("qwen3-4b", "q8_0"): ("Qwen/Qwen3-4B", "qwen3-4b-uqff-q8_0/q8_0-0.uqff"),
    ("gemma4-e4b", "q4k"): ("google/gemma-4-E4B-it", "gemma4-uqff-q4k/q4k-0.uqff"),
    ("gemma4-e4b", "q6k"): ("google/gemma-4-E4B-it", "gemma4-uqff-q6k/q6k-0.uqff"),
    ("gemma4-e4b", "q8_0"): ("google/gemma-4-E4B-it", "gemma4-uqff-q8_0/q8_0-0.uqff"),
    ("lfm2.5-230m", "q4k"): ("LiquidAI/LFM2.5-230M", "lfm25-uqff-q4k/q4k-0.uqff"),
    ("lfm2.5-230m", "q6k"): ("LiquidAI/LFM2.5-230M", "lfm25-uqff-q6k/q6k-0.uqff"),
    ("lfm2.5-230m", "q8_0"): ("LiquidAI/LFM2.5-230M", "lfm25-uqff-q8_0/q8_0-0.uqff"),
}

MODELS = {
    "qwen3-4b": {
        "q4k": {
            "mrs": ["-m", "Qwen/Qwen3-4B", "--isq", "q4k"],
            "lcpp": str(MODELS_DIR / "qwen3-4b-gguf/Qwen3-4B-Q4_K_M.gguf"),
        },
        "q6k": {
            "mrs": ["-m", "Qwen/Qwen3-4B", "--isq", "q6k"],
            "lcpp": str(MODELS_DIR / "qwen3-4b-gguf/Qwen3-4B-Q6_K.gguf"),
        },
        "q8_0": {
            "mrs": ["-m", "Qwen/Qwen3-4B", "--isq", "q8_0"],
            "lcpp": str(MODELS_DIR / "qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf"),
        },
    },
    "gemma4-e4b": {
        "q4k": {
            "mrs": ["-m", "google/gemma-4-E4B-it", "--isq", "q4k"],
            "lcpp": str(MODELS_DIR / "gemma4-e4b-gguf/gemma-4-E4B-it-Q4_K_M.gguf"),
        },
        "q6k": {
            "mrs": ["-m", "google/gemma-4-E4B-it", "--isq", "q6k"],
            "lcpp": str(MODELS_DIR / "gemma4-e4b-gguf/gemma-4-E4B-it-Q6_K.gguf"),
        },
        "q8_0": {
            "mrs": ["-m", "google/gemma-4-E4B-it", "--isq", "q8_0"],
            "lcpp": str(MODELS_DIR / "gemma4-e4b-gguf/gemma-4-E4B-it-Q8_0.gguf"),
        },
    },
    "lfm2.5-230m": {
        "q4k": {
            "mrs": ["-m", "LiquidAI/LFM2.5-230M", "--isq", "q4k"],
            "lcpp": str(MODELS_DIR / "lfm2.5-230m-gguf/LFM2.5-230M-Q4_K_M.gguf"),
        },
        "q6k": {
            "mrs": ["-m", "LiquidAI/LFM2.5-230M", "--isq", "q6k"],
            "lcpp": str(MODELS_DIR / "lfm2.5-230m-gguf/LFM2.5-230M-Q6_K.gguf"),
        },
        "q8_0": {
            "mrs": ["-m", "LiquidAI/LFM2.5-230M", "--isq", "q8_0"],
            "lcpp": str(MODELS_DIR / "lfm2.5-230m-gguf/LFM2.5-230M-Q8_0.gguf"),
        },
    },
}

# affinity strategies per engine: (extra env, wrapper argv prefix, extra engine args)
MRS_MODES = {
    "default": ({}, [], []),
    "mask": ({"CANDLE_CPU_MASK": BIG_CORES}, [], []),
    "taskset": ({}, ["taskset", "-c", BIG_CORES], []),
}
LCPP_MODES = {
    "default": ({}, [], ["-t", "20"]),
    "t10": ({}, [], ["-t", str(N_BIG)]),
    "mask": ({}, [], ["-t", str(N_BIG), "-C", BIG_CORES_HEX, "--cpu-strict", "1"]),
    "taskset": ({}, ["taskset", "-c", BIG_CORES], ["-t", str(N_BIG)]),
}

MRS_PREFILL_RE = re.compile(r"Prefill \((\d+) tokens\)\s*[^\d]*([\d.]+)")
MRS_DECODE_RE = re.compile(r"Decode \((\d+) tokens @ d(\d+)\)\s*[^\d]*([\d.]+)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_rows(out_path, rows):
    with open(out_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_cmd(cmd, env_extra, raw_path, timeout):
    env = os.environ.copy()
    env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    dur = time.time() - t0
    Path(raw_path).write_text(
        f"# cmd: {' '.join(map(str, cmd))}\n# env: {env_extra}\n# wall_s: {dur:.1f}\n# rc: {proc.returncode}\n"
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr[-8000:]}\n"
    )
    if proc.returncode != 0:
        log(f"FAILED rc={proc.returncode}: {' '.join(map(str, cmd))}")
        log(proc.stderr[-2000:])
        return None, dur
    return proc.stdout, dur


def run_mistralrs(model, quant, mode, prompt_lens, depths, gen_len, iters, warmup, raw_dir, timeout):
    env_extra, wrapper, _ = MRS_MODES[mode]
    spec = MODELS[model][quant]["mrs"]
    cmd = wrapper + [
        str(MISTRALRS), "bench", "--cpu",
        "--prompt-len", ",".join(map(str, prompt_lens)),
        "--depth", ",".join(map(str, depths)),
        "--gen-len", str(gen_len),
        "--iterations", str(iters), "--warmup", str(warmup),
    ] + spec
    raw = raw_dir / f"mrs_{model}_{quant}_{mode}.txt"
    out, dur = run_cmd(cmd, env_extra, raw, timeout)
    if out is None:
        return []
    rows = []
    base = dict(engine="mistralrs", model=model, quant=quant, affinity=mode,
                gen_len=gen_len, iterations=iters, warmup=warmup, wall_s=round(dur, 1))
    for m in MRS_PREFILL_RE.finditer(out):
        rows.append({**base, "mode": "prefill", "length": int(m.group(1)), "tps": float(m.group(2))})
    for m in MRS_DECODE_RE.finditer(out):
        rows.append({**base, "mode": "decode", "length": int(m.group(2)), "tps": float(m.group(3))})
    return rows


def run_llamacpp(model, quant, mode, prompt_lens, depths, gen_len, iters, raw_dir, timeout,
                 decode_mode=None):
    decode_mode = decode_mode or mode
    gguf = MODELS[model][quant]["lcpp"]
    rows = []
    base = dict(engine="llamacpp", model=model, quant=quant,
                gen_len=gen_len, iterations=iters, warmup=1)

    # prefill: pp rows at each length
    env_extra, wrapper, targs = LCPP_MODES[mode]
    cmd = wrapper + [str(LLAMA_BENCH), "-m", gguf, "-p", ",".join(map(str, prompt_lens)),
                     "-n", "0", "-r", str(iters), "-o", "json", "--progress"] + targs
    raw = raw_dir / f"lcpp_{model}_{quant}_{mode}_prefill.txt"
    out, dur = run_cmd(cmd, env_extra, raw, timeout)
    if out is not None:
        for rec in json.loads(out):
            rows.append({**base, "affinity": mode, "mode": "prefill", "length": rec["n_prompt"],
                         "tps": rec["avg_ts"], "stddev": rec.get("stddev_ts"), "wall_s": round(dur, 1)})

    # decode: tg at each depth (-d prefills the context first, excluded from timing)
    env_extra, wrapper, targs = LCPP_MODES[decode_mode]
    cmd = wrapper + [str(LLAMA_BENCH), "-m", gguf, "-p", "0", "-n", str(gen_len),
                     "-d", ",".join(map(str, depths)), "-r", str(iters), "-o", "json", "--progress"] + targs
    raw = raw_dir / f"lcpp_{model}_{quant}_{decode_mode}_decode.txt"
    out, dur = run_cmd(cmd, env_extra, raw, timeout)
    if out is not None:
        for rec in json.loads(out):
            rows.append({**base, "affinity": decode_mode, "mode": "decode", "length": rec["n_depth"],
                         "tps": rec["avg_ts"], "stddev": rec.get("stddev_ts"), "wall_s": round(dur, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["affinity", "full", "smoke"], required=True)
    ap.add_argument("--out", default=str(REPO / "releases/v0.9.0/raw"))
    ap.add_argument("--models", default="qwen3-4b,gemma4-e4b,lfm2.5-230m")
    ap.add_argument("--quants", default="q4k,q6k,q8_0")
    ap.add_argument("--lengths", default="128,512,2048,4096,8192,16384")
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--mrs-mode", default="default")
    ap.add_argument("--lcpp-mode", default="default")
    ap.add_argument("--lcpp-decode-mode", default=None)
    ap.add_argument("--engines", default="mistralrs,llamacpp")
    ap.add_argument("--mrs-uqff", action="store_true", help="load prequantized UQFF instead of --isq (same weights, faster load)")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / f"raw_{args.phase}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / f"results_{args.phase}.jsonl"
    lengths = [int(x) for x in args.lengths.split(",")]

    if args.phase == "smoke":
        rows = run_mistralrs("qwen3-4b", "q4k", "default", [512], [512], 32, 1, 0, raw_dir, args.timeout)
        rows += run_llamacpp("qwen3-4b", "q4k", "default", [512], [512], 32, 1, raw_dir, args.timeout)
        append_rows(results, rows)
        print(json.dumps(rows, indent=2))
        return

    if args.phase == "affinity":
        # representative workload: mid-size prefill + shallow decode, two quants
        for quant in ["q4k", "q8_0"]:
            for mode in MRS_MODES:
                log(f"affinity mistralrs qwen3-4b {quant} {mode}")
                append_rows(results, run_mistralrs("qwen3-4b", quant, mode, [2048], [512],
                                                   args.gen_len, args.iters, args.warmup, raw_dir, args.timeout))
            for mode in LCPP_MODES:
                log(f"affinity llamacpp qwen3-4b {quant} {mode}")
                append_rows(results, run_llamacpp("qwen3-4b", quant, mode, [2048], [512],
                                                  args.gen_len, args.iters, raw_dir, args.timeout))
        return

    # full sweep
    engines = args.engines.split(",")
    for model in args.models.split(","):
        for quant in args.quants.split(","):
            if args.mrs_uqff:
                base, rel = UQFF[(model, quant)]
                MODELS[model][quant]["mrs"] = ["-m", base, "--from-uqff", str(UQFF_DIR / rel)]
            if "mistralrs" in engines:
                log(f"full mistralrs {model} {quant} ({args.mrs_mode})")
                append_rows(results, run_mistralrs(model, quant, args.mrs_mode, lengths, lengths,
                                                   args.gen_len, args.iters, args.warmup, raw_dir, args.timeout))
            if "llamacpp" in engines:
                log(f"full llamacpp {model} {quant} ({args.lcpp_mode}/{args.lcpp_decode_mode or args.lcpp_mode})")
                append_rows(results, run_llamacpp(model, quant, args.lcpp_mode, lengths, lengths,
                                                  args.gen_len, args.iters, raw_dir, args.timeout,
                                                  decode_mode=args.lcpp_decode_mode))
    log("sweep complete")


if __name__ == "__main__":
    main()
