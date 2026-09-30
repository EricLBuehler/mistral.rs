#!/usr/bin/env python3
"""Compare dense FP8 and FP8-to-Q4K backend scaling on GB10."""

import argparse
import datetime
import hashlib
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/ericbuehler/mistral.rs")
HARNESS_DIR = REPO / "benchmarks/qwen3.8-flash-next"
sys.path.insert(0, str(HARNESS_DIR))
bench_concurrency = importlib.import_module("bench_concurrency")
bench_serving = importlib.import_module("bench_serving")

MODEL = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
SNAPSHOT = (
    Path(
        "/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots"
    )
    / REVISION
)
PROMPTS_PATH = REPO / "releases/v0.9.3/raw/prompts.jsonl"
REQUESTS = 8
OUTPUT_TOKENS = 128
WARMUPS = 1
TRIALS = 3
CONCURRENCIES = (1, 8)
CACHE_TOKENS = 16384
STARTUP_SECONDS = 2400
POLL_SECONDS = 2
STOP_SECONDS = 30


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git_output(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


def measure(args, metadata):
    tokenizer = bench_serving.Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    prompt_rows = [json.loads(line) for line in PROMPTS_PATH.read_text().splitlines()]
    prompts = {
        f"canonical_{i:02}": row["prompt"]
        for i, row in enumerate(prompt_rows[:REQUESTS])
    }
    bench_serving.PROMPTS = prompts
    request_args = argparse.Namespace(
        base_url=f"http://127.0.0.1:{args.port}",
        max_tokens=OUTPUT_TOKENS,
        requests=REQUESTS,
        timeout=bench_serving.TIMEOUT_SECONDS,
        expected_prompt_tokens={
            name: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            for name, prompt in prompts.items()
        },
    )
    data = {
        "label": args.label,
        "date": now(),
        "purpose": __doc__,
        "harness_sha256": sha256(__file__),
        "component_hashes": metadata["component_hashes"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "prompts_source_sha256": sha256(PROMPTS_PATH),
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, sort_keys=True).encode()
        ).hexdigest(),
        "prompts": prompts,
        "settings": vars(request_args)
        | {
            "concurrencies": list(CONCURRENCIES),
            "warmup_trials": WARMUPS,
            "measured_trials": TRIALS,
            "sampling_seed": bench_serving.SEED,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": False,
            "logprobs_requested": False,
            "prompt_selection": "First eight original canonical release prompts, in file order.",
            "prompt_format": "Raw canonical text, without chat formatting. These cached-tokenizer counts differ from the original vLLM client's reported input counts.",
            "mode": "closed-loop; C8 has only one finite wave because requests equals concurrency.",
        },
        "complete": False,
        "cases": {},
    }
    raw_path = args.output_dir / "raw.json"
    save(raw_path, data)
    for concurrency in CONCURRENCIES:
        case = {"concurrency": concurrency, "runs": []}
        data["cases"][str(concurrency)] = case
        for trial in range(WARMUPS + TRIALS):
            run = bench_concurrency.run_trial(request_args, "closed-loop", concurrency)
            run.update({"trial": trial, "warmup": trial < WARMUPS})
            case["runs"].append(run)
            case["summary"] = bench_concurrency.summarize(case["runs"])
            save(raw_path, data)
            print(
                f"C{concurrency} trial={trial} warmup={run['warmup']} complete={run['complete']} "
                f"aggregate={run['aggregate_output_tokens_per_second']:.3f} "
                f"mean_active={run['mean_active_requests']:.3f}",
                flush=True,
            )
            if not run["complete"]:
                raise RuntimeError(
                    f"C{concurrency} trial {trial} failed; responses saved to {raw_path}"
                )
    data["complete"] = True
    save(raw_path, data)
    metrics = {c: data["cases"][str(c)]["summary"] for c in CONCURRENCIES}
    summary = {
        "label": args.label,
        "complete": True,
        "raw_sha256": sha256(raw_path),
        "definitions": {
            "aggregate": "Returned output tokens / whole-trial client wall seconds.",
            "per_active_request": "Returned output tokens / summed request latency.",
            "mean_active": "Summed request latency / whole-trial client wall seconds.",
            "statistics": "Arithmetic mean and sample standard deviation of three measured trials.",
        },
        "interpretation": "Current dense-model scaling on GB10 only. Different model, weights, prompts, and speculative mode from Flash-Next. One finite C8 wave, not sustained traffic. No old-versus-new engine regression isolation.",
        "metrics_by_concurrency": metrics,
        "c8_over_c1_aggregate": metrics[8]["aggregate_output_tokens_per_second"]["mean"]
        / metrics[1]["aggregate_output_tokens_per_second"]["mean"],
        "c8_over_c1_per_active_request": metrics[8][
            "latency_weighted_request_output_tokens_per_second"
        ]["mean"]
        / metrics[1]["latency_weighted_request_output_tokens_per_second"]["mean"],
    }
    save(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=REPO / "target/release/mistralrs"
    )
    parser.add_argument("--source-repo", type=Path, default=REPO)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "current_dense_fp8"
    )
    parser.add_argument("--label", default="current_dense_fp8_bf16kv")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--cache-type", choices=["auto", "f8e4m3"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isq", choices=["q4k"])
    args = parser.parse_args()
    assert args.binary.is_file(), args.binary
    assert sha256(SNAPSHOT / "config.json") == CONFIG_SHA256
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    assert all(
        (SNAPSHOT / name).is_file() for name in set(index["weight_map"].values())
    )
    command = [
        str(args.binary),
        "serve",
        "-m",
        str(SNAPSHOT),
        "--dtype",
        "bf16",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-ui",
        "--disable-access-log",
        "--max-seqs",
        "8",
        "--max-model-len",
        str(CACHE_TOKENS),
        "--pa-context-len",
        str(CACHE_TOKENS),
        "--pa-cache-type",
        args.cache_type,
        "--max-num-batched-tokens",
        "4096",
        "--max-prefill-chunk-tokens",
        "512",
        "--prefix-cache-n",
        "0",
    ]
    if args.isq:
        command.extend(["--isq", args.isq])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "output_dir": str(args.output_dir),
                    "requests": REQUESTS,
                    "output_tokens": OUTPUT_TOKENS,
                    "concurrencies": CONCURRENCIES,
                    "warmups": WARMUPS,
                    "trials": TRIALS,
                },
                indent=2,
            )
        )
        return
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
            raise RuntimeError(
                f"Port {args.port} is occupied; refusing to touch another server"
            )
    except ConnectionRefusedError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", RUST_LOG="info")
    metadata = {
        "date": now(),
        "command": command,
        "binary_sha256": sha256(args.binary),
        "model": MODEL,
        "model_revision": REVISION,
        "isq": args.isq,
        "weight_source_note": "Q4K, when selected, requantizes this FP8 checkpoint; timings do not establish BF16-source Q4K quality.",
        "model_config_sha256": CONFIG_SHA256,
        "tokenizer_sha256": sha256(SNAPSHOT / "tokenizer.json"),
        "weight_index_sha256": sha256(SNAPSHOT / "model.safetensors.index.json"),
        "source_repo": str(args.source_repo),
        "source_head_at_launch": git_output(args.source_repo, "rev-parse", "HEAD")
        .decode()
        .strip(),
        "source_diff_sha256_at_launch": hashlib.sha256(
            git_output(args.source_repo, "diff", "HEAD")
        ).hexdigest(),
        "source_note": "Working-tree source observed at launch; executable SHA-256 identifies the loaded binary.",
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "component_hashes": {
            name: sha256(path)
            for name, path in {
                "control_script": __file__,
                "bench_concurrency": bench_concurrency.__file__,
                "bench_serving": bench_serving.__file__,
            }.items()
        },
        "environment": {
            key: value
            for key, value in env.items()
            if key.startswith(("MISTRALRS_", "CUDA_", "CUTILE_"))
            or key in ("HF_HUB_OFFLINE", "RUST_LOG", "LD_LIBRARY_PATH")
        },
        "kv_cache": "BF16" if args.cache_type == "auto" else "FP8 E4M3",
        "target_only": True,
        "profiler": None,
        "complete": False,
    }
    metadata_path = args.output_dir / "metadata.json"
    save(metadata_path, metadata)
    started = time.monotonic()
    with (args.output_dir / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=args.source_repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        metadata["server_pid"] = server.pid
        save(metadata_path, metadata)
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {server.returncode} during startup"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/v1/models", timeout=2
                    ) as response:
                        if response.status == 200:
                            metadata["models_response"] = json.load(response)
                            break
                except OSError:
                    pass
                if time.monotonic() - started > STARTUP_SECONDS:
                    raise TimeoutError("Model startup exceeded deadline")
                time.sleep(POLL_SECONDS)
            metadata["startup_seconds"] = time.monotonic() - started
            metadata["loaded_executable_sha256"] = sha256(
                Path(f"/proc/{server.pid}/exe")
            )
            assert metadata["loaded_executable_sha256"] == metadata["binary_sha256"], (
                "Binary changed between hashing and launch"
            )
            save(metadata_path, metadata)
            print(f"Server ready after {metadata['startup_seconds']:.1f}s", flush=True)
            measure(args, metadata)
            metadata["complete"] = True
        except BaseException as error:
            metadata["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            metadata["finished_at"] = now()
            metadata["raw_sha256"] = (
                sha256(args.output_dir / "raw.json")
                if (args.output_dir / "raw.json").exists()
                else None
            )
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            metadata["server_exit_code"] = server.returncode
            save(metadata_path, metadata)


if __name__ == "__main__":
    main()
