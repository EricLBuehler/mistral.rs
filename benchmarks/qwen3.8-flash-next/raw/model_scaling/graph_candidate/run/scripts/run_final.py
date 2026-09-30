"""Run pinned Flash-Next MTP serving measurements after the GPU slot is released."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from observe import isolation_snapshot, memory_snapshot, stamp

ROOT = Path("/home/ericbuehler/mistral.rs")
ANCESTOR = Path(
    "/home/ericbuehler/qwen4exp_work/tuner_exploration_20260930/run_server.py"
)
SNAPSHOT = Path(
    "/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)
HARNESS_DIR = Path(__file__).resolve().parent / "harnesses"
HARNESSES = (
    "bench_serving.py",
    "bench_concurrency.py",
    "summarize_concurrency.py",
    "text_smoke.py",
    "mixed_context_smoke.py",
    "bench_homogeneous.py",
)
MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
)
STARTUP_TIMEOUT_SECONDS = 2400
PHASE_TIMEOUT_SECONDS = 7200
HTTP_TIMEOUT_SECONDS = 30
STOP_TIMEOUT_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2
MIN_FREE_DISK_BYTES = 1024**3
SOURCE_PATHS = (
    ":(glob)mistralrs*/**",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/",
)
STARTUP_PATTERNS = {
    "small_group_mmq": r"Using grouped GGUF MoE for small batches",
    "all_48_layers_cuda0": r"Layers 0-47: cuda\[0\]",
    "model_bf16": r"DType selected is BF16",
    "isq_q4k_sensitive_q6k": r"Quantizing model weights to q4k, with sensitive tensors using q6k",
    "ple_q4": r"PLE table resident in device memory format=Q4(?: |\n)",
    "kv_513_bf16_blocks": r"block size 32 and 513 GPU blocks: available context length is 16384 tokens",
    "kv_bf16": r"PagedAttention KV cache type is BF16",
    "mtp_builtin_six": r"MTP assistant `built-in` with n_predict=6",
    "scheduler_8": r"max_num_seqs=8 max_num_batched_tokens=4096 max_prefill_chunk_tokens=512 max_decode_steps_before_prefill=8",
    "graphs_36": r"Captured 36 CUDA decode graphs through batch bucket 8",
}


def check_free_disk(path):
    if shutil.disk_usage(path).free < MIN_FREE_DISK_BYTES:
        raise RuntimeError("At least 1 GiB free disk is required for the serving run")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True, timeout=60).strip()


def environment():
    env = dict(os.environ, HF_HUB_OFFLINE="1", RUST_LOG="info")
    forbidden = {
        key: value
        for key, value in env.items()
        if value
        and (
            key.startswith(("MISTRALRS_", "NSYS_", "NCU_"))
            or key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH")
        )
    }
    if forbidden:
        raise RuntimeError(
            f"Remove experiment/profiler environment overrides: {sorted(forbidden)}"
        )
    recorded = {
        key: value
        for key, value in env.items()
        if key.startswith(("CUDA_", "CUTILE_", "TILEIR_"))
        or key
        in ("HF_HUB_OFFLINE", "RUST_LOG", "LD_LIBRARY_PATH", "PATH", "OMP_NUM_THREADS")
    }
    return env, recorded


def server_command(binary, port):
    return [
        str(binary),
        "serve",
        "--no-ui",
        "-p",
        str(port),
        "--max-model-len",
        "16384",
        "--max-seqs",
        "8",
        "--prefix-cache-n",
        "0",
        "-m",
        str(SNAPSHOT),
        "--isq",
        "q4k",
        "--mtp",
    ]


def phases(args, scripts, output):
    base_url = f"http://127.0.0.1:{args.port}"
    tokenizer = output / "checkpoint/tokenizer.json"
    common = [
        "--base-url",
        base_url,
        "--tokenizer",
        str(tokenizer),
        "--label",
        args.label,
    ]
    jobs = []
    if args.include_serving:
        jobs.append(
            (
                "serving",
                [
                    sys.executable,
                    str(scripts / "bench_serving.py"),
                    *common,
                    "--output",
                    str(output / "serving.json"),
                    "--iterations",
                    "5",
                    "--warmup",
                    "2",
                    "--max-tokens",
                    "128",
                ],
            )
        )
    if args.include_smokes:
        jobs.append(
            (
                "text",
                [
                    sys.executable,
                    str(scripts / "text_smoke.py"),
                    str(output / "text.json"),
                    "--base-url",
                    base_url,
                ],
            )
        )
    for name, concurrency, requests in (
        ("c1", ["1"], "8"),
        ("c6_c8", ["6", "8"], "24"),
    ):
        jobs.append(
            (
                name,
                [
                    sys.executable,
                    str(scripts / "bench_concurrency.py"),
                    *common,
                    "--output",
                    str(output / f"concurrency.{name}.json"),
                    "--concurrencies",
                    *concurrency,
                    "--requests",
                    requests,
                    "--modes",
                    "closed-loop",
                    "--warmup",
                    "2",
                    "--trials",
                    "5",
                    "--max-tokens",
                    "128",
                ],
            )
        )
    if args.include_smokes:
        jobs.append(
            (
                "mixed_context",
                [
                    sys.executable,
                    str(scripts / "mixed_context_smoke.py"),
                    str(output / "mixed_context.json"),
                    "--base-url",
                    base_url,
                    "--tokenizer",
                    str(tokenizer),
                ],
            )
        )
    for prompt_name in ("python", "math"):
        for concurrency in (1, 8):
            name = f"homogeneous_{prompt_name}_c{concurrency}"
            jobs.append(
                (
                    name,
                    [
                        sys.executable,
                        str(scripts / "bench_homogeneous.py"),
                        "run",
                        *common,
                        "--output",
                        str(output / f"{name}.json"),
                        "--prompt-name",
                        prompt_name,
                        "--concurrency",
                        str(concurrency),
                    ],
                )
            )
    return jobs


def stop_process(process):
    if process is None:
        return None
    forced = False
    # The process group also contains request workers if a harness fails or is interrupted.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        code = process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        forced = True
        os.killpg(process.pid, signal.SIGKILL)
        code = process.wait(timeout=STOP_TIMEOUT_SECONDS)
    return {"pid": process.pid, "returncode": code, "forced_kill": forced, **stamp()}


def fetch(base_url, endpoint):
    with urllib.request.urlopen(
        base_url + endpoint, timeout=HTTP_TIMEOUT_SECONDS
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {response.status}: {endpoint}")
        return response.read()


def isolation_check(output, name, owned_pids=()):
    snapshot = isolation_snapshot(owned_pids)
    save(output / f"{name}.processes.json", snapshot)
    if snapshot["unexpected_active"]:
        raise RuntimeError(
            f"Active compiler/profiler/model processes; see {name}.processes.json"
        )


def snapshot_phase(output, base_url, server, name, point):
    save(output / f"{name}.memory.{point}.json", memory_snapshot(server.pid))
    (output / f"{name}.metrics.{point}.txt").write_bytes(fetch(base_url, "/metrics"))
    isolation_check(output, f"{name}.{point}", (server.pid,))


def capture_provenance(args, output, metadata):
    scripts = output / "scripts"
    scripts.mkdir()
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    copies = [(HARNESS_DIR / name, scripts / name) for name in HARNESSES]
    copies += [
        (Path(__file__).resolve(), scripts / "run_final.py"),
        (Path(__file__).with_name("observe.py"), scripts / "observe.py"),
        (ANCESTOR, scripts / "ancestor_run_server.py"),
    ]
    copies += [
        (SNAPSHOT / name, checkpoint / name)
        for name in MODEL_FILES
        if (SNAPSHOT / name).is_file()
    ]
    for required in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        if not (SNAPSHOT / required).is_file():
            raise FileNotFoundError(SNAPSHOT / required)
    if args.build_metadata:
        copies.append((args.build_metadata, output / "candidate_build_metadata.json"))
    files = []
    for source, destination in copies:
        shutil.copyfile(source, destination)
        source_hash = digest(source)
        if digest(destination) != source_hash:
            raise RuntimeError(f"Copy changed: {source}")
        files.append(
            {
                "source": str(source),
                "copy": str(destination.relative_to(output)),
                "sha256": source_hash,
            }
        )
    metadata["files"] = files
    metadata["checkpoint"] = str(SNAPSHOT)
    metadata["git_head"] = command_output(["git", "rev-parse", "HEAD"])
    metadata["git_status"] = command_output(["git", "status", "--short"])
    source_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", *SOURCE_PATHS], cwd=ROOT
    )
    (output / "source.diff").write_bytes(source_diff)
    metadata["source_diff_sha256"] = digest(output / "source.diff")
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *SOURCE_PATHS,
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    sources = {
        name: digest(ROOT / name) for name in paths if name and (ROOT / name).is_file()
    }
    save(output / "source_files.sha256.json", sources)
    metadata["source_manifest_sha256"] = digest(output / "source_files.sha256.json")
    metadata["source_provenance_limit"] = (
        "Source snapshot is taken before launch; executable identity is its SHA256. Only supplied build metadata links the executable to a build."
    )
    metadata["binary"] = str(args.binary)
    metadata["binary_bytes"] = args.binary.stat().st_size
    metadata["binary_sha256"] = digest(args.binary)
    if metadata["binary_sha256"] != args.expected_binary_sha256:
        raise RuntimeError("Candidate binary SHA256 differs from the required value")
    metadata["binary_version"] = command_output([str(args.binary), "--version"])
    metadata["python_version"] = sys.version
    metadata["gpu"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,nounits",
        ]
    )
    return scripts


def run_phase(
    name, command, output, base_url, server, gpu_monitor, monitor_errors, metadata
):
    check_free_disk(output)
    snapshot_phase(output, base_url, server, name, "before")
    record = {"name": name, "command": command, "started": stamp(), "complete": False}
    metadata["phases"].append(record)
    save(output / "metadata.json", metadata)
    print(f"Starting {name}", flush=True)
    process = None
    try:
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + PHASE_TIMEOUT_SECONDS
            while process.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Model server exited during measurement")
                if gpu_monitor.poll() is not None or monitor_errors:
                    raise RuntimeError(f"Monitoring failed: {monitor_errors}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} exceeded {PHASE_TIMEOUT_SECONDS}s")
                time.sleep(1)
        record["process_finished"] = stamp()
        record["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(
                f"{name} failed: exit {process.returncode}; see {name}.log"
            )
        snapshot_phase(output, base_url, server, name, "after")
        record["complete"] = True
    finally:
        if process is not None and process.poll() is None:
            record["cleanup"] = stop_process(process)
        record["finished"] = stamp()
        save(output / "metadata.json", metadata)
    print(f"Finished {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--label", default="isq_mtp_adaptive_graphs36")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--include-serving", action="store_true")
    parser.add_argument("--include-smokes", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no files, hashing, HTTP, or GPU access",
    )
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    output = args.output.resolve()
    if not re.fullmatch(r"[a-f0-9]{64}", args.expected_binary_sha256):
        parser.error("expected-binary-sha256 must be a lowercase 64-digit SHA256")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server": server_command(args.binary, args.port),
                    "phases": phases(args, output / "scripts", output),
                },
                indent=2,
            )
        )
        return
    env, recorded_env = environment()
    check_free_disk(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "complete": False,
        "started": stamp(),
        "env": recorded_env,
        "phases": [],
        "label": args.label,
        "profiled": False,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "counter_scope": "Whole command including warmups; global swap counters do not identify model paging or timing impact.",
        "isolation_scope": "Boundary process snapshots, not a continuous monitor of all processes.",
    }
    save(output / "metadata.json", metadata)
    server = None
    gpu_monitor = None
    memory_thread = None
    stop_monitor = threading.Event()
    monitor_errors = []
    success = False

    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        isolation_check(output, "preflight")
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", args.port))
        scripts = capture_provenance(args, output, metadata)
        metadata["command"] = server_command(args.binary, args.port)
        save(output / "metadata.json", metadata)
        with ExitStack() as stack:
            gpu_log = stack.enter_context((output / "gpu.csv").open("w"))
            gpu_monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv,nounits",
                    "--loop-ms=1000",
                ],
                stdout=gpu_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log = stack.enter_context((output / "server.log").open("w"))
            server = subprocess.Popen(
                metadata["command"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            metadata["model_pid"] = server.pid
            save(output / "metadata.json", metadata)

            def monitor_memory():
                try:
                    with (output / "memory.jsonl").open("w") as memory_log:
                        while not stop_monitor.is_set():
                            check_free_disk(output)
                            memory_log.write(
                                json.dumps(memory_snapshot(server.pid)) + "\n"
                            )
                            memory_log.flush()
                            stop_monitor.wait(MONITOR_INTERVAL_SECONDS)
                except Exception as error:
                    monitor_errors.append(repr(error))

            memory_thread = threading.Thread(target=monitor_memory, daemon=True)
            memory_thread.start()
            base_url = f"http://127.0.0.1:{args.port}"
            start = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {server.returncode} during startup"
                    )
                if gpu_monitor.poll() is not None or monitor_errors:
                    raise RuntimeError(f"Monitoring failed: {monitor_errors}")
                try:
                    models = json.loads(fetch(base_url, "/v1/models"))
                    break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() - start > STARTUP_TIMEOUT_SECONDS:
                    raise TimeoutError("Server startup timed out")
                time.sleep(2)
            save(output / "models.json", models)
            startup_log = (output / "server.log").read_text()
            checks = {
                key: bool(re.search(pattern, startup_log))
                for key, pattern in STARTUP_PATTERNS.items()
            }
            save(
                output / "startup_validation.json",
                {
                    "complete": all(checks.values()),
                    "checks": checks,
                    "patterns": STARTUP_PATTERNS,
                },
            )
            if not all(checks.values()):
                raise RuntimeError(
                    f"Startup topology changed: {[key for key, ok in checks.items() if not ok]}"
                )
            metadata["startup_seconds"] = time.monotonic() - start
            metadata["ready"] = stamp()
            save(output / "metadata.json", metadata)
            print(f"Ready in {metadata['startup_seconds']:.1f}s", flush=True)
            for name, command in phases(args, scripts, output):
                run_phase(
                    name,
                    command,
                    output,
                    base_url,
                    server,
                    gpu_monitor,
                    monitor_errors,
                    metadata,
                )
            summary_command = [
                sys.executable,
                str(scripts / "summarize_concurrency.py"),
                str(output / "concurrency.c1.json"),
                str(output / "concurrency.c6_c8.json"),
                "--output",
                str(output / "concurrency.summary.json"),
            ]
            with (output / "concurrency.summary.txt").open("w") as summary_log:
                subprocess.run(
                    summary_command,
                    cwd=ROOT,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    check=True,
                    timeout=60,
                    stdout=summary_log,
                    stderr=subprocess.STDOUT,
                )
            metadata["summary_command"] = summary_command
            homogeneous_summary_command = [
                sys.executable,
                str(scripts / "bench_homogeneous.py"),
                "summarize",
                *[
                    str(output / f"homogeneous_{prompt}_c{concurrency}.json")
                    for prompt in ("python", "math")
                    for concurrency in (1, 8)
                ],
                "--output",
                str(output / "homogeneous.summary.json"),
            ]
            with (output / "homogeneous.summary.txt").open("w") as summary_log:
                subprocess.run(
                    homogeneous_summary_command,
                    cwd=ROOT,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    check=True,
                    timeout=60,
                    stdout=summary_log,
                    stderr=subprocess.STDOUT,
                )
            metadata["homogeneous_summary_command"] = homogeneous_summary_command
            if args.include_serving:
                serving = json.loads((output / "serving.json").read_text())
                if (
                    "repetitive_prompt" not in serving
                    or len(serving["correctness_checks"]) != 8
                ):
                    raise RuntimeError(
                        "Serving harness did not finish validation cases"
                    )
            if (
                args.include_smokes
                and not json.loads((output / "mixed_context.json").read_text())[
                    "complete"
                ]
            ):
                raise RuntimeError("Mixed-context smoke incomplete")
            snapshot_phase(output, base_url, server, "completed", "before_shutdown")
            metadata["measurements_finished"] = stamp()
            success = True
    except BaseException as error:
        metadata["error"] = repr(error)
        raise
    finally:
        try:
            metadata["server_shutdown"] = stop_process(server)
        except Exception as error:
            metadata["server_shutdown"] = {"error": repr(error)}
        stop_monitor.set()
        if memory_thread is not None:
            memory_thread.join(timeout=STOP_TIMEOUT_SECONDS)
            if memory_thread.is_alive():
                monitor_errors.append("Memory monitor did not stop")
        try:
            metadata["gpu_monitor_shutdown"] = stop_process(gpu_monitor)
        except Exception as error:
            metadata["gpu_monitor_shutdown"] = {"error": repr(error)}
            monitor_errors.append(repr(error))
        metadata["monitor_errors"] = monitor_errors
        save(output / "shutdown.memory.json", memory_snapshot())
        metadata["finished"] = stamp()
        metadata["binary_sha256_after"] = (
            digest(args.binary) if args.binary.is_file() else None
        )
        metadata["git_head_after"] = command_output(["git", "rev-parse", "HEAD"])
        same_binary = metadata["binary_sha256_after"] == metadata.get("binary_sha256")
        clean_shutdown = (
            server is not None
            and metadata["server_shutdown"].get("forced_kill") is False
            and metadata["server_shutdown"].get("returncode") in (0, -signal.SIGTERM)
        )
        monitor_shutdown = metadata["gpu_monitor_shutdown"] or {}
        if monitor_shutdown.get("returncode") not in (0, -signal.SIGTERM):
            monitor_errors.append("GPU monitor did not finish normally")
        metadata["complete"] = (
            success and same_binary and clean_shutdown and not monitor_errors
        )
        save(output / "metadata.json", metadata)
        files = {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name != "SHA256SUMS.json"
            and "__pycache__" not in path.parts
        }
        save(output / "SHA256SUMS.json", files)
    if not metadata["complete"]:
        raise SystemExit("Run failed completion checks; see metadata.json")
    print(f"Complete; server stopped; results: {output}", flush=True)


if __name__ == "__main__":
    main()
