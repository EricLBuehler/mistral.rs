"""Run target-only Qwen3-family MoE concurrency measurements with a pinned local checkpoint."""

import argparse
from collections import Counter
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
SNAPSHOT = None
HARNESS_DIR = Path(__file__).resolve().parent / "harnesses"
HARNESSES = ("bench_serving.py", "bench_concurrency.py", "summarize_concurrency.py")
MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
)
STARTUP_TIMEOUT_SECONDS = 2400
PHASE_TIMEOUT_SECONDS = 7200
HTTP_TIMEOUT_SECONDS = 30
STOP_TIMEOUT_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2
SOURCE_PATHS = (
    ":(glob)mistralrs*/**",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/",
)
BACKEND_PATTERNS = {
    "cutile-bf16": r"Warming [1-9][0-9]* cuTile MoE kernels\.",
    "cutile-fp8": r"MoE experts backend: cuTile blockwise FP8 grouped GEMM",
    "gguf-q4k": r"Quantizing model weights to q4k, with sensitive tensors using q6k",
}
MIN_FREE_DISK_BYTES = 1024**3
MAX_TENSOR_HEADER_BYTES = 16 * 1024**2


def checkpoint_validation(snapshot, backend):
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") not in ("qwen3_moe", "qwen3_5_moe"):
        raise ValueError("Expected an actual Qwen3-family MoE checkpoint")
    text_config = config.get("text_config", config)
    if (
        text_config.get("num_experts", 0) <= 1
        or text_config.get("num_experts_per_tok", 0) < 1
    ):
        raise ValueError("Missing Qwen3 MoE expert geometry")
    quant = text_config.get("quantization_config", config.get("quantization_config"))
    if backend in ("cutile-bf16", "gguf-q4k"):
        if quant:
            raise ValueError("BF16 control requires an unquantized checkpoint")
    elif not (
        quant
        and quant.get("quant_method") == "fp8"
        and quant.get("weight_block_size") == [128, 128]
        and quant.get("activation_scheme") == "dynamic"
    ):
        raise ValueError("FP8 control requires dynamic blockwise 128x128 FP8")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    headers = []
    expert_dtypes = Counter()
    for name in shards:
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0:
            raise FileNotFoundError(snapshot / name)
        with (snapshot / name).open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            if not 0 < length <= MAX_TENSOR_HEADER_BYTES:
                raise ValueError(f"Unexpected safetensor header size: {name}")
            raw_header = source.read(length)
        header = json.loads(raw_header)
        for key, tensor in header.items():
            if (
                key != "__metadata__"
                and ".experts." in key
                and "scale" not in key
                and len(tensor["shape"]) >= 2
            ):
                expert_dtypes[tensor["dtype"]] += 1
        headers.append(
            {
                "shard": name,
                "header_bytes": length,
                "header_sha256": hashlib.sha256(raw_header).hexdigest(),
            }
        )
    if backend in ("cutile-bf16", "gguf-q4k") and set(expert_dtypes) != {"BF16"}:
        raise ValueError(f"Expected native BF16 expert tensors: {dict(expert_dtypes)}")
    return {
        "config": config,
        "text_config": text_config,
        "shards": [
            {"name": name, "bytes": (snapshot / name).stat().st_size} for name in shards
        ],
        "headers": headers,
        "expert_tensor_dtypes": dict(expert_dtypes),
        "tensor_inventory_scope": "Shard presence/sizes and safetensor header hashes/dtypes only; no full tensor-data hashing.",
    }


def startup_validation(log, config, backend):
    last_layer = config["num_hidden_layers"] - 1
    patterns = {
        "all_layers_cuda0": rf"Layers 0-{last_layer}: cuda\[0\]",
        "model_bf16": r"DType selected is BF16",
        "cache_bf16": r"PagedAttention KV cache type is BF16",
        "context_16k": r"available context length is 16384 tokens",
        "scheduler_8": r"max_num_seqs=8 max_num_batched_tokens=4096 max_prefill_chunk_tokens=512",
        "graphs_batch8": r"Captured [1-9][0-9]* CUDA decode graphs through batch bucket 8",
        "requested_moe_backend": BACKEND_PATTERNS[backend],
    }
    checks = {key: bool(re.search(pattern, log)) for key, pattern in patterns.items()}
    checks["no_cutile_warmup_failure"] = "cuTile MoE warmup failed" not in log
    checks["no_mtp_assistant"] = "MTP assistant" not in log
    checks["no_cutlass_fallback"] = (
        "using CUTLASS MoE kernels" not in log
        and "MoE experts backend: CUTLASS" not in log
    )
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "patterns": patterns,
        "backend_evidence_scope": "Startup selection/JIT logs with source provenance; no kernel-time profiling in this benchmark.",
    }


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True, timeout=60).strip()


def environment(backend=None):
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
    if backend == "cutile-bf16":
        env["MISTRALRS_MOE_BACKEND"] = "cutile"
        recorded["MISTRALRS_MOE_BACKEND"] = "cutile"
    return env, recorded


def server_command(binary, port, isq=None):
    command = [
        str(binary),
        "serve",
        "--no-ui",
        "--host",
        "127.0.0.1",
        "-p",
        str(port),
        "--disable-access-log",
        "--max-model-len",
        "16384",
        "--pa-context-len",
        "16384",
        "--max-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--max-prefill-chunk-tokens",
        "512",
        "--prefix-cache-n",
        "0",
        "--pa-cache-type",
        "auto",
        "--dtype",
        "bf16",
        "-m",
        str(SNAPSHOT),
    ]

    if isq:
        command.extend(["--isq", isq])
    return command


def phases(args, scripts, output):
    jobs = []
    for concurrency, requests in ((1, 8), (6, 24), (8, 24)):
        jobs.append(
            (
                f"c{concurrency}",
                [
                    sys.executable,
                    str(scripts / "bench_concurrency.py"),
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--tokenizer",
                    str(output / "checkpoint/tokenizer.json"),
                    "--label",
                    args.label,
                    "--output",
                    str(output / f"concurrency.c{concurrency}.json"),
                    "--concurrencies",
                    str(concurrency),
                    "--requests",
                    str(requests),
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
        (Path(__file__).resolve(), scripts / "run_qwen3_moe.py"),
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
    global SNAPSHOT
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--expected-backend", required=True, choices=BACKEND_PATTERNS)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--isq", choices=("q4k",))
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no files, hashing, HTTP, or GPU access",
    )
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    SNAPSHOT = args.snapshot.resolve()
    output = args.output.resolve()
    if not re.fullmatch(r"[a-f0-9]{64}", args.expected_binary_sha256):
        parser.error("expected-binary-sha256 must be a lowercase 64-digit SHA256")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if (args.expected_backend == "gguf-q4k") != (args.isq == "q4k"):
        parser.error(
            "gguf-q4k backend requires --isq q4k, and native cuTile controls prohibit ISQ"
        )
    if not re.fullmatch(r"[a-f0-9]{64}", args.expected_config_sha256):
        parser.error("expected-config-sha256 must be a lowercase 64-digit SHA256")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server": server_command(args.binary, args.port, args.isq),
                    "phases": phases(args, output / "scripts", output),
                },
                indent=2,
            )
        )
        return
    if digest(SNAPSHOT / "config.json") != args.expected_config_sha256:
        raise RuntimeError("Checkpoint config differs from required SHA256")
    checkpoint = checkpoint_validation(SNAPSHOT, args.expected_backend)
    if shutil.disk_usage(output.parent).free < MIN_FREE_DISK_BYTES:
        raise RuntimeError("At least 1 GiB free disk is required")
    env, recorded_env = environment(args.expected_backend)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "complete": False,
        "started": stamp(),
        "env": recorded_env,
        "phases": [],
        "label": args.label,
        "profiled": False,
        "target_only": True,
        "isq": args.isq,
        "expected_config_sha256": args.expected_config_sha256,
        "expected_backend": args.expected_backend,
        "allowed_backend_override": {"MISTRALRS_MOE_BACKEND": "cutile"}
        if args.expected_backend == "cutile-bf16"
        else {},
        "checkpoint_validation": checkpoint,
        "model_comparison_limit": "A different model/checkpoint is an architecture/backend control, not an isolated regression test against Flash-Next.",
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
            sock.bind(("127.0.0.1", args.port))
        scripts = capture_provenance(args, output, metadata)
        metadata["command"] = server_command(args.binary, args.port, args.isq)
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
            validation = startup_validation(
                startup_log, checkpoint["text_config"], args.expected_backend
            )
            save(output / "startup_validation.json", validation)
            if not validation["complete"]:
                raise RuntimeError(
                    f"Startup topology/backend check failed: {validation['checks']}"
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
                str(output / "concurrency.c6.json"),
                str(output / "concurrency.c8.json"),
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
