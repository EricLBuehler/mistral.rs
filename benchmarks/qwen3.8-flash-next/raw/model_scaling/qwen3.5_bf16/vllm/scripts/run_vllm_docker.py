"""Benchmark the pinned cached vLLM image after the coordinator releases the GPU."""

import argparse
from contextlib import ExitStack
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
import uuid

import run_qwen3_moe as shared
from observe import memory_snapshot, stamp

HERE = Path(__file__).resolve().parent
IMAGE_ID = "sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f"
IMAGE_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
CHECKPOINT = Path("/home/ericbuehler/hf_models/qwen3.5_35b_a3b")
CONFIG_SHA256 = "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"
STOP_SECONDS = 30
STARTUP_SECONDS = 3600
KV_CACHE_BYTES = 4 * 1024**3


class ContainerServer:
    def __init__(self, process, pid):
        self.process = process
        self.pid = pid

    def poll(self):
        return self.process.poll()


def docker_json(*arguments):
    return json.loads(
        subprocess.check_output(["docker", *arguments], text=True, timeout=60)
    )


def container_info(cid):
    return docker_json("inspect", cid)[0]


def command(args, output, name):
    return [
        "docker",
        "run",
        "--pull=never",
        "--name",
        name,
        "--workdir",
        "/tmp",
        "--label",
        f"mistralrs-benchmark-run={name}",
        "--cidfile",
        str(output / "container.cid"),
        "--gpus",
        "device=0",
        "--ipc=host",
        "--network=host",
        "--mount",
        f"type=bind,src={CHECKPOINT},dst=/model,readonly",
        "--mount",
        f"type=bind,src={output / 'cache'},dst=/benchmark-cache",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "HF_HOME=/benchmark-cache/huggingface",
        "--env",
        "VLLM_CACHE_ROOT=/benchmark-cache/vllm",
        "--env",
        "TRITON_CACHE_DIR=/benchmark-cache/triton",
        "--env",
        "CUDA_CACHE_PATH=/benchmark-cache/cuda",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/benchmark-cache/torchinductor",
        "--env",
        "FLASHINFER_WORKSPACE_BASE=/benchmark-cache",
        "--entrypoint",
        "vllm",
        IMAGE_ID,
        "serve",
        "/model",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "default",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--mamba-ssm-cache-dtype",
        "auto",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_bytes),
        "--no-enable-prefix-caching",
        "--no-enable-log-requests",
    ]


def optional_metrics(base_url, path):
    try:
        path.write_bytes(shared.fetch(base_url, "/metrics"))
        return {"available": True, "path": path.name, "sha256": shared.digest(path)}
    except (OSError, TimeoutError) as error:
        return {
            "available": False,
            "error": repr(error),
            "interpretation": "Missing metrics are unavailable, not zero.",
        }


def phase_snapshot(output, base_url, server, name, point, metadata):
    shared.save(output / f"{name}.memory.{point}.json", memory_snapshot(server.pid))
    record = optional_metrics(base_url, output / f"{name}.metrics.{point}.txt")
    metadata.setdefault("metrics", {}).setdefault(name, {})[point] = record
    shared.isolation_check(output, f"{name}.{point}", (server.pid,))


def run_phase(name, cmd, output, base_url, server, monitor, errors, metadata):
    phase_snapshot(output, base_url, server, name, "before", metadata)
    row = {"name": name, "command": cmd, "started": stamp(), "complete": False}
    metadata["phases"].append(row)
    shared.save(output / "metadata.json", metadata)
    child = None
    print(f"Starting {name}", flush=True)
    try:
        with (output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                start_new_session=True,
            )
            deadline = time.monotonic() + shared.PHASE_TIMEOUT_SECONDS
            while child.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Owned vLLM container exited during requests")
                if monitor.poll() is not None or errors:
                    raise RuntimeError(f"Monitoring failed: {errors}")
                if time.monotonic() > deadline:
                    raise TimeoutError(name)
                time.sleep(1)
        row["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"Failed request phase: {name}; see {name}.log")
        phase_snapshot(output, base_url, server, name, "after", metadata)
        row["complete"] = True
    finally:
        if child is not None and child.poll() is None:
            row["cleanup"] = shared.stop_process(child)
        row["finished"] = stamp()
        shared.save(output / "metadata.json", metadata)
    print(f"Finished {name}", flush=True)


def stop_owned_container(cid, name, output):
    if cid is None:
        return None
    before = container_info(cid)
    if (
        before["Id"] != cid
        or before["Name"] != "/" + name
        or before["Config"]["Labels"].get("mistralrs-benchmark-run") != name
    ):
        raise RuntimeError("Container identity changed; refusing to stop it")
    shared.save(output / "container.before_stop.json", before)
    if before["State"]["Running"]:
        subprocess.run(
            ["docker", "stop", "--time", str(STOP_SECONDS), cid],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=STOP_SECONDS + 30,
        )
    after = container_info(cid)
    shared.save(output / "container.after_stop.json", after)
    if after["State"]["Running"]:
        raise RuntimeError("Owned container is still running")
    subprocess.run(
        ["docker", "rm", cid],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return {
        "id": cid,
        "name": name,
        "removed": True,
        "state": after["State"],
        **stamp(),
    }


def prepare(args, output, metadata):
    image = docker_json("image", "inspect", IMAGE_ID)[0]
    if (
        image["Id"] != IMAGE_ID
        or image["Architecture"] != "arm64"
        or image["Os"] != "linux"
    ):
        raise RuntimeError("Unexpected cached image identity/architecture")
    labels = image["Config"]["Labels"]
    if (
        labels.get("ai.vllm.build.commit") != IMAGE_COMMIT
        or labels.get("org.opencontainers.image.revision") != IMAGE_COMMIT
    ):
        raise RuntimeError("Cached image source-commit labels differ")
    shared.save(output / "image.inspect.json", image)
    metadata["image_id"] = image["Id"]
    metadata["image_commit"] = IMAGE_COMMIT
    metadata["image_repo_digests"] = image.get("RepoDigests", [])
    metadata["docker_version"] = subprocess.check_output(
        ["docker", "version", "--format", "{{json .}}"], text=True, timeout=30
    ).strip()
    if shared.digest(CHECKPOINT / "config.json") != CONFIG_SHA256:
        raise RuntimeError("Checkpoint config changed")
    metadata["checkpoint_validation"] = shared.checkpoint_validation(
        CHECKPOINT, "cutile-bf16"
    )
    scripts, checkpoint = output / "scripts", output / "checkpoint"
    scripts.mkdir()
    checkpoint.mkdir()
    copies = [(HERE / "harnesses" / name, scripts / name) for name in shared.HARNESSES]
    copies += [
        (HERE / name, scripts / name)
        for name in (
            "run_vllm_docker.py",
            "run_qwen3_moe.py",
            "observe.py",
            "image_preflight.py",
        )
    ]
    copies += [
        (CHECKPOINT / name, checkpoint / name)
        for name in shared.MODEL_FILES
        if (CHECKPOINT / name).is_file()
    ]
    metadata["files"] = []
    for source, target in copies:
        shutil.copyfile(source, target)
        digest = shared.digest(source)
        assert shared.digest(target) == digest
        metadata["files"].append(
            {
                "source": str(source),
                "copy": str(target.relative_to(output)),
                "sha256": digest,
            }
        )
    sys.path.insert(0, str(scripts))
    import bench_serving

    shared.save(scripts / "preflight_prompts.json", bench_serving.PROMPTS)
    preflight = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--mount",
        f"type=bind,src={CHECKPOINT},dst=/model,readonly",
        "--mount",
        f"type=bind,src={scripts},dst=/benchmark,readonly",
        "--entrypoint",
        "python3",
        IMAGE_ID,
        "/benchmark/image_preflight.py",
        "/model",
        "/benchmark/preflight_prompts.json",
    ]
    result = subprocess.run(
        preflight,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    (output / "image_preflight.stderr.log").write_text(result.stderr)
    (output / "image_preflight.stdout.log").write_text(result.stdout)
    metadata["image_preflight_command"] = preflight
    result.check_returncode()
    evidence = json.loads(result.stdout)
    assert evidence["complete"] and evidence["config_sha256"] == CONFIG_SHA256
    assert evidence["tokenizer_sha256"] == shared.digest(checkpoint / "tokenizer.json")
    shared.save(output / "image_preflight.json", evidence)
    metadata["versions"] = evidence["versions"]
    metadata["gpu"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,nounits",
        ],
        text=True,
        timeout=30,
    ).strip()
    return scripts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="qwen3_5_35b_a3b_bf16_vllm_0_28")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--kv-cache-bytes", type=int, default=KV_CACHE_BYTES)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not 1 <= args.port <= 65535
        or args.kv_cache_bytes <= 0
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error("Invalid port or KV-cache budget")
    name = "qwen35-bf16-" + uuid.uuid4().hex[:12]
    cmd = command(args, output, name)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server": cmd,
                    "phases": shared.phases(args, output / "scripts", output),
                    "scope": "Plan only; no Docker/GPU/HTTP activity",
                },
                indent=2,
            )
        )
        return
    shared.environment()
    if shutil.disk_usage(output.parent).free < shared.MIN_FREE_DISK_BYTES:
        raise RuntimeError("At least 1 GiB free disk is required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cache").mkdir()
    metadata = {
        "complete": False,
        "started": stamp(),
        "command": cmd,
        "label": args.label,
        "container_name": name,
        "phases": [],
        "target_only": True,
        "profiled": False,
        "checkpoint": str(CHECKPOINT),
        "expected_config_sha256": CONFIG_SHA256,
        "kv_cache_bytes": args.kv_cache_bytes,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "counter_scope": "Whole command including warmups; raw engine-specific vLLM counters only, optional and not zero-filled.",
        "memory_scope": "Global system counters and container init process VmSwap; worker memory may live in child processes.",
        "comparison_limits": [
            "Same checkpoint/request protocol, but engine/kernel/scheduler implementations differ.",
            "Explicit vLLM KV-byte budget differs from mistral.rs context-capped physical allocation; record actual startup capacity.",
            "Each engine runs sequentially; this is not a randomized causal experiment.",
        ],
    }
    shared.save(output / "metadata.json", metadata)
    process = server = monitor = memory_thread = None
    cid = None
    stop_monitor = threading.Event()
    errors = []
    success = False

    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        shared.isolation_check(output, "preflight")
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", args.port))
        scripts = prepare(args, output, metadata)
        shared.save(output / "metadata.json", metadata)
        with ExitStack() as stack:
            log = stack.enter_context((output / "server.log").open("w"))
            gpu_log = stack.enter_context((output / "gpu.csv").open("w"))
            process = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + STARTUP_SECONDS
            while True:
                cid_path = output / "container.cid"
                if cid_path.is_file():
                    candidate = cid_path.read_text().strip()
                    if re.fullmatch(r"[a-f0-9]{64}", candidate):
                        cid = candidate
                        break
                if process.poll() is not None:
                    raise RuntimeError(
                        "Docker run failed before creating its container"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("Container creation timed out")
                time.sleep(0.2)
            cid = (output / "container.cid").read_text().strip()
            if not re.fullmatch(r"[a-f0-9]{64}", cid):
                raise RuntimeError("Malformed owned container ID")
            info = container_info(cid)
            if (
                info["Id"] != cid
                or info["Name"] != "/" + name
                or info["Image"] != IMAGE_ID
            ):
                raise RuntimeError("Unexpected container identity")
            shared.save(output / "container.started.json", info)
            if info["State"]["Pid"] <= 0:
                raise RuntimeError("Container did not enter running state")
            server = ContainerServer(process, info["State"]["Pid"])
            metadata["container_id"] = cid
            metadata["model_pid"] = server.pid
            monitor = subprocess.Popen(
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

            def monitor_memory():
                try:
                    with (output / "memory.jsonl").open("w") as memory_log:
                        while not stop_monitor.is_set():
                            if (
                                shutil.disk_usage(output).free
                                < shared.MIN_FREE_DISK_BYTES
                            ):
                                raise RuntimeError(
                                    "Free disk fell below 1 GiB during vLLM cache compilation/measurement"
                                )
                            memory_log.write(
                                json.dumps(memory_snapshot(server.pid)) + "\n"
                            )
                            memory_log.flush()
                            stop_monitor.wait(shared.MONITOR_INTERVAL_SECONDS)
                except Exception as error:
                    errors.append(repr(error))

            memory_thread = threading.Thread(target=monitor_memory, daemon=True)
            memory_thread.start()
            base_url = f"http://127.0.0.1:{args.port}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM container exited during startup: {process.returncode}"
                    )
                if monitor.poll() is not None or errors:
                    raise RuntimeError(f"Monitoring failed: {errors}")
                try:
                    models = json.loads(shared.fetch(base_url, "/v1/models"))
                    if any(model["id"] == "default" for model in models["data"]):
                        break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("vLLM startup timed out")
                time.sleep(2)
            shared.save(output / "models.json", models)
            startup_log = (output / "server.log").read_text()
            evidence = [
                line
                for line in startup_log.splitlines()
                if re.search(
                    r"dtype|quant|MoE|moe|backend|cache|graph|model.*load",
                    line,
                    re.IGNORECASE,
                )
            ]
            shared.save(
                output / "startup_evidence.json",
                {
                    "complete": True,
                    "lines": evidence,
                    "scope": "Actual startup log excerpts plus explicit command. Backend names require inspection; no kernel trace was collected here.",
                },
            )
            capacity = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", startup_log)
            if not capacity or int(capacity[-1].replace(",", "")) < 16384:
                raise RuntimeError(
                    "Expected a logged GPU KV cache capacity of at least 16k tokens"
                )
            metadata["reported_gpu_kv_cache_tokens"] = int(
                capacity[-1].replace(",", "")
            )
            metadata["ready"] = stamp()
            shared.save(output / "metadata.json", metadata)
            for phase, phase_command in shared.phases(args, scripts, output):
                run_phase(
                    phase,
                    phase_command,
                    output,
                    base_url,
                    server,
                    monitor,
                    errors,
                    metadata,
                )
            summary = [
                sys.executable,
                str(scripts / "summarize_concurrency.py"),
                *[str(output / f"concurrency.c{c}.json") for c in (1, 6, 8)],
                "--output",
                str(output / "concurrency.summary.json"),
            ]
            with (output / "concurrency.summary.txt").open("w") as summary_log:
                subprocess.run(
                    summary,
                    check=True,
                    stdout=summary_log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                )
            metadata["summary_command"] = summary
            metadata["measurements_finished"] = stamp()
            success = True
    except BaseException as error:
        metadata["error"] = repr(error)
        raise
    finally:
        if cid is None and process is not None and (output / "container.cid").is_file():
            recovered = (output / "container.cid").read_text().strip()
            if re.fullmatch(r"[a-f0-9]{64}", recovered):
                cid = recovered
        stop_monitor.set()
        if memory_thread is not None:
            memory_thread.join(timeout=STOP_SECONDS)
            if memory_thread.is_alive():
                errors.append("Memory monitor did not stop")
        try:
            metadata["container_shutdown"] = stop_owned_container(cid, name, output)
        except Exception as error:
            metadata["container_shutdown"] = {"error": repr(error)}
        if process is not None:
            try:
                metadata["docker_returncode"] = process.wait(timeout=STOP_SECONDS)
            except subprocess.TimeoutExpired:
                metadata["docker_cleanup"] = shared.stop_process(process)
        try:
            metadata["gpu_monitor_shutdown"] = shared.stop_process(monitor)
        except Exception as error:
            errors.append(repr(error))
        monitor_result = metadata.get("gpu_monitor_shutdown") or {}
        if monitor is not None and monitor_result.get("returncode") not in (
            0,
            -signal.SIGTERM,
        ):
            errors.append("GPU monitor did not finish normally")
        metadata["monitor_errors"] = errors
        metadata["finished"] = stamp()
        shared.save(output / "shutdown.memory.json", memory_snapshot())
        cleanup = metadata.get("container_shutdown") or {}
        state = cleanup.get("state", {})
        metadata["complete"] = bool(
            success
            and cleanup.get("removed")
            and not state.get("OOMKilled")
            and state.get("ExitCode") in (0, 143)
            and not errors
        )
        shared.save(output / "metadata.json", metadata)
        shared.save(
            output / "SHA256SUMS.json",
            {
                str(path.relative_to(output)): shared.digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
                and path.name != "SHA256SUMS.json"
                and "cache" not in path.relative_to(output).parts
                and "__pycache__" not in path.parts
            },
        )
    if not metadata["complete"]:
        raise SystemExit("Run failed completion checks; see metadata.json")
    print(f"Complete; owned container removed: {output}", flush=True)


if __name__ == "__main__":
    main()
