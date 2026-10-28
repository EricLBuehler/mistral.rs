#!/usr/bin/env python3
"""Replay the final warmed C1 FP4 protocol against a separately launched server."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_ROOT.parents[1]
HARNESS = REPO_ROOT / "scripts/bench_nvfp4_serving.py"
MODEL_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REVISION = "b9ce18a6887e18c089fdea8509a33e63429a3300"
SNAPSHOT = Path("hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots") / REVISION
VLLM_IMAGE = (
    "eugr/spark-vllm-b12x@sha256:"
    "57c24fc397c66d4d28a7d0d5f1d792934fd01a70f4574cff6315fb7bc329d5cb"
)
WORKLOAD = RELEASE_ROOT / "raw/prompts/fp4.json"
WARMUP = RELEASE_ROOT / "raw/prompts/fp4-independent-warmup.json"
REFERENCE = RELEASE_ROOT / "raw/prompts/fp4-reference.json"
WORKLOAD_SHA256 = "04d0560b4e013c34baf727dd2c79f5e285af7da9af9e6fad5945807b547033f4"
WARMUP_SHA256 = "5178aa72f9c9ee00f97b90dcd18e24acfbeb6387f01b12055b9a495e8869a2c4"
DEFAULT_PORTS = {"mistralrs": 8239, "vllm": 8238}
REFERENCE_COUNT = 4
TOP_LOGPROBS = 5
REFERENCE_TIMEOUT_SECONDS = 180
READY_TIMEOUT_SECONDS = 1800
READY_POLL_SECONDS = 2
READY_REQUEST_TIMEOUT_SECONDS = 2
PHASES = (
    ("reference", 4),
    ("repeat-logprobs", 9),
    ("independent-prewarm", 5),
    ("conditioning", 57),
    ("measured", 57),
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch_recipe(args):
    port = urllib.parse.urlparse(args.base_url).port or DEFAULT_PORTS[args.engine]
    if args.engine == "mistralrs":
        cuda_root = args.cuda_root
        environment = {
            "NVCC": cuda_root / "bin/nvcc",
            "CUDA_ROOT": cuda_root,
            "CUDA_HOME": cuda_root,
            "CUDA_TOOLKIT_PATH": cuda_root,
            "CUTILE_TILEIRAS_PATH": cuda_root / "bin/tileiras",
            "LD_LIBRARY_PATH": f"{cuda_root}/compat:{cuda_root}/lib64",
            "RUST_BACKTRACE": "1",
        }
        lines = [f"cd {shlex.quote(str(REPO_ROOT))}"]
        lines.extend(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in environment.items()
        )
        lines.append(f'export PATH={shlex.quote(str(cuda_root / "bin"))}:"$PATH"')
        lines.append(
            shlex.join(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "mistralrs-cli",
                    "--features",
                    "cuda,flash-attn,cutile",
                    "--config",
                    "profile.release.package.candle-core.codegen-units=1",
                    "-j2",
                ]
            )
        )
        command = [
            "target/release/mistralrs",
            "serve",
            "-m",
            str(args.hf_cache / SNAPSHOT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        command.extend(
            """--dtype bf16 --max-model-len 16384 --max-seqs 1
            --max-num-batched-tokens 4096 --max-prefill-chunk-tokens 4096
            --prefix-cache-n 0 --kv-cache 2GiB --no-ui --disable-access-log""".split()
        )
        lines.append(shlex.join(command))
        return lines
    cache_mounts = {
        "vllm": "/root/.cache/vllm",
        "flashinfer": "/root/.cache/flashinfer",
        "triton": "/root/.triton",
        "b12x": "/root/.cache/b12x",
    }
    command = """docker run --rm --name qwen38-fp4-vllm-exact-device
        --gpus all --ipc host --network host --shm-size=16g
        --ulimit memlock=-1 --ulimit stack=67108864
        --security-opt seccomp=unconfined --security-opt apparmor=unconfined""".split()
    command.extend(["-v", f"{args.hf_cache}:/root/.cache/huggingface:ro"])
    for name, target in cache_mounts.items():
        command.extend(["-v", f"{args.kernel_cache / name}:{target}"])
    environment = """HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.9 B12X_DISK_BACKEND=io_uring
        CUTE_DSL_ARCH=sm_121a SAFETENSORS_FAST_GPU=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
        VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_USE_AOT_COMPILE=1 VLLM_USE_MEGA_AOT_ARTIFACT=1
        VLLM_USE_V2_MODEL_RUNNER=1 B12X_POLICY_MODE=auto VLLM_MXFP8_LM_HEAD=0
        VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 VLLM_B12X_MOE_FP4_FORCE_A16=1
        B12X_W4A16_A4_PREFILL=1""".split()
    for variable in environment:
        command.extend(["-e", variable])
    command.extend(
        [
            "--entrypoint",
            "vllm",
            VLLM_IMAGE,
            "serve",
            str(Path("/root/.cache/huggingface") / SNAPSHOT),
            "--served-model-name",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )
    command.extend(
        """--tensor-parallel-size 1 --pipeline-parallel-size 1
        --mamba-cache-mode align --no-enable-prefix-caching --enable-chunked-prefill
        --dtype bfloat16 --kv-cache-dtype bfloat16 --quantization modelopt_mixed
        --block-size 16 --load-format b12x --max-model-len 16384
        --max-num-seqs 1 --max-num-batched-tokens 4096 --max-cudagraph-capture-size 1
        --kv-cache-memory-bytes 2147483648 --gdn-decode-kernel b12x
        --linear-backend b12x --moe-backend b12x --no-enable-flashinfer-autotune
        --mm-encoder-tp-mode data --reasoning-parser qwen3 --tool-call-parser qwen3_xml
        --enable-auto-tool-choice --gpu-memory-utilization 0.85""".split()
    )
    command.extend(
        [
            "--compilation-config",
            '{"pass_config":{"fuse_act_quant":true}}',
            "--additional-config",
            '{"ple_table_memory":"device"}',
        ]
    )
    return [
        shlex.join(
            ["mkdir", "-p", *(str(args.kernel_cache / name) for name in cache_mounts)]
        ),
        shlex.join(command),
    ]


def benchmark_command(args, phase, workload):
    return [
        sys.executable,
        str(HARNESS),
        "run",
        "--workload",
        str(workload),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--engine",
        f"{args.engine}-fp4-final-{phase}",
        "--output-dir",
        str(args.output_dir / phase),
    ]


def validate_inputs(args):
    for path, expected in ((WORKLOAD, WORKLOAD_SHA256), (WARMUP, WARMUP_SHA256)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"workload checksum mismatch: {path}: {digest}")
    reference = json.loads(REFERENCE.read_text())
    expected = json.loads(args.reference_results.read_text())
    if len(reference["cases"]) != REFERENCE_COUNT or len(expected) != REFERENCE_COUNT:
        raise ValueError("expected the four original reference cases and results")
    if [case["id"] for case in reference["cases"]] != [row["id"] for row in expected]:
        raise ValueError("reference case IDs do not match saved results")
    if not HARNESS.is_file():
        raise FileNotFoundError(HARNESS)


def run_reference(args):
    source = json.loads(REFERENCE.read_text())
    common = {key: value for key, value in source.items() if key != "cases"}
    common["model"] = args.model
    results = []
    for case in source["cases"]:
        payload = dict(
            common, messages=case["messages"], logprobs=True, top_logprobs=TOP_LOGPROBS
        )
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(
            request, timeout=REFERENCE_TIMEOUT_SECONDS
        ) as response:
            result = json.load(response)
        results.append(
            {
                "id": case["id"],
                "request": payload,
                "response": result,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        write_json(args.output_dir / "reference.json", results)
        print(
            json.dumps({"id": case["id"], "text": result["choices"][0]["message"]}),
            flush=True,
        )
    expected = json.loads(args.reference_results.read_text())
    comparisons = [
        {
            "id": actual["id"],
            "text_equal": actual["response"]["choices"][0]["message"]["content"]
            == saved["response"]["choices"][0]["message"]["content"],
        }
        for actual, saved in zip(results, expected)
    ]
    write_json(args.output_dir / "reference-comparison.json", comparisons)
    if not all(row["text_equal"] for row in comparisons):
        raise RuntimeError(
            "reference text differs from the recorded original-checkpoint responses"
        )


def wait_for_server(args):
    deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            with urllib.request.urlopen(
                args.base_url.rstrip("/") + "/models",
                timeout=READY_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"server did not become ready: {args.base_url}")
        time.sleep(READY_POLL_SECONDS)


def run(args):
    from check_fp4_repeats import check_repeats

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    protocol = {
        "engine": args.engine,
        "model": args.model,
        "base_url": args.base_url,
        "checkpoint": MODEL_ID,
        "checkpoint_revision": REVISION,
        "workload_sha256": WORKLOAD_SHA256,
        "independent_prewarm_sha256": WARMUP_SHA256,
        "reference_results": str(args.reference_results),
        "reference_results_sha256": hashlib.sha256(
            args.reference_results.read_bytes()
        ).hexdigest(),
        "launch_recipe": launch_recipe(args),
        "phases": [{"phase": name, "requests": count} for name, count in PHASES],
        "started_at": utc_now(),
    }
    write_json(args.output_dir / "protocol.json", protocol)

    def phase(name, **extra):
        value = {"phase": name, "time": utc_now(), **extra}
        write_json(args.output_dir / "phase.json", value)
        print(json.dumps(value), flush=True)

    active_phase = "waiting_for_server"
    try:
        phase(active_phase)
        wait_for_server(args)
        active_phase = "reference"
        phase(active_phase)
        with (
            (args.output_dir / "reference.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            run_reference(args)
        active_phase = "repeat-logprobs"
        phase(active_phase)
        with (
            (args.output_dir / "repeat-logprobs.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            check_repeats(
                args.base_url, args.model, args.output_dir / active_phase, WORKLOAD
            )
        for active_phase, workload in (
            ("independent-prewarm", WARMUP),
            ("conditioning", WORKLOAD),
            ("measured", WORKLOAD),
        ):
            phase(active_phase)
            with (args.output_dir / f"{active_phase}.log").open("w") as log:
                subprocess.run(
                    benchmark_command(args, active_phase, workload),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        phase("completed")
    except Exception as error:
        phase("failed", failed_phase=active_phase, error=str(error))
        raise


def parse_args():
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=DEFAULT_PORTS)
    parser.add_argument(
        "--base-url",
        help="API root ending in /v1; defaults to the recorded engine port",
    )
    parser.add_argument(
        "--model", help="served model name; defaults to the recorded engine name"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and protocol without writing or requesting anything",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(os.environ.get("HF_HOME", cache_home / "huggingface")),
        help="HF cache root containing hub/; used only in printed server commands",
    )
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path("/usr/local/cuda-13.3"),
        help="CUDA 13.3 installation used in printed mistral.rs commands",
    )
    parser.add_argument(
        "--kernel-cache",
        type=Path,
        default=cache_home / "mistralrs-fp4-release",
        help="persistent vLLM kernel cache root used in printed commands",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=RELEASE_ROOT / "raw/source/fp4/vllm/reference.json",
    )
    parser.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_SECONDS)
    args = parser.parse_args()
    args.base_url = args.base_url or f"http://127.0.0.1:{DEFAULT_PORTS[args.engine]}/v1"
    args.model = args.model or ("default" if args.engine == "mistralrs" else MODEL_ID)
    for field in (
        "output_dir",
        "hf_cache",
        "cuda_root",
        "kernel_cache",
        "reference_results",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    url = urllib.parse.urlparse(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.path.rstrip("/") != "/v1"
    ):
        parser.error("base-url must be an HTTP(S) API root ending in /v1")
    if args.ready_timeout <= 0:
        parser.error("ready-timeout must be positive")
    return args


def main():
    args = parse_args()
    validate_inputs(args)
    print(
        "Launch one engine at a time with the pinned checkpoint already in the HF cache."
    )
    print("Keep other GPU work and compilers idle during all requests.")
    print("Server/build recipe (printed only; run in a separate terminal):")
    print("\n".join(launch_recipe(args)))
    print("Protocol: " + " -> ".join(f"{name} ({count})" for name, count in PHASES))
    print(
        "References: max_tokens=48, normal EOS. Repeats/performance: 128 tokens, ignore_eos=true."
    )
    print(
        "All requests: temperature=0, seed=20260904, thinking off; prefix caching disabled in server."
    )
    print(
        "The measured sweep includes three warmups and 54 scored requests (18 per context)."
    )
    print(
        "Preserve conditioning and measured outputs; compare every scored request, not the best round."
    )
    print(
        "Record RAM/swap, GPU clocks/temperature and process activity alongside the run."
    )
    for phase, workload in (
        ("independent-prewarm", WARMUP),
        ("conditioning", WORKLOAD),
        ("measured", WORKLOAD),
    ):
        print(shlex.join(benchmark_command(args, phase, workload)))
    if not args.dry_run:
        run(args)


if __name__ == "__main__":
    main()
