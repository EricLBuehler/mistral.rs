#!/usr/bin/env python3
"""Run the v0.9.3 mistral.rs, vLLM, and SGLang decode-serving sweep."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

SCHEMA_VERSION = 1
RELEASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = Path(os.environ.get("MISTRALRS_REPO", RELEASE_DIR.parents[1]))
DEFAULT_SUMMARY = RELEASE_DIR / "raw" / "summary.json"
DEFAULT_FIGURE_DIR = RELEASE_DIR / "figures"
MODEL = "Qwen/Qwen3.8-27B-FP8"
DRAFT_MODEL = "incoai/Qwen3.8-27B-DFlash2"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
MODEL_CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
MISTRALRS_SOURCE_SHA256 = (
    "da48c78130710f387e7fe32a571149a157f543b8f9ad25970b3f76d45dd525e6"
)
VLLM_SERVER_COMMIT = "b389ac29465b33f9e9c534df221ea3c129e9793f"
VLLM_SERVER_WHEEL_SHA256 = (
    "a2cc284fbdefba0d8b42d97fece25ac4762407438a4fb8c9f351ed0136a42384"
)
VLLM_SERVER_WHEEL_URL = (
    "https://wheels.vllm.ai/b389ac29465b33f9e9c534df221ea3c129e9793f/"
    "vllm-0.26.1rc1.dev1048%2Bgb389ac294-cp38-abi3-manylinux_2_28_aarch64.whl"
)
SGLANG_COMMIT = "1cf2b8c54d81802abc15dcf23a29b9cc687bc01e"
VLLM_BENCH_VERSION = "0.27.1"
VLLM_SERVER_PACKAGES = {
    "flashinfer-python": "0.6.17",
    "humming-kernels": "0.1.12",
    "instanttensor": "0.1.9",
    "ninja": "1.13.0",
    "nvidia-cutlass-dsl": "4.6.2",
    "nvidia-cutlass-dsl-libs-base": "4.6.2",
    "nvidia-cutlass-dsl-libs-core": "4.6.2",
    "nvidia-cutlass-dsl-libs-cu12": "4.6.2",
    "nvidia-cutlass-dsl-libs-cu13": "4.6.2",
    "quack-kernels": "0.6.4",
    "torch": "2.13.0",
    "vllm": "0.26.1rc1.dev1048+gb389ac294",
}
CANONICAL_PROMPT_SHA256 = (
    "2afa8b3e16c26fa36c8d7bf474aac90bdabc32b909f90e685bd1d7bc0d2e64f4"
)
SEED = 20260825
OUTPUT_LEN = 512
MAX_MODEL_LEN = 4096
MAX_BATCHED_TOKENS = 4096
MAX_PREFILL_CHUNK_TOKENS = 512
TARGET_MEMORY_FRACTION = 0.66
DFLASH_MEMORY_FRACTION = 0.85
CONCURRENCIES = (1, 8, 16, 32, 64, 96, 128)
ENGINES = ("mistralrs", "vllm", "sglang")
MODES = ("target", "dflash")
STARTUP_LOG_TAIL_BYTES = 24000
SERVER_STOP_GRACE_SECONDS = 30
SERVER_TERM_GRACE_SECONDS = 15
SERVER_KILL_GRACE_SECONDS = 5
PORT_RELEASE_TIMEOUT_SECONDS = 60
OOM_RE = re.compile(
    r"out of memory|not enough gpu memory|insufficient.*gpu memory|"
    r"computed max_mamba_cache_size|cuda_error_out_of_memory|"
    r"torch\.outofmemoryerror|cuda out of memory",
    re.IGNORECASE,
)
UNSUPPORTED_RE = re.compile(r"not supported|unsupported", re.IGNORECASE)

TOPICS = [
    "How a public transit agency can redesign a crowded bus network while preserving access for elderly and disabled riders.",
    "Why database indexes improve some workloads but harm others, including write amplification, cache behavior, and query planning.",
    "How a coastal city should compare sea walls, wetland restoration, zoning changes, and managed retreat under uncertain climate forecasts.",
    "The scientific evidence for sleep's role in memory, immune function, mood, and athletic recovery, including important limitations.",
    "How a small manufacturer can improve quality without creating a blame culture, using measurement, feedback loops, and process design.",
    "The architectural tradeoffs between a modular monolith, microservices, and event-driven systems for a rapidly growing product team.",
    "How historians distinguish reliable primary evidence from propaganda, selective archives, retrospective testimony, and modern mythmaking.",
    "A practical framework for evaluating a major renewable-energy project across cost, reliability, land use, grid integration, and local consent.",
    "How an engineering organization can respond to a severe production outage while protecting users, learning quickly, and avoiding superficial fixes.",
    "The economic and social effects of housing shortages, with attention to permitting, construction capacity, infrastructure, and displacement.",
    "How a teacher can design an introductory statistics course that builds intuition while still teaching uncertainty and mathematical rigor.",
    "The opportunities and risks of using machine learning in clinical decision support, including calibration, distribution shift, and accountability.",
    "How a fictional expedition might survive a year in an isolated polar research station after its resupply route unexpectedly fails.",
    "Why biodiversity matters to agriculture and water systems, and how policy can align conservation incentives with rural livelihoods.",
    "How a security team should build a threat model for a multi-tenant cloud service without turning the exercise into a compliance checklist.",
    "The causes of organizational decision paralysis and a concrete operating model for making reversible and irreversible decisions well.",
]

ANGLES = [
    "Present the strongest competing viewpoints, identify hidden assumptions, and end with a staged recommendation and measurable checkpoints.",
    "Use a concrete running example, explain the causal mechanisms step by step, and include failure modes plus mitigations for each one.",
    "Write for an informed but non-specialist reader, define technical terms in context, and distinguish robust conclusions from uncertain judgments.",
    "Organize the response as an executive briefing with evidence, tradeoffs, implementation details, and a final decision framework.",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/mistralrs-v0.9.3-sweep")
    )
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--concurrencies", nargs="+", type=int, default=list(CONCURRENCIES)
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument(
        "--target-memory-fraction", type=float, default=TARGET_MEMORY_FRACTION
    )
    parser.add_argument(
        "--dflash-memory-fraction", type=float, default=DFLASH_MEMORY_FRACTION
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=MAX_BATCHED_TOKENS
    )
    parser.add_argument(
        "--max-prefill-chunk-tokens", type=int, default=MAX_PREFILL_CHUNK_TOKENS
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--model-config-sha256", default=MODEL_CONFIG_SHA256)
    parser.add_argument("--mistralrs-source-sha256", default=MISTRALRS_SOURCE_SHA256)
    parser.add_argument("--allow-mistralrs-source-mismatch", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--draft-snapshot", type=Path)
    parser.add_argument("--original-config", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument(
        "--mistralrs-bin",
        type=Path,
        default=Path(os.environ["MISTRALRS_BIN"])
        if "MISTRALRS_BIN" in os.environ
        else None,
    )
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=Path(os.environ.get("VLLM_BENCH_BIN", "/tmp/vllm-bench-venv/bin/vllm")),
    )
    parser.add_argument(
        "--vllm-python",
        type=Path,
        default=Path(os.environ.get("VLLM_PYTHON", "/tmp/vllm-bench-venv/bin/python")),
    )
    parser.add_argument(
        "--vllm-overlay",
        type=Path,
        default=Path(os.environ.get("VLLM_OVERLAY", "/tmp/vllm-dflash2-overlay")),
    )
    parser.add_argument(
        "--sglang-python",
        type=Path,
        default=Path(
            os.environ.get(
                "SGLANG_PYTHON", f"/tmp/sglang-{SGLANG_COMMIT}-venv/bin/python"
            )
        ),
    )
    parser.add_argument(
        "--sglang-source",
        type=Path,
        default=Path(os.environ.get("SGLANG_SOURCE", f"/tmp/sglang-{SGLANG_COMMIT}")),
    )
    parser.add_argument("--sglang-revision", default=SGLANG_COMMIT)
    parser.add_argument("--allow-sglang-revision-mismatch", action="store_true")
    parser.add_argument(
        "--sglang-enable-mixed-chunk",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sglang-sequence-headroom", type=int, default=1)
    parser.add_argument("--sglang-headroom-min-concurrency", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--continue-on-failure", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.mistralrs_bin is None:
        args.mistralrs_bin = args.repo_root / "target/release/mistralrs"
    args.mistralrs_bin = args.mistralrs_bin.expanduser().absolute()
    args.bench_bin = args.bench_bin.expanduser().absolute()
    args.vllm_python = args.vllm_python.expanduser().absolute()
    args.vllm_overlay = args.vllm_overlay.resolve()
    args.sglang_python = args.sglang_python.expanduser().absolute()
    args.sglang_source = args.sglang_source.resolve()
    if not args.repo_root.is_dir():
        raise SystemExit(f"repository not found: {args.repo_root}")
    actual_source_hash = git_source_sha256(args.repo_root)
    if (
        not args.allow_mistralrs_source_mismatch
        and actual_source_hash != args.mistralrs_source_sha256
    ):
        raise SystemExit(
            f"mistral.rs source hash is {actual_source_hash}, expected "
            f"{args.mistralrs_source_sha256}; pin the source or pass "
            "--allow-mistralrs-source-mismatch"
        )
    if not args.allow_mistralrs_source_mismatch and not git_source_is_clean(
        args.repo_root
    ):
        raise SystemExit(
            "mistral.rs has tracked source changes outside releases/; restore them or pass "
            "--allow-mistralrs-source-mismatch"
        )
    if args.repetitions <= 0 or args.output_len <= 1:
        raise SystemExit(
            "--repetitions must be positive and --output-len must exceed one"
        )
    if len(args.engines) != len(set(args.engines)) or len(args.modes) != len(
        set(args.modes)
    ):
        raise SystemExit("--engines and --modes cannot contain duplicates")
    if not args.concurrencies or len(args.concurrencies) != len(
        set(args.concurrencies)
    ):
        raise SystemExit("--concurrencies must contain unique positive values")
    if any(concurrency <= 0 or concurrency > 128 for concurrency in args.concurrencies):
        raise SystemExit("--concurrencies must be between 1 and 128")
    if args.sglang_sequence_headroom < 0:
        raise SystemExit("--sglang-sequence-headroom cannot be negative")
    if not 1 <= args.sglang_headroom_min_concurrency <= 128:
        raise SystemExit("--sglang-headroom-min-concurrency must be between 1 and 128")
    if not 0 < args.port < 65536:
        raise SystemExit("--port must be between 1 and 65535")
    if (
        not 0 < args.target_memory_fraction <= 1
        or not 0 < args.dflash_memory_fraction <= 1
    ):
        raise SystemExit("memory fractions must be in (0, 1]")
    required = {"benchmark client": args.bench_bin}
    if "mistralrs" in args.engines:
        required["mistral.rs binary"] = args.mistralrs_bin
    if "vllm" in args.engines:
        required["vLLM Python"] = args.vllm_python
    if "sglang" in args.engines:
        required["SGLang Python"] = args.sglang_python
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if "vllm" in args.engines and not (args.vllm_overlay / "vllm").is_dir():
        raise SystemExit(f"vLLM overlay not found: {args.vllm_overlay}")
    if "vllm" in args.engines:
        versions = package_versions(args.vllm_python, ["vllm"])
        if versions["vllm"] != VLLM_BENCH_VERSION:
            raise SystemExit(
                f"vLLM benchmark client is {versions['vllm']}, expected {VLLM_BENCH_VERSION}"
            )
        direct_urls = list(args.vllm_overlay.glob("vllm-*.dist-info/direct_url.json"))
        if len(direct_urls) != 1:
            raise SystemExit("vLLM overlay must contain exactly one direct_url.json")
        direct_url = json.loads(direct_urls[0].read_text(encoding="utf-8"))
        wheel_hash = direct_url.get("archive_info", {}).get("hashes", {}).get("sha256")
        if (
            direct_url.get("url") != VLLM_SERVER_WHEEL_URL
            or wheel_hash != VLLM_SERVER_WHEEL_SHA256
        ):
            raise SystemExit("vLLM overlay does not match the pinned DFlash2 wheel")
        server_versions = package_versions(
            args.vllm_python,
            list(VLLM_SERVER_PACKAGES),
            pythonpath=args.vllm_overlay,
        )
        if server_versions != VLLM_SERVER_PACKAGES:
            raise SystemExit(
                f"vLLM server packages are {server_versions}, expected "
                f"{VLLM_SERVER_PACKAGES}"
            )
    if "sglang" in args.engines:
        if not (args.sglang_source / "python/sglang").is_dir():
            raise SystemExit(f"SGLang source tree not found: {args.sglang_source}")
        actual_revision = command_output(
            ["git", "-C", str(args.sglang_source), "rev-parse", "HEAD"]
        )
        if (
            not args.allow_sglang_revision_mismatch
            and actual_revision != args.sglang_revision
        ):
            raise SystemExit(
                f"SGLang source is {actual_revision}, expected {args.sglang_revision}; "
                "pin the source or pass --allow-sglang-revision-mismatch"
            )
        source_python = args.sglang_source / "python"
        import_env = os.environ.copy()
        import_env["PYTHONPATH"] = str(source_python)
        import_check = subprocess.run(
            [
                str(args.sglang_python),
                "-c",
                "from sglang.srt.models.dflash import DFlash2DraftModel",
            ],
            env=import_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if import_check.returncode != 0:
            detail = import_check.stderr.strip() or import_check.stdout.strip()
            raise SystemExit(
                f"pinned SGLang environment cannot import DFlash2: {detail}"
            )
    if args.resume and args.dry_run:
        raise SystemExit("--resume and --dry-run are mutually exclusive")


def command_output(
    command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = completed.stdout.strip()
        return output if output else completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def git_diff_sha256(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=60,
        )
        return sha256_bytes(completed.stdout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def git_source_sha256(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    digest = hashlib.sha256()
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        _, path = record.split(b"\t", 1)
        if path.startswith(b"releases/"):
            continue
        digest.update(record)
        digest.update(b"\0")
    return digest.hexdigest()


def git_source_is_clean(root: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                ".",
                ":(exclude)releases/**",
            ],
            cwd=root,
            check=False,
            timeout=60,
        )
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    resolved = path.resolve()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def package_versions(
    python: Path, names: list[str], pythonpath: Path | None = None
) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        f"names={names!r}; "
        "print(json.dumps({n:(m.version(n) if n in {d.metadata['Name'] for d in m.distributions()} else 'unavailable') for n in names}))"
    )
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    output = command_output([str(python), "-c", code], env=env)
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(output)
    return {name: "unavailable" for name in names}


def fixed_prompts() -> list[str]:
    return [
        "Write a self-contained response of at least 900 words. Maintain a natural, varied style and develop every point fully rather than ending with a short summary. "
        f"Subject: {topic} Instructions: {angle}"
        for topic in TOPICS
        for angle in ANGLES
    ]


def prompt_jsonl(prompts: list[str], output_len: int) -> str:
    return "".join(
        json.dumps({"prompt": prompt, "output_tokens": output_len}) + "\n"
        for prompt in prompts
    )


def canonical_prompt_hash(output_len: int) -> str:
    digest = sha256_bytes(prompt_jsonl(fixed_prompts(), output_len).encode())
    if output_len == OUTPUT_LEN and digest != CANONICAL_PROMPT_SHA256:
        raise RuntimeError(
            f"canonical prompt hash is {digest}, expected {CANONICAL_PROMPT_SHA256}"
        )
    return digest


def expanded_prompts() -> list[str]:
    canonical = fixed_prompts()
    return [canonical[index % len(canonical)] for index in range(max(CONCURRENCIES))]


def requests_for_concurrency(concurrency: int) -> int:
    return max(64, concurrency)


def hf_cache_root(args: argparse.Namespace) -> Path:
    if args.hf_cache is not None:
        return args.hf_cache.expanduser().resolve()
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return (hf_home / "hub").expanduser().resolve()


def cached_snapshot(repo_id: str, revision: str, cache: Path) -> Path:
    return cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def resolve_snapshot(
    explicit: Path | None,
    repo_id: str,
    revision: str,
    args: argparse.Namespace,
) -> Path:
    if explicit is not None:
        snapshot = explicit.expanduser().resolve()
    else:
        snapshot = cached_snapshot(repo_id, revision, hf_cache_root(args))
    if snapshot.is_dir():
        return snapshot
    if args.dry_run:
        raise SystemExit(f"cached snapshot not found for dry run: {snapshot}")
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({repo_id!r}, revision={revision!r}))"
    )
    completed = subprocess.run(
        [str(args.vllm_python), "-c", code],
        cwd=args.repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip().splitlines()[-1]).resolve()


def resolve_models(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_snapshot = resolve_snapshot(
        args.model_snapshot, args.model, args.model_revision, args
    )
    draft_snapshot = resolve_snapshot(
        args.draft_snapshot, args.draft_model, args.draft_revision, args
    )
    if args.original_config is not None:
        candidates = (args.original_config.expanduser().resolve(),)
    else:
        candidates = (
            model_snapshot / "config.json",
            model_snapshot / "config.json.original",
        )
    original_config = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file()
            and sha256_file(candidate) == args.model_config_sha256
        ),
        None,
    )
    if original_config is None:
        observed = {
            str(candidate): sha256_file(candidate) if candidate.is_file() else None
            for candidate in candidates
        }
        raise SystemExit(
            f"no model config matches {args.model_config_sha256}; observed {observed}; "
            "provide --original-config"
        )
    return model_snapshot, draft_snapshot, original_config


@contextlib.contextmanager
def immutable_model_view(model_snapshot: Path, original_config: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="mistralrs-final-model-") as temporary:
        view = Path(temporary)
        for source in model_snapshot.iterdir():
            if source.name in {
                "config.json",
                "config.json.original",
                "config.json.long",
            }:
                continue
            (view / source.name).symlink_to(source.resolve())
        shutil.copyfile(original_config, view / "config.json")
        (view / "config.json").chmod(0o444)
        yield view


def mode_memory(args: argparse.Namespace, mode: str) -> float:
    return (
        args.target_memory_fraction if mode == "target" else args.dflash_memory_fraction
    )


def sglang_sequence_capacity(
    args: argparse.Namespace, mode: str, concurrency: int
) -> int:
    if mode == "target" and concurrency >= args.sglang_headroom_min_concurrency:
        return concurrency + args.sglang_sequence_headroom
    return concurrency


def server_command(
    engine: str,
    mode: str,
    concurrency: int,
    model_view: str,
    draft_snapshot: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str]]:
    memory = mode_memory(args, mode)
    env = os.environ.copy()
    if engine == "mistralrs":
        command = [
            str(args.mistralrs_bin),
            "serve",
            "-m",
            model_view,
            "--dtype",
            "bf16",
            "--max-model-len",
            str(args.max_model_len),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--no-ui",
            "--disable-access-log",
            "--max-seqs",
            str(concurrency),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-prefill-chunk-tokens",
            str(args.max_prefill_chunk_tokens),
            "--pa-memory-fraction",
            str(memory),
            "--pa-cache-type",
            "f8e4m3",
            "--prefix-cache-n",
            "0",
        ]
        if mode == "dflash":
            command.extend(
                [
                    "--mtp-model",
                    draft_snapshot,
                    "--mtp-n-predict",
                    "7",
                    "--mtp-draft-sampling",
                    "probabilistic",
                ]
            )
        return command, env
    if engine == "vllm":
        command = [
            str(args.vllm_python),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model_view,
            "--served-model-name",
            args.model,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(args.max_model_len),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--gpu-memory-utilization",
            str(memory),
            "--max-num-seqs",
            str(concurrency),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--language-model-only",
            "--async-scheduling",
            "--no-enable-prefix-caching",
        ]
        if mode == "dflash":
            speculative = {
                "method": "dflash",
                "model": draft_snapshot,
                "num_speculative_tokens": 7,
                "draft_sample_method": "probabilistic",
            }
            command.extend(
                ["--speculative-config", json.dumps(speculative, separators=(",", ":"))]
            )
        env["PYTHONPATH"] = str(args.vllm_overlay)
        env["PATH"] = f"{args.vllm_python.parent}:{env.get('PATH', '')}"
        return command, env
    sequence_capacity = sglang_sequence_capacity(args, mode, concurrency)
    command = [
        str(args.sglang_python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_view,
        "--revision",
        args.model_revision,
        "--served-model-name",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--context-length",
        str(args.max_model_len),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(memory),
        "--max-running-requests",
        str(sequence_capacity),
        "--max-mamba-cache-size",
        str(sequence_capacity),
        "--mamba-ssm-dtype",
        "float32",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--attention-backend",
        "flashinfer",
        "--chunked-prefill-size",
        str(args.max_num_batched_tokens),
        "--max-prefill-tokens",
        str(args.max_num_batched_tokens),
        "--cuda-graph-max-bs-decode",
        str(concurrency),
        "--disable-radix-cache",
        "--trust-remote-code",
        "--json-model-override-args",
        '{"language_model_only":true}',
    ]
    if mode == "dflash":
        command.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft_snapshot,
                "--speculative-draft-model-revision",
                args.draft_revision,
                "--speculative-num-draft-tokens",
                "8",
            ]
        )
    elif args.sglang_enable_mixed_chunk:
        command.append("--enable-mixed-chunk")
    source_python = args.sglang_source / "python"
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{source_python}:{prior_pythonpath}"
        if prior_pythonpath
        else str(source_python)
    )
    env["PATH"] = f"{args.sglang_python.parent}:{env.get('PATH', '')}"
    return command, env


def bench_command(
    engine: str,
    mode: str,
    concurrency: int,
    repetition: str,
    num_prompts: int,
    prompt_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    run_signature: str,
) -> list[str]:
    return [
        str(args.bench_bin),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--endpoint",
        "/v1/completions",
        "--tokenizer",
        str(prompt_path.parent / "model-view"),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(prompt_path),
        "--disable-shuffle",
        "--custom-output-len",
        str(args.output_len),
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--ignore-eos",
        "--temperature",
        "1",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--repetition-penalty",
        "1",
        "--extra-body",
        json.dumps({"seed": SEED}, separators=(",", ":")),
        "--seed",
        str(SEED),
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(output_path.parent),
        "--result-filename",
        output_path.name,
        "--metadata",
        f"engine={engine}",
        f"mode={mode}",
        f"concurrency={concurrency}",
        f"repetition={repetition}",
        f"run_signature={run_signature}",
    ]


def prompt_model_view_link(output_dir: Path, model_view: Path) -> None:
    link = output_dir / "model-view"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(model_view)


def provenance(args: argparse.Namespace) -> dict:
    result = {
        "harness": str(Path(__file__).resolve()),
        "harness_sha256": sha256_file(Path(__file__)),
        "mistralrs_commit": command_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root
        ),
        "mistralrs_source_sha256": git_source_sha256(args.repo_root),
        "mistralrs_tracked_status": command_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=args.repo_root
        ),
        "mistralrs_tracked_diff_sha256": git_diff_sha256(args.repo_root),
        "mistralrs_binary": str(args.mistralrs_bin),
        "mistralrs_binary_sha256": sha256_file(args.mistralrs_bin)
        if args.mistralrs_bin.is_file()
        else None,
        "benchmark_client": str(args.bench_bin),
        "benchmark_client_sha256": sha256_file(args.bench_bin),
        "vllm_python": str(args.vllm_python),
        "vllm_python_sha256": sha256_file(args.vllm_python)
        if args.vllm_python.is_file()
        else None,
        "vllm_server_commit": VLLM_SERVER_COMMIT,
        "vllm_server_wheel_url": VLLM_SERVER_WHEEL_URL,
        "vllm_server_wheel_sha256": VLLM_SERVER_WHEEL_SHA256,
        "sglang_python": str(args.sglang_python),
        "sglang_python_sha256": sha256_file(args.sglang_python)
        if args.sglang_python.is_file()
        else None,
        "sglang_source": str(args.sglang_source),
        "sglang_commit": command_output(
            ["git", "-C", str(args.sglang_source), "rev-parse", "HEAD"]
        ),
        "sglang_tracked_status": command_output(
            [
                "git",
                "-C",
                str(args.sglang_source),
                "status",
                "--short",
                "--untracked-files=no",
            ]
        ),
        "sglang_tracked_diff_sha256": git_diff_sha256(args.sglang_source),
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
    }
    if args.vllm_python.is_file():
        result["vllm_benchmark_packages"] = package_versions(
            args.vllm_python, ["vllm", "torch", "flashinfer-python"]
        )
        result["vllm_server_packages"] = package_versions(
            args.vllm_python,
            list(VLLM_SERVER_PACKAGES),
            pythonpath=args.vllm_overlay,
        )
        result["vllm_server_source_sha256"] = sha256_tree(
            args.vllm_overlay / "vllm", (".py", ".so")
        )
    if args.sglang_python.is_file():
        result["sglang_packages"] = package_versions(
            args.sglang_python,
            ["sglang", "sglang-kernel", "torch", "flashinfer-python", "sgl-deep-gemm"],
        )
    return result


def command_template(
    engine: str, mode: str, concurrency: int, args: argparse.Namespace
) -> list[str]:
    command, _ = server_command(
        engine,
        mode,
        concurrency,
        "<IMMUTABLE_MODEL_VIEW>",
        "<PINNED_DRAFT_SNAPSHOT>",
        args,
    )
    return command


def cell_key(engine: str, mode: str, concurrency: int) -> str:
    return f"{engine}:{mode}:c{concurrency}"


def build_payload(
    args: argparse.Namespace,
    model_snapshot: Path,
    draft_snapshot: Path,
    original_config: Path,
    prompt_hash: str,
) -> dict:
    cells = []
    for mode in args.modes:
        for concurrency in args.concurrencies:
            for engine in args.engines:
                cells.append(
                    {
                        "key": cell_key(engine, mode, concurrency),
                        "engine": engine,
                        "mode": mode,
                        "concurrency": concurrency,
                        "num_prompts": requests_for_concurrency(concurrency),
                        "memory_fraction": mode_memory(args, mode),
                        "server_command": command_template(
                            engine, mode, concurrency, args
                        ),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance(args),
        "models": {
            "model": args.model,
            "model_revision": args.model_revision,
            "model_snapshot": str(model_snapshot),
            "original_config": str(original_config),
            "original_config_sha256": sha256_file(original_config),
            "draft_model": args.draft_model,
            "draft_revision": args.draft_revision,
            "draft_snapshot": str(draft_snapshot),
            "view_policy": "snapshot entries symlinked except config variants; verified config copied read-only as config.json",
        },
        "workload": {
            "prompt_sha256": prompt_hash,
            "canonical_prompt_sha256": canonical_prompt_hash(args.output_len),
            "canonical_prompts": len(fixed_prompts()),
            "stored_prompts": len(expanded_prompts()),
            "request_count_policy": "max(64, concurrency), yielding 64/96/128 requests",
            "warmup": "one full-concurrency wave per cell",
            "repetitions": args.repetitions,
            "output_len": args.output_len,
            "seed": SEED,
            "sampling": {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "repetition_penalty": 1,
                "ignore_eos": True,
            },
        },
        "serving": {
            "restart_per_cell": True,
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8_e4m3",
            "gdn_state_dtype": "float32",
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
            "prefix_cache": False,
            "language_model_only_note": (
                "vLLM uses its standalone language-model-only mode; pinned SGLang uses its equivalent model-config override because its Qwen3.5 implementation supports language_model_only but the dedicated CLI flag has not added Qwen3.5 to its validation allowlist; mistral.rs serves the monolithic checkpoint without invoking vision for text requests"
            ),
            "target_memory_fraction": args.target_memory_fraction,
            "dflash_memory_fraction": args.dflash_memory_fraction,
            "dflash_proposals": 7,
            "dflash_policy_note": (
                "all engines use probabilistic DFlash2 draft selection with sampling-correct target verification; SGLang's block size 8 is the anchor plus seven proposals"
            ),
            "sglang_mixed_chunk": args.sglang_enable_mixed_chunk,
            "sglang_sequence_headroom": args.sglang_sequence_headroom,
            "sglang_headroom_min_concurrency": args.sglang_headroom_min_concurrency,
        },
        "cells": cells,
    }


def prepare_prompts(args: argparse.Namespace) -> tuple[Path, str]:
    prompt_path = args.output_dir / "prompts.jsonl"
    content = prompt_jsonl(expanded_prompts(), args.output_len)
    prompt_hash = sha256_bytes(content.encode())
    if args.resume:
        if not prompt_path.is_file() or sha256_file(prompt_path) != prompt_hash:
            raise SystemExit("existing prompt workload does not match this run")
    else:
        atomic_text(prompt_path, content)
    return prompt_path, prompt_hash


def initialize_manifest(args: argparse.Namespace, payload: dict) -> tuple[Path, dict]:
    manifest_path = args.output_dir / "manifest.json"
    signature_payload = {key: value for key, value in payload.items() if key != "cells"}
    signature_payload["cell_plan"] = payload["cells"]
    run_signature = canonical_sha256(signature_payload)
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f"cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature_sha256") != run_signature:
            raise SystemExit(
                "resume configuration or binary provenance does not match the manifest"
            )
        return manifest_path, manifest
    if manifest_path.exists() or any(args.output_dir.iterdir()):
        allowed = {args.output_dir / "prompts.jsonl"}
        unexpected = [path for path in args.output_dir.iterdir() if path not in allowed]
        if unexpected:
            raise SystemExit(
                f"output directory is not empty: {args.output_dir}; use --resume or a new directory"
            )
    manifest = {
        **payload,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "run_signature_sha256": run_signature,
        "results": {},
    }
    atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def update_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_json(manifest_path, manifest)


def check_port_free(port: int) -> None:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")


def http_get(url: str, timeout: float = 2) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return response.read().decode("utf-8", errors="replace")


def log_tail(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - STARTUP_LOG_TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def wait_ready(
    process: subprocess.Popen, port: int, timeout: float, server_log: Path
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with status {process.returncode}\n{log_tail(server_log)}"
            )
        try:
            http_get(f"http://127.0.0.1:{port}/v1/models")
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            time.sleep(1)
    raise TimeoutError(
        f"server did not become ready within {timeout:.0f}s\n{log_tail(server_log)}"
    )


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for server_signal, timeout in (
        (signal.SIGINT, SERVER_STOP_GRACE_SECONDS),
        (signal.SIGTERM, SERVER_TERM_GRACE_SECONDS),
        (signal.SIGKILL, SERVER_KILL_GRACE_SECONDS),
    ):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, server_signal)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def wait_port_release(port: int) -> None:
    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(1)
    raise RuntimeError(f"port {port} remained in use after server shutdown")


def run_benchmark(
    command: list[str], output_path: Path, log_path: Path, args: argparse.Namespace
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=args.benchmark_timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark client exited with status {completed.returncode}\n{log_tail(log_path)}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"benchmark client did not write {output_path}")


def normalized_result(
    path: Path,
    engine: str,
    mode: str,
    concurrency: int,
    repetition: int | str,
    expected_prompts: int,
    output_len: int,
    run_signature: str,
) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_metadata = {
        "engine": engine,
        "mode": mode,
        "concurrency": str(concurrency),
        "repetition": str(repetition),
        "run_signature": run_signature,
    }
    for key, expected in expected_metadata.items():
        if data.get(key) != expected:
            raise RuntimeError(
                f"{path} has {key}={data.get(key)!r}, expected {expected!r}"
            )
    if (
        data.get("num_prompts") != expected_prompts
        or data.get("max_concurrency") != concurrency
        or data.get("completed") != expected_prompts
        or data.get("failed") != 0
    ):
        raise RuntimeError(
            f"{path} num_prompts={data.get('num_prompts')} max_concurrency={data.get('max_concurrency')} "
            f"completed={data.get('completed')} failed={data.get('failed')}, expected "
            f"{expected_prompts}/{concurrency}/{expected_prompts}/0"
        )
    errors = [error for error in data.get("errors", []) if error]
    if errors:
        raise RuntimeError(f"{path} contains request errors: {errors[:3]}")
    output_lens = data.get("output_lens", [])
    if len(output_lens) != expected_prompts or any(
        abs(length - output_len) > 2 for length in output_lens
    ):
        raise RuntimeError(f"{path} contains unexpected output lengths")
    if data.get("total_output_tokens") != sum(output_lens):
        raise RuntimeError(f"{path} total_output_tokens does not match output_lens")
    keys = (
        "duration",
        "completed",
        "failed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "spec_decode_acceptance_rate",
        "spec_decode_acceptance_length",
        "max_concurrent_requests",
    )
    result = {key: data.get(key) for key in keys}
    result.update(
        {
            "engine": engine,
            "mode": mode,
            "concurrency": concurrency,
            "repetition": repetition,
            "num_prompts": expected_prompts,
            "raw_path": str(path),
            "raw_sha256": sha256_file(path),
            "output_len_min": min(output_lens),
            "output_len_max": max(output_lens),
        }
    )
    return result


def aggregate_runs(runs: list[dict]) -> dict:
    metrics = (
        "output_throughput",
        "request_throughput",
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "spec_decode_acceptance_rate",
        "spec_decode_acceptance_length",
    )
    medians = {}
    for metric in metrics:
        values = [run[metric] for run in runs if run.get(metric) is not None]
        medians[metric] = statistics.median(values) if values else None
    return {"medians": medians, "repetitions": runs}


def classify_failure(stage: str, error: BaseException, server_log: Path) -> str:
    text = f"{error}\n{log_tail(server_log)}"
    if OOM_RE.search(text):
        return "startup_oom" if stage == "startup" else "runtime_oom"
    if UNSUPPORTED_RE.search(text):
        return "unsupported"
    if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)):
        return f"{stage}_timeout"
    return f"{stage}_failure"


def result_paths(
    args: argparse.Namespace, engine: str, mode: str, concurrency: int
) -> tuple[Path, Path]:
    raw_dir = args.output_dir / "raw" / engine / mode / f"c{concurrency}"
    log_dir = args.output_dir / "logs" / engine / mode / f"c{concurrency}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, log_dir


def run_cell(
    engine: str,
    mode: str,
    concurrency: int,
    prompt_path: Path,
    model_view: Path,
    draft_snapshot: Path,
    args: argparse.Namespace,
    manifest: dict,
    manifest_path: Path,
) -> bool:
    key = cell_key(engine, mode, concurrency)
    prior = manifest["results"].get(key, {})
    if prior.get("status") == "completed":
        print(f"Skipping completed {key}", flush=True)
        return True
    attempt = int(prior.get("attempts", 0)) + 1
    raw_dir, log_dir = result_paths(args, engine, mode, concurrency)
    attempt_log_dir = log_dir / f"attempt-{attempt}"
    attempt_log_dir.mkdir(parents=True, exist_ok=True)
    server_log = attempt_log_dir / "server.log"
    command, env = server_command(
        engine,
        mode,
        concurrency,
        str(model_view),
        str(draft_snapshot),
        args,
    )
    command_record = {
        "command": command,
        "shell": shlex.join(command),
        "cwd": str(args.repo_root),
        "environment_overrides": {
            key: env[key]
            for key in ("PATH", "PYTHONPATH")
            if env.get(key) != os.environ.get(key)
        },
        "engine": engine,
        "mode": mode,
        "concurrency": concurrency,
        "attempt": attempt,
    }
    atomic_json(attempt_log_dir / "server_command.json", command_record)
    manifest["results"][key] = {
        "status": "running",
        "attempts": attempt,
        "started_at_utc": utc_now(),
        "server_log": str(server_log),
        "command_record": str(attempt_log_dir / "server_command.json"),
    }
    update_manifest(manifest_path, manifest)
    stage = "startup"
    process: subprocess.Popen | None = None
    try:
        check_port_free(args.port)
        print(f"Starting {key} (attempt {attempt})", flush=True)
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=args.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        wait_ready(process, args.port, args.startup_timeout, server_log)
        stage = "warmup"
        warmup_output = raw_dir / f"warmup-attempt-{attempt}.json"
        warmup_command = bench_command(
            engine,
            mode,
            concurrency,
            f"warmup-{attempt}",
            concurrency,
            prompt_path,
            warmup_output,
            args,
            manifest["run_signature_sha256"],
        )
        atomic_json(
            attempt_log_dir / "warmup_command.json",
            {"command": warmup_command, "shell": shlex.join(warmup_command)},
        )
        run_benchmark(
            warmup_command, warmup_output, attempt_log_dir / "warmup-client.log", args
        )
        normalized_result(
            warmup_output,
            engine,
            mode,
            concurrency,
            f"warmup-{attempt}",
            concurrency,
            args.output_len,
            manifest["run_signature_sha256"],
        )
        stage = "benchmark"
        runs = []
        num_prompts = requests_for_concurrency(concurrency)
        for repetition in range(1, args.repetitions + 1):
            output = raw_dir / f"r{repetition}.json"
            try:
                run = normalized_result(
                    output,
                    engine,
                    mode,
                    concurrency,
                    repetition,
                    num_prompts,
                    args.output_len,
                    manifest["run_signature_sha256"],
                )
                print(f"Reusing {key} repetition {repetition}", flush=True)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                client_log = attempt_log_dir / f"r{repetition}-client.log"
                client_command = bench_command(
                    engine,
                    mode,
                    concurrency,
                    str(repetition),
                    num_prompts,
                    prompt_path,
                    output,
                    args,
                    manifest["run_signature_sha256"],
                )
                atomic_json(
                    attempt_log_dir / f"r{repetition}-command.json",
                    {"command": client_command, "shell": shlex.join(client_command)},
                )
                run_benchmark(client_command, output, client_log, args)
                run = normalized_result(
                    output,
                    engine,
                    mode,
                    concurrency,
                    repetition,
                    num_prompts,
                    args.output_len,
                    manifest["run_signature_sha256"],
                )
            runs.append(run)
            print(
                f"{key} r{repetition}: {run['output_throughput']:.2f} output tok/s, "
                f"TTFT median/p99 {run['median_ttft_ms']:.2f}/{run['p99_ttft_ms']:.2f} ms, "
                f"TPOT median/p99 {run['median_tpot_ms']:.3f}/{run['p99_tpot_ms']:.3f} ms",
                flush=True,
            )
        aggregate = aggregate_runs(runs)
        aggregate.update(
            {
                "status": "completed",
                "attempts": attempt,
                "completed_at_utc": utc_now(),
                "engine": engine,
                "mode": mode,
                "concurrency": concurrency,
                "num_prompts": num_prompts,
                "memory_fraction": mode_memory(args, mode),
                "server_log": str(server_log),
                "warmup_raw": str(warmup_output),
            }
        )
        manifest["results"][key] = aggregate
        update_manifest(manifest_path, manifest)
        return True
    except KeyboardInterrupt as error:
        manifest["results"][key] = {
            **manifest["results"][key],
            "status": "interrupted",
            "stage": stage,
            "classification": "interrupted",
            "error": str(error),
            "finished_at_utc": utc_now(),
        }
        update_manifest(manifest_path, manifest)
        raise
    except Exception as error:  # noqa: BLE001
        failure = {
            **manifest["results"][key],
            "status": "failed",
            "stage": stage,
            "classification": classify_failure(stage, error, server_log),
            "error_type": type(error).__name__,
            "error": str(error),
            "server_log_tail": log_tail(server_log),
            "finished_at_utc": utc_now(),
        }
        manifest["results"][key] = failure
        atomic_json(raw_dir / f"failure-attempt-{attempt}.json", failure)
        update_manifest(manifest_path, manifest)
        print(f"{key} failed ({failure['classification']}): {error}", flush=True)
        return False
    finally:
        if process is not None:
            stop_server(process)
            wait_port_release(args.port)


def write_summaries(args: argparse.Namespace, manifest: dict) -> None:
    rows = []
    cells = {}
    for key, result in sorted(manifest["results"].items()):
        if result.get("status") != "completed":
            cells[key] = {
                "status": result.get("status"),
                "classification": result.get("classification"),
                "error": result.get("error"),
            }
            continue
        cells[key] = {
            "status": "completed",
            "engine": result["engine"],
            "mode": result["mode"],
            "concurrency": result["concurrency"],
            "num_prompts": result["num_prompts"],
            **result["medians"],
        }
        rows.extend(result["repetitions"])
    comparisons = {}
    for mode in args.modes:
        for concurrency in args.concurrencies:
            group = {}
            for engine in args.engines:
                value = cells.get(cell_key(engine, mode, concurrency), {})
                if value.get("status") == "completed":
                    group[engine] = value["output_throughput"]
            comparison = {"output_throughput": group}
            if "mistralrs" in group:
                for other in ("vllm", "sglang"):
                    if other in group:
                        comparison[f"mistralrs_vs_{other}_percent"] = 100 * (
                            group["mistralrs"] / group[other] - 1
                        )
            comparisons[f"{mode}:c{concurrency}"] = comparison
    summary = {
        "run_signature_sha256": manifest["run_signature_sha256"],
        "generated_at_utc": utc_now(),
        "cells": cells,
        "comparisons": comparisons,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_text(args.output_dir / "results.jsonl", lines)


def plot_summary(summary_path: Path, figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = summary["cells"]
    concurrencies = sorted(
        {cell["concurrency"] for cell in cells.values() if "concurrency" in cell}
    )
    engines = ("mistralrs", "vllm", "sglang")
    labels = {"mistralrs": "mistral.rs", "vllm": "vLLM", "sglang": "SGLang"}
    colors = {"mistralrs": "#2563eb", "vllm": "#6b7280", "sglang": "#f59e0b"}
    mode_labels = {"target": "Target-only", "dflash": "DFlash2"}
    width = 0.25
    xs = list(range(len(concurrencies)))
    figure_dir.mkdir(parents=True, exist_ok=True)

    available_modes = [
        mode
        for mode in MODES
        if any(
            cell.get("mode") == mode and cell.get("status") == "completed"
            for cell in cells.values()
        )
    ]
    for mode in available_modes:
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        maximum = max(
            cell["output_throughput"]
            for cell in cells.values()
            if cell.get("mode") == mode and cell.get("status") == "completed"
        )
        for engine_index, engine in enumerate(engines):
            offset = (engine_index - 1) * width
            values = []
            for concurrency in concurrencies:
                cell = cells.get(cell_key(engine, mode, concurrency), {})
                values.append(
                    cell.get("output_throughput")
                    if cell.get("status") == "completed"
                    else None
                )
            bars = ax.bar(
                [x + offset for x in xs],
                [value or 0 for value in values],
                width,
                color=colors[engine],
                label=labels[engine],
            )
            ax.bar_label(
                bars,
                labels=[
                    f"{value:,.0f}" if value is not None else "" for value in values
                ],
                padding=3,
                fontsize=8,
            )
            for x, value in zip(xs, values):
                if value is None:
                    ax.text(
                        x + offset,
                        maximum * 0.012,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=colors[engine],
                    )
        ax.set_xticks(xs, [f"C{concurrency}" for concurrency in concurrencies])
        ax.set_ylim(0, maximum * 1.14)
        ax.set_ylabel("Output tokens/s")
        ax.set_title(f"Qwen3.8-27B FP8 {mode_labels[mode]} Decode Serving")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{mode}_throughput.png", dpi=180)
        plt.close(fig)


def dry_run_plan(args: argparse.Namespace, payload: dict) -> None:
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cells": payload["cells"],
                "workload": payload["workload"],
                "serving": payload["serving"],
                "models": payload["models"],
                "provenance": payload["provenance"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.plot_only:
        figure_dir = args.figure_dir or DEFAULT_FIGURE_DIR
        plot_summary(
            args.summary.expanduser().resolve(), figure_dir.expanduser().resolve()
        )
        return
    validate_args(args)
    model_snapshot, draft_snapshot, original_config = resolve_models(args)
    prompt_content = prompt_jsonl(expanded_prompts(), args.output_len)
    payload = build_payload(
        args,
        model_snapshot,
        draft_snapshot,
        original_config,
        sha256_bytes(prompt_content.encode()),
    )
    if args.dry_run:
        dry_run_plan(args, payload)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path, prompt_hash = prepare_prompts(args)
    if prompt_hash != payload["workload"]["prompt_sha256"]:
        raise RuntimeError("prompt hash changed while preparing the run")
    manifest_path, manifest = initialize_manifest(args, payload)
    with immutable_model_view(model_snapshot, original_config) as model_view:
        prompt_model_view_link(args.output_dir, model_view)
        try:
            for mode in args.modes:
                for concurrency in args.concurrencies:
                    for engine in args.engines:
                        success = run_cell(
                            engine,
                            mode,
                            concurrency,
                            prompt_path,
                            model_view,
                            draft_snapshot,
                            args,
                            manifest,
                            manifest_path,
                        )
                        write_summaries(args, manifest)
                        if not success and not args.continue_on_failure:
                            raise SystemExit(1)
        finally:
            (args.output_dir / "model-view").unlink(missing_ok=True)
    write_summaries(args, manifest)
    plot_summary(
        args.output_dir / "summary.json", args.figure_dir or args.output_dir / "figures"
    )
    failed = [
        key
        for key, value in manifest["results"].items()
        if value.get("status") != "completed"
    ]
    if failed:
        print(
            f"Sweep completed with {len(failed)} failed or unsupported cells; see summary.json",
            flush=True,
        )
    else:
        print(
            f"Sweep completed successfully: {args.output_dir / 'summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
