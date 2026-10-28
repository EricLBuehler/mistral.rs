#!/usr/bin/env python3
"""Repeat the published GGUF C1 protocol against an already running server."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_ROOT.parents[1]
CLIENT = REPO_ROOT / "scripts/bench_c1_serving.py"
PROMPTS = RELEASE_ROOT / "raw/prompts/gguf.json"
PROMPTS_SHA256 = "049f78cd048bd286deb0deb54bb5715b0b80397ddee98a9bd49eed45cbc69f44"
MODEL = "Qwen/Qwen3.8-Flash-Next"
LENGTHS = (512, 2048, 8192)
OUTPUT_TOKENS = 128
WARMUP = 1
REPETITIONS = 3
MAX_MODEL_LEN = 16384
CHUNKS = (512, 8192)
SHARDS = 4
THREADS = 10


def command_text(command):
    return shlex.join(str(arg) for arg in command)


def build_commands(args):
    return [
        [
            "cargo",
            "--config",
            "profile.release.package.candle-core.codegen-units=1",
            "build",
            "--release",
            "-p",
            "mistralrs-cli",
            "--features",
            "cuda,flash-attn,cutile",
            f"-j{THREADS}",
        ],
        [
            "cmake",
            "-S",
            args.llama_dir,
            "-B",
            args.llama_dir / "build-release",
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=121",
        ],
        [
            "cmake",
            "--build",
            args.llama_dir / "build-release",
            "--target",
            "llama-server",
            "-j",
            THREADS,
        ],
    ]


def server_commands(args):
    shards = [
        args.gguf_dir
        / "UD-Q4_K_XL"
        / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-{SHARDS:05d}.gguf"
        for i in range(1, SHARDS + 1)
    ]
    commands = [
        (
            "mistral.rs, 8192-token chunks",
            [
                REPO_ROOT / "target/release/mistralrs",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "1234",
                "--no-ui",
                "--max-seqs",
                "1",
                "--max-model-len",
                MAX_MODEL_LEN,
                "--prefix-cache-n",
                "0",
                "--max-num-batched-tokens",
                max(CHUNKS),
                "--max-prefill-chunk-tokens",
                max(CHUNKS),
                "--dtype",
                "bf16",
                "--kv-cache",
                f"{MAX_MODEL_LEN}tokens",
                "-m",
                MODEL,
                "-f",
                ";".join(str(path) for path in shards),
                "--mmproj",
                args.gguf_dir / "mmproj-BF16.gguf",
            ],
        )
    ]
    for chunk in CHUNKS:
        commands.append(
            (
                f"llama.cpp, {chunk}-token chunks",
                [
                    args.llama_dir / "build-release/bin/llama-server",
                    "-m",
                    shards[0],
                    "-ngl",
                    "99",
                    "-fa",
                    "on",
                    "-ctk",
                    "bf16",
                    "-ctv",
                    "bf16",
                    "-b",
                    chunk,
                    "-ub",
                    chunk,
                    "-t",
                    THREADS,
                    "-c",
                    MAX_MODEL_LEN,
                    "-np",
                    "1",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "1235",
                    "--alias",
                    "qwen38",
                    "--no-cache-prompt",
                    "--cache-ram",
                    "0",
                    "--no-webui",
                ],
            )
        )
    return commands


def client_commands(args):
    common = [
        sys.executable,
        str(CLIENT),
        "--url",
        args.base_url,
        "--model",
        args.model,
        "--prompts",
        str(PROMPTS),
        "--lengths",
        *(str(length) for length in LENGTHS),
        "--warmup",
        str(WARMUP),
        "--repetitions",
        str(REPETITIONS),
        "--output-tokens",
        str(OUTPUT_TOKENS),
    ]
    if args.engine == "mistralrs":
        common += ["--mistralrs-metrics", "--text-tokenizer", str(args.text_tokenizer)]
    return [
        common + ["--output", str(args.output_dir / f"{phase}.json")]
        for phase in ("prewarm", "serving")
    ]


def validate_result(path):
    report = json.loads(path.read_text())
    if not report["complete"] or report["prompts_sha256"] != PROMPTS_SHA256:
        raise RuntimeError(f"Incomplete sweep or incorrect fixture: {path}")
    for length in LENGTHS:
        rows = [row for row in report["results"] if row["prompt_tokens"] == length]
        if (
            len(rows) != WARMUP + REPETITIONS
            or sum(row["warmup"] for row in rows) != WARMUP
        ):
            raise RuntimeError(f"Incorrect request count for {length} tokens: {path}")
    if any(
        row["output_tokens"] != OUTPUT_TOKENS or row["text_chunks"] != OUTPUT_TOKENS
        for row in report["results"]
    ):
        raise RuntimeError(
            f"The canonical protocol requires one text chunk per generated token: {path}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mistralrs", "llama"), required=True)
    parser.add_argument("--base-url", help="Server origin, optionally ending in /v1.")
    parser.add_argument(
        "--model", help="Served model name; defaults to the printed server command."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New or empty directory for the two sweeps."
    )
    parser.add_argument(
        "--text-tokenizer",
        type=Path,
        help="Local pinned tokenizer.json; required for mistral.rs.",
    )
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=Path("/path/to/gguf"),
        help="Model root used in printed server commands.",
    )
    parser.add_argument(
        "--llama-dir",
        type=Path,
        default=REPO_ROOT.parent / "llama.cpp",
        help="Checkout used in printed build/server commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and protocol without writes or requests.",
    )
    args = parser.parse_args()
    args.base_url = (
        args.base_url
        or f"http://127.0.0.1:{1234 if args.engine == 'mistralrs' else 1235}"
    ).rstrip("/")
    args.base_url = args.base_url.removesuffix("/v1")
    args.model = args.model or (MODEL if args.engine == "mistralrs" else "qwen38")
    if args.dry_run:
        args.output_dir = args.output_dir or Path("/path/to/new-run")
        args.text_tokenizer = args.text_tokenizer or Path(
            "/path/to/pinned/tokenizer.json"
        )
        print(f"Build commands (run from {REPO_ROOT}; GB10 CUDA architecture 121):")
        for command in build_commands(args):
            print(command_text(command))
        print(
            "\nServer alternatives: run one at a time, with no other inference or compilation."
        )
        for label, command in server_commands(args):
            print(f"{label}:\n{command_text(command)}")
        print(
            "\nClient protocol: C1; 512, 2048, 8192 prompt tokens; 128 generated tokens; temperature 0;"
        )
        print(
            "ignore EOS; prefix cache off; first run all 12 requests as prewarm, then run a separate sweep."
        )
        print(
            "In the serving sweep, discard one warmup and take the median of three requests per length."
        )
        for command in client_commands(args):
            print(command_text(command))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    if args.engine == "mistralrs" and (
        args.text_tokenizer is None or not args.text_tokenizer.is_file()
    ):
        parser.error(
            "mistral.rs requires --text-tokenizer pointing to the pinned local tokenizer.json"
        )
    if args.text_tokenizer is not None:
        args.text_tokenizer = args.text_tokenizer.resolve()
    if hashlib.sha256(PROMPTS.read_bytes()).hexdigest() != PROMPTS_SHA256:
        parser.error("the published GGUF prompt fixture has changed")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = client_commands(args)
    manifest = {
        "engine": args.engine,
        "base_url": args.base_url,
        "model": args.model,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompts_sha256": PROMPTS_SHA256,
        "client_sha256": hashlib.sha256(CLIENT.read_bytes()).hexdigest(),
        "commands": commands,
        "complete": False,
    }
    if args.engine == "mistralrs":
        manifest["tokenizer_sha256"] = hashlib.sha256(
            args.text_tokenizer.read_bytes()
        ).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for command in commands:
        print(command_text(command), flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)
        validate_result(Path(command[-1]))
    manifest["complete"] = True
    manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
