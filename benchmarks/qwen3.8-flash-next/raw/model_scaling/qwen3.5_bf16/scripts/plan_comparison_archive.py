"""Prepare a hash-identified, sanitized compact archive plan after both runs complete."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_bf16 import ROOT, compare, digest, require

MAX_COPY_BYTES = 2 * 1024**2
EXTERNAL_NAMES = {"tokenizer.json", "tokenizer_config.json"}
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".py",
    ".md",
    ".log",
    ".txt",
    ".csv",
    ".diff",
    ".cid",
}
SENSITIVE_KEY = re.compile(
    r"^(?:hf_token|api[-_]?key|access[-_]?token|auth(?:orization)?|password|secret|credentials?)$",
    re.I,
)
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|API_KEY|ACCESS_TOKEN|PASSWORD|SECRET)=[^\s\"']+"
    ),
    re.compile(r"(?i)--(?:hf-token|api-key|access-token|password)\s+\S+"),
)


def redact_text(value):
    for pattern in SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE_KEY.fullmatch(key) else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        output, redact_next = [], False
        for item in value:
            output.append("[REDACTED]" if redact_next else sanitize(item))
            redact_next = isinstance(item, str) and item in (
                "--hf-token",
                "--api-key",
                "--access-token",
                "--password",
            )
        return output
    return redact_text(value) if isinstance(value, str) else value


def sanitized_bytes(path):
    original = path.read_bytes()
    text = original.decode()
    if path.suffix == ".json":
        obj = json.loads(text)
        redacted = sanitize(obj)
        return (
            original
            if redacted == obj
            else (json.dumps(redacted, indent=2, allow_nan=False) + "\n").encode()
        )
    if path.suffix == ".jsonl":
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        redacted = [sanitize(line) for line in lines]
        return (
            original
            if redacted == lines
            else (
                "\n".join(json.dumps(line, allow_nan=False) for line in redacted) + "\n"
            ).encode()
        )
    return redact_text(text).encode()


def excluded_reason(path, relative):
    if "cache" in relative.parts or "__pycache__" in relative.parts:
        return "Compiler/runtime cache, external only"
    if path.name in EXTERNAL_NAMES:
        return "Tokenizer copy, identity retained by hash"
    if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > MAX_COPY_BYTES:
        return "Large or non-text artifact, external only"
    return None


def plan(root, native=None, vllm=None, failed_runs=()):
    comparison = compare(root, native, vllm)
    included, external = [], []
    directories = [
        (Path(path), Path(engine))
        for engine, path in comparison["run_directories"].items()
    ]
    for directory in failed_runs:
        require(
            directory.resolve() not in [path.resolve() for path, _ in directories],
            "Repeated run directory",
        )
        require(
            json.loads((directory / "metadata.json").read_text())["complete"] is False,
            "Failed-attempt input reports complete",
        )
        directories.append((directory, Path("failed_attempts") / directory.name))
    for directory, archive_root in directories:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = archive_root / path.relative_to(directory)
            record = {
                "source": str(path),
                "relative": str(relative),
                "bytes": path.stat().st_size,
            }
            reason = excluded_reason(path, path.relative_to(directory))
            if reason:
                try:
                    record["sha256"] = digest(path)
                except PermissionError:
                    record["sha256"] = None
                    record["hash_unavailable"] = (
                        "PermissionError: external cache is owned by the container"
                    )
                record["reason"] = reason
                external.append(record)
                continue
            record["sha256"] = digest(path)
            redacted = sanitized_bytes(path)
            record.update(
                {
                    "archive_sha256": hashlib.sha256(redacted).hexdigest(),
                    "archive_bytes": len(redacted),
                    "sanitized": redacted != path.read_bytes(),
                }
            )
            included.append(record)
    for name in (
        "compare_bf16.py",
        "plan_comparison_archive.py",
        "test_comparison.py",
        "comparison_plan.md",
        "precision_evidence.json",
        "comparison_preparation.json",
    ):
        path = root / name
        require(path.is_file(), f"Missing support script {name}")
        included.append(
            {
                "source": str(path),
                "relative": f"scripts/{name}",
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "archive_sha256": digest(path),
                "archive_bytes": path.stat().st_size,
                "sanitized": False,
            }
        )
    return {
        "complete": True,
        "execution": "Plan only; does not copy files or alter runs",
        "source_root": str(root),
        "selected_run_directories": comparison["run_directories"],
        "failed_attempts_excluded_from_statistics": [
            str(path.resolve()) for path in failed_runs
        ],
        "included": included,
        "external_artifacts": external,
        "copy_bytes": sum(x["archive_bytes"] for x in included),
        "limits": [
            "Original source manifests remain evidence; a new archive manifest must cover sanitized copies.",
            "Model weights and Docker images remain at their pinned external locations; no weight or binary copies.",
            "Caches, tokenizer copies, traces, and files over 2 MiB remain external with file hashes.",
            "Container-owned unreadable cache files retain paths and sizes with explicit unavailable hashes; cache inventory is not an exhaustive permission audit.",
            "Add completed comparison.json/comparison.md and root-reviewed report only after comparison validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--vllm", type=Path)
    parser.add_argument("--failed-run", type=Path, action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "comparison.archive-plan.json"
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite archive plan")
    result = plan(args.root, args.native, args.vllm, args.failed_run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(args.output),
                "copy_bytes": result["copy_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
