"""Archive the completed engine comparison and separate native kernel proof."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from compare_bf16 import ROOT, digest, require
from plan_comparison_archive import excluded_reason, plan, sanitized_bytes

DESTINATION = Path(
    "/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/model_scaling/qwen3.5_bf16"
)


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_trace(trace):
    capture = load(trace / "capture/metadata.json")
    require(
        capture["complete"] and capture["cleanup"]["model_exited"], "Trace still active"
    )
    provenance = load(trace / "analysis/provenance.json")
    require(provenance["complete"], "Trace analysis incomplete")
    proof = load(trace / "analysis/kernel_proof.json")
    require(proof["complete"], "Kernel proof incomplete")
    require(
        {row["name"] for row in proof["phases"]} == {"profile_c1", "profile_c8"}
        and all(row["graph_kernel_launches"] > 0 for row in proof["phases"]),
        "Expected graph execution not established",
    )
    for directory, name in (
        ("capture", "manifest.json"),
        ("analysis", "manifest.json"),
    ):
        for relative, record in load(trace / directory / name).items():
            expected = record["sha256"] if isinstance(record, dict) else record
            require(
                digest(trace / directory / relative) == expected,
                f"Trace changed: {relative}",
            )
    proof_manifest = load(trace / "analysis/kernel_proof.manifest.json")
    for relative, key in (
        ("analysis/kernel_proof.json", "kernel_proof.json"),
        ("analysis/manifest.json", "analysis_manifest_sha256"),
        ("prove_kernels.py", "proof_analyzer_sha256"),
    ):
        require(
            digest(trace / relative) == proof_manifest[key],
            f"Proof changed: {relative}",
        )
    return proof


def record(path, relative):
    payload = sanitized_bytes(path)
    return {
        "source": str(path),
        "relative": str(relative),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_bytes": len(payload),
        "sanitized": payload != path.read_bytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    require(args.execute, "Pass --execute only after all trace work has completed")
    require(not args.destination.exists(), "Refusing to overwrite archive")
    trace = ROOT / "kernel_trace"
    proof = validate_trace(trace)
    archive = plan(
        ROOT,
        ROOT / "mistralrs_bf16",
        ROOT / "vllm_bf16_retry1",
        [ROOT / "vllm_bf16"],
    )
    for name in ("comparison.json", "comparison.md", "presented_text.json"):
        archive["included"].append(record(ROOT / "comparison" / name, name))
    archive["included"].append(record(Path(__file__), "scripts/archive_comparison.py"))
    archive["included"].append(
        record(ROOT / "compare_presented_text.py", "scripts/compare_presented_text.py")
    )
    for path in sorted(trace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(trace)
        reason = excluded_reason(path, relative)
        destination = Path("kernel_trace") / relative
        if reason:
            archive["external_artifacts"].append(
                {
                    "source": str(path),
                    "relative": str(destination),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                    "reason": reason,
                }
            )
        else:
            archive["included"].append(record(path, destination))
    archive["copy_bytes"] = sum(row["archive_bytes"] for row in archive["included"])
    archive["execution"] = (
        "Completed copy; source runs and profiler reports are unchanged"
    )
    archive["kernel_proof"] = {
        row["name"]: row["graph_kernel_launches"] for row in proof["phases"]
    }
    staging = args.destination.with_name(args.destination.name + ".partial")
    require(not staging.exists(), "Previous partial archive exists")
    staging.mkdir(parents=True)
    for row in archive["included"]:
        source = Path(row["source"])
        require(
            digest(source) == row["sha256"], f"Source changed during copy: {source}"
        )
        target = staging / row["relative"]
        require(not target.exists(), f"Repeated archive path: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitized_bytes(source)
        require(
            hashlib.sha256(payload).hexdigest() == row["archive_sha256"],
            "Redaction drift",
        )
        target.write_bytes(payload)
    save(staging / "archive_provenance.json", archive)
    (staging / "README.md").write_text(
        "# Same-checkpoint Qwen3.5 BF16 engine comparison\n\n"
        "The comparison covers completed target-only mistral.rs and vLLM runs on one GB10, "
        "with the same local Qwen3.5-35B-A3B BF16 checkpoint and frozen request harness. "
        "Five measured trials follow two excluded warmups at C1, C6, and C8. "
        "See [comparison.md](comparison.md) and [comparison.json](comparison.json) for "
        "recomputed statistics, exact identities, protocol validation, operating ranges, "
        "and limitations.\n\n"
        "The separate kernel_trace subtree establishes observed native cuTile MoE graph "
        "execution at C1 and C8. Its instrumented timings do not replace the unprofiled "
        "throughput results. Capture and analysis manifests retain the original hashes; "
        "kernel_proof.manifest.json additionally covers the later proof.\n\n"
        "The first vLLM attempt failed during a port preflight before Docker or GPU launch. "
        "It is preserved under failed_attempts and excluded from statistics. The successful "
        "vLLM source directory is vllm_bf16_retry1.\n\n"
        "Tokenizer copies, compiler caches, large profiler reports, SQLite databases, "
        "model weights, and executable binaries remain external. archive_provenance.json "
        "records source paths and hashes for excluded run/trace artifacts; model, binary, "
        "and image identities also remain in the run metadata. No model weights were "
        "copied or rehashed by this archive step. Potential secret fields are redacted "
        "from copies with original and sanitized digests retained.\n\n"
        "SHA256SUMS.json covers every archived file except itself. Source manifests refer "
        "to original run layouts, including externally retained files.\n"
    )
    manifest = {
        str(path.relative_to(staging)): digest(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    save(staging / "SHA256SUMS.json", manifest)
    require(
        all(
            digest(staging / relative) == expected
            for relative, expected in manifest.items()
        ),
        "Archive verification failed",
    )
    shutil.move(staging, args.destination)
    result = {
        "complete": True,
        "destination": str(args.destination),
        "files": len(manifest),
        "bytes": sum(
            path.stat().st_size
            for path in args.destination.rglob("*")
            if path.is_file()
        ),
        "manifest_sha256": digest(args.destination / "SHA256SUMS.json"),
        "external_files": len(archive["external_artifacts"]),
    }
    save(ROOT / "comparison/archive_result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
