"""Archive compact current-profile evidence only after completed capture and analysis."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DESTINATION = Path(
    "/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/model_scaling/current_profile"
)
EXCLUDED_SUFFIXES = {".sqlite", ".nsys-rep"}
MAX_COPIED_FILE_BYTES = 2 * 1024**2


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected_hash(value):
    return value["sha256"] if isinstance(value, dict) else value


def selection(root):
    pairs = []
    external = []
    for directory, target in (
        ("current_capture", "capture"),
        ("analysis", "analysis"),
        ("current_analysis/results", "semantic"),
    ):
        source = root / directory
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            omit = (
                path.suffix in EXCLUDED_SUFFIXES
                or path.name.endswith(".analysis.json")
                or path.name == "semantic_breakdown.json"
            )
            if omit:
                external.append(path)
            else:
                pairs.append((path, Path(target) / path.name))
    for name in (
        "run_capture.py",
        "profile_requests.py",
        "clock_anchor.py",
        "exact_timestamps.patch",
        "analyze_capture.py",
        "self_check.json",
        "README.md",
    ):
        pairs.append((root / name, Path("scripts") / name))
    for directory in ("dependencies", "current_analysis"):
        for path in sorted((root / directory).iterdir()):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                pairs.append((path, Path("scripts") / directory / path.name))
    return pairs, external


def verify_source(root):
    run = root / "current_capture"
    metadata = load(run / "metadata.json")
    assert metadata["complete"] and metadata["cleanup"]["model_exited"]
    process = metadata["model_process"]
    stat = Path("/proc") / str(process["pid"]) / "stat"
    if stat.exists():
        fields = stat.read_text().rsplit(") ", 1)[1].split()
        assert int(fields[19]) != process["starttime_ticks"] or fields[0] in ("Z", "X")
    assert load(root / "analysis/provenance.json")["complete"]
    assert load(root / "current_analysis/results/provenance.json")["complete"]
    for name in ("interpretation.json", "interpretation.md"):
        assert (root / "current_analysis/results" / name).is_file(), (
            f"Wait for semantic report: {name}"
        )
    for directory in ("current_capture", "analysis", "current_analysis/results"):
        folder = root / directory
        for name, row in load(folder / "manifest.json").items():
            assert sha(folder / name) == expected_hash(row), (
                f"Changed source: {directory}/{name}"
            )
    assert sha(root / "run_capture.py") == metadata["runner_sha256"]
    for name, digest in metadata["support_sha256"].items():
        assert sha(root / name) == digest
    for name, row in metadata["dependencies"].items():
        assert sha(root / "dependencies" / name) == row["sha256"]
    semantic = load(root / "current_analysis/results/provenance.json")
    assert semantic["run_metadata_sha256"] == sha(run / "metadata.json")
    assert semantic["export_provenance_sha256"] == sha(
        root / "analysis/provenance.json"
    )
    for row in semantic["scripts"]:
        assert sha(Path(row["path"])) == row["sha256"]
    return metadata


def archive(args):
    root, destination = args.source.resolve(), args.output.resolve()
    pairs, omitted = selection(root)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "output": str(destination),
                    "copy_files": len(pairs) + 4,
                    "copy_bytes": sum(path.stat().st_size for path, _ in pairs),
                    "external_only": [str(path) for path in omitted],
                    "gate": "Capture/server cleanup, export and semantic analysis complete; all source manifests verified; interpretation files present.",
                },
                indent=2,
            )
        )
        return
    assert not destination.exists(), "Refusing to overwrite an archive"
    metadata = verify_source(root)
    command = metadata["plan"]["launcher_command"]
    checkpoint = Path(command[command.index("-m") + 1])
    config = checkpoint / "config.json"
    assert sha(config) == metadata["checkpoint_metadata_sha256"]["config.json"]
    pairs.append((config, Path("checkpoint/config.json")))
    pairs.append((Path(__file__).resolve(), Path("scripts/archive_current_profile.py")))
    destination.mkdir(parents=True)
    provenance = []
    for source, relative in pairs:
        assert source.stat().st_size <= MAX_COPIED_FILE_BYTES, (
            f"Unexpected large copy: {source}"
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha(source)
        assert sha(target) == digest
        provenance.append(
            {
                "source": str(source),
                "copy": str(relative),
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        )
    large = [
        {
            "external_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "reason": "Large trace/database or detailed kernel list; compact derived evidence is retained.",
        }
        for path in omitted
    ]
    save(destination / "external_artifacts.json", large)
    save(
        destination / "archive_provenance.json",
        {
            "complete": True,
            "source_root": str(root),
            "binary_sha256": metadata["plan"]["binary_sha256"],
            "files": provenance,
            "limits": [
                "No weights, frozen binaries, profiler reports or SQLite databases are copied.",
                "Original source manifests include intentionally omitted large files; SHA256SUMS.json covers only this compact archive.",
                "Historical trace analysis is not included. Current semantic-classifier source provenance identifies reused source definitions separately.",
                "Recomputing kernel attribution requires the hash-identified external SQLite/report artifacts; requests and counters can be checked from this archive alone.",
            ],
        },
    )
    (destination / "README.md").write_text("""# Current whole-model profile

This is the frozen masked-MMQ/small-group adaptive build, before the proposed graph
policy change. It is an instrumented diagnostic, separate from official serving
benchmarks and any later candidate run.

- [Interpretation](semantic/interpretation.md) and [compact measured summary](semantic/compact_summary.json).
- [Same-server capture-off versus recorded phases](analysis/capture_off_vs_recorded.json).
- [C1 alignment and profiler warnings](analysis/profile_c1.alignment.json) and [C8 alignment and warnings](analysis/profile_c8.alignment.json).
- [Exact request/counter commands, startup and cleanup provenance](capture/metadata.json), [server log](capture/server.log), and raw request/metrics files in `capture/`.
- [Semantic source/method](scripts/current_analysis/README.md), [capture method](scripts/README.md), and exact scripts under `scripts/`.
- [Omitted trace/database paths, sizes and hashes](external_artifacts.json), [archive provenance](archive_provenance.json), and [local manifest](SHA256SUMS.json).

Each recorded phase has one warmed closed-loop trial: C1 has eight requests and
1024 outputs; C8 has 24 requests and 3072 outputs. Rates and per-output kernel
costs include prefill, request startup and drain. Warmup recording is disabled but
profiler instrumentation remains attached. Similar warmup/recorded elapsed times
are not a pure instrumentation-overhead estimate: adaptive depth, output text,
cache state and routes can differ. Full-phase denominators are used only with the
whole request envelope, not the completion-boundary interior.

Nsight warns that some CUDA/OS runtime events might be missing; it provides no
NVTX events, CPU scheduling trace or unified-memory trace here. The analysis is
of recorded kernel activity, not guaranteed complete hardware time, DRAM traffic,
or a hardware ceiling. Exact compact source artifacts are preserved; reproducing
the full SQLite analysis needs the external files listed by hash. No historical
trace output has been substituted for this current capture.
""")
    manifest = {
        str(path.relative_to(destination)): sha(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    save(destination / "SHA256SUMS.json", manifest)
    for name, digest in manifest.items():
        assert sha(destination / name) == digest
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(destination),
                "files": len(manifest),
                "bytes": sum((destination / name).stat().st_size for name in manifest),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--execute", action="store_true")
    archive(parser.parse_args())


if __name__ == "__main__":
    main()
