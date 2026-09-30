"""Validate and archive a completed adaptive graph-policy run without GPU access."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import compare_stage

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "adaptive_run"
ANALYSIS = ROOT / "adaptive_run.analysis"
EXPECTED_BINARY = "92c050e15bbfe15fc99ef11077366a8fb40db5af96223c67e0f9299fce23f2fa"
ARCHIVE = Path(
    "/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/model_scaling/graph_candidate"
)
HOMOGENEOUS_PHASES = tuple(
    f"homogeneous_{prompt}_c{concurrency}"
    for prompt in ("python", "math")
    for concurrency in (1, 8)
)
MAX_COPY_BYTES = 2 * 1024**2
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".txt",
    ".csv",
    ".log",
    ".py",
    ".md",
    ".diff",
    ".patch",
}


def load(path):
    return compare_stage.validator.load(path)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_run():
    metadata = load(RUN / "metadata.json")
    require(
        metadata["complete"] and not metadata.get("error"),
        "Wait for complete measurements",
    )
    require(not metadata["monitor_errors"], "Runtime monitor failed")
    require(
        metadata["server_shutdown"]["forced_kill"] is False
        and metadata["server_shutdown"]["returncode"] in (0, -15),
        "Wait for clean shutdown",
    )
    require(
        metadata["binary_sha256"] == metadata["binary_sha256_after"] == EXPECTED_BINARY,
        "Unexpected binary identity",
    )
    expected = {"serving", "text", "c1", "c6_c8", "mixed_context", *HOMOGENEOUS_PHASES}
    require(len(metadata["phases"]) == len(expected), "Unexpected phase count")
    names = [row["name"] for row in metadata["phases"]]
    require(len(set(names)) == len(names), "Duplicate phase names")
    phases = {row["name"]: row for row in metadata["phases"]}
    require(set(phases) == expected, "Unexpected phase set")
    require(
        all(row["complete"] and row["returncode"] == 0 for row in phases.values()),
        "Incomplete phase",
    )
    compare_stage.validator.verify_manifest(RUN, load(RUN / "SHA256SUMS.json"))
    return metadata


def run(command, log):
    with log.open("w") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600
        )


def analyze():
    completed_run()
    require(not ANALYSIS.exists(), "Refusing to overwrite analysis")
    ANALYSIS.mkdir()
    commands = []
    command = [
        sys.executable,
        str(ROOT / "compare_stage.py"),
        str(RUN),
        "--expected-binary-sha256",
        EXPECTED_BINARY,
        "--output",
        str(ANALYSIS / "stage_comparison"),
    ]
    run(command, ANALYSIS / "stage_comparison.log")
    commands.append(command)
    command = [
        sys.executable,
        str(RUN / "scripts/bench_homogeneous.py"),
        "summarize",
        *(str(RUN / f"{name}.json") for name in HOMOGENEOUS_PHASES),
        "--output",
        str(ANALYSIS / "homogeneous.summary.json"),
    ]
    run(command, ANALYSIS / "homogeneous.summary.txt")
    commands.append(command)
    require(
        load(ANALYSIS / "homogeneous.summary.json")
        == load(RUN / "homogeneous.summary.json"),
        "Frozen homogeneous summary differs from independent rerun",
    )
    windows = {}
    for name in HOMOGENEOUS_PHASES:
        counters = [
            (RUN / f"{name}.metrics.{point}.txt").read_text()
            for point in ("before", "after")
        ]
        memory = [
            load(RUN / f"{name}.memory.{point}.json") for point in ("before", "after")
        ]
        require(
            memory[0]["page_size_bytes"] == memory[1]["page_size_bytes"],
            "Page size changed",
        )
        windows[name] = {
            "counters": compare_stage.validator.counter_window(name, *counters),
            "memory": compare_stage.validator.memory_delta(
                *memory, memory[0]["page_size_bytes"]
            ),
        }
        windows[name]["counters"]["scope"] = (
            "This homogeneous prompt/concurrency command, including two warmups and three measured trials"
        )
        for point in ("before", "after"):
            require(
                not load(RUN / f"{name}.{point}.processes.json")["unexpected_active"],
                f"Process interference: {name}/{point}",
            )
    save(ANALYSIS / "homogeneous.windows.json", windows)
    release = load(ROOT / "prior_checkpoint_cache_release.json")
    require(
        len(release["files"]) == 14, "Unexpected prior checkpoint cache-release scope"
    )
    save(
        ANALYSIS / "completion.json",
        {
            "complete": True,
            "candidate_metadata_sha256": sha(RUN / "metadata.json"),
            "candidate_manifest_sha256": sha(RUN / "SHA256SUMS.json"),
            "binary_sha256": EXPECTED_BINARY,
            "before_binary_sha256": compare_stage.BEFORE_SHA,
            "commands": commands,
            "analyzer_sha256": sha(Path(__file__)),
            "prior_checkpoint_cache_release_sha256": sha(
                ROOT / "prior_checkpoint_cache_release.json"
            ),
            "limitations": [
                "Staged adaptive MTP comparison, not a randomized or fixed-depth-four causal A/B.",
                "Original serving/closed-loop phases use five measured trials after two warmups; homogeneous finite-wave phases use three after two.",
                "C6/C8 standard counters are combined and include warmups; homogeneous counters have separate prompt/concurrency command boundaries.",
                "Graph counts do not measure kernel-time coverage; global swap does not establish GPU paging.",
                "Same prompts or matching decoded text do not establish expert-route identity or actual weight reuse.",
                "Semantic review of saved chat and mixed-context smoke answers remains distinct from HTTP/finite-logprob checks.",
                "Clean page cache from the completed distinct Qwen3.5 checkpoint was released before startup; file contents were unchanged, and the action was outside timing.",
            ],
        },
    )
    save(
        ANALYSIS / "SHA256SUMS.json",
        {
            str(path.relative_to(ANALYSIS)): sha(path)
            for path in sorted(ANALYSIS.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps({"complete": True, "analysis": str(ANALYSIS)}))


def archive():
    completed_run()
    require(load(ANALYSIS / "completion.json")["complete"], "Analyze before archiving")
    compare_stage.validator.verify_manifest(
        ANALYSIS, load(ANALYSIS / "SHA256SUMS.json")
    )
    require(not ARCHIVE.exists(), "Refusing to overwrite archive")
    staging = ARCHIVE.with_name(ARCHIVE.name + ".partial")
    require(not staging.exists(), "Previous staging tree exists")
    included, external = [], []
    roots = [(RUN, Path("run")), (ANALYSIS, Path("analysis"))]
    support = [
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    ]
    roots += [
        (ROOT / "validation_helpers", Path("validation_helpers")),
        (ROOT / "harnesses", Path("harnesses")),
    ]
    pairs = [
        (path, prefix / path.relative_to(source))
        for source, prefix in roots
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    pairs += [(path, Path("scripts") / path.name) for path in sorted(support)]
    staging.mkdir(parents=True)
    for source, relative in pairs:
        row = {
            "source": str(source),
            "relative": str(relative),
            "bytes": source.stat().st_size,
            "sha256": sha(source),
        }
        excluded = (
            source.name in {"tokenizer.json", "tokenizer_config.json"}
            or "__pycache__" in relative.parts
            or source.suffix not in TEXT_SUFFIXES
            or row["bytes"] > MAX_COPY_BYTES
        )
        if excluded:
            row["reason"] = (
                "Tokenizer, cache, large or binary artifact retained externally"
            )
            external.append(row)
            continue
        target = staging / relative
        require(not target.exists(), f"Duplicate archive path: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(target) == row["sha256"], "Copy changed bytes")
        included.append(row)
    save(
        staging / "archive_provenance.json",
        {
            "complete": True,
            "included": included,
            "external_artifacts": external,
            "before_directory": str(compare_stage.DEFAULT_BEFORE),
            "before_binary_sha256": compare_stage.BEFORE_SHA,
            "before_repo_archive": "raw/optimization/final_serving/run",
            "weights_or_binaries_copied_or_rehashed": False,
        },
    )
    (staging / "README.md").write_text(
        "# Adaptive depth-four graph candidate\n\n"
        "The completed frozen candidate adds eligible B7/B8, Q5 target graphs. "
        "analysis/stage_comparison/comparison.json compares it directly with the prior "
        "d2c85e serving stage. Ancestor-validation reports are retained only for audit. "
        "The prior baseline archive remains unchanged.\n\n"
        "analysis/homogeneous.summary.json contains separate same-prompt finite-wave "
        "controls; they are not added to the ordinary serving rates. Command counters "
        "include warmups, and standard C6/C8 counters are combined. Raw responses, "
        "smokes, process/memory snapshots and commands remain under run/.\n\n"
        "scripts/prior_checkpoint_cache_release.json records release of clean cache "
        "from the completed different Qwen3.5 checkpoint before model loading. It did "
        "not change checkpoint contents and is outside timing.\n\n"
        "Tokenizer copies and other large artifacts remain external with hashes. "
        "Model weights and executables are not copied or rehashed here. The run's "
        "original manifests preserve original layouts; SHA256SUMS.json covers every "
        "archived file except itself.\n"
    )
    manifest = {
        str(path.relative_to(staging)): sha(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    save(staging / "SHA256SUMS.json", manifest)
    compare_stage.validator.verify_manifest(staging, manifest)
    shutil.move(staging, ARCHIVE)
    print(
        json.dumps(
            {
                "complete": True,
                "archive": str(ARCHIVE),
                "files": len(manifest),
                "manifest_sha256": sha(ARCHIVE / "SHA256SUMS.json"),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "archive"))
    args = parser.parse_args()
    analyze() if args.action == "analyze" else archive()


if __name__ == "__main__":
    main()
