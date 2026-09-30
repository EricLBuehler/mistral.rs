#!/usr/bin/env python3
"""Archive the completed dense backend control without model weights or binaries."""

import datetime
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE = Path(__file__).parent
REPO = Path("/home/ericbuehler/mistral.rs")
DESTINATION = (
    REPO / "benchmarks/qwen3.8-flash-next/raw/scaling_investigation/backend_comparison"
)
HARNESS = REPO / "benchmarks/qwen3.8-flash-next"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    for arm in ("dense_fp8", "dense_fp8_to_q4k"):
        metadata = read(SOURCE / arm / "metadata.json")
        assert metadata["complete"] and "server_exit_code" in metadata
        assert not Path(f"/proc/{metadata['server_pid']}").exists()
    comparison = read(SOURCE / "comparison.json")
    assert comparison["complete"]
    inventory = read(SOURCE / "checkpoint_inventory.json")["models"]["Qwen3.8-27B-FP8"]
    snapshot = Path(inventory["snapshot"])
    mapping = {}
    for arm in ("dense_fp8", "dense_fp8_to_q4k"):
        for name in ("raw.json", "summary.json", "metadata.json", "server.log"):
            mapping[f"{arm}/{name}"] = SOURCE / arm / name
    for name in (
        "comparison.json",
        "comparison.txt",
        "run_dense_backend_control.py",
        "compare_backend_results.py",
        "checkpoint_inventory.json",
        "inventory_local_checkpoints.py",
        "methodology.md",
        "startup_verification.json",
        "paused_editor_processes.json",
        "paused_editor_processes_after_dense.json",
        "package_backend_results.py",
        "dense_fp8.controller.log",
        "dense_fp8_to_q4k.controller.log",
    ):
        mapping[name] = SOURCE / name
    for name in (
        "bench_concurrency.py",
        "bench_serving.py",
        "summarize_concurrency.py",
    ):
        mapping[f"harness/{name}"] = HARNESS / name
    mapping["canonical_prompts.jsonl"] = REPO / "releases/v0.9.3/raw/prompts.jsonl"
    for name in ("config.json", "model.safetensors.index.json"):
        mapping[f"checkpoint_metadata/{name}"] = snapshot / name
    for source in SOURCE.glob("*stopped*.json"):
        mapping[source.name] = source
    copied = []
    for name, source in mapping.items():
        target = DESTINATION / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hash = digest(source)
        assert digest(target) == source_hash
        copied.append(
            {
                "file": name,
                "source": str(source),
                "bytes": target.stat().st_size,
                "sha256": source_hash,
            }
        )
    write(
        DESTINATION / "provenance.json",
        {
            "packaged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_directory": str(SOURCE),
            "files_copied_and_byte_verified": copied,
            "excluded": [
                "Checkpoint tensor payloads",
                "Executable binaries",
                "Tokenizer payload (pinned fingerprint retained)",
            ],
            "process_isolation": {
                "editor_pause_record": "paused_editor_processes.json",
                "post_pair_stopped_state": "paused_editor_processes_after_dense.json",
                "root_observations": "The orchestrator separately observed the identified editor process in stopped T/Tl state before the pair and in ps snapshots around 13:25 UTC and 13:35 UTC, with no compiler observed. All three agents held builds and GPU work throughout the pair.",
                "initial_state_note": "Immediate R states in the pause-request record reflect signal-delivery timing, not evidence that the pause failed.",
                "limits": "No continuous system-wide process, GPU utilization, or paging monitor was captured for this pair. Snapshot observations and coordinated agents do not prove total machine isolation or absence of timing interference.",
            },
        },
    )
    (DESTINATION / "README.md").write_text("""# Dense FP8 versus FP8-to-Q4K control

Both arms use the same pinned Qwen3.8-27B-FP8 checkpoint, executable, GB10,
BF16 KV cache, eight-sequence capacity, and greedy request settings. Only the
Q4K arm adds `--isq q4k`. Each C1/C8 case has one warmup and three measured
trials of eight requests with 128 output tokens. C8 is one finite wave.

[comparison.json](comparison.json) independently validates all saved responses,
raw timestamps, counts, settings, source hashes, and launch provenance, then
recomputes the rates. Full responses and logs are preserved in
[dense_fp8](dense_fp8/raw.json) and [dense_fp8_to_q4k](dense_fp8_to_q4k/raw.json).
[startup_verification.json](startup_verification.json) distinguishes log facts
from source-derived backend dispatch; packed affine Marlin is disabled in both
production environments. [methodology.md](methodology.md) preserves the design.

Q4K first dequantizes the FP8 weights and requantizes eligible layers. It also
changes eligible lm_head precision and stored projection layout. Separate Q4K
gate/up tensors can still use fused MMVQ computation. This is a full execution
path comparison, with no claim of equivalent quality or isolated kernel cost.

The identified editor processes were paused and the agents held builds/GPU
work during measurement. [provenance.json](provenance.json) records the
orchestrator's stopped-state observations, a [post-pair snapshot](paused_editor_processes_after_dense.json) confirming all three process identities still stopped, and the isolation limits; the initial
pause record's immediate R state is not a failed-pause determination. No
continuous machine-wide process or paging trace was collected for this pair.

To reproduce validation without running inference, install `tokenizers` and
run from this directory:

```bash
python3 compare_backend_results.py --snapshot /path/to/pinned-snapshot --output /tmp/backend-comparison.json
```

The snapshot directory needs only `config.json`, `tokenizer.json`, and
`model.safetensors.index.json` from revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; hashes are checked. Config and index
are included under `checkpoint_metadata/`, while the tokenizer payload is
omitted. A matching existing cache is the default snapshot path. The validator
uses the archived `harness/` and canonical prompts, not current repository
sources. Tensor shards and the executable are never read during validation;
launch-time executable hashes are checked for agreement.

The frozen controller retains its original absolute source/cache paths as
measured. Its [metadata](dense_fp8/metadata.json) and the
[Q4K metadata](dense_fp8_to_q4k/metadata.json) preserve exact launch commands.
[manifest.json](manifest.json) and `SHA256SUMS` cover every archived file except
the two manifests themselves. [archive_validation.json](archive_validation.json)
records a successful rerun using the archived validator and shared helpers.
""")
    with tempfile.TemporaryDirectory(prefix="backend-archive-validation-") as temporary:
        output = Path(temporary) / "comparison.json"
        command = [
            "python3",
            str(DESTINATION / "compare_backend_results.py"),
            "--snapshot",
            str(snapshot),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        recomputed = read(output)
        for name in ("fp8", "q4k"):
            for key in (
                "metrics_by_concurrency",
                "c8_over_c1_aggregate",
                "c8_over_c1_per_active_request",
            ):
                assert recomputed["arms"][name][key] == comparison["arms"][name][key]
        for key in (
            "q4k_over_fp8",
            "q4k_over_fp8_c8_c1_scaling_ratio",
            "output_comparisons",
        ):
            assert recomputed[key] == comparison[key]
        assert (
            recomputed["validation"]["source_hashes"]
            == comparison["validation"]["source_hashes"]
        )
        write(
            DESTINATION / "archive_validation.json",
            {
                "complete": True,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "all_metrics_and_ratios_exactly_reproduced": True,
                "archived_source_hashes_match_original_validation": True,
                "scope": "Archived raw data and helper scripts; only pinned tokenizer/config/index read outside archive. No model tensor payloads, binary hashes, or GPU/server operations.",
            },
        )
    files = [
        path
        for path in sorted(DESTINATION.rglob("*"))
        if path.is_file()
        and path not in (DESTINATION / "manifest.json", DESTINATION / "SHA256SUMS")
        and "__pycache__" not in path.parts
    ]
    for pycache in DESTINATION.rglob("__pycache__"):
        shutil.rmtree(pycache)
    manifest = [
        {
            "file": str(path.relative_to(DESTINATION)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]
    write(
        DESTINATION / "manifest.json",
        {
            "files": manifest,
            "scope": "All archived regular files except manifest.json and SHA256SUMS.",
        },
    )
    (DESTINATION / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in manifest)
    )
    for entry in manifest:
        assert digest(DESTINATION / entry["file"]) == entry["sha256"]
    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "files_verified": len(manifest),
                "total_bytes": sum(entry["bytes"] for entry in manifest),
                "archive_reproduction": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
