import csv
import hashlib
import io
import json
import shutil
from pathlib import Path


SOURCE = Path(__file__).parent / "ncu_native56"
TARGET = Path(
    "/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/real_routing/ncu_native56"
)
NAMES = [
    "gemv.log",
    "gemv.ncu-rep",
    "gemv.profiled_test_output_not_benchmark.json",
    "gemv_raw.log",
    "gemv_raw_corrected.log",
    "gemv_raw_final.log",
    "kernels.json",
    "metadata.first_import_failure.json",
    "metadata.json",
    "mmq.log",
    "mmq.ncu-rep",
    "mmq.profiled_test_output_not_benchmark.json",
    "mmq_raw.log",
    "profile_selected_replay.initial.py",
    "profile_selected_replay.measured.py",
    "resume_selected_profile.py",
    "version.log",
]
METRICS = [
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_max_registers_per_multiprocessor",
    "device__attribute_max_warps_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "profiler__replayer_passes",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    kernels = json.loads((SOURCE / "kernels.json").read_text())
    assert metadata["complete"] and not metadata["dram_counters_available"]
    TARGET.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in NAMES:
        source = SOURCE / name
        assert source.stat().st_size < 1_000_000
        target = TARGET / name
        shutil.copyfile(source, target)
        sha = digest(source)
        assert digest(target) == sha
        copied.append(
            {
                "source": str(source),
                "path": name,
                "sha256": sha,
                "bytes": source.stat().st_size,
            }
        )
    assert (
        digest(TARGET / "profile_selected_replay.initial.py")
        == metadata["script_sha256"]
    )
    assert (
        digest(TARGET / "profile_selected_replay.measured.py")
        == metadata["current_reproduction_script_sha256"]
    )
    assert (
        digest(TARGET / "resume_selected_profile.py")
        == metadata["resumption"]["script_sha256"]
    )
    for command in metadata["commands"]:
        path = TARGET / (command["name"] + ".log")
        if "log_sha256" in command and path.exists():
            assert digest(path) == command["log_sha256"]
    rows = []
    for backend, name in [("gemv", "gemv_raw_final.log"), ("mmq", "mmq_raw.log")]:
        content = (TARGET / name).read_text()
        data = list(csv.DictReader(io.StringIO(content[content.index('"ID",') :])))[1:]
        recorded = next(
            item["kernels"] for item in kernels["results"] if item["backend"] == backend
        )
        assert len(data) == len(recorded)
        for row, kernel in zip(data, recorded, strict=True):
            assert row["Kernel Name"] == kernel["kernel"]
            for metric, value in kernel["metrics"].items():
                assert row[metric] == value["value"]
            rows.append(
                {
                    "backend": backend,
                    "kernel": row["Kernel Name"],
                    "source_csv": name,
                    "launch_id": int(row["ID"]),
                    "metrics": {key: float(row[key]) for key in METRICS},
                }
            )
    write_json(
        TARGET / "resources.summary.json",
        {
            "scope": metadata["scope"],
            "sample": Path(metadata["sample"]["path"]).name,
            "sample_sha256": metadata["sample"]["sha256"],
            "dram_counters_available": False,
            "rows": rows,
            "validation": "All recorded kernel metrics agree exactly with the raw CSV values.",
            "limits": metadata["limits"],
        },
    )
    query = next(
        command
        for command in metadata["commands"]
        if command["name"] == "query_metrics"
    )
    write_json(
        TARGET / "provenance.json",
        {
            "source_directory": str(SOURCE),
            "copied_files": copied,
            "external_artifacts": [
                {
                    "path": str(SOURCE / "query_metrics.log"),
                    "bytes": (SOURCE / "query_metrics.log").stat().st_size,
                    "sha256": query["log_sha256"],
                    "hash_scope": "Previously recorded by the measured runner; not rehashed while GPU experiments run.",
                    "reason_omitted": "Large metric-availability listing; selected available and unavailable names remain in metadata.json.",
                },
                {
                    **metadata["binary"],
                    "reason_omitted": "Diagnostic executable; source and build provenance are archived in the parent replay package.",
                },
            ],
            "manifest_scope": "Fresh hashes of this whitelist and generated summaries; the older external SHA256SUMS was not used as the package manifest.",
            "report_sizes": "Both native NCU reports are small (less than 61 KB each) and included for offline inspection.",
        },
    )
    shutil.copyfile(__file__, TARGET / "package_ncu_evidence.py")
    (
        TARGET / "README.md"
    ).write_text("""# Nsight Compute: one captured 56-row expert input

This is a diagnostic profile of five projection launches, using the native B8xQ7 layer-8 sample with 213 selected experts, the median selected-expert count among the five captured C8 samples. It is separate from both the final serving benchmark and the earlier Nsight Systems full-model traces.

| Projection path | Registers/thread | Threads/block | Register-limited blocks/SM | Achieved warp occupancy |
| --- | ---: | ---: | ---: | ---: |
| GEMV fused gate/up | 40 | 128 | 12 | 89.61% |
| GEMV down | 40 | 128 | 12 | 97.82% |
| Grouped MMQ gate | 254 | 256 | 1 | 16.78% |
| Grouped MMQ up | 254 | 256 | 1 | 16.75% |
| Grouped MMQ down | 254 | 256 | 1 | 16.98% |

The grouped 64-column kernels allocate 256 registers per thread after allocation rounding, using all 65,536 registers available per SM for one block. The raw launch data also reports a one-block shared-memory limit for this configuration. Eight resident warps out of the device's maximum 48 give a theoretical occupancy of 16.67%. This identifies concrete resource constraints; higher occupancy alone does not establish a faster implementation. The unprofiled actual-route replay remains the timing evidence, and grouped MMQ wins its native 42/56-row comparisons despite GEMV's higher occupancy.

[resources.summary.json](resources.summary.json) preserves the additional launch-resource fields from the raw CSV; every metric in [kernels.json](kernels.json) was checked against that CSV. [metadata.json](metadata.json) records Nsight Compute 2026.2.1, exact commands, sample and executable hashes, and stopped-editor observations before and after capture.

The requested DRAM read/write counters were unavailable on this device. L2 counters do not establish off-chip bandwidth or a physical throughput ceiling. The profiler serializes launches and performs 10 replay passes, with cache control and clock control set to `none`; replay can still alter cache state. Profiled durations and the files named `profiled_test_output_not_benchmark` are not serving or unprofiled replay measurements. This covers one input at one layer and omits the shared expert and all other model operations.

The successful GEMV capture survived two offline CSV-import errors. Their logs and initial metadata remain alongside corrected CSV output, the exact initial script, the corrected reproduction script, and the resumption script. Those scripts retain original external paths and require the archived diagnostic executable and external weight file described by the parent [replay package](../README.md). Both small `.ncu-rep` reports are included for offline inspection. The 93 MB metric-availability listing is omitted; its original path and previously recorded hash are in [provenance.json](provenance.json). [SHA256SUMS](SHA256SUMS) independently hashes every file in this subtree except itself.
""")
    paths = sorted(
        path
        for path in TARGET.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (TARGET / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in paths)
    )
    print(
        json.dumps(
            {
                "files_hashed": len(paths),
                "total_files": len(paths) + 1,
                "bytes": sum(path.stat().st_size for path in TARGET.iterdir()),
                "target": str(TARGET),
            }
        )
    )


if __name__ == "__main__":
    main()
