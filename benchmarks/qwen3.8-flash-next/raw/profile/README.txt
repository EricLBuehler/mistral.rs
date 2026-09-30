Qwen3.8 concurrency profiling evidence

These diagnostic captures used Nsight Systems 2026.1.3 software tracing on GB10.
Kernel timelines were recovered, with a generic "Not all CUDA events might have
been collected" warning. The reports have no dropped-incomplete-CUPTI count.
Treat GPU busy time as the union of available traced events. Kernel category
shares use summed kernel durations, not elapsed time. No GPU bandwidth counters
were collected, and these traces do not establish a bandwidth limitation.
Use the separate unprofiled benchmark results for performance claims.

Files copied from the original run are listed in archive_sources.json. Those
copies are byte-for-byte unchanged. SHA256SUMS covers every archived file except
itself; verify it from this directory with: sha256sum -c SHA256SUMS

profile_findings.txt gives the interpretation. profile_compact_summary.json
contains the kernel/API summaries. draft_sampling_transfer_evidence.json retains
the exact SQL query and baseline/new full-vocabulary D2H counts, bytes and transfer
durations. Transfer duration excludes CPU sampling and synchronization costs.
allocator_after_profile.prom is the post-profile allocator/dispatch snapshot.

profile_analysis_provenance.json retains the server command, binary/source hashes,
capture start metadata, export/analysis commands, and original report paths and
SHA256 hashes. Large .nsys-rep and .sqlite files are intentionally external:

  /home/ericbuehler/qwen4exp_work/batched_draft_20260929/
    mtp_concurrency.nsys-rep  (C1)
    profile_c8.nsys-rep      (C8)
    profile_c1.sqlite
    profile_c8.sqlite

The baseline exports used for D2H comparison remain under:
  /home/ericbuehler/qwen4exp_work/concurrency_20260929/

Window alignment is recorded in profile_c1.alignment.json and
profile_c8.alignment.json. Request offsets did not include an absolute trial
origin, so sustained windows discard the bounded origin uncertainty at each end.
Kernel-span summaries do not depend on request alignment.

Reproduce C8 analysis from the repository root, using the retained export:

  python3 benchmarks/qwen3.8-flash-next/raw/profile/analyze_nsys.py \
    /home/ericbuehler/qwen4exp_work/batched_draft_20260929/profile_c8.sqlite \
    --windows=benchmarks/qwen3.8-flash-next/raw/profile/profile_c8.windows.json \
    --output=/tmp/qwen38-profile-c8.analysis.json

Use the corresponding C1 filenames for C1. The preserved analyze_profiles.py
records the original gated export/orchestration procedure; it expects the
original run directory and is not needed to analyze already exported databases.
analyze_nsys_selftest.py checks interval union, clipping, categorization, graph
metadata, missing-kernel handling and GPU metric identity isolation without a GPU.

Reproduce before/after C8 transfer counts from the retained exports:

  python3 benchmarks/qwen3.8-flash-next/raw/profile/reproduce_d2h.py \
    --sqlite=/home/ericbuehler/qwen4exp_work/concurrency_20260929/profile_c8.sqlite \
    --sqlite=/home/ericbuehler/qwen4exp_work/batched_draft_20260929/profile_c8.sqlite \
    --output=/tmp/qwen38-draft-d2h.json

Both reproduction tools read SQLite in read-only mode and write only their
requested output. Their database inputs can be copied elsewhere and passed using
different paths. The original external evidence remains unchanged.
