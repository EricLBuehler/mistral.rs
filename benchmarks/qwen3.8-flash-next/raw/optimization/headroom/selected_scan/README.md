# Selected-weight scan evidence

Read [report.md](report.md) for the fixed 8 KiB comparison and all chunk sensitivities. [summary.json](summary.json) contains per-capture paired FFN comparisons, all scan sizes, drift, and the retained cross-kernel numerical diagnostics.

This package contains text evidence only. The original compressed weights and compiled probe remain at their external paths; their SHA256 identities are recorded in `timed/metadata.json` and `build.metadata.json`. Capture paths, hashes, exact selected expert IDs, GGUF offsets, launch grids, nominal byte counts, and source identities are recorded in the timed metadata and results.

The scan uses 256 threads per block and coalesced 16-byte loads, with four observable XOR checksums per block. CPU checksums independently validate every selected byte range. `memcheck.log` records zero errors for four chunk sizes and three changing routes on shared allocations. The timed suite checks all 312 capture/projection/chunk combinations. `sass_evidence.json` records the offline disassembly command and surviving vector loads/stores; `resource_usage.txt` records resource use.

Each CUDA-event measurement runs entirely within the C++ loop. The primary warmed scan has seven rounds of ten repeats. The separate 96 MiB flush control runs before each scan event, outside the event window; it is an eviction attempt, not proof of cold DRAM. Repeated scans revisit one layer. Checksum work, different launch counts, access patterns, and cache histories prevent interpreting this control as a physical lower bound or model-wide throughput ceiling. Byte rates are logical selected-payload rates, not measured DRAM bandwidth.

`ffn_before/` and `ffn_after/` contain the complete 71-case frozen replay runs bracketing the scan. The comparison uses the current native source path: GEMV for M1/M7, MMQ for M42/M56. All 26 native source-path guards pass in both runs. Six pre-existing GEMV/MMQ numerical diagnostic failures remain visible; no alternate output is silently substituted. The fixed-Q7 layer-8 captures do not represent the current adaptive-depth whole-model workload.

Exact build, validation, timing, and FFN commands are recorded in the respective metadata. `package_provenance.json` maps each copied artifact back to its source. `SHA256.json` hashes every other packaged file; compiled probes and model weights are deliberately excluded.
