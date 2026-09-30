# Nsight Compute: one captured 56-row expert input

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
