# Selected-weight scan control

The scan reads the exact compressed weights of every expert selected by each native layer-8 capture. It performs no model computation. Four XOR checksums per block keep all vector loads observable; CPU per-expert checksums matched for all 312 projection/chunk combinations. Focused sanitizer validation passed with zero errors.

The table uses the same 8 KiB chunk configuration for every capture. All four tested chunk sizes remain in the raw results. FFN values are the midpoint of the before/after per-capture medians, then the median across captures. The scan-to-FFN ratios are paired per capture.

| Native rows | Captures | Median selected experts | Selected MiB | Warm scan ms | >L2 flush control ms | FFN ms | Paired FFN/scan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 10 | 27.34 | 0.0958 | 0.1867 | 0.1120 | 1.166x |
| 7 | 8 | 31 | 84.77 | 0.3679 | 0.4322 | 0.5477 | 1.505x |
| 42 | 5 | 164 | 448.44 | 1.9053 | 1.9755 | 2.5407 | 1.359x |
| 56 | 5 | 213 | 582.42 | 2.4657 | 2.5585 | 3.4027 | 1.380x |

The M56 bracketing FFN measurements differed by -1.37% to +0.63% per capture. M42 differed by -0.96% to +3.49%; the much shorter M1 cases varied more (-23.29% to +12.38%). Treat the small-case ratios cautiously.

Both 71-case FFN replays retain six pre-existing GEMV-versus-MMQ numerical diagnostic failures: four native M1/M7 cases and two derived M8 cases. All 26 native source-path guards pass in each replay. No alternate MMQ output is substituted for the current M1/M7 GEMV default in this comparison. The scan has no model output.

The device reported 24 MiB of L2. The separate 96 MiB flush writes finish before each scan event window; this is an eviction attempt, not proof of cold DRAM. Repeated same-layer scans and FFNs have different access patterns, instruction costs, launch counts, and cache histories. The checksum adds work. Logical payload divided by event time must not be labeled measured DRAM bandwidth.

The selected payload is 2,867,200 bytes per expert: Q4K gate and up each use 921,600 bytes, and Q4_1 down uses 1,024,000. Output-feature tiles partition these weight rows. For the captured native M42/M56 cases, the current MMQ query tile covers each active expert in one tile; reducing query width can introduce additional reads of that expert.

These fixed-Q7, single-layer diagnostic captures do not describe every layer or the current adaptive-depth serving distribution. The remaining scan/FFN gap is an empirical comparison, not a guaranteed optimization budget or a model-wide ceiling.

## Chunk sensitivity

| Native rows | Chunk KiB | Warm total ms | >L2 flush control total ms |
| --- | --- | --- | --- |
| 1 | 8 | 0.0958 | 0.1867 |
| 1 | 32 | 0.0945 | 0.1922 |
| 1 | 64 | 0.0938 | 0.1920 |
| 1 | 256 | 0.1211 | 0.1974 |
| 7 | 8 | 0.3679 | 0.4322 |
| 7 | 32 | 0.3731 | 0.4398 |
| 7 | 64 | 0.3770 | 0.4428 |
| 7 | 256 | 0.3794 | 0.4454 |
| 42 | 8 | 1.9053 | 1.9755 |
| 42 | 32 | 1.9298 | 1.9997 |
| 42 | 64 | 1.9403 | 2.0114 |
| 42 | 256 | 1.9641 | 2.0313 |
| 56 | 8 | 2.4657 | 2.5585 |
| 56 | 32 | 2.4905 | 2.5624 |
| 56 | 64 | 2.5475 | 2.5801 |
| 56 | 256 | 2.5333 | 2.6029 |
