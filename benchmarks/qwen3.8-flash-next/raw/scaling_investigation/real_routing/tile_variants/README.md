# Smaller grouped-MMQ tile diagnostic

The 16- and 32-column tile variants reproduce all 25 baseline outputs exactly, but do not show a consistent timing gain. Production tile selection remains unchanged.

The diagnostic overrides only grouped Q4K/Q4_1 MMQ dispatch (`expert_bounds != nullptr`). It retains the full `ncols_max` and recomputes the grid for the selected width, so experts with more rows than the tile still receive full coverage. External copies of two CUDA units were compiled with the same code-generation flags and linked through a shadow native archive. Production sources, cached native archive and CLI were unchanged. Exact commands, source hashes and binaries are in `build/provenance.json`; `final_provenance.json` verifies preservation and a passing targeted cargo check.

Every run uses the same 25 real-route cases: five native 42, five native 56 and five each derived B8xQ3/4/5 layouts. A separate saved grouped-MMQ baseline tensor plus exact source-row identity is checked before timing each candidate. The guard is RMS<1e-3/cosine>.999999; all candidate errors were zero. Native captured-output checks remain enabled. Timings use the same 3 warmups, 7 alternating rounds and 10 repetitions as the original layer replay.

The main `summary.json` compares uniformly monitored runs:

| Shape | Patched code, override unset | Tile 16 | Tile 32 |
| --- | ---: | ---: | ---: |
| Native M42 | 0.996 | 0.941 | 0.955 |
| Native M56 | 0.990 | 0.964 | 1.014 |
| Derived B8xQ3, M24 | 1.005 | 0.958 | 0.982 |
| Derived B8xQ4, M32 | 0.998 | 1.003 | 1.003 |
| Derived B8xQ5, M40 | 0.989 | 0.975 | 0.984 |

Values are median paired baseline-MMQ/variant-MMQ timing ratios; above 1 means the variant was faster. The small native 56 tile 32 difference is not a reliable gain: unchanged GEMV timing in that phase also shifts by a median factor 1.017. Across monitored phases/groups, the unchanged GEMV median factors range 0.993..1.021. Raw per-sample ratios and ranges remain in the summary.

Compiled register counts decrease from 252..255 registers/thread at 40/48/64-column tiles to 201/205 at 16 and 224/227 at 32 for Q4K/Q4_1. With 256 threads per block, even the smaller choices still exceed the 128-register/thread threshold for two such blocks in a65536-register SM. Smaller tiles alone therefore do not establish improved register-limited residency. All compared template resource records are identical between baseline and variant builds; only host selection changes.

Each monitored phase has 15 one-second `nvidia-smi` samples spanning loading, correctness checks, warmups and timings. Median SM clocks are 2515 MHz for baseline, unset and 16, and 2502 MHz for 32. Median temperatures rise 52/56/59/62 C through that sequence; median power is 60.78/61.41/56.56/60.28 W. GPU memory clocks were unavailable. These are whole-child observations, not kernel-local clocks, energy measurements or bandwidth counters. Initial unmonitored baseline/unset results are preserved separately in `initial_unmonitored_summary.json` and excluded from the main comparison.

Raw replay/lifecycle JSON, monitor CSV, source snapshots, scripts, patch, resource listings and hashes are packaged. `external_artifacts.json` points to saved output tensors and executables; large archives/binaries/tensors are not duplicated here. The full actual weight dump and original capture provenance are referenced by the parent real-routing package.

This measures warmed repeated target-layer 8 expert work. It does not establish full-model performance, cover other layers or route distributions, or authorize a production optimization. `SHA256.json` covers this subtree except itself.
