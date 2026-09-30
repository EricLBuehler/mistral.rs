# Register live-range diagnostic with 16-column tiles

All 25 grouped outputs are bit-exact against separately saved baseline tensors. This variant did not show a reliable gain across the matched real-route cases, and is not a production change.

The external CUDA header streams one k01 A/dmA fragment at a time instead of retaining all four fragments. Each accumulator keeps the same ascending k01 updates and product-then-min-correction order. It also selects 16-column grouped Q4K/Q4_1 tiles while preserving ncols_max and complete grid coverage. The shared helper is also used by Q5_1/Q5K/IQ1_S, but these two-unit external builds test only Q4K/Q4_1. No broader-format claim follows.

Native 42/56 and derived B8xQ3/4/5 inputs come from the same target-layer 8 captures. Times cover 3 warmups and 7 alternating rounds of 10 repeated eager expert forwards. Source rows, routes and quantized weights are identical to the baseline. Strict baseline comparisons precede timing; all 25 relative RMS errors are zero.

| Shape | Paired baseline/variant ratio | Per-input range |
| --- | ---: | ---: |
| Native M42 | 0.992 | 0.919..1.010 |
| Native M56 | 1.011 | 0.970..1.023 |
| Derived M24 | 0.973 | 0.932..1.001 |
| Derived M32 | 1.006 | 0.963..1.011 |
| Derived M40 | 1.020 | 0.938..1.028 |

Ratios above 1 mean the variant is faster. They are medians of per-input paired timing ratios, not ratios of group time medians. Unchanged GEMV measurements and uniform one-second GPU clock/power/temperature samples remain in `summary.json` and the monitored phase. Their scope includes load/checks/warmups/timings. They do not establish kernel-local clocks or bandwidth.

A bounded Nsight Compute follow-up used the same median-U native 56 input as the baseline and collected exactly the gate/up/down MMQ launches. Actual active-warp occupancy rose 16.78/16.75/16.98% to 31.11/32.77/32.14%. Registers per thread fell 254 to 112/112/126; the driver used 100 KiB shared-memory carveout and reported register/shared residency limits of two blocks. Thus this specific rewrite did increase observed occupancy, but still did not produce consistent unprofiled layer-timing gains. Higher occupancy by itself was insufficient for this tested implementation. This does not rule out other scheduling improvements or establish a physical performance ceiling.

`ncu_comparison.json` records paired counters. `ncu_native56/` contains the small native report, raw CSV log, parsed kernels and exact command metadata. The profiled test output is explicitly not benchmark timing. Kernel replay used 10 counter passes with cache/clock control disabled. DRAM counters were unavailable on this device, so these data do not establish DRAM saturation. The primary script and imported helpers are archived with their original paths/hashes under `helper_sources/` and `dependency_sources.json`.

External source copies/shadow archives leave the production CUDA sources, cached native archive and CLI unchanged; exact compilation/link commands and hashes are in `build/provenance.json`, preservation checks in `final_provenance.json`. Large executables, native archives and saved output tensors stay external with path/hash records. `SHA256.json` verifies every packaged file except itself.

These are 25 inputs from one layer, including explicitly derived smaller layouts. They are warm repeated expert-pipeline measurements, not full-model throughput or model-quality guarantees.
