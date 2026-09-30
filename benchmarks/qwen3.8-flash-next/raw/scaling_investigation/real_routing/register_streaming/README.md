# Register live-range diagnostic

All 25 grouped outputs are bit-exact against separately saved baseline tensors. This variant did not show a reliable gain across the matched real-route cases, and is not a production change.

The external CUDA header streams one k01 A/dmA fragment at a time instead of retaining all four fragments. Each accumulator keeps the same ascending k01 updates and product-then-min-correction order. Host tile selection and full grid coverage remain unchanged. The shared helper is also used by Q5_1/Q5K/IQ1_S, but these two-unit external builds test only Q4K/Q4_1. No broader-format claim follows.

Native 42/56 and derived B8xQ3/4/5 inputs come from the same target-layer 8 captures. Times cover 3 warmups and 7 alternating rounds of 10 repeated eager expert forwards. Source rows, routes and quantized weights are identical to the baseline. Strict baseline comparisons precede timing; all 25 relative RMS errors are zero.

| Shape | Paired baseline/variant ratio | Per-input range |
| --- | ---: | ---: |
| Native M42 | 0.940 | 0.918..0.961 |
| Native M56 | 0.938 | 0.930..0.961 |
| Derived M24 | 0.959 | 0.929..0.972 |
| Derived M32 | 0.962 | 0.942..0.966 |
| Derived M40 | 0.964 | 0.952..0.973 |

Ratios above 1 mean the variant is faster. They are medians of per-input paired timing ratios, not ratios of group time medians. Unchanged GEMV measurements and uniform one-second GPU clock/power/temperature samples remain in `summary.json` and the monitored phase. Their scope includes load/checks/warmups/timings. They do not establish kernel-local clocks or bandwidth.

External source copies/shadow archives leave the production CUDA sources, cached native archive and CLI unchanged; exact compilation/link commands and hashes are in `build/provenance.json`, preservation checks in `final_provenance.json`. Large executables, native archives and saved output tensors stay external with path/hash records. `SHA256.json` verifies every packaged file except itself.

These are 25 inputs from one layer, including explicitly derived smaller layouts. They are warm repeated expert-pipeline measurements, not full-model throughput or model-quality guarantees.
