# Adaptive graph-policy stage comparison

| C | Before aggregate tok/s | After aggregate tok/s | Change | After per-active tok/s |
|---:|---:|---:|---:|---:|
| 1 | 53.775 | 54.703 | +1.73% | 54.711 |
| 6 | 119.948 | 119.848 | -0.08% | 20.800 |
| 8 | 131.940 | 131.742 | -0.15% | 17.321 |

Raw samples and sample standard deviations are in comparison.json.

- Both stages run adaptive MTP; this is not a fixed-depth-four A/B experiment.
- Sequential runs are not a randomized causal estimate; inspect acceptance, mean depth, graph dispatches, generated text, and memory/swap differences.
- The intended policy difference adds two eligible depth-four graph shapes, with 36 startup graphs versus 34. Other startup checks must match.
- Counter windows include all warmups and measured trials; the C6/C8 command combines both concurrencies.
- Graph dispatch counts do not measure kernel-time coverage; global swap counters cannot identify model page-ins or timing impact.
- Full raw before/after outputs remain in the run archives. Smoke answers require human semantic inspection; finite logprobs do not prove MTP equivalence.
- Intermediate validation reports compare each stage with the historical common ancestor; use this file for the direct graph-policy stage comparison.
