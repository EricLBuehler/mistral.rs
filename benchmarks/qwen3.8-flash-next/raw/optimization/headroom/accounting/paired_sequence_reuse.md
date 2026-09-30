# Matched cross-sequence selected-weight reuse

Each native capture is split into its own constituent sequences, retaining all seven query positions and ten routes per position. Each separate sequence already reuses weights across its own seven positions. We compare the sum of those sequence-specific expert unions with the union for that exact batch. This avoids comparing unrelated C1 and C8 requests.

| Capture | Unique experts per sequence | Sum | Batch union | Reuse factor | Logical payload reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| c6_target_verify_b6_q7_00.json | 35, 33, 26, 27, 36, 25 | 182 | 107 | 1.701x | 41.21% |
| c6_target_verify_b6_q7_01.json | 39, 33, 31, 29, 31, 37 | 200 | 136 | 1.471x | 32.00% |
| c6_target_verify_b6_q7_02.json | 43, 49, 37, 28, 30, 40 | 227 | 164 | 1.384x | 27.75% |
| c6_target_verify_b6_q7_03.json | 43, 35, 41, 30, 37, 32 | 218 | 168 | 1.298x | 22.94% |
| c6_target_verify_b6_q7_04.json | 41, 46, 38, 45, 42, 42 | 254 | 195 | 1.303x | 23.23% |
| c8_target_verify_b8_q7_00.json | 32, 35, 28, 30, 33, 30, 33, 38 | 259 | 132 | 1.962x | 49.03% |
| c8_target_verify_b8_q7_01.json | 31, 37, 40, 37, 37, 43, 28, 33 | 286 | 184 | 1.554x | 35.66% |
| c8_target_verify_b8_q7_02.json | 27, 27, 36, 33, 45, 34, 37, 52 | 291 | 216 | 1.347x | 25.77% |
| c8_target_verify_b8_q7_03.json | 29, 33, 35, 40, 28, 41, 43, 53 | 302 | 219 | 1.379x | 27.48% |
| c8_target_verify_b8_q7_04.json | 45, 31, 30, 45, 46, 46, 42, 28 | 313 | 213 | 1.469x | 31.95% |

Across five native captures per batch size:

| Batch | Median reuse | Range | Median logical payload reduction | Batch payload / mean sequence payload |
| --- | ---: | ---: | ---: | ---: |
| B6xQ7 | 1.384x | 1.298x to 1.701x | 27.75% | 4.335x |
| B8xQ7 | 1.469x | 1.347x to 1.962x | 31.95% | 5.444x |

For the selected native B8 sample04, the eight sequences individually select 313 expert payloads in total, while their union selects 213. The logical selected-weight payload is 897,433,600 bytes separately versus 610,713,600 bytes batched: 286,720,000 bytes of cross-sequence reuse, or 31.95%. The union is 5.444 times the mean individual sequence payload. This illustrates why eight sequences do not automatically share nearly all expert weights.

The calculation is a property of these exact saved routes and compressed weight geometry. It is not actual memory traffic, a serving speedup prediction, a physical throughput ceiling, or a model-quality result. It does not account for cache persistence across steps, different routing in other layers, or the final adaptive-depth workload.

Reproduce with `python3 build_paired_sequence_reuse.py`. The JSON includes sequence row indices, exact expert sets, per-case byte counts, source hashes and summary statistics. No model weights are loaded and no CUDA calls are made.
