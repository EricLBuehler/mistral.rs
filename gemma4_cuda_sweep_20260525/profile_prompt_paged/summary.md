# Gemma 4 E4B Q8 prompt profile with PagedAttention

All runs kept PagedAttention enabled. The profiling was done with temporary env-gated stream synchronizations around Gemma4 and PagedAttention phases, then the instrumentation was removed.

## Fast prefill check

Prompt length 2048, PagedAttention on:

| Run | Prefill T/s | TTFT |
| --- | ---: | ---: |
| Normal | ~6896 T/s | ~297 ms |
| `MISTRALRS_GEMMA4_DISABLE_FAST_PREFILL=1` | 4531 T/s | 452 ms |

The shared-KV fast prefill path is active and material.

## Phase attribution

These are synchronized profiler timings, so absolute totals include profiling overhead. The useful signal is attribution and scaling.

### 2048 tokens

| Phase | Count | Total ms |
| --- | ---: | ---: |
| Full-sequence dense MLP (`seq_len=2048`) | 24 | 270.425 |
| Full-sequence regular SDPA | 24 | 19.790 |
| Full-sequence attention projections | 24 | 31.479 |
| Full-sequence attention output projection | 24 | 26.544 |
| Full-sequence RoPE/norm | 24 | 9.401 |
| Full-sequence PLE projection | 24 | 22.751 |
| Shared tail layers, all work | 18 | 14.497 |
| Shared tail donor gather | 18 | 0.815 |
| Shared tail prefix SDPA | 18 | 1.673 |

### 16384 tokens

| Phase | Count | Total ms |
| --- | ---: | ---: |
| Full-sequence dense MLP (`seq_len=16384`) | 24 | 3011.285 |
| Full-sequence regular SDPA | 24 | 814.175 |
| Full-sequence attention projections | 24 | 437.645 |
| Full-sequence attention output projection | 24 | 374.316 |
| Full-sequence RoPE/norm | 24 | 99.182 |
| Full-sequence PLE projection | 24 | 297.210 |
| Shared tail layers, all work | 18 | 64.504 |
| Shared tail donor gather | 18 | 31.962 |
| Shared tail prefix SDPA | 18 | 10.000 |

## Conclusion

PagedAttention is not the source of the long-context prompt slope. The shared-KV tail is already reduced to one token after layer 23, and at 16k it is only about 64 ms total in the synchronized profile. The dominant cost is the 24 full-sequence layers before the KV-sharing cutoff, especially Q8 dense MLP/MMQ work, followed by full-sequence attention.

An experimental single-query direct FlashInfer decode path for shared-tail prompt layers was tested. It was neutral to slightly slower in unprofiled 8k/16k prompt runs and the tensor-core FlashInfer variant hung at long contexts with prompt-tail metadata, so it was not kept.
