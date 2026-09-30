# Final speculative metric snapshots

These metrics bracket each whole benchmark subprocess, including warmups and every trial. C6 and C8 share one subprocess and cannot be separated using these counters. All data comes from saved snapshots; the summarizer does not contact the server.

| Subprocess | Sequence proposals | Proposed draft tokens | Accepted draft tokens | Acceptance | Mean proposed depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| isq_mtp_tuner.serving | 5879 | 17995 | 13836 | 76.89% | 3.061 |
| isq_mtp_tuner.text | 2 | 6 | 6 | 100.00% | 3.000 |
| concurrency_c1 | 1969 | 6461 | 5126 | 79.34% | 3.281 |
| concurrency_c6_c8 | 9584 | 49683 | 33011 | 66.44% | 5.184 |
| burst_c6 | 4679 | 24590 | 16611 | 67.55% | 5.255 |
| mixed_context | 397 | 1184 | 616 | 52.03% | 2.982 |

A sequence proposal is one verified sequence row, not one batch step. Acceptance is accepted draft tokens divided by proposed draft tokens. Mean depth is proposed draft tokens divided by sequence proposals.

The JSON also reports accepted drafts plus one theoretical continuation per sequence proposal. This is not committed output length: EOS or output limits can discard verified lookahead or omit continuation. Per-position rates are unconditional over sequence proposals; conditional acceptance cannot be reconstructed without proposed-per-position counts.

Target graph dispatch counts were 2,025 replay and zero eager for C1, versus 720 replay and 1,065 eager (batch unsupported) for combined C6/C8. Separate prefill skips were 56 and 314. These are dispatch counts, not kernel-time coverage or per-request graph-use fractions.

Actual expert occupancy, depth distributions, and batch-size distributions were not recorded. Endpoint gauges cannot reconstruct average batch size. Client request intervals supply mean active requests separately.

All counter deltas were finite and nonnegative; per-position accepted counts matched total accepted counts. Raw snapshots and their bundle retain exact contents. The manifest hashes every packaged file except the manifest itself. Run `python3 summarize_metrics.py` to recompute the metric summary from the snapshots; its source-directory field will reflect the current location.
