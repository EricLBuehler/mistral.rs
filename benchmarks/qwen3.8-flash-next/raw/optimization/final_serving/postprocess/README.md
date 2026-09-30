# Postprocess the completed optimized server run

Run only after `optimized_final/metadata.json` has `complete: true` and the
server has stopped:

```bash
python3 -B compare_final.py ../optimized_final \
  --output ../optimized_final.comparison
```

This is offline CPU work. The script never contacts a server, launches a build,
or queries the GPU. It refuses partial runs and existing output directories.
The expected candidate SHA is pinned to the currently launched binary; a later
candidate requires an explicit `--expected-binary-sha256` override. Historical
repository benchmarks and the candidate's completed files are read-only.

The baseline is the archived final autotuner run, binary `d245efbb...`, before
these MoE optimizations. Both stages use the same snapshot, server settings,
harnesses, tokenizer, prompt hashes/token counts, 2 warmup trials, 5 measured
trials, and 128 outputs. C1 has 8 requests/trial and C6/C8 have 24. Server ports,
output paths, script paths, labels and executable paths are provenance rather
than workload settings. The earlier archive did not record every inherited
environment variable; only shared recorded values are compared.

`comparison.json` contains recomputed means/sample standard deviations, ratios
of means, aggregate/per-active rates, latency, mean active requests, scaling,
text-difference diagnostics, counter windows, global swap and process VmSwap.
The original serving and concurrency validators are copied here with hashes in
`dependencies.json`; their validated summaries are saved separately. Candidate
artifact hashes and the archived baseline raw-file hashes are verified.

`comparison.md` is a separate report draft. It labels serving C4/C8 as bursts
and concurrency rates as finite closed-loop trials. The prefill estimate counts
input tokens over total request wall time, including the one output token.
Counters cover entire commands including warmups. C6/C8 counters remain one
combined window, and graph dispatch counts are not kernel-time coverage.
System-wide swap cannot identify model paging or its effect on a timed trial.

The draft includes both repeated-phrase chat answers for human inspection:
check that each is a coherent sentence about the river, is not stuck repeating
the prompt, and ends reasonably. Structural and finite-logprob checks are
automatic; semantic quality is not. Greedy output differences remain diagnostic
and do not make a run fail by themselves.

The candidate combines masked-Y MMQ with the guarded small-group dispatch and
documented intervening fixes. A staged improvement is not a randomized causal
estimate of either individual change. Keep the historical headline/raw files
intact and publish any accepted result as a new optimization section with its
own artifact links. Adoption still requires interpreting correctness, startup
memory/graphs, and throughput together.
