# Adaptive MTP graph candidate

Prepared only; this directory does not launch the candidate or build it. The
coordinator supplies the frozen candidate binary path, its exact SHA256, and build
metadata after the current profiler run has stopped.

```bash
python3 run_final.py \
  --binary "$CANDIDATE_BINARY" \
  --expected-binary-sha256 "$CANDIDATE_SHA256" \
  --build-metadata "$BUILD_METADATA" \
  --output /home/ericbuehler/qwen4exp_work/model_scaling_20260930/graph_candidate/adaptive_run \
  --include-serving --include-smokes
```

Add `--dry-run` to print commands without server, HTTP, hashing, or GPU access.
The runner is copied from the completed final serving stage. `runner.patch`
records the runner changes: require 36 startup CUDA graphs instead of 34,
read the exact frozen previous-stage harnesses, use the new descriptive label,
and append the separate homogeneous-input diagnostic described below.
The two added graphs cover eligible depth-four verification shapes. The server
still runs adaptive MTP with no fixed-depth option or environment override.

The same pinned model, Q4K ISQ, 16k context, max-seqs 8, prefix cache off, BF16
cache, Q4 PLE, 48 GPU layers, and 513 cache blocks are required. The serving suite
and replenished C1/C6/C8 trials retain two warmups, five measured repetitions,
128 outputs, C1 eight requests, and C6/C8 24 requests. Text and mixed-context
smokes follow the same existing protocol. Profiler/backend overrides are rejected.
The source snapshot, binary hash before/after, inherited execution environment,
per-command counters, memory/swap observations, raw responses, and clean shutdown
are retained as in the previous runner. Root owns editor pause/restoration for
this unprofiled candidate run; this runner does not signal editor processes.
The preflight and two-second runtime memory monitor require at least 1 GiB free
disk; the existing monitor-failure path stops owned processes if this limit is crossed.

The previous runner's CPU mock checks pass with the extended expected phase list. Additional checks confirm
that the new startup rule accepts 36 graphs and rejects 34, with unchanged adaptive
server arguments. The copied measurement harness hashes match the completed
`optimized_final/scripts/` artifacts. No CPU self-check contacts a server.

After the candidate completes and its server stops:

```bash
python3 compare_stage.py adaptive_run \
  --expected-binary-sha256 "$CANDIDATE_SHA256" \
  --output adaptive_run.comparison
```

The direct baseline is the completed masked-MMQ/small-group adaptive run with
binary `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d`.
The existing frozen validators independently revalidate both stages against their
common historical ancestor, then `compare_stage.py` compares the two validated
stage rates directly. Intermediate ancestor reports are retained for audit, not
used as the graph-policy headline. Matched server arguments, recorded environment,
prompt/tokenizer/harness identities, output counts, and startup checks are required;
the explicit graph count changes from 34 to 36.

This is a staged adaptive-policy comparison, not a fixed-depth-four A/B or a
randomized isolated causal estimate. Report changes in acceptance, proposed depth,
graph dispatch counts, responses, and memory pressure alongside timing. Counter
windows include warmups; C6/C8 counters are combined. The original serving and
concurrency artifacts remain unchanged.

## Homogeneous-input diagnostic

After every original phase finishes, four separate commands run the existing
`python` and `math` prompts at C1 and C8. Each request repeats exactly that selected
prompt with temperature zero, the existing fixed seed, EOS ignored, prefix caching
off, no logprobs, and 128 output tokens. C1 has eight sequential requests per
trial; C8 has three synchronized waves of eight requests, waiting for all responses
before starting the next wave. Each command has two warmups and three measured
trials. This finite-wave input control is separate from the normal mixed-prompt
serving and replenished concurrency benchmarks.

`harnesses/bench_homogeneous.py` imports the unchanged concurrency scheduler/request
helper and replaces the prompt cycle only inside its own subprocess. The exact
source prompt text, prompt/tokenizer/script hashes, all request bodies/responses,
response and text hashes, per-wave output agreement, trial mean active requests,
aggregate throughput, and latency-weighted per-active throughput are saved. The
last rate is total output tokens divided by summed request wall time. Rates include
prefill, client overhead, wave startup, and drain. Each command also gets the
runner's separate metrics/memory/process snapshots; those counter windows include
both warmups and measured trials.

The runner automatically creates `homogeneous.summary.json` and `.txt` after
revalidating all four raw files, token counts, request settings, hashes, timings,
and concurrency limits. The summary reports same-prompt C1 and C8 absolute rates,
sample standard deviations, C8/C1 ratios, and within-wave/cross-C output agreement.
It can be reproduced after completion without server access:

```bash
python3 adaptive_run/scripts/bench_homogeneous.py summarize \
  adaptive_run/homogeneous_python_c1.json \
  adaptive_run/homogeneous_python_c8.json \
  adaptive_run/homogeneous_math_c1.json \
  adaptive_run/homogeneous_math_c8.json \
  --output homogeneous.summary.json
```

Identical prompts do not establish identical expert routes or actual weight reuse.
Batch-dependent numerics and adaptive MTP can change outputs, acceptance, or depth;
text differences are retained rather than failed. C1/C8 phases are sequential, not
randomized causal trials. `compare_stage.py` and the original concurrency summary
explicitly read only the original mixed workload files and phase names; these four
added diagnostic commands do not enter their rates or counter windows.

`check_homogeneous.py` uses mocked HTTP responses only. It verifies all four case
counts, peak concurrency, active-integral calculations, no overlap between waves
or phases, independent ratio arithmetic, exact/differing text agreement, hash
corruption rejection, failed-wave persistence, and stopping after a short output.
Its saved result is `check_homogeneous.json`.
