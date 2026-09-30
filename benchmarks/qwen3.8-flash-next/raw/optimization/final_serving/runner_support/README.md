# Final masked-Y serving measurements

`run_final.py` reuses the pinned model command from
`../../tuner_exploration_20260930/run_server.py` and copies the repository's
unchanged serving/concurrency harnesses into each output directory. It launches
one server and stops it after validation. No profiler is attached. The candidate
executable must be supplied after the final kernels are selected and the GPU
slot is free; preparing these scripts does not launch a server.

```bash
python3 run_final.py \
  --binary "$CANDIDATE" \
  --expected-binary-sha256 "$CANDIDATE_SHA" \
  --output ./masked_only_final \
  --include-serving --include-smokes
```

Set `CANDIDATE` to the frozen release executable and `CANDIDATE_SHA` to its
previously recorded SHA256. The output directory must not exist. An optional
`--build-metadata PATH` archives the build provenance alongside the binary hash.
`--dry-run` prints the commands without hashing, writing files, contacting a
server, or querying a GPU. Python needs the same `tokenizers` dependency as the
existing benchmark harnesses. `nvidia-smi` must be available when executing.

The default run uses closed-loop C1 with 8 requests and C6/C8 with 24 requests
each. Each concurrency has 2 warmup trials followed by 5 measured trials, using
the same eight ordinary prompts in balanced cycles, 128 generated tokens,
greedy sampling, ignored EOS, no logprobs, and disabled prefix caching. The
server uses max-seqs 8 for every case, a 16384-token context, Q4K ISQ, and built-in
MTP. These are finite closed-loop trials: whole-trial throughput includes
startup and drain. Per-active-request throughput divides output tokens by
summed request latency; the secondary completion-boundary window is not a
streaming token rate.

`--include-serving` runs the original serving suite first, including prefill,
single-request decode, C4/C8 bursts, finite-logprob checks, and the repetitive
prompt. `--include-smokes` runs the separate chat smoke before concurrency and
the mixed-context correctness smoke last. Chat output still needs semantic
inspection; a successful HTTP response alone is not a quality test.

Before timing, the runner requires the prior topology: all 48 layers on CUDA 0,
BF16 model/cache, Q4 PLE, Q4K ISQ with sensitive Q6K tensors, 513 KV blocks of 32,
the same scheduler settings, built-in MTP initialized at depth 6, and 34 decode
graphs. A changed startup log fails the run for review instead of silently
changing the comparison. Adaptive MTP remains enabled and can change depth.

Each output contains exact copied scripts and small checkpoint files, source
revision/diff/file hashes, binary version and SHA256, selected runtime
environment, complete raw responses, validated concurrency summaries, server
log, per-command metrics and memory snapshots, a 2-second process-memory series,
and 1-second GPU clock/power/utilization polling. Final hashes cover all saved
artifacts. Source inspection is not proof of which source built a binary;
provide build metadata to establish that link.

Process snapshots reject active known compiler, profiler, or other model
processes before startup and around each phase. Stopped editor processes are
recorded and allowed. The runner never stops unrelated processes. These boundary
checks do not constitute continuous system isolation. Nonempty `MISTRALRS_*`,
Nsight, preload, or CUDA injection overrides are rejected so an earlier
experiment cannot silently select a different backend.

Metrics and memory counter differences cover entire command windows, including
warmups and boundary collection overhead. System swap counters cannot identify
which process paged or establish whether page-ins affected a timed trial;
VmSwap is saved separately for the server. GPU polling is diagnostic telemetry,
not Nsight profiling. No bandwidth or hardware-ceiling claim follows from it.

Interruptions and failures terminate only owned process groups and preserve
partial artifacts with `complete: false`. Completion requires successful raw
validation, an unchanged binary hash, and a server shutdown without SIGKILL.
The output can then be compared with the archived tuner measurements using the
same repository summarizer. Old and new results should be treated as staged
measurements, not a randomized causal A/B experiment.
