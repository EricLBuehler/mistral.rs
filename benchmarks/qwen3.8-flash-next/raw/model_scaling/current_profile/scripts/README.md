# Current full-model C1/C8 Nsight capture

This is a diagnostic capture of the final frozen server binary, not another official
throughput benchmark. It complements the fixed-layer expert controls with current
whole-model execution, including the adaptive MTP policy.

The coordinator launched the prepared supervisor after its review. No runner or
dependency source is changed after launch. `self_check.json` records a CPU-only
mock of C1/C8 scheduling, balanced prompt cycles, exact timestamp ordering, token
counts, and completion before the next trial.

## Invocation

```bash
python3 /home/ericbuehler/qwen4exp_work/model_scaling_20260930/run_capture.py \
  --released \
  --restore-editor-record /home/ericbuehler/qwen4exp_work/model_scaling_20260930/paused_editor_processes.json
```

Without `--released`, the command only prints the plan. Default output is
`current_capture/`, and existing output is never overwritten. The runner never
pauses editor processes. Passing the explicit pause record delegates restoration
of those identities to its final cleanup; otherwise the coordinator retains that
responsibility. Restoration checks PID, start time, and command name and does not
resume processes that were already stopped before the coordinator's pause.

## Exact workload

- Frozen binary:
  `/home/ericbuehler/qwen4exp_work/moe_optimization_20260930/final_serving/build/mistralrs`,
  SHA256 `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d`.
- Pinned `Qwen/Qwen3.8-Flash-Next` snapshot
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`; Q4K ISQ plus built-in MTP.
- 16,384-token maximum context, eight scheduler sequences for both phases,
  prefix cache size zero, the same eight canonical serving prompts.
- Each request uses greedy sampling, seed from the frozen shared harness,
  `ignore_eos=true`, `cache_prompt=false`, and 128 output tokens, without logprobs.
  Prompt token counts are checked with the pinned tokenizer; prefix reuse is rejected.
- C1: one unrecorded eight-request warmup, then one recorded eight-request
  closed-loop trial, totaling 1,024 recorded output tokens.
- C8: one unrecorded 24-request warmup, then one recorded 24-request closed-loop
  trial, totaling 3,072 recorded output tokens. Completed requests are replenished
  until the finite request count is exhausted.

The phases have equal prompt proportions but different total request counts,
matching the established serving comparison. These finite trials include startup
and drain; C8 is not guaranteed to remain at eight active requests throughout.
One warmup does not establish a stationary adaptive-depth distribution. Full raw
responses, request settings, prompt hashes, tokenizer/harness hashes, and expected
and observed token counts are preserved for every phase.

## Capture and timing

Nsight Systems 2026.1.3 launches the server with the previously working options:

```text
profile --start-later=true --trace=cuda-sw,nvtx,osrt
--cuda-graph-trace=node:host-only --sample=none --cpuctxsw=none --kill=none
```

Model loading and warmup are outside recording. Each recorded phase has its own
`nsys start`, request command, and `nsys stop`. Named reports are
`profile_c1.nsys-rep` and `profile_c8.nsys-rep`. Startup instrumentation can still
affect the process; disabled recording does not make it an ordinary uninstrumented
benchmark. All profiler stdout/stderr, server warnings, and control commands remain
in the output directory. SQLite export is deferred until the live run ends.

The externally copied concurrency harness only adds integer nanosecond timestamps;
`exact_timestamps.patch` and `dependencies.json` preserve the change and both source
hashes. It records the exact `perf_counter_ns` trial origin and each HTTP request's
begin/end. `profile_requests.py` brackets the trial with realtime, monotonic, and
`CLOCK_MONOTONIC_RAW` clock pairs. Each realtime sample is bounded by monotonic
reads, preserving mapping uncertainty and allowing drift checks. Do not assume
Nsight's `systemClockNs` is Python's raw clock. Use the exported capture's
`utcEpochNs` and these paired clocks, verify that capture start lies within its
control-command window, and retain any mapping uncertainty.

`request_window_perf_counter_ns` spans the first request begin through the last
complete response read. It includes prompt processing and client overhead. These
are non-streaming requests; there are no per-token arrival timestamps and no claim
of a decode-only or first-token measurement. Request windows permit exact active
request integrals. The harness's optional interior window counts whole responses
at completion boundaries; it is not a streaming token rate.

Each warmup and recorded trial has separate `/metrics` snapshots and validated
counter deltas. The recorded phases therefore exclude warmups from acceptance,
draft-depth, and graph/eager-dispatch counter summaries. Counts still do not
measure kernel-time coverage or a distribution of MTP depth or expert occupancy.
Memory and swap boundary records and a two-second memory/disk monitor preserve
possible interference; system-wide swap counters cannot identify model-specific
page-ins or their effect on individual requests.

## Space and cleanup

Preparation observed about 3.58 GB free. The historical C8 report occupied 46.5 MB,
and historical C1/C8 SQLite exports occupied 110/121 MB. The runner reserves a
conservative 1 GiB trace/temporary budget, requires 2 GiB free before startup,
and aborts if free space falls below 1 GiB. This estimate is not a size guarantee;
sampling and context-switch traces are disabled, only one trial is recorded per
phase, and no SQLite export runs while the server is alive.

Success requires completed requests, both reports, profiler shutdown, server exit,
and error-free cleanup in `metadata.json`. Even on failure, the supervisor attempts
to stop an active capture, shut down only its named profiler session, terminate its
owned server/launcher, and restore explicitly delegated editor identities. The
metadata records failures and forced shutdown. Check `metadata.complete`, not only
the process exit code, before analysis. Original reports and logs are retained and
hashed in `manifest.json`; no weights or server binaries are copied.

After shutdown, export each report separately and inspect all `DIAGNOSTIC_EVENT`
records for missing kernels, dropped data, and software-trace warnings. The frozen
historical `dependencies/analyze_nsys.py` can help classify kernels, but its broad
generic-MMQ category must not be assumed to identify all projections precisely.
Root and the kernel reviewer own final trace alignment and attribution.

## Offline export and comparison

After `metadata.complete=true` and model exit, run:

```bash
python3 /home/ericbuehler/qwen4exp_work/model_scaling_20260930/analyze_capture.py
```

This refuses a live or incomplete capture, verifies the raw-run manifest, exports
each report into a new `analysis/` directory, preserves every profiler diagnostic,
and builds trace windows from the exact client timestamps. It uses the observed
realtime-minus-monotonic offset envelope from the client and memory-monitor clock
pairs, trimming window edges by that envelope. Unobserved wall-clock steps are not
ruled out; the exact monotonic client intervals remain separately available.

`capture_off_vs_recorded.json` compares each warmup with its recorded phase using
matched prompts/settings and output counts. It reports both client durations,
their ratio, text agreement, and separate acceptance/depth/graph counter deltas.
This is a sequential same-server comparison with instrumentation attached in both
phases. It can reveal a material observed slowdown during recording, but adaptive
MTP, generated text, routes, and cache state also change, so the ratio is not an
isolated estimate of profiler overhead or a normal benchmark result.
