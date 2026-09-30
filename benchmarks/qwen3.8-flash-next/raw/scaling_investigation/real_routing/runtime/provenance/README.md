# Real Flash-Next routing capture and kernel replay

This is a temporary diagnostic, separate from the production benchmark binary.
No throughput claim should use these capture requests: graphs are disabled,
fixed MTP depth is 6, and disk writes/device downloads synchronize execution.

`capture.patch` adds an opt-in hook for target layer 8. It captures the BF16
expert input and output before adding the shared expert, the GPU-selected U32
expert IDs, and the exact F32 routing weights used by the kernels. It writes the
actual post-ISQ QTensor bytes once as `layer8.gguf`, without requantization.
This costs about 1.37 GiB plus at most tens of MiB of activation samples.
`replay.patch` adds an ignored CUDA test to the existing MoE benchmark.

Both patches remain external until the coordinator releases the build/GPU.
The original/revised source hashes are in `source_manifest.json`, and exact
original files are retained under `base_tree/`. `patch_tree/` is reviewable.

## Coordinated lifecycle

`build_diagnostic.py --released` implements the build-only steps below. It must
not run until the coordinator explicitly releases the build slot. It verifies
the production SHA256 `d245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588`,
preserves a hard link, checks baseline source hashes, applies both patches, builds
the CLI and replay test sequentially, and runs cargo check for both. It archives
the diagnostic CLI and exact Cargo-reported replay executable. Finally it restores
the capture-hook sources and the production CLI path, even on failure or normal
termination signals; it retains the ignored replay-test extension. All commands,
source/binary hashes and restoration results go into `build/lifecycle.json`.
Only a fully restored successful build writes `build/build-complete`.
No model, replay test, profiler or GPU query is executed by this script.

The resulting explicit paths are `build/mistralrs.capture` for diagnostic serving,
`build/moe_dispatch_bench.capture` for replay, and `build/mistralrs.production`
for the preserved production artifact. The usual `target/release/mistralrs`
path is restored to the original production binary before the script succeeds.

1. Preserve the current production CLI using a hard link outside target, and
   record its SHA256. Cargo's linker should replace the target inode, but verify
   the preserved production hash after building; do not assume this behavior.
2. Check/apply both patches. Build the CUDA/flash-attn/cutile CLI and the quant
   replay test using existing release artifacts. Archive the diagnostic binary
   with a hard link and hash it before restoring the production path.
3. Restore the capture-hook source exactly from `base_tree/` after the binary
   build and confirm its prepatch hashes. The ignored replay test can remain
   only if the coordinator wants it included; it has no runtime hook.
4. Use one diagnostic server with the same Flash-Next ISQ q4k settings as the
   final benchmark, adding `--mtp --mtp-n-predict 6`, and launch environment:

   ```sh
   MISTRALRS_CUDA_GRAPHS=0
   MISTRALRS_MOE_CAPTURE_DIR=/home/ericbuehler/qwen4exp_work/real_routing_20260930/captures
   ```

   Create a fresh empty capture directory; do not create `control.json` before
   startup. Prefix cache remains disabled. The hook rejects graph-enabled
   capture and unsupported sharded/nonquantized/LoRA configurations.
5. When the server is ready, run `capture_requests.py`. It makes exactly eight
   ordinary diverse `bench_serving.PROMPTS` requests at each C1/C6/C8, 128 output
   tokens each by default. `ignore_eos` ensures the requested length. Primary
   capture requests omit logprobs to preserve the ordinary greedy verification
   path. Actual completion counts, nonempty text and full responses are saved.
   The validator separately checks every captured input/routing/output value
   for finiteness.
   C1 arms a new case per prompt and takes one verify sample per prompt; C6/C8
   each collect up to eight samples of their native shapes spaced four visits
   apart. No benchmark metrics are computed. All control writes are atomic.

   ```sh
   python3 /home/ericbuehler/qwen4exp_work/real_routing_20260930/capture_requests.py \
     --directory /home/ericbuehler/qwen4exp_work/real_routing_20260930/captures \
     --output /home/ericbuehler/qwen4exp_work/real_routing_20260930/requests.json
   ```

6. Stop the diagnostic server before kernel replay. Validate and hash captures
   with `validate_capture.py CAPTURE_DIR --requests REQUESTS_JSON --output
   capture.validation.json`. This uses no GPU libraries. It checks all tensor
   finiteness, dimensions, dtypes, routing bounds/distinct IDs, occupancy counts,
   request completion and native 7/42/56-row coverage. If native shapes are absent,
   it fails instead of presenting only derived controls as observed data.
7. Run the compiled ignored test with exclusive GPU access:

   ```sh
   MISTRALRS_MOE_REPLAY_DIR=/home/ericbuehler/qwen4exp_work/real_routing_20260930/captures \
   MISTRALRS_MOE_BENCH_OUTPUT=/home/ericbuehler/qwen4exp_work/real_routing_20260930/replay.json \
   cargo test --release -p mistralrs-quant --features cuda,cutile \
     --test moe_dispatch_bench flash_next_real_routing_replay -- --ignored --exact --nocapture
   ```

   The coordinator should use the already compiled test executable if restoring
   source would otherwise trigger a rebuild. Root owns exact feature unification
   to reuse existing artifacts. Check and Clippy are coordinated with the build.
8. Verify restored production sources/binary hashes. Final production serving
   must use the original uninstrumented binary (or a verified rebuild from the
   restored source), never the capture binary.

## Replay interpretation

Each native input is replayed against the exact same quantized expert weights
through GEMV and packed grouped MMQ, with alternating repeated timing order.
Both paths are checked against each other and the captured output. Native
same-dispatch replay additionally requires relative RMS <1e-3 and cosine >.999999.
The cross-kernel tolerance remains the existing benchmark's RMS <.03 and
cosine >.999. Finite outputs and routing validity are mandatory.

For native B8xQ7 captures, extra controls gather rows 0,7,14,... to get one
query-position-zero row per sequence at widths 1/6/8. Separate controls use first
flattened rows at widths 1/6/8/24/32/40. Every derived record lists exact source row
indices and is labeled `sequence_query_zero` or `prefix`. These are true captured
operands, but they are not independently observed decode batches. Full native
records remain separate. Native target-decode shapes are captured if encountered;
the target layer hook does not capture the MTP draft head's separate layer0.

Timings cover a warmed, repeatedly reused isolated expert layer and include
host launch gaps in CUDA-event intervals. They do not measure full-model latency,
CUDA graph replay, shared experts, attention/GDN, or an unperturbed autotuner.
Occupancy is exact for saved samples, not a whole-workload routing distribution.

## Runtime controller

After the build completes and the coordinator releases the GPU, run
`python3 run_diagnostic.py --released`. It loads `build/mistralrs.capture` with
`--mtp-n-predict 6`, checks the final model/memory configuration, bounds startup
at2400 seconds, captures24 ordinary greedy requests without logprobs, validates
at least4 native7/42/56-row verify samples, stops the owned server and confirms
its PID/port are gone, then executes `build/moe_dispatch_bench.capture`.
`runtime/` retains raw requests, tensors, weight dump, logs, exact source copies,
binary/source hashes, replay timings and a concise median/occupancy summary.
Summary occupancy-derived reuse and distinct expert weight payload are logical
properties of the samples, not measured DRAM traffic or bandwidth lower bounds.
