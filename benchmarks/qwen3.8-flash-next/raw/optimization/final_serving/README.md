# Final optimized serving archive

The completed September 30, 2026 run uses Q4K ISQ with adaptive MTP on GB10. See [comparison/comparison.md](comparison/comparison.md) for validated results and limits, and [manifest.json](manifest.json) for file hashes and exclusions.

- `run/` contains raw requests, timing samples, memory and counter snapshots, startup checks, source hashes/diff, and build metadata.
- `comparison/` contains independently recomputed serving/concurrency summaries and staged before/after comparisons.
- `postprocess/` contains the exact comparison script and frozen validators used to produce those summaries.
- `runner_support/` contains orchestration and archive provenance.

The measured executable has SHA256 `d2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d`. Startup's embedded git revision predates some changes; the executable hash, build metadata, and source hashes identify this run. The runner records HEAD `f5f225aab31b603213118d79618fa8fc5c4f0745` plus the saved uncommitted source diff. The server received SIGTERM after successful validation and exited without a forced kill.

The executable and 12.8 MB tokenizer are omitted. Their paths and hashes remain in the manifest. Model weights are never included. The original `run/SHA256SUMS.json` retains the tokenizer entry, so checking that original complete-run manifest or rerunning the comparison requires restoring the exact tokenizer in a separate copy of the archive. Obtain it from safetensors revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, and verify SHA256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` before placing it at `run/checkpoint/tokenizer.json`. The archive's top-level manifest covers the included files directly.

The postprocessor accepts `--baseline` to select the repository's historical `benchmarks/qwen3.8-flash-next/raw` directory and requires a new `--output` directory. Historical results and the candidate must retain their original raw-file hashes. Its checks include model/tokenizer identity, settings, prompts, requested/returned token counts, complete samples, finite validation logprobs, and counter/swap windows. Greedy text differences are diagnostic, not an answer-quality evaluation.

No new llama.cpp or same-GGUF result is mixed into this ISQ + MTP comparison. The before/after runs are staged rather than randomized, and changes in adaptive MTP behavior and swap-in are recorded alongside throughput.

The six editor processes paused for measurement were restored after the run and validation completed, with PID/start-time checks before each signal. The later [restoration record](editor_restoration.json) and this README are explanatory additions outside the immutable raw-evidence manifest.
