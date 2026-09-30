# Dense FP8 versus FP8-to-Q4K control

Both arms use the same pinned Qwen3.8-27B-FP8 checkpoint, executable, GB10,
BF16 KV cache, eight-sequence capacity, and greedy request settings. Only the
Q4K arm adds `--isq q4k`. Each C1/C8 case has one warmup and three measured
trials of eight requests with 128 output tokens. C8 is one finite wave.

[comparison.json](comparison.json) independently validates all saved responses,
raw timestamps, counts, settings, source hashes, and launch provenance, then
recomputes the rates. Full responses and logs are preserved in
[dense_fp8](dense_fp8/raw.json) and [dense_fp8_to_q4k](dense_fp8_to_q4k/raw.json).
[startup_verification.json](startup_verification.json) distinguishes log facts
from source-derived backend dispatch; packed affine Marlin is disabled in both
production environments. [methodology.md](methodology.md) preserves the design.

Q4K first dequantizes the FP8 weights and requantizes eligible layers. It also
changes eligible lm_head precision and stored projection layout. Separate Q4K
gate/up tensors can still use fused MMVQ computation. This is a full execution
path comparison, with no claim of equivalent quality or isolated kernel cost.

The identified editor processes were paused and the agents held builds/GPU
work during measurement. [provenance.json](provenance.json) records the
orchestrator's stopped-state observations, a [post-pair snapshot](paused_editor_processes_after_dense.json) confirming all three process identities still stopped, and the isolation limits; the initial
pause record's immediate R state is not a failed-pause determination. No
continuous machine-wide process or paging trace was collected for this pair.

To reproduce validation without running inference, install `tokenizers` and
run from this directory:

```bash
python3 compare_backend_results.py --snapshot /path/to/pinned-snapshot --output /tmp/backend-comparison.json
```

The snapshot directory needs only `config.json`, `tokenizer.json`, and
`model.safetensors.index.json` from revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; hashes are checked. Config and index
are included under `checkpoint_metadata/`, while the tokenizer payload is
omitted. A matching existing cache is the default snapshot path. The validator
uses the archived `harness/` and canonical prompts, not current repository
sources. Tensor shards and the executable are never read during validation;
launch-time executable hashes are checked for agreement.

The frozen controller retains its original absolute source/cache paths as
measured. Its [metadata](dense_fp8/metadata.json) and the
[Q4K metadata](dense_fp8_to_q4k/metadata.json) preserve exact launch commands.
[manifest.json](manifest.json) and `SHA256SUMS` cover every archived file except
the two manifests themselves. [archive_validation.json](archive_validation.json)
records a successful rerun using the archived validator and shared helpers.
