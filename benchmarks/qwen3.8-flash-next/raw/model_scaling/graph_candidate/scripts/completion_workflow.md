# Final candidate evidence workflow

No candidate measurements are interpreted until `adaptive_run/metadata.json` is complete, all nine phase processes succeeded, the server stopped without forced kill, monitor checks passed, and the binary identity is unchanged. The coordinator also waits for the runner process to exit so its final manifest is closed.

Run the CPU-only analysis after that boundary:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ericbuehler/mistral.rs/.venv/bin/python finalize_candidate.py analyze
```

`adaptive_run.analysis/stage_comparison/comparison.json` is the direct comparison with the immutable prior `d2c85e` run. The frozen `compare_stage.py` independently revalidates both raw stages with existing serving/concurrency/smoke validators, checks exact protocol/environment identities, and requires the intended startup graph count change from 34 to 36. Its historical ancestor reports remain audit evidence and do not supply the direct headline.

The workflow also reruns the frozen homogeneous-input summary and requires equality with the runner's saved result. The four added commands use two warmups and three measured trials; ordinary concurrency keeps two warmups and five measured trials. Homogeneous results have separate counter and memory windows. Their finite C8 waves are not treated as replenished mixed-prompt trials, and matching text is not proof of matching expert routes.

Review the saved chat answer and eight mixed-context answers after numerical/protocol validation. Record semantic observations separately: successful HTTP responses and finite logprobs are necessary checks, not a broad model-quality result. Summarize acceptance, average proposed depth, graph/eager dispatch counts, and memory/swap alongside rates. The ordinary C6/C8 counters are combined, include warmups, and do not establish kernel-time coverage.

The record `prior_checkpoint_cache_release.json` is included unchanged. The coordinator released clean cached pages for the completed, distinct Qwen3.5 checkpoint's 14 shards before candidate startup. It changed no file contents. MemFree increased from about 71 to 117 GiB while MemAvailable remained about 118 GiB. This preparation is outside measurement windows and is not evidence of benchmark-time model paging.

Archive only after analysis and response review:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ericbuehler/mistral.rs/.venv/bin/python finalize_candidate.py archive
```

The destination is `benchmarks/qwen3.8-flash-next/raw/model_scaling/graph_candidate`. Raw trials, counter snapshots, memory/process observations, commands, startup checks, source/build hashes, exact validators, and the cache-release record are retained. Tokenizer copies and large artifacts remain external with hashes; model weights and binaries are neither copied nor rehashed. A new archive manifest covers all copied evidence. The prior `raw/optimization/final_serving` baseline is not changed.

After completion, replace the unmeasured depth-four statement in `model-scaling.md`, and append a compact candidate section in `optimization.md`. Report five-trial mean/sample standard deviation, C1/C6/C8 rates, ratios to the prior stage, and adaptive-depth/counter changes. Keep the homogeneous-input control separate. This is an adaptive staged comparison, not a fixed-depth-four or randomized causal experiment; a rate change is not automatically attributable to the two additional graph shapes.
