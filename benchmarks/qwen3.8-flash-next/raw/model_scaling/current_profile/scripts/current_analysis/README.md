# Current capture semantic adapter

Completed for the `current_d2c85e_masked_mmq24_autotuner` snapshot. The model exited before exports and semantic analysis. See `results/interpretation.md` and `results/interpretation.json` for the checked comparison.

After the server exits and `analyze_capture.py` completes:

```sh
python3 /home/ericbuehler/qwen4exp_work/model_scaling_20260930/current_analysis/analyze_current.py \
  --run /home/ericbuehler/qwen4exp_work/model_scaling_20260930/current_capture \
  --analysis /home/ericbuehler/qwen4exp_work/model_scaling_20260930/analysis \
  --output /home/ericbuehler/qwen4exp_work/model_scaling_20260930/current_analysis/results \
  --snapshot-label current_d2c85e_masked_mmq24_autotuner \
  --execute
```

Choose an explicit snapshot label matching the measured binary. Without `--execute`, this prints the plan without reading a live capture or its exports. It refuses existing output directories.

The completion gate requires completed run metadata, original model PID/start-time identity gone, completed export provenance tied to the same metadata, and matching input manifests. The model config hash must match the Flash-Next geometry used for semantic classification. Saved phase counters are recomputed independently from the before/after Prometheus snapshots and checked against the saved summary.

The adapter uses the exported `client_request_envelope` for whole-phase output-token and proposal denominators. This contains the exact monotonic request interval within the sampled clock-offset uncertainty. It includes prefill, client setup and final drain; it is not pure decode. `client_request_interior` is used only to report the kernel-time sensitivity to the offset envelope. No full-phase token denominator is assigned to either trimmed interior, especially the completion-boundary interior.

All semantic categories retain graph/eager time and MMQ correction-pass time. Target/draft stage topology is identified by 97 versus 3 hyper-connection mixes, with mixed intervals kept unknown. Draft-shaped time per recorded proposed token is a whole-phase accounting ratio: draft prefill/replay and discarded proposals can prevent trace vocabulary-row counts from matching proposal counters. Both counts and their difference are reported, and no equivalence is assumed.

Outputs are `semantic_breakdown.json`, `compact_summary.json`, `provenance.json`, and `manifest.json`. The snapshot label, profiled binary/run metadata, all input hashes, classifier/reader hashes and source-mapping references are retained. Profiler collection warnings remain in the full output. Results must not be used as unprofiled serving benchmarks.

`self_check.py` exercises only synthetic CPU data: clipping, semantic topology, proposal/output normalization, count invariants and the incomplete-run gate. It never exports or reads a real trace, launches CUDA, or sends a request.
