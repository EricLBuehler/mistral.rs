# Current whole-model profile

This is the frozen masked-MMQ/small-group adaptive build, before the proposed graph
policy change. It is an instrumented diagnostic, separate from official serving
benchmarks and any later candidate run.

- [Interpretation](semantic/interpretation.md) and [compact measured summary](semantic/compact_summary.json).
- [Same-server capture-off versus recorded phases](analysis/capture_off_vs_recorded.json).
- [C1 alignment and profiler warnings](analysis/profile_c1.alignment.json) and [C8 alignment and warnings](analysis/profile_c8.alignment.json).
- [Exact request/counter commands, startup and cleanup provenance](capture/metadata.json), [server log](capture/server.log), and raw request/metrics files in `capture/`.
- [Semantic source/method](scripts/current_analysis/README.md), [capture method](scripts/README.md), and exact scripts under `scripts/`.
- [Omitted trace/database paths, sizes and hashes](external_artifacts.json), [archive provenance](archive_provenance.json), and [local manifest](SHA256SUMS.json).

Each recorded phase has one warmed closed-loop trial: C1 has eight requests and
1024 outputs; C8 has 24 requests and 3072 outputs. Rates and per-output kernel
costs include prefill, request startup and drain. Warmup recording is disabled but
profiler instrumentation remains attached. Similar warmup/recorded elapsed times
are not a pure instrumentation-overhead estimate: adaptive depth, output text,
cache state and routes can differ. Full-phase denominators are used only with the
whole request envelope, not the completion-boundary interior.

Nsight warns that some CUDA/OS runtime events might be missing; it provides no
NVTX events, CPU scheduling trace or unified-memory trace here. The analysis is
of recorded kernel activity, not guaranteed complete hardware time, DRAM traffic,
or a hardware ceiling. Exact compact source artifacts are preserved; reproducing
the full SQLite analysis needs the external files listed by hash. No historical
trace output has been substituted for this current capture.
