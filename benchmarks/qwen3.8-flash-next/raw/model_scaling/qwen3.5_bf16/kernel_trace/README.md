# Qwen3.5 BF16 cuTile kernel proof

Prepared, not launched. The coordinator must explicitly release the GPU after both throughput servers finish. This separate diagnostic reloads the exact native comparison binary/checkpoint with the same BF16, cuTile backend, scheduler, and context settings. It does not alter the throughput runner or its artifacts.

The frozen CLI is `/home/ericbuehler/qwen4exp_work/verify_graph_depth4_20260930/build/mistralrs`, SHA256 `92c050e15bbfe15fc99ef11077366a8fb40db5af96223c67e0f9299fce23f2fa`. The checkpoint is `/home/ericbuehler/hf_models/qwen3.5_35b_a3b`. `MISTRALRS_MOE_BACKEND=cutile` is preserved from the native comparison, along with its CUDA/tileiras paths. This tests the explicitly selected production BF16 path; it is not a backend-default selection experiment.

Nsight Systems 2026.1.3 launches the model with recording delayed through loading/JIT and graph capture. At C1 and C8, one unrecorded warmup precedes eight balanced canonical requests with 32 outputs each. Only CUDA software tracing and graph nodes are enabled; no GPU metrics, CPU samples, or OS runtime trace is collected. These phases prove observed kernel execution, not unprofiled performance.

Preparation can be inspected without launching anything:

```sh
python3 run_capture.py
```

After explicit coordinator release, use the actual completed throughput directories:

```sh
python3 run_capture.py --released \
  --after-run /home/ericbuehler/qwen4exp_work/qwen3_moe_scaling_20260930/mistralrs_bf16 \
  --after-run /path/to/completed/vllm/run
```

The runner requires both metadata files to report complete, checks for conflicting processes and an existing server, hashes the binary/checkpoint/helpers, refuses an existing output directory, and keeps 2 GiB/1 GiB starting/running disk guards. It owns only its new profiler session/model process. The coordinator retains editor process restoration unless it explicitly delegates a pause record. Expected additional report/export budget is under 512 MiB, an estimate guarded by live free-space checks rather than a guarantee.

After capture, profiler shutdown, and model exit:

```sh
python3 analyze_capture.py
python3 prove_kernels.py
```

The exporter refuses live/incomplete captures and verifies input hashes. It retains exact client timestamp alignment, observed clock uncertainty, all profiler diagnostics, and the warmup/recorded comparison. `prove_kernels.py` requires positive `fused_moe_kernel` launches with nonzero graph IDs in both C1 and C8. This establishes execution during observed decode graph work, beyond startup/JIT selection logs. It verifies the export manifest and records every matching kernel name, launch geometry, graph ID, count, and clipped diagnostic duration. `kernel_proof.manifest.json` separately hashes the added proof, its analyzer, and the original analysis manifest. Absence of a fallback name is not proof of exhaustive trace coverage.

The full finite request envelopes include prefill and drain. No kernel-share, throughput, hardware-utilization, or bandwidth claim follows from this presence check. Source, command, request, metric, memory, report, and export hashes remain external with the evidence.
