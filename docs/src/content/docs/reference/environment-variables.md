---
title: Environment variables
description: Environment variables read by mistralrs at build time or runtime.
---

User-facing environment variables read by `mistralrs` or its build scripts. Standard Cargo build variables such as `OUT_DIR` and `TARGET` are omitted.

## Hugging Face

| Variable | Purpose |
|---|---|
| `HF_HOME` | Root of the Hugging Face cache. Default `~/.cache/huggingface`. |
| `HF_HUB_CACHE` | Hugging Face hub cache location. |
| `HF_TOKEN` | Auth token. Overrides any token saved by `mistralrs login` at `$HF_HOME/token`. |
| `HF_HUB_TOKEN` | Auth token fallback when `HF_TOKEN` is not set. |
| `HF_HUB_OFFLINE` | Set to `1`/`true`/`yes`/`on` to disable all Hugging Face Hub network calls. Files and listings are then served only from `$HF_HUB_CACHE`/`$HF_HOME/hub`, and a missing file errors out. Also skips the `mistralrs doctor` connectivity check. |

If `--token-source env:NAME` is used, mistral.rs reads the environment variable named by `NAME` as the token source.

For the offline workflow (pre-downloading models, local paths), see [run any model](/mistral.rs/guides/models/run-any-model/).

## Logging

| Variable | Purpose |
|---|---|
| `RUST_LOG` | Override the `tracing` log filter. Examples: `mistralrs_core=debug,tower_http=info`, `trace`. CLI users can usually use `-v` or `-vv` instead. |
| `MISTRALRS_DEBUG` | `MISTRALRS_DEBUG=1` enables extra debug-level engine tracing. |

## Quantization and loading

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_MMAP` | `MISTRALRS_NO_MMAP=1` loads safetensors without mmap. |
| `MISTRALRS_ISQ_SINGLETHREAD` | If set, runs [ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/) single-threaded. |

## CPU runtime

| Variable | Purpose |
|---|---|
| `RAYON_NUM_THREADS` | Sets the default CPU worker count used by Candle and Rayon-backed CPU kernels unless a more specific variable is set. |
| `CANDLE_NUM_THREADS` | Sets Candle's CPU worker count. This overrides the fallback from `RAYON_NUM_THREADS` for Candle's own thread pools. |
| `CANDLE_CPU_MASK` | Linux only. Pins CPU worker threads to a cpulist such as `15-19` or `5-9,15-19`. If no explicit thread-count variable is set, the mask size also becomes the default worker count. |
| `CANDLE_CPU_AFFINITY` | Linux only. Set to `1` to try Candle's automatic high-capacity CPU affinity mask on heterogeneous CPUs. Default is off. |
| `CANDLE_BARRIER_POOL_SPIN_LIMIT` | Advanced CPU tuning. Overrides the spin count used by Candle's persistent barrier pool before worker threads park. |

See [CPU threads and affinity](/mistral.rs/guides/perf/throughput-tuning/#cpu-threads-and-affinity) for examples.

## Sandbox

| Variable | Purpose |
|---|---|
| `MISTRALRS_SANDBOX` | `auto`, `on`, or `off`. Overrides the sandbox only when the resolved mode is `auto`; `on` and `off` in CLI/TOML win. See [sandbox reference](/mistral.rs/reference/sandbox/). |

## Server and UI

| Variable | Purpose |
|---|---|
| `MCP_CONFIG_PATH` | [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) client configuration path used when `--mcp-config` is not passed. |
| `KEEP_ALIVE_INTERVAL` | SSE (Server-Sent Events) keep-alive interval in milliseconds. Falls back to the default if missing or invalid. |
| `XDG_CACHE_HOME` | Base cache directory for web UI state. The UI uses `$XDG_CACHE_HOME/mistralrs`. |
| `HOME` | Fallback for web UI cache path when `XDG_CACHE_HOME` is not set. |

## CUDA and attention kernels

| Variable | Purpose |
|---|---|
| `MISTRALRS_CUDA_GRAPHS` | CUDA decode graph capture and replay is enabled by default for supported paged-attention decode steps. Set to `0`, `false`, `no`, or `off` to disable. See [CUDA graphs](/mistral.rs/guides/perf/paged-attention/#cuda-graphs). |
| `MISTRALRS_FLASHINFER_DECODE` | Set to `0`, `false`, `no`, or `off` to disable the FlashInfer (paged-attention kernel library) paged decode/cache layout and use the generic paged KV-cache layout instead. Defaults to enabled on CUDA when compatible. |
| `MISTRALRS_NO_MLA` | `MISTRALRS_NO_MLA=1` disables the MLA (Multi-head Latent Attention) path for DeepSeek V2/V3. Generic attention is used instead. |
| `MISTRALRS_MOE_BACKEND` | Forces the MoE (Mixture of Experts) expert backend: `cutile`, `cutlass`, `fused` (also `wmma`, `native`, `legacy`), or `fast`. Default is automatic selection. See [MoE expert backends](/mistral.rs/developer/moe-backends/). |
| `CUTILE_TILEIRAS_PATH` | Path to a specific `tileiras` binary for the cuTile JIT instead of resolving it from `PATH`. |

## Multi-GPU and multi-node

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_NCCL` | `MISTRALRS_NO_NCCL=1` disables NCCL at runtime; single-machine CUDA multi-GPU then falls back to layer mapping. When using the ring backend on a binary also built with `nccl`, set this so the ring backend is selected. |
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total NCCL tensor-parallel world size across nodes. Presence of this variable enables multi-node NCCL mode. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Local NCCL tensor-parallel size contributed by each node. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Set on the head node: number of worker nodes. |
| `MISTRALRS_MN_HEAD_PORT` | Set on the head node: listening port for worker connections. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Set on worker nodes: address of the head node. |
| `MISTRALRS_MN_WORKER_ID` | Set on worker nodes: worker index (0-based). |
| `RING_CONFIG` | Path to the ring backend JSON config. Setting it selects the ring backend when built with the `ring` feature. If the binary also has `nccl`, set `MISTRALRS_NO_NCCL=1` as well. |

See the [distributed inference guide](/mistral.rs/guides/perf/distributed-inference/) for use.

## GPU memory

| Variable | Purpose |
|---|---|
| `MISTRALRS_IGPU_MEMORY_FRACTION` | Fraction of integrated GPU memory usable on CUDA systems with iGPUs. Default 0.75. |

## Build-time

These are read by build scripts, not at runtime.

| Variable | Purpose |
|---|---|
| `MISTRALRS_METAL_PRECOMPILE` | `MISTRALRS_METAL_PRECOMPILE=0` skips Metal kernel precompilation at build time; kernels are compiled at runtime on first use. Also accepts `false`, `no`, and `off`. |
| `MISTRALRS_METAL_PLATFORMS` | Limits which Metal platform metallibs are precompiled. Accepts comma-separated `macos`, `ios`, `tvos`, or `all`; defaults to all platforms. For local macOS development, use `MISTRALRS_METAL_PLATFORMS=macos`. |
| `CUDA_NVCC_FLAGS` | Extra compiler options passed to CUDA builds. |
| `MISTRALRS_CUTLASS_COMMIT` | Overrides the CUTLASS git commit used by CUDA build scripts for flash-attention and CUTLASS MoE kernels. Defaults to the project-pinned commit. |
| `MISTRALRS_INSTALL_TAG` | Pins the installers to a specific release tag (e.g. `v0.8.23`): the prebuilt is downloaded from that release, and a source build checks out that git tag. Default is the latest stable release (prebuilt) or latest `master` (source). |
| `MISTRALRS_INSTALL_FROM_SOURCE` | `MISTRALRS_INSTALL_FROM_SOURCE=1` makes the shell and PowerShell installers skip the prebuilt download and build from the latest `master` (bleeding edge) instead of the latest stable release. |
| `MISTRALRS_INSTALL_NCCL` | `MISTRALRS_INSTALL_NCCL=1` forces the shell and PowerShell installers to add the `nccl` feature for CUDA builds even if NCCL is not detected. |
| `MISTRALRS_INSTALL_NO_NCCL` | `MISTRALRS_INSTALL_NO_NCCL=1` makes the shell and PowerShell installers skip the `nccl` feature. |
| `MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH` | `MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1` lets a source build continue when local `nvcc` is newer than the CUDA version reported by the NVIDIA driver. |
| `MISTRALRS_INSTALL_YES` | `MISTRALRS_INSTALL_YES=1` auto-confirms every installer prompt (non-interactive installs for CI/containers; used by `mistralrs update`). |
| `MISTRALRS_INSTALL_IGNORE_FFMPEG` | `MISTRALRS_INSTALL_IGNORE_FFMPEG=1` skips the installer's FFmpeg step, leaving any existing FFmpeg untouched. |
| `MISTRALRS_GIT_REVISION` | Git revision embedded in the binary by the build script. |

## Internal

Not intended for direct use.

| Variable | Purpose |
|---|---|
| `__MISTRALRS_DAEMON_INTERNAL` | Set by the engine on spawned worker processes. |
