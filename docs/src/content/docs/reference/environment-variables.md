---
title: Environment variables
description: Environment variables read by mistralrs at build time or runtime.
sidebar:
  order: 15
---

User-facing environment variables read by `mistralrs` or its build scripts. Standard Cargo build variables such as `OUT_DIR` and `TARGET` are omitted.

## Hugging Face

| Variable | Purpose |
|---|---|
| `HF_HOME` | Root of the Hugging Face cache. Default `~/.cache/huggingface`. |
| `HF_HUB_CACHE` | Hugging Face hub cache location. |
| `HF_TOKEN` | Auth token. Overrides any token saved by `mistralrs login` at `$HF_HOME/token`. |
| `HF_HUB_TOKEN` | Auth token fallback when `HF_TOKEN` is not set. |
| `HF_HUB_OFFLINE` | `HF_HUB_OFFLINE=1` (or `true`/`yes`/`on`) disables all network calls to the Hugging Face Hub. Files and repo listings are served from `$HF_HUB_CACHE`/`$HF_HOME/hub` only; missing files fail fast with a clear error. The `mistralrs doctor` connectivity check is also skipped. |

If `--token-source env:NAME` is used, mistral.rs reads the environment variable named by `NAME` as the token source.

### Fully offline operation

Set `HF_HUB_OFFLINE=1` to guarantee no network calls are made to the Hugging Face Hub. mistral.rs will only resolve files from the local cache (`$HF_HUB_CACHE`, falling back to `$HF_HOME/hub`, falling back to `~/.cache/huggingface/hub`). Pre-download the model on a machine with network access (e.g. with `huggingface-cli download <repo>` or by running mistral.rs once online), then launch with `HF_HUB_OFFLINE=1`. A local model path (`-m /path/to/dir`) always reads from disk and never hits the network, so it works in offline mode without any cache lookup.

## Logging

| Variable | Purpose |
|---|---|
| `RUST_LOG` | Override the `tracing` log filter. Examples: `mistralrs_core=debug,tower_http=info`, `trace`. CLI users can usually use `-v` or `-vv` instead. |
| `MISTRALRS_DEBUG` | `MISTRALRS_DEBUG=1` enables extra debug-level engine tracing. |

## Quantization and loading

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_MMAP` | `MISTRALRS_NO_MMAP=1` loads safetensors without mmap. |
| `MISTRALRS_ISQ_SINGLETHREAD` | If set, runs ISQ quantization single-threaded. |

## Sandbox

| Variable | Purpose |
|---|---|
| `MISTRALRS_SANDBOX` | `auto`, `on`, or `off`. Overrides the sandbox only when the resolved mode is `auto`; `on` and `off` in CLI/TOML win. See [sandbox reference](/mistral.rs/reference/sandbox/). |

## Server and UI

| Variable | Purpose |
|---|---|
| `MCP_CONFIG_PATH` | MCP client configuration path used when `--mcp-config` is not passed. |
| `KEEP_ALIVE_INTERVAL` | SSE keep-alive interval in milliseconds. Falls back to the default if missing or invalid. |
| `XDG_CACHE_HOME` | Base cache directory for web UI state. The UI uses `$XDG_CACHE_HOME/mistralrs`. |
| `HOME` | Fallback for web UI cache path when `XDG_CACHE_HOME` is not set. |

## CUDA and attention kernels

| Variable | Purpose |
|---|---|
| `MISTRALRS_CUDA_GRAPHS` | CUDA decode graph capture and replay is enabled by default for supported paged-attention decode steps. Set to `0`, `false`, `no`, or `off` to disable. See [CUDA graphs](/mistral.rs/guides/perf/use-cuda-graphs/). |
| `MISTRALRS_FLASHINFER_DECODE` | Set to `0`, `false`, `no`, or `off` to disable the FlashInfer paged decode/cache layout and use the generic paged KV-cache layout instead. Defaults to enabled on CUDA when compatible. |
| `MISTRALRS_NO_MLA` | `MISTRALRS_NO_MLA=1` disables the MLA-specific attention path for DeepSeek V2/V3. Generic attention is used instead. |
| `MISTRALRS_GEMMA4_DISABLE_FAST_PREFILL` | If set, disables the Gemma 4 KV-sharing fast prefill path. Intended for debugging or comparison. |

## Multi-GPU and multi-node

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_NCCL` | `MISTRALRS_NO_NCCL=1` disables NCCL at runtime. Single-machine CUDA multi-GPU then uses layer mapping; ring builds also use this to force the ring backend when `nccl` is compiled in. |
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total NCCL tensor-parallel world size across nodes. Presence of this variable enables multi-node NCCL mode. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Local NCCL tensor-parallel size contributed by each node. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Set on the head node: number of worker nodes. |
| `MISTRALRS_MN_HEAD_PORT` | Set on the head node: listening port for worker connections. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Set on worker nodes: address of the head node. |
| `MISTRALRS_MN_WORKER_ID` | Set on worker nodes: worker index (0-based). |
| `RING_CONFIG` | Path to the ring backend JSON config. Presence of this variable selects ring when built with `ring`; set `MISTRALRS_NO_NCCL=1` too if the binary also has `nccl`. |

See [multi-GPU and distributed inference](/mistral.rs/guides/perf/multi-gpu-distributed/), [multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/), and the [ring backend guide](/mistral.rs/guides/perf/multi-machine-ring/) for use.

## GPU memory

| Variable | Purpose |
|---|---|
| `MISTRALRS_IGPU_MEMORY_FRACTION` | Fraction of integrated GPU memory usable on CUDA systems with iGPUs. Default 0.75. |

## Build-time

These are read by build scripts, not at runtime.

| Variable | Purpose |
|---|---|
| `MISTRALRS_METAL_PRECOMPILE` | `MISTRALRS_METAL_PRECOMPILE=0` skips Metal kernel precompilation at build time; kernels are compiled at runtime on first use. |
| `CUDA_NVCC_FLAGS` | Extra compiler options passed to CUDA builds. |
| `MISTRALRS_INSTALL_NCCL` | `MISTRALRS_INSTALL_NCCL=1` forces the shell and PowerShell installers to add the `nccl` feature for CUDA builds even if NCCL is not detected. |
| `MISTRALRS_INSTALL_NO_NCCL` | `MISTRALRS_INSTALL_NO_NCCL=1` makes the shell and PowerShell installers skip the `nccl` feature. |
| `MISTRALRS_BUILD_NCCL` | `MISTRALRS_BUILD_NCCL=1` forces `scripts/build_wheels.py` to add `nccl` to Linux CUDA wheels. |
| `MISTRALRS_BUILD_NO_NCCL` | `MISTRALRS_BUILD_NO_NCCL=1` makes `scripts/build_wheels.py` skip `nccl` for CUDA wheels. |
| `MISTRALRS_GIT_REVISION` | Git revision embedded in the binary by the build script. |

## Internal

Not intended for direct use.

| Variable | Purpose |
|---|---|
| `__MISTRALRS_DAEMON_INTERNAL` | Set by the engine on spawned worker processes. |
