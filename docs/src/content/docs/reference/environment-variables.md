---
title: Environment variables
description: Environment variables read by mistralrs at build time, runtime, or during installer automation.
---

User-facing environment variables read by `mistralrs`, its build scripts, or its installers. Standard Cargo build variables such as `OUT_DIR` and `TARGET` are omitted.

## Hugging Face

| Variable | Purpose |
|---|---|
| `HF_HOME` | Hugging Face home directory. Defaults to `$XDG_CACHE_HOME/huggingface` if `XDG_CACHE_HOME` is set, otherwise `~/.cache/huggingface`. |
| `HF_HUB_CACHE` | Hub model cache location. Defaults to the `hub` directory under `HF_HOME`; `--hf-cache <dir>` overrides it. |
| `HF_TOKEN` | Auth token for the default `cache` token source. Overrides any token saved by `hf auth login`. |
| `HF_HUB_TOKEN` | Auth token fallback when `HF_TOKEN` is unset or blank. |
| `HF_TOKEN_PATH` | Saved token file. Defaults to the `token` file under `HF_HOME`. Used when neither `HF_TOKEN` nor `HF_HUB_TOKEN` supplies a token. |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Set to `1`, `true`, `yes`, or `on` to disable automatic credential lookup for the default `cache` token source. Explicit `literal:`, `env:`, and `path:` token sources still apply. |
| `HF_HUB_OFFLINE` | Set to `1`, `true`, `yes`, or `on` to stop all Hugging Face Hub network calls. Files then come only from the local cache (`$HF_HUB_CACHE` or `$HF_HOME/hub`), and a missing file is an error. `mistralrs doctor` also skips its connectivity check. |
| `HF_XET_CACHE` | Xet cache location. Defaults to the `xet` directory under `HF_HOME`. Separate from the Hub model cache. |
| `HF_XET_HIGH_PERFORMANCE` | Set to `1` to enable Xet's high-performance transfer settings, allowing more concurrent work and larger buffers. Off by default; Xet downloads work without it. |

If `--token-source env:NAME` is used, mistral.rs reads the environment variable named by `NAME` as the token source. `--token-source none` sends no Hugging Face account token, even when credentials are saved or set in the environment.

For the offline workflow (pre-downloading models, local paths), see [run models](/guides/models/run-any-model/).

## Logging

| Variable | Purpose |
|---|---|
| `RUST_LOG` | Override the `tracing` log filter. Examples: `mistralrs_core=debug,tower_http=info`, `trace`. CLI users can usually use `-v` or `-vv` instead. |
| `MISTRALRS_DEBUG` | `MISTRALRS_DEBUG=1` enables extra debug-level engine tracing. |

## Quantization and loading

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_MMAP` | `MISTRALRS_NO_MMAP=1` loads safetensors without mmap. |
| `MISTRALRS_ISQ_SINGLETHREAD` | If set, runs [ISQ (in-situ quantization)](/reference/quantization-types/) single-threaded. |

## CPU runtime

| Variable | Purpose |
|---|---|
| `RAYON_NUM_THREADS` | Sets the default CPU worker count used by Candle and Rayon-backed CPU kernels unless a more specific variable is set. |
| `CANDLE_NUM_THREADS` | Sets Candle's CPU worker count. This overrides the fallback from `RAYON_NUM_THREADS` for Candle's own thread pools. |
| `CANDLE_CPU_MASK` | Linux only. Pins CPU worker threads to a cpulist such as `15-19` or `5-9,15-19`. If no explicit thread-count variable is set, the mask size also becomes the default worker count. |
| `CANDLE_CPU_AFFINITY` | Linux only. Set to `1` to try Candle's automatic high-capacity CPU affinity mask on heterogeneous CPUs. Default is off. |
| `CANDLE_BARRIER_POOL_SPIN_LIMIT` | Advanced CPU tuning. Overrides the spin count used by Candle's persistent barrier pool before worker threads park. |

See [CPU threads and affinity](/guides/perf/throughput-tuning/#cpu-threads-and-affinity) for examples.

## Sandbox

| Variable | Purpose |
|---|---|
| `MISTRALRS_SANDBOX` | Sets the [sandbox](/reference/sandbox/) mode: `auto`, `on`, or `off`. Applies only when the CLI flag or TOML key leaves the mode at `auto`. |
| `MISTRALRS_MAX_OUTPUT_BYTES` | Largest output file, in bytes, that code execution or the shell tool returns to the client. Default 256 MiB. A larger file comes back as an entry with an error instead of its contents. |

## Server and UI

| Variable | Purpose |
|---|---|
| `MCP_CONFIG_PATH` | [MCP (Model Context Protocol)](/guides/agents/connect-mcp-server/) client configuration path used when `--mcp-config` is not passed. |
| `KEEP_ALIVE_INTERVAL` | SSE (Server-Sent Events) keep-alive interval in milliseconds. Falls back to the default if missing or invalid. |
| `MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING` | Set to `1`, `true`, `yes`, or `on` to allow loading and unloading LoRA adapters over HTTP. Off by default, and the model must be started with dynamic LoRA. See [LoRA adapters](/guides/customize/lora-adapters/#enable-http-mutation). |
| `MISTRALRS_LORA_ADAPTER_ROOT` | Only adapter directories inside this path can be loaded at runtime, and relative paths resolve inside it. Set it whenever runtime LoRA updating is on in production. |
| `XDG_CACHE_HOME` | Base cache directory. The UI uses `$XDG_CACHE_HOME/mistralrs`; Hugging Face uses `$XDG_CACHE_HOME/huggingface` when `HF_HOME` is unset. |
| `HOME` | Fallback for web UI cache path when `XDG_CACHE_HOME` is not set. |

## CUDA acceleration

| Variable | Purpose |
|---|---|
| `MISTRALRS_CUDA_GRAPHS` | CUDA graphs are on by default. Set to `0`, `false`, `no`, or `off` to turn them off. See [CUDA graphs](/guides/perf/paged-attention/#cuda-graphs). |
| `MISTRALRS_DFLASH_ADAPTIVE` | How a [DFlash](/guides/perf/speculative-decoding/#dflash-and-dflash-2) drafter on CUDA picks its draft depth when `--mtp-n-predict` isn't set. By default it drafts the full depth while up to 8 requests are running, and at most 3 tokens beyond that. Set to `0` or `false` to pick the depth from measured throughput instead. |
| `MISTRALRS_DFLASH_ISQ` | Quantization for DFlash drafter weights, such as `q4k` or `q6k`, or `none` for BF16. Defaults to the target model's ISQ type. |
| `MISTRALRS_FLASHINFER_DECODE` | FlashInfer decode kernels are on by default. Set to `0`, `false`, `no`, or `off` to turn them off for compatibility troubleshooting. |
| `MISTRALRS_GDN_DECODE_KERNEL` | Forces a GDN decode kernel. Default `auto`. See [GDN kernel overrides](#gdn-kernel-overrides). |
| `MISTRALRS_GDN_PREFILL_KERNEL` | Forces a GDN prefill kernel. Default `auto`. See [GDN kernel overrides](#gdn-kernel-overrides). |
| `MISTRALRS_FP8_SM90_PROVIDER` | Kernel for blockwise FP8 weights on Hopper: `deepgemm` (default, also `auto`) or `cutlass`. DeepGEMM needs a Linux build with CUDA 12.8 or newer, and falls back to CUTLASS with a warning if it can't start. |
| `MISTRALRS_DEEPGEMM_CACHE_DIR` | Where DeepGEMM saves the kernels it compiles, so later starts skip compiling. Defaults to `$XDG_CACHE_HOME/mistralrs/deepgemm-sm90`, or `~/.cache/mistralrs/deepgemm-sm90`. Must be a directory your user owns. |
| `MISTRALRS_DEEPGEMM_NVCC` | `nvcc` used to compile DeepGEMM kernels that aren't cached yet. Defaults to `$CUDA_HOME/bin/nvcc`, then `nvcc` on `PATH`. |
| `MISTRALRS_NO_MLA` | Set to `1` to turn off MLA (multi-head latent attention) acceleration for DeepSeek V2 and V3, for compatibility troubleshooting. |
| `MISTRALRS_GGUF_AFFINE_BACKEND` | Set to `on` to speed up GGUF matmuls at batch sizes of 8 or more. Off by default because it keeps a second copy of the quantized weights, which takes memory from the KV cache. Worth enabling for high-concurrency serving. |
| `MISTRALRS_MOE_BACKEND` | Forces the kernel used for MoE experts. Unset, mistral.rs picks the fastest one the model and GPU support. See [MoE expert backends](#moe-expert-backends). |
| `MISTRALRS_CUTILE_TUNE` | Whether `cutile` builds benchmark kernel settings for your GPU during warmup. Default `cached`. See [cuTile tuning](#cutile-tuning). |
| `MISTRALRS_CUTILE_TUNE_CACHE` | Where cuTile tuning results are saved, one file per kernel, shape, and GPU model. Defaults to `$XDG_CACHE_HOME/mistralrs/cutile_tune`, or `~/.cache/mistralrs/cutile_tune`. |
| `MISTRALRS_CUTILE_TUNE_BUDGET_SECS` | Total seconds cuTile tuning may spend benchmarking per process. Default 300. Kernels left untuned when it runs out use built-in settings. |
| `CUDA_TOOLKIT_PATH` | CUDA toolkit root used by cuTile for headers, `tileiras`, and Tile IR bytecode compatibility. Recommended when multiple CUDA toolkits are installed. |
| `CUTILE_TILEIRAS_PATH` | Path to a specific `tileiras` executable. Takes precedence over `CUDA_TOOLKIT_PATH` and automatic discovery. |
| `CUTILE_SETUP_DIAGNOSTICS` | Set to `1` to print cuTile toolkit and `tileiras` discovery diagnostics. |
| `CUTILE_JIT_LOG` | Set to `1` to print cuTile JIT and persistent-cache diagnostics. |
| `CUTILE_JIT_TIMING` | If set, prints per-stage cuTile compilation timings and bounds-check placement counts. |
| `CUTILE_BYTECODE_VERSION` | Advanced override for the emitted Tile IR bytecode version, such as `13.2`. Normally inferred from the selected toolkit or `tileiras`. |

### GDN kernel overrides

GDN (Gated DeltaNet) layers are the recurrent layers in hybrid models such as Qwen3-Next and Qwen3.5. mistral.rs picks their CUDA kernels automatically. These overrides are for benchmarking and troubleshooting. Forcing a kernel that can't run the model or GPU is an error.

`MISTRALRS_GDN_DECODE_KERNEL` accepts `auto` (default), `baseline`, `cooperative`, `pipelined`, `vmajor4`, or `vmajor32`.

`MISTRALRS_GDN_PREFILL_KERNEL` accepts `auto` (default) or one of:

- `vmajor1`, `vmajor2`, `vmajor4`, `vmajor8`, `legacy-chunked`
- `mma`: tensor-core prefill. Needs compute capability 8.0 or newer, BF16 inputs, and FP32 state with head dim 128.
- `mma-fused`: `mma` with input normalization and gates fused in. `auto` can pick it for single prompts of 64 or more tokens on compute capability 12.x GPUs.
- `cutile`: needs a `cutile` build and head dim 128.
- `flashinfer-sm90`: Hopper builds only.

### MoE expert backends

`MISTRALRS_MOE_BACKEND` accepts:

- `fused` (also `native`, `legacy`, `wmma`): fused CUDA kernels. Needs unquantized F16 or BF16 experts.
- `cutile`: cuTile grouped GEMM. Needs a `cutile` build and unquantized BF16 experts.
- `cutlass`: CUTLASS grouped GEMM. Needs unquantized BF16 experts on CUDA.
- `fast`: gather-based kernels. Works on CUDA, Metal, and CPU, including quantized and ISQ experts.

Forcing a backend the model or GPU can't use is an error. On `cutile` builds, checkpoints whose experts use 128x128 block-FP8 scales get a cuTile FP8 kernel by default; any value other than `cutile` turns that off.

### cuTile tuning

`MISTRALRS_CUTILE_TUNE` controls the autotuner for cuTile kernels, including MoE experts, FP8, MXFP8, and NVFP4 GEMMs, and routed LoRA:

- `cached` (default): use saved results that match this GPU and kernel version. Kernels without one use built-in settings, and nothing is benchmarked.
- `auto`: also benchmark kernels that have no saved result, and save the winner. The first start takes longer, up to `MISTRALRS_CUTILE_TUNE_BUDGET_SECS`.
- `force`: benchmark every kernel again, even if a result is saved.
- `off`: always use built-in settings.

## Metal acceleration

| Variable | Purpose |
|---|---|
| `MISTRALRS_METAL_PAGED_LAYOUT` | Paged KV cache layout on Metal: `hnd` (default; `auto` is the same) or `legacy`. `legacy` keeps paging on. Use only to isolate a suspected layout bug. |

## Multi-GPU and multi-node

| Variable | Purpose |
|---|---|
| `MISTRALRS_NO_NCCL` | Set to `1` to turn off NCCL. On a binary built with NCCL, this is how you use layer mapping (`-n`/`--device-layers`, which NCCL ignores) or the ring backend. |
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total NCCL tensor-parallel size across all nodes. Setting it turns on multi-node NCCL. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | NCCL tensor-parallel size each node contributes. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Set on the head node: number of worker nodes. |
| `MISTRALRS_MN_HEAD_PORT` | Set on the head node: listening port for worker connections. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Set on worker nodes: address of the head node. |
| `MISTRALRS_MN_WORKER_ID` | Set on worker nodes: worker index (0-based). |
| `RING_CONFIG` | Path to the ring backend JSON config. Setting it selects the ring backend when built with the `ring` feature. If the binary also has `nccl`, set `MISTRALRS_NO_NCCL=1` as well. |

See the [distributed inference guide](/guides/perf/distributed-inference/) for use.

## Build-time

These are read by build scripts, not at runtime.

| Variable | Purpose |
|---|---|
| `MISTRALRS_METAL_PRECOMPILE` | Set to `0`, `false`, `no`, or `off` to skip precompiling Metal shaders at build time. They then compile at runtime, the first time each one loads. |
| `MISTRALRS_METAL_PLATFORMS` | Which Apple platforms to precompile Metal shaders for: comma-separated `macos`, `ios`, `tvos`, or `all` (default). For local macOS development, use `macos`. |
| `CUDA_NVCC_FLAGS` | Extra compiler options passed to CUDA builds. |
| `MISTRALRS_GIT_REVISION` | Git revision embedded in the binary by the build script. |

## Installer automation

For normal setup, use the [guided installer or its command-line options](/reference/installer/). These variables are for scripted installs, such as in CI or containers. Unless a variable says otherwise, installs use the latest stable release.

| Variable | Purpose |
|---|---|
| `MISTRALRS_INSTALL_TAG` | Selects a release tag (e.g. `v0.9.5`) for either a prebuilt or source install. Equivalent to `--release` on macOS/Linux or `-Release` on Windows. |
| `MISTRALRS_INSTALL_FROM_SOURCE` | Set to `1` to build from source. It builds `master` unless `MISTRALRS_INSTALL_TAG` is set. To build the latest stable release from source, pass `--from-source` or `-FromSource` instead. |
| `MISTRALRS_INSTALL_NCCL` | Set to `1` to build CUDA source installs with the `nccl` feature even if NCCL isn't detected. It doesn't install NCCL, which the linker must still find. `--no-nccl` / `-NoNccl` takes precedence. |
| `MISTRALRS_INSTALL_NO_NCCL` | Set to `1` to build without the `nccl` feature. |
| `MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH` | Set to `1` to let a source build continue when the local `nvcc` is newer than the CUDA version your NVIDIA driver supports. |
| `MISTRALRS_INSTALL_YES` | Set to `1` to answer yes to every installer prompt, for non-interactive installs. `mistralrs update` uses it. |
| `MISTRALRS_INSTALL_IGNORE_FFMPEG` | Set to `1` to skip the FFmpeg step, even with `--with-ffmpeg` / `-WithFFmpeg`. FFmpeg isn't installed by default. |

## Internal

Not intended for direct use.

| Variable | Purpose |
|---|---|
| `__MISTRALRS_DAEMON_INTERNAL` | Set by the engine on spawned worker processes. |
