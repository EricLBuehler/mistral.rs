---
title: cuTile setup
description: Install the optional cuTile runtime tool for supported NVIDIA GPUs.
---

Supported CUDA builds can use cuTile acceleration for MoE and routed LoRA workloads. The installer
selects a cuTile-capable binary automatically when one matches the GPU and driver. NVIDIA's
`tileiras` tool is installed separately. mistral.rs checks it automatically and continues without
cuTile when the requirements are not met. Source builds use the workspace-pinned cuTile Rust 0.3.0
release.

## Install tileiras

For Ampere, Ada, and Blackwell, install NVIDIA's cuTile package:

```bash
python3 -m pip install --upgrade "cuda-tile[tileiras]"
```

Hopper requires the CUDA 13.3 or newer toolkit components:

```bash
python3 -m pip install --upgrade "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3"
```

The pip packages do not add `tileiras` to `PATH`. Point mistral.rs at the installed binary, and add
the same export to the shell profile or service environment that starts the server:

```bash
export CUTILE_TILEIRAS_PATH="$(python3 -c 'import nvidia.cu13.bin as b; print(next(iter(b.__path__)))')/tileiras"
```

A system CUDA installation containing a compatible `tileiras` works as well. Keep the NVIDIA CUDA
package components on the same major/minor release. Set `CUDA_TOOLKIT_PATH` to the toolkit root for
a reproducible source build, or set `CUTILE_TILEIRAS_PATH` to the executable directly. cuTile also
searches standard CUDA 13.3 and 13.2 installations before falling back to `PATH`. Release archives
do not redistribute `tileiras`.

Run `mistralrs doctor` to check cuTile availability for every detected GPU. See NVIDIA's
[cuTile installation guide](https://docs.nvidia.com/cuda/cutile-python/quickstart.html).

## Requirements

- Ampere and Ada require CUDA 13.2 or newer.
- Hopper requires CUDA 13.3 or newer.
- Blackwell requires CUDA 13.2 or newer for source builds. cuTile 0.3 can emit CUDA 13.1 Tile IR,
  but its published CUDA bindings require CUDA 13.2 or newer headers.
- The `tileiras` installation must support the active GPU.
- The mistral.rs binary must include the `cutile` feature.
- Source builds require `libclang` because cuTile generates CUDA bindings during the build.

`CUTILE_TILEIRAS_PATH` selects a specific `tileiras` binary and takes precedence over
`CUDA_TOOLKIT_PATH` and automatic discovery.

## Autotuning

cuTile kernels are JIT-compiled for the GPU in the machine, so their launch configs are measured
there too. The first time a model loads, the warmup step times candidate tile shapes and compiler
knobs for each MoE expert shape and FP8 GEMM shape, about a minute for a large MoE model, and
records the winners under `cutile_tune` in the mistral.rs cache directory. Later loads reuse a
record whose provenance matches: the kernel source, the GPU architecture, the `tileiras` build,
and the candidate set. A record that no longer matches is re-measured, never approximated. Routed
LoRA tunes each route bucket the first time it is launched and persists the result the same way.

Every candidate is checked against the built-in policy config's output before it is timed, and a
winner replaces the policy only when it is measurably faster, so tuning can only leave a machine
where it started or better. `MISTRALRS_CUTILE_TUNE=off` keeps the built-in policies,
`force` re-measures, and `MISTRALRS_CUTILE_TUNE_CACHE` moves the records directory.

See also: [environment variables](/reference/environment-variables/),
[cargo features](/reference/cargo-features/).
