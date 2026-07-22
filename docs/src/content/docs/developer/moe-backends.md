---
title: cuTile setup
description: Install the optional cuTile runtime tool for supported NVIDIA GPUs.
---

Supported CUDA builds can use cuTile acceleration for MoE and routed LoRA workloads. The installer
selects a cuTile-capable binary automatically when one matches the GPU and driver. NVIDIA's
`tileiras` tool is installed separately. mistral.rs checks it automatically and continues without
cuTile when the requirements are not met.

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
package components on the same major/minor release. Put the executable on `PATH` or set
`CUTILE_TILEIRAS_PATH` to it. Release archives do not redistribute `tileiras`.

Run `mistralrs doctor` to check cuTile availability for every detected GPU. See NVIDIA's
[cuTile installation guide](https://docs.nvidia.com/cuda/cutile-python/quickstart.html).

## Requirements

- Ampere and Ada require CUDA 13.2 or newer.
- Hopper requires CUDA 13.3 or newer.
- Blackwell requires CUDA 13.2 or newer.
- The `tileiras` installation must support the active GPU.
- The mistral.rs binary must include the `cutile` feature.

`CUTILE_TILEIRAS_PATH` selects a specific `tileiras` binary instead of resolving it from `PATH`.

See also: [environment variables](/mistral.rs/reference/environment-variables/),
[cargo features](/mistral.rs/reference/cargo-features/).
