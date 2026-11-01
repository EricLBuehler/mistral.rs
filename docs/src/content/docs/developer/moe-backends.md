---
title: cuTile setup
description: Install NVIDIA's tileiras tool so supported NVIDIA GPUs can use cuTile kernels.
---

cuTile is NVIDIA's tile-based way of writing CUDA kernels. Some mistral.rs kernels are written in cuTile, and NVIDIA's `tileiras` compiler turns them into code for your exact GPU at runtime.

With cuTile available, mistral.rs uses faster kernels for:

- Mixture-of-experts (MoE) layers.
- FP8, MXFP8, and NVFP4 quantized linear layers.
- Routed LoRA.

Without cuTile, these layers use CUDA or CUTLASS kernels instead. For NVFP4, cuTile on Blackwell runs W4A4 checkpoints (4-bit weights and activations) on FP4 tensor cores. Without cuTile, NVFP4 uses slower kernels that keep activations in BF16 or F16, on compute capability 8.0 and newer.

To use cuTile you need a mistral.rs binary built with the `cutile` feature, and a separate install of `tileiras`. The installer picks a cuTile build automatically when your GPU and driver support one, but release archives and Docker images don't include `tileiras`.

## Requirements

| GPU | Compute capability | CUDA toolkit used to build mistral.rs |
|---|---|---|
| Ampere, Ada | 8.x | 13.2 or newer |
| Hopper | 9.0 | 13.3 or newer |
| Blackwell | 10.x, 12.x | 13.2 or newer. The block-scaled NVFP4 and MXFP8 kernels need 13.3 or newer. |

For a prebuilt binary, the toolkit is the artifact's CUDA lane. See [hardware support](/reference/hardware-support/#cuda-artifacts) for the published lanes.

You also need:

- A `tileiras` that is version 13.1 or newer and supports your GPU. The Blackwell block-scaled kernels need `tileiras` 13.3 or newer.
- For source builds, `libclang`, because cuTile generates CUDA bindings during the build.

## Install tileiras

For Ampere, Ada, and Blackwell, install NVIDIA's cuTile package:

```bash
python3 -m pip install --upgrade "cuda-tile[tileiras]"
```

Hopper, and [NVFP4 inference](/reference/quantization-types/#nvfp4) on Blackwell, need the CUDA 13.3 or newer toolkit components:

```bash
python3 -m pip install --upgrade "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3"
```

The pip packages don't put `tileiras` on `PATH`. Point mistral.rs at the installed binary, and add the same line to the shell profile or service environment that starts the server:

```bash
export CUTILE_TILEIRAS_PATH="$(python3 -c 'import nvidia.cu13.bin as b; print(next(iter(b.__path__)))')/tileiras"
```

A system CUDA install that includes a compatible `tileiras` works too. Keep all NVIDIA CUDA package components on the same major.minor release.

cuTile looks for `tileiras` in this order:

1. `CUTILE_TILEIRAS_PATH`, the path to the executable.
2. The `bin` directory of `CUDA_TOOLKIT_PATH`, the toolkit root.
3. Standard CUDA 13.3 and 13.2 installs.
4. `PATH`.

Setting `CUDA_TOOLKIT_PATH` makes source builds reproducible on machines with several toolkits.

### CUDA 13.4

cuTile's standard search doesn't include CUDA 13.4, so on a machine that also has an older toolkit it finds the older one first. To use 13.4:

- Set `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.4` and put `/usr/local/cuda-13.4/bin` first on `PATH`. The [source build instructions](/developer/from-source/#clone-and-build) list the full set of exports.
- If you set `CUTILE_TILEIRAS_PATH`, point it at `/usr/local/cuda-13.4/bin/tileiras`.

### Check the install

Run `mistralrs doctor`. In a binary built with `cutile`, it reports for each GPU whether the build supports that GPU and whether `tileiras` can compile for it. NVIDIA's [cuTile installation guide](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) covers the NVIDIA side.

## Autotuning

cuTile kernels are compiled for the exact GPU in the machine, so the fastest launch settings (tile shapes and compiler options) are measured on that GPU too. Each kernel ships with a built-in default, called its policy config. Autotuning looks for a faster one.

`MISTRALRS_CUTILE_TUNE` controls it:

| Value | What happens |
|---|---|
| `cached` (default) | Use a saved result if one matches. Otherwise use the policy config. Never measures. |
| `auto` | Use a saved result if one matches. Otherwise measure during warmup and save the winner. |
| `force` | Ignore saved results, measure again, and save. |
| `off` | Ignore saved results and always use the policy config. |

What gets tuned:

- MoE expert GEMMs and FP8, MXFP8, and NVFP4 GEMMs, once per shape during warmup.
- Routed LoRA, once per route bucket the first time it's launched.

`MISTRALRS_CUTILE_TUNE_BUDGET_SECS` caps the total measuring time per process. The default is 300 seconds, and each bucket gets at most 20 seconds. When the budget runs out, the remaining shapes keep the policy config.

### Saved results

Results are saved in `MISTRALRS_CUTILE_TUNE_CACHE` if it's set, otherwise in `$XDG_CACHE_HOME/mistralrs/cutile_tune` or `~/.cache/mistralrs/cutile_tune`.

A saved result is used only if all of these match the current run:

- The kernel source.
- The GPU architecture.
- The `tileiras` build.
- The set of candidate configs.

A result that doesn't match is ignored, never adapted.

### Why tuning can't make things slower

A tuned config replaces the policy config only if it passes every check:

1. Its output must stay within 2% (relative) of the policy config's output.
2. It must be at least 3% faster than the policy config during the search.
3. It must win again, by the same margin, in a final paired A/B run against the policy config.

If any check fails, or the search errors out, the policy config stays.

See also: [environment variables](/reference/environment-variables/), [cargo features](/reference/cargo-features/).
