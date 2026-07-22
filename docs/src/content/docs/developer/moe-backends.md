---
title: MoE expert backends
description: How mistralrs picks the fastest mixture-of-experts kernel for your hardware, and how to override it.
---

MoE (Mixture of Experts) models spend most of their prefill time in the expert feed-forward
layers. mistralrs ships several interchangeable implementations of this computation and picks
the fastest one your machine can actually run, automatically, at model load. CUTLASS selection
emits a log line; the other backends are silent unless they fall back.

## The backends

| Backend | What it is | When it runs |
|---|---|---|
| cuTile | JIT-compiled grouped-GEMM kernels specialized for your exact GPU architecture and model shapes at load time. The fastest option where supported. | The `cutile` feature build, a supported CUDA/SM pair, and the `tileiras` JIT assembler available at runtime. |
| CUTLASS | Ahead-of-time compiled grouped GEMMs. Runs on any GPU from Ampere onward, with any CUDA toolkit, from a plain `cuda` build. | Default for unquantized BF16 MoE models with gated SiLU or tanh-approx GeLU (NewGelu / GeluPytorchTanh) when cuTile is unavailable. |
| Fused (WMMA) | Hand-written CUDA kernels for small batches, where grouped GEMMs are the wrong tool. | Small-batch decode under CUTLASS (below 64 tokens), and prefill when neither backend above applies (including erf-based GeLU and other activations). |
| Gather | Generic implementation built on the quantized-layer machinery. | Quantized experts ([ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/), [UQFF (Universal Quantized File Format)](/mistral.rs/reference/uqff-format/), pre-quantized), Metal, and CPU. |

The ordering matters: cuTile outperforms CUTLASS, which substantially outperforms the fused
fallback for prefill. A build without the `cutile` feature still gets a strong MoE path through
CUTLASS. The installer selects a cutile-enabled binary automatically for supported CUDA/SM pairs,
but NVIDIA's `tileiras` developer tool is installed separately.

## Install the cuTile runtime tool

Install NVIDIA's official `tileiras` package. For Ampere, Ada, and Blackwell, NVIDIA's cuTile
package supplies it:

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
package components on the same major/minor release. Put that executable on `PATH` or set
`CUTILE_TILEIRAS_PATH` to it. Release archives do not redistribute `tileiras`; without a compatible
installation, the cutile-enabled binary uses its native CUDA routed-LoRA and CUTLASS MoE fallbacks.
Run `mistralrs doctor` to probe the runtime tool and target support for every detected GPU. See NVIDIA's
[cuTile installation guide](https://docs.nvidia.com/cuda/cutile-python/quickstart.html).

## Selection and graceful degradation

Backend selection happens once per model load and validates the available runtime tooling before
choosing an optimized backend:

- cuTile requires a supported build CUDA and GPU pair: Ampere/Ada (`sm_8x`) needs CUDA >= 13.2,
  Hopper (`sm90`) needs CUDA >= 13.3, and Blackwell+ (`sm_10x`/`sm_12x`) needs CUDA >= 13.2.
  It also needs the `tileiras` JIT assembler at runtime, and that assembler must list the active
  GPU target. If either probe fails, selection quietly moves to CUTLASS and logs why.
- CUTLASS requires a build targeting compute capability 8.0 or newer. Below that, selection
  moves on.
- Under CUTLASS, batches below 64 tokens delegate to the fused kernels: the grouped-GEMM setup
  cost exceeds the work itself at small batch sizes. cuTile runs its grouped-GEMM path at all
  batch sizes.

## Overriding the choice

Set `MISTRALRS_MOE_BACKEND` to force a specific backend: `cutile`, `cutlass`, `fused` (also
accepted: `wmma`, `native`, `legacy`), or `fast`. This is intended for debugging and A/B
comparisons. Forcing a backend the build or model cannot support behaves in one of two ways:

- Unsupported by the build (e.g. `cutile` on a non-cutile build): falls back to automatic selection.
- Compiled but ineligible because of the device, dtype, activation, weight format, or JIT tooling:
  fails during model loading with an error naming the unmet requirement.

`CUTILE_TILEIRAS_PATH` points the cuTile JIT at a specific `tileiras` binary instead of
resolving it from `PATH`.

See also: [environment variables](/mistral.rs/reference/environment-variables/),
[cargo features](/mistral.rs/reference/cargo-features/).
