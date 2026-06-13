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
| cuTile | JIT-compiled grouped-GEMM kernels specialized for your exact GPU architecture and model shapes at load time. The fastest option where supported. | The `cutile` feature build (CUDA >= 13.1), an Ampere/Ada (sm_8x) or Blackwell+ (sm_10x/sm_12x) GPU (not Hopper), and the `tileiras` JIT assembler available at runtime. |
| CUTLASS | Ahead-of-time compiled grouped GEMMs. Runs on any GPU from Ampere onward, with any CUDA toolkit, from a plain `cuda` build. | Default for unquantized BF16 MoE models with gated SiLU or tanh-approx GeLU (NewGelu / GeluPytorchTanh) when cuTile is unavailable. |
| Fused (WMMA) | Hand-written CUDA kernels for small batches, where grouped GEMMs are the wrong tool. | Small-batch decode under CUTLASS (below 64 tokens), and prefill when neither backend above applies (including erf-based GeLU and other activations). |
| Gather | Generic implementation built on the quantized-layer machinery. | Quantized experts ([ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/), [UQFF (Universal Quantized File Format)](/mistral.rs/reference/uqff-format/), pre-quantized), Metal, and CPU. |

The ordering matters: cuTile outperforms CUTLASS, which substantially outperforms the fused
fallback for prefill. A build without the `cutile` feature still gets a strong MoE path through
CUTLASS - but enabling `cutile` on supported hardware is meaningfully faster, which is why the
installer adds it automatically when it detects CUDA >= 13.1 and a supported GPU.

## Selection and graceful degradation

Backend selection happens once per model load and never strands you on a broken
configuration:

- cuTile requires architecture support (`major == 8` covers Ampere and Ada; `major >= 10` covers
  Blackwell and newer; Hopper is excluded) **and** the
  `tileiras` JIT assembler, which ships with CUDA toolkits 13.1 and newer. Both are probed at
  runtime. If either probe fails - for example, a cutile-enabled binary deployed to a machine
  without the toolkit - selection quietly moves to CUTLASS and logs why.
- CUTLASS requires a build targeting compute capability 8.0 or newer. Below that, selection
  moves on.
- Under CUTLASS, batches below 64 tokens delegate to the fused kernels: the grouped-GEMM setup
  cost exceeds the work itself at small batch sizes. cuTile runs its grouped-GEMM path at all
  batch sizes.

## Overriding the choice

Set `MISTRALRS_MOE_BACKEND` to force a specific backend: `cutile`, `cutlass`, `fused` (also
accepted: `wmma`, `native`, `legacy`), or `fast`. This is intended for debugging and A/B
comparisons. Forcing a backend the build or hardware cannot support behaves in one of two ways:

- Unsupported by the build (e.g. `cutile` on a non-cutile build): falls back to automatic selection.
- Unsupported by the hardware (e.g. `cutlass` on a pre-Ampere build): fails at the first forward pass.

`CUTILE_TILEIRAS_PATH` points the cuTile JIT at a specific `tileiras` binary instead of
resolving it from `PATH`.

See also: [environment variables](/mistral.rs/reference/environment-variables/),
[cargo features](/mistral.rs/reference/cargo-features/).
