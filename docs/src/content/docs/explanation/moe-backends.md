---
title: MoE expert backends
description: How mistralrs picks the fastest mixture-of-experts kernel for your hardware, and how to override it.
sidebar:
  order: 12
---

Mixture-of-experts (MoE) models spend most of their prefill time in the expert feed-forward
layers. mistralrs ships several interchangeable implementations of this computation and picks
the fastest one your machine can actually run, automatically, at model load. One log line
reports which backend won.

## The backends

| Backend | What it is | When it runs |
|---|---|---|
| cuTile | JIT-compiled grouped-GEMM kernels specialized for your exact GPU architecture and model shapes at load time. The fastest option where supported. | Built with the `cutile` feature (CUDA >= 13.1), on Ampere or Blackwell GPUs, when the `tileiras` JIT assembler is available at runtime. |
| CUTLASS | Ahead-of-time compiled grouped GEMMs. Runs on any GPU from Ampere onward, with any CUDA toolkit, from a plain `cuda` build. | Default for unquantized BF16 MoE models with gated activations (GeLU and SiLU families) when cuTile is unavailable. |
| Fused (WMMA) | Hand-written CUDA kernels. Owns the decode path (single-token batches), where grouped GEMMs are the wrong tool regardless of implementation. | Decode steps for all CUDA backends, and prefill when neither backend above applies. |
| Gather | Generic implementation built on the quantized-layer machinery. | Quantized experts (ISQ, UQFF, pre-quantized), Metal, and CPU. |

The ordering matters: cuTile outperforms CUTLASS, which substantially outperforms the fused
fallback for prefill. A build without the `cutile` feature still gets a strong MoE path through
CUTLASS - but enabling `cutile` on supported hardware is meaningfully faster, which is why the
installer adds it automatically when it detects CUDA >= 13.1 and a supported GPU.

## Selection and graceful degradation

Backend selection happens once per model load and never strands you on a broken
configuration:

- cuTile requires architecture support (Ampere and Blackwell; not Hopper) **and** the
  `tileiras` JIT assembler, which ships with CUDA toolkits 13.1 and newer. Both are probed at
  runtime. If either probe fails - for example, a cutile-enabled binary deployed to a machine
  without the toolkit - selection quietly moves to CUTLASS and logs why.
- CUTLASS requires a build targeting compute capability 8.0 or newer. Below that, selection
  moves on.
- Decode always uses the fused kernels: below a small batch threshold the grouped-GEMM setup
  cost exceeds the work itself, so the prefill backends delegate.

## Overriding the choice

Set `MISTRALRS_MOE_BACKEND` to force a specific backend: `cutile`, `cutlass`, `fused` (also
accepted: `wmma`, `native`, `legacy`), or `fast`. This is intended for debugging and A/B
comparisons. Forcing a backend the build or hardware cannot support either falls back to
automatic selection (`cutile` on a non-cutile build) or fails at the first forward pass
(`cutlass` on a pre-Ampere build).

`CUTILE_TILEIRAS_PATH` points the cuTile JIT at a specific `tileiras` binary instead of
resolving it from `PATH`.

See also: [environment variables](/mistral.rs/reference/environment-variables/),
[cargo features](/mistral.rs/reference/cargo-features/).
