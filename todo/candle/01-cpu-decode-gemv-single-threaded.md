# CPU decode GEMV (m=1 matmul) does not scale with thread count

- **Repo:** `sarah-ek/gemm`, reached through `huggingface/candle`
- **Crates:** `gemm-common` 0.19.0 (current, newest published 2025-11-14), `candle-core` 0.11.0
- **candle rev checked:** `12bb5d1` (upstream master, 2026-10-14)
- **Status:** already reported upstream. See "Existing reports" below; this file is the
  measurement to add to them, not a new issue.
- **Impact:** batch-1 CPU inference, most severely MoE

## Already reported, do not refile

| where | state | since |
|---|---|---|
| `huggingface/candle#1103` "Llama example uses ~10% of the CPU" | open | 2023 |
| `sarah-ek/gemm#20` "Candle example uses 10% of CPU when fma is active for x86" | open | 2023-10-16 |

Both describe the same symptom: a decode-shaped matmul occupying about one core out of many.
#20 speculates that gemm bounds its threads "for some reason that might have to do with the
stack". That guess is ruled out by the `user/real` measurement below - there is no thread
pool activity at all at `m=1`.

The fix is also already written: `sarah-ek/gemm` PR #41. This file is the aarch64
confirmation to add to those threads, not a new issue.


## What

`(1, K) x (K, N)`, the shape of a decode step, gets no benefit from extra threads at any size.
Larger `m` scales normally, so this is specific to the GEMV shape.

## Measurement

`Tensor::matmul`, f32, one NUMA node, governor `performance` (3.4 GHz), `RAYON_NUM_THREADS` set,
eager `Tensor::from_vec` inputs, 50 iterations after warmup:

| shape | 1 thr | 8 thr | 32 thr | scaling |
|---|---|---|---|---|
| m=1, n=5632, k=2048 (46 MB) | 14.4 GB/s | 14.3 | 14.4 | **1.00x** |
| m=1, n=512, k=256 (0.5 MB) | 25.6 GB/s | 26.0 | 25.9 | **1.01x** |
| m=64, n=5632, k=2048 | 1.1 GB/s | 1.1 | 2.8 | 2.5x |
| m=512, n=5632, k=2048 | 0.2 GB/s | 1.5 | 4.2 | 21x |

Reproduced across 3 runs (13.8-14.3 GB/s at both 1 and 32 threads).

The 0.5 MB row is the load-bearing one: cache-resident, so this is not a bandwidth limit. One
thread already extracts ~26 GB/s from L2 and 32 threads do no better. `m=512` scaling 21x
shows gemm parallelises correctly once `m` is large enough.

## Two hypotheses ruled out

**Not the threading threshold.** `gemm_common::DEFAULT_THREADING_THRESHOLD` is
`48 * 48 * 256 = 589824`, and `n_threads` is forced to 1 when `(m * n_chunk) * k_chunk` falls
under it. Since `m=1` divides that product by `MR`, this looked like the cause. It is not -
setting the threshold to 0 at runtime changes nothing:

| threads | default | threshold=0 |
|---|---|---|
| 1 | 14.2 | 14.3 |
| 8 | 13.5 | 13.4 |
| 32 | 14.3 | 14.1 |

That is consistent with the real cause: the gemv path never reaches the parallel branch, so the
gate governing it is irrelevant either way.

## What is actually happening

The work is **entirely single-threaded**. `user / real` for `m=1` alone in one process, no
other shape running:

| threads | real | user | user/real |
|---|---|---|---|
| 1 | 0.184 s | 0.167 s | 0.9x |
| 8 | 0.187 s | 0.182 s | 1.0x |
| 32 | 0.186 s | 0.182 s | 1.0x |

So the gemv kernels ignore the `parallelism` argument outright. That is what
`sarah-ek/gemm` PR #41 states, and this measurement confirms it on aarch64.

A caveat on getting this number: an earlier measurement in this file reported ~9x user/real
and concluded the threads were busy. That was wrong - the process also ran an `m=64` shape,
which does parallelise, and it dominated the CPU time. Measuring `m=1` in isolation is what
exposes it.

## Already fixed, pending

`sarah-ek/gemm` PR #41 "feat: parallelize gemv" (drbh, 2026-06-04) implements exactly the
direction below, for both the colmajor (k-split) and rowmajor (m-split) gemv kernels. It is
mergeable and open.

We commented on it with the aarch64 confirmation and the two corrections:
<https://github.com/sarah-quinones/gemm/pull/41#issuecomment-6233559624> (2026-10-17).

The repository is worth watching rather than filing against: **0 commits on master since
2025-11-14** while carrying 13 open PRs, so fixes appear to be queuing rather than landing. The
project has since moved to `sarah-quinones/gemm`.



## Are there alternatives inside candle?

No, and that is why this matters. `cpu_backend/mod.rs` dispatches over exactly three backends:

```rust
#[cfg(all(not(feature = "mkl"), not(feature = "accelerate")))]   // gemm  (the default)
#[cfg(feature = "accelerate")]                                   // Apple's Accelerate
#[cfg(feature = "mkl")]                                          // Intel MKL
```

`accelerate` is Apple frameworks, so macOS and iOS only. `mkl` comes from `intel-mkl-src`, an
x86 binary redistribution last published 2022-09-13 whose features are all
`mkl-{static,dynamic}-{lp64,ilp64}-{iomp,seq}` - Intel OpenMP, no ARM64 build. There is no
ggml, oneDNN, OpenBLAS or cblas path in candle.

So on aarch64 Linux, gemm is the only matmul candle can offer, and the m=1 serial path is
forced. This is also the concrete reason mistral.rs needed its own decode GEMV rather than a
feature flag.

Worth contrasting: the reference oracle this work is measured against, vLLM, goes through
PyTorch's CPU backend, which on aarch64 dispatches to oneDNN and parallelises GEMV properly.
That is a large part of why it reaches usable decode throughput on CPU where candle-as-is does
not.

## Impact

Measured in mistral.rs on Qwen3-4B bf16, decode at depth 512, one node:

| build | decode tok/s |
|---|---|
| candle as-is | 0.6 |
| with an output-parallel GEMV | 4.9 |

Qwen3.5-9B bf16: 0.3 to 2.1 tok/s. This single kernel is most of the reason CPU decode in
mistral.rs was unusable. The MoE case is filed separately as `03-aarch64-packed-moe-gemv-serial.md`.

## Suggested direction

Partition `n` across threads for `m == 1`, one output row per thread, each walking `k` in the
original order. That is what ggml does and what the local workaround does.

The ordering is the catch and the reason it is not a one-liner: partitioning over output rows is
bit-identical to the serial result, partitioning over a row's `k` reduction is not. ggml's row
chunking is the reference and its correctness argument transfers.

## Correction to an earlier draft of this file

An earlier version claimed the matmul "is simply not being spread across cores" and ran at "13%
of what the hardware offers". Both are wrong. It *is* spread across cores (9x user/real), and
14.4 GB/s against this node's ~68 GB/s ceiling is ~21%. The earlier 16.9 GB/s figure was also
taken under `schedutil` at 1.0 GHz; see `../evidence-host.md`.
