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
stack". That guess can be ruled out, and the cause can be narrowed - see below.

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

This also rules out the stack-management guess in gemm#20 for this shape.

**Not idle threads.** `user / real` at 16 threads is 7.68s / 0.83s, about 9x, so rayon is
spawning and running work. The work simply does not shorten the critical path.

## Where it probably is

On aarch64, `gemm-common/src/gemm.rs`:

```rust
let do_pack_rhs = _requires_row_major_rhs || m > get_rhs_packing_threshold() * MR;
```

With a row-major rhs the first term holds regardless of `m`, so every decode GEMV packs the
whole weight matrix before any arithmetic. At `m=1` that pass dominates the runtime, and it
does not appear to be split across threads. That would account for both observations at once:
no scaling at any size, and 9x of busy threads that do not help.

**Not confirmed.** What would settle it: timing `pack_rhs` separately, or re-running with an
rhs already in the layout the kernel prefers.

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
