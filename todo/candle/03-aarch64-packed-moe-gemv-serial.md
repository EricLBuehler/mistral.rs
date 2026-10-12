# The aarch64 packed MoE GEMV is serial

- **Repo:** huggingface/candle
- **Pinned rev:** `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0)
- **Status:** open, blocks CPU MoE decode
- **Impact:** CPU MoE inference on aarch64

## What

`QTensor::indexed_gemv` has an aarch64 CPU implementation, which is the fast path that
avoids dequantizing the whole expert stack. It is a single core.

`candle-core/src/quantized/repack.rs` is 1062 lines and contains **zero** parallel constructs:

```
$ grep -cE 'rayon|par_iter|par_chunks|into_par|thread::|scope\(|spawn' \
    candle-core/src/quantized/repack.rs
0
```

The function `try_indexed_gemv` that the aarch64 path calls has zero of the same. There is an
x86_64 sibling (`repack_x86.rs`) and both are behind `#[cfg(target_arch = ...)]`, so this is
an aarch64-specific gap rather than a whole-platform one, but neither file parallelises.

The entry point is `QTensor::indexed_gemv` at `candle-core/src/quantized/mod.rs:1053`. The
aarch64 arm normalises shapes, widens bf16 to f32, and calls into `repack::try_indexed_gemv`,
which does the work in a plain loop.

## Measured

Qwen3.6-35B-A3B (GGUF Q4_K_M, 21 GB) run through mistral.rs on the CPU node, sampled from
`/proc` during decode:

```
178 threads alive
177 sleeping, 1 running
1.91 of 128 cores busy
```

The benchmark never completed a single case in 30 minutes, so there is no throughput number to
report. What is reported is the concurrency, which is unambiguous.

## Impact

This is the blocking defect for CPU MoE inference. Decode has an obvious parallel axis: the
per-expert dot products are independent, and the function already has `n_experts`, `batch` and
`topk` in hand. Not using it means a MoE decode runs at roughly the speed of a single-core
dense decode while carrying a much larger weight tensor.

## Suggested direction

Parallelize `try_indexed_gemv` over the `(token, expert)` pairs, or over expert rows, with
one task per pair. The loop already materializes `ids_v: Vec<u32>` per call (mod.rs:1076-1079),
so the work items are enumerable before the inner loop starts.

Two constraints:

- **Bit-exactness.** Expert outputs are summed across the top-k selection downstream, so the
  order of the per-expert contributions must not change. Partitioning over independent
  (token, expert) pairs is safe; reducing within a pair is not.
- **Stay inside the existing repack layout.** `repack.rs` exists to make each expert's rows
  contiguous for exactly this loop, and the aarch64 layout may differ from x86_64's. The
  partitioning should not disturb the row packing, only the order in which rows are visited.

## Related

- `../mistralrs/02-cpu-gguf-moe-dequant-every-call.md` is the other half: when this fast path
  is not taken, the caller dequantizes everything, per forward.
- `../candle/01-cpu-decode-gemv-single-threaded.md` is the dense analogue.