# CPU decode GEMV runs at one core's bandwidth regardless of thread count

- **Repo:** huggingface/candle
- **Pinned rev:** `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0)
- **Status:** open, worked around locally
- **Impact:** every batch-1 CPU inference, most severely for MoE

## What

A decode step is `[1, K] x [K, N]`: one token against a weight matrix that must be streamed
in full. It is therefore purely bandwidth-bound, and the only thing that matters is how many
cores stream the weight concurrently.

Measured on the CPU node (see `../evidence-host.md`), decode-shaped matmul, `[1,2048] x
[2048,5632]` f32:

| threads | candle GB/s |
|---|---|
| 1 | 16.9 |
| 2 | 17.4 |
| 4 | 18.3 |
| 8 | 18.4 |
| 16 | 15.1 |
| 32 | 16.6 |

Flat. The same probe reads the same weight tensor with a plain column sum:

| threads | read GB/s |
|---|---|
| 1 | 13.0 |
| 32 | 13.2 |

Also flat, and at the same rate as the matmul. So the matmul is not slower than reading the
data, which means it is not doing avoidable arithmetic; it is simply not being spread across
cores. The node can stream at 122.7 GB/s with 32 threads (lower bound, see
`../evidence-host.md`), so this is roughly 13% of what the hardware offers.

The bf16 shape is worse still: 2.8 GB/s at every thread count, about 5x below the f32 shape.

## How it was found

Isolating the op from the model. `mistralrs-quant/examples/cpu_decode_gemv.rs` times the
matmul on its own and times a `sum` over the same tensor in the same process, which separates
"the kernel is slow" from "the kernel is not parallel".

## Impact

End to end, on Qwen3-4B bf16 through mistral.rs, decode at depth 512 on one node:

| build | decode tok/s |
|---|---|
| master | 0.6 |
| with a local output-parallel GEMV | 4.9 |

and on Qwen3.5-9B bf16, 0.3 to 2.1 tok/s. So this single kernel is most of the reason CPU
decode in mistral.rs was unusable. The MoE case is worse and is filed separately as
`03-aarch64-packed-moe-gemv-serial.md`.

## Likely cause

Not investigated inside candle beyond this measurement. The matmul reaches
`candle_core::cpu_backend` for dense f32 and the gemm crate's `Parallelism::Rayon` path, which
does take a thread count (`cpu_backend/mod.rs:1482-1487`). For a GEMV shape `m == 1` there is
no `m` dimension to tile over, so the parallelism has to come from the `n` dimension, and that
is the dimension the tiling most likely does not exploit at this shape. Reading the gemm crate
did not settle it.

## Suggested direction

Partition `n` across threads for the `m == 1` case, one output row per thread, each thread
walking `k` in the original order. That is what ggml does, and it is what the local workaround
does.

The ordering constraint matters and is the reason this is not a one-liner: partitioning over
output rows is bit-identical to the serial result, partitioning over a row's `k` reduction is
not. ggml's row chunking is the reference for getting this right, and its correctness argument
transfers directly.