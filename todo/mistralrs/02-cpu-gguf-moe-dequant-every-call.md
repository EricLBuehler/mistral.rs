# CPU GGUF MoE dequantizes every expert on every forward

- **Repo:** mistral.rs
- **Component:** `mistralrs-quant`, GGUF indexed MoE forward
- **Status:** open, not started
- **Impact:** every CPU MoE run from a GGUF checkpoint

## What

`qtensor_indexed_moe_forward` falls back to dequantizing the whole expert stack per call:

`mistralrs-quant/src/gguf/cpu.rs:72-80`

```rust
let device = x.device();

// Dequantize all weights to f32
let weights = qtensor.dequantize(device)?;

let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

unquant.gather_forward(x, ids)
```

The module doc says the same thing: "This dequantizes the weights and delegates to
UnquantLinear's gather_forward."

There is a fast path attempted first (`:60-69`, `qtensor.indexed_gemv`), but that path is
serial on aarch64, so in practice this file runs one core. See the candle entry below.

## Measured consequence

`/mnt/models/unsloth-qwen3.6-35B-A3B-gguf` (Qwen3.6-35B-A3B, GGUF Q4_K_M, 21 GB) on the CPU
node, sampled from `/proc` during a decode:

```
178 threads alive
177 sleeping, 1 running
1.91 of 128 cores busy
```

The model was not measurable as a throughput number: the first benchmark configuration ran
over 30 minutes without completing a case, so it was killed. This is the single most blocking
defect found in this work and the reason the MoE path was not attempted before the dense one.

Read together with `../mistralrs/01-cpu-fp8-dequant-every-forward.md`, which is the same
defect on the safetensors fp8 path, this is one problem in two places: on CPU, quantized MoE
weights are re-expanded per forward and the consumer of the expanded weights is serial.

## Suggested direction

Two changes, and the second is worth more:

1. Do not dequantize the full stack. Dequantize the experts the current token actually routed
   to, or hold a bounded cache of already-dequantized experts.
2. Parallelize the consumer. `qtensor.indexed_gemv` already partitions over
   `(token, expert)` pairs, so there is a natural parallel axis, but the aarch64
   implementation it calls has no parallel constructs at all. This is
   `../candle/03-aarch64-packed-moe-gemv-serial.md` and is in candle, not here.

Note the two interact: if the dequant is cached, the parallel consumer is what determines
throughput; if the dequant is avoided, the fallback in this file is still serial. Doing only
one leaves most of the win on the table.

## Bit-exactness

Expert outputs are summed across the top-k selection, so any repartitioning changes
accumulation order and therefore logits. The `../cpu-decode-bandwidth` branch partitions over
output rows only, never over a single row's reduction, which keeps results bit-identical; the
same discipline applies here and is the reason that approach was chosen.

## Related

- `../mistralrs/01-cpu-fp8-dequant-every-forward.md`
- `../candle/03-aarch64-packed-moe-gemv-serial.md`
- `../candle/01-cpu-decode-gemv-single-threaded.md`