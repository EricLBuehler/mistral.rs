# CPU GGUF MoE dequantizes every expert on every forward

- **Repo:** mistral.rs
- **Component:** `mistralrs-quant`, GGUF indexed MoE forward
- **Status:** **the GGUF path is fixed and bit-exact** (see below). The blockwise-fp8 path is
  not: it is the one that costs 74 s per token (`todo/01`), and it needs the same treatment
  applied differently, because its experts arrive as separate per-expert tensors that are then
  stacked.
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

## Fixed for the GGUF path: dequantise only the routed experts

The quantised stack is one contiguous block of bytes with the expert axis outermost, so the
per-expert stride is `storage_size_in_bytes() / n_experts` and a slice of that many bytes
dequantises to exactly the values the whole stack's slice of the same expert dequantises to.
`QTensor::new` checks the shape against the block size, so a slice that cannot be represented
is rejected rather than silently mis-dequantised.

`PerExpertQuantizedLinear` (`mistralrs-quant/src/per_expert_quantized.rs`) resolves only the
experts a gather names, remaps the ids onto that compacted set, and runs the existing per-expert
gather. `gguf/cpu.rs` uses it in place of the whole-stack dequantise.

Two conditions, both load-bearing:

- **The 5D input form only.** The 3D form's whole-stack path is a matmul over the selected
  experts rather than the gather kernel, and a matmul and the gather are not the same
  arithmetic, so replacing it there would change the output rather than only what is
  dequantised. The 3D form is the CUDA-shaped input; the CPU path uses the 5D form.
- **Only where the gather kernel would have been used.** The whole-stack path falls back to a
  matmul when the shape is too small for the gather, and the scoped path has to agree with
  that, for the same reason.

The result is bit-identical to the whole-stack path, which is the acceptance criterion:
`gguf::cpu::tests::scoped_gather_matches_the_whole_stack_bit_exactly` compares the two
outputs element for element, and `moe_gather_cpu::tests` now also covers the per-expert
broadcast form against the fused one at `repeat > 1`, which was previously untested and which
this depends on.

**Not yet measured.** The speed is unmeasured, because it needs a model load and the box is
under a measurement window. What is established is that the scoped path resolves the experts a
token routes to rather than the whole stack, so the dequantised bytes per forward drop from the
whole model to the routed experts, and the peak anonymous memory with them.

## The fp8 path is the same defect in a different shape

`BlockwiseFP8Linear` stacks the per-expert fp8 tensors and then dequantises the whole stack,
which is where the 74 s per token comes from (`todo/01`). The fp8 checkpoint already stores
the experts as separate tensors, so the fix is the mirror image of the GGUF one: keep them
per-expert and use the per-expert gather, rather than stacking and dequantising.

That is the single change that would take the fp8 path from 74 s per token to something
I/O-shaped, close the 290x gap to the vllm.cpp CPU tier, and make the 32 GB box viable
(`todo/gaps/04`). It is the highest-value remaining perf item on this side.
