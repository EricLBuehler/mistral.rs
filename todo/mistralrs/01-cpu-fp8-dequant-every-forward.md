# CPU FP8 dequantizes the whole weight on every forward

- **Repo:** mistral.rs
- **Component:** `mistralrs-quant`, blockwise FP8 linear
- **Status:** open, not started
- **Impact:** any fp8 checkpoint on CPU, every decode step

## What

`BlockwiseFP8Linear::forward_raw` ends in a fallback that dequantizes the entire weight tensor
on every call, then runs an ordinary matmul on the result:

`mistralrs-quant/src/blockwise_fp8/mod.rs:447-454`

```rust
// Fallback: dequantize and use unquantized matmul
let weight = self.dequantize_w()?;
let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
    weight,
    self.bias.clone(),
)))?;
unquant.forward(x)
```

`dequantize_w` (`:287-294`) is a plain call to `ops::fp8_blockwise_dequantize` with no
caching:

```rust
fn dequantize_w(&self) -> Result<candle_core::Tensor> {
    ops::fp8_blockwise_dequantize(
        &self.weight,
        &self.weight_scale_inv,
        self.weight_block_size.to_vec(),
        self.dequant_dtype,
    )
}
```

There is no `OnceLock`, `Mutex` or cache field anywhere in the type. Every forward of every
layer re-expands the full weight matrix from fp8 to the dequant dtype.

## Why it does not show up on GPU

Every GPU path in the same function returns before reaching line 447: the cuTile
`CutileFp8W8A16` arm (`:297`), the DeepGEMM arm (`:340`), the CUTLASS activation-quantized
arm (`:381`), the TensorCore GEMV arm (`:389`), and the native blockwise GEMM (`:405`). All are
`#[cfg(feature = "cuda")]` or gated on `x.device().is_cuda()`.

So the fallback is dead code in a CUDA build and the hot path in a CPU build, which is exactly
the shape of a defect that survives testing on GPU hardware.

## Impact

For a checkpoint quantized with `weight_block_size: [128, 128]`, decode touches the top-6 of 384
experts per layer, so the working set per token is a small fraction of the model but the
*work* is not: each selected expert's weights are fully dequantized, used, and thrown away
before the next token. Dequantizing to bf16 also doubles the bytes touched, so the fallback
pays both the expansion cost and the doubled bandwidth on every call.

Not measured end to end here: the FP8 Kolibri-1 checkpoint was not run on CPU, because the
model is not implemented in mistral.rs yet. The claim about wasted work is read off the code
path, not from a benchmark.

## Suggested direction

Cache the dequantized weight rather than recomputing it. The shape of the problem:

- The cache must be bounded by bytes, not entries, and it must own its allocations. The
  sibling project `../vllm.cpp` implemented exactly this for the same model family
  (`kolibri1_dequant_cache`) and measured 2.52x on CPU decode; its lease object exists
  because a naive version returned a raw pointer that a concurrent eviction could free
  mid-GEMM.
- A lease, or an equivalent, is required if the cache can evict while a matmul is in flight.
  Holding the GIL-equivalent is not enough here; this is plain `Send`/`Sync` Rust.
- With a cache, the GEMV itself still needs to be parallel over output rows to reach memory
  bandwidth, otherwise the cached path becomes a bandwidth-starved serial dequant. The
  workaround in `../cpu-decode-bandwidth` (`mistralrs-quant/src/gemv_cpu.rs`) does that for
  the dense path.

An alternative that avoids the cache entirely is to keep the weights packed and dequantize
into registers inside a fused kernel. That is more work but has no memory ceiling. Worth
deciding before implementing, since the two have very different memory profiles and only one
of them suits a 78 GB model.

### The sibling cache is already CPU-proven, and its first version shipped a corruption

Worth reading before writing any of it. `../vllm.cpp`'s `kolibri1_dequant_cache.h` is
**device-free** and is called from the CPU forward (`kolibri1_forward.cpp:184`), and the
reland writeup (`docs/bench-evidence/kolibri1-dequant-cache-reland-20261007.md`) ran with
`-DVLLM_CPP_TENSTORRONT=OFF`. So the design is validated on the same class of machine.

It is also the **second** implementation. The first was reverted when the token gate went
red with 7 hard flips, and the failure is worth carrying over rather than rediscovering:

1. **Entries must own their bytes.** v1 stored each decoded block in a shared pool block held
   through a lease; the pool had no idea the block was still in use, re-handed the same
   address to a later tensor, and a live cache entry silently became another tensor's bytes
   (`POOL DOUBLE-HAND-OUT` / `CACHE MISMATCH`). The fix is an independent `Vec` per entry, so
   a cache hit cannot alias pool traffic by construction.
2. **The key needs a model generation.** v2 keyed on the weight's data pointers plus shape
   scalars. After an unload and reload the allocator hands back same-size buffers at the
   *same* addresses, so a reloaded model of the same geometry was served the previous
   incarnation's decoded bytes. Adding a generation counter bumped at every model load
   closes it. In Rust the analogue is keying on something the loader owns rather than on a
   pointer.
3. **Return a lease, not a raw pointer.** Unlocking between lookup and consume let a
   concurrent caller evict the entry and reuse its bytes before the matmul finished.
   Serializing the whole lookup-and-consume was rejected because it holds the lock across the
   expensive half.

Budget semantics worth copying: the default is **0, meaning disabled**, so it is opt-in and
the unquantized path stays the default; an entry larger than the whole budget is never
cached; and the budget counts an evicted entry as evicted, so a live lease can transiently
overshoot by at most one entry.

## Related

- `../mistralrs/02-cpu-gguf-moe-dequant-every-call.md` is the same defect on the GGUF MoE path.
- `../candle/01-cpu-decode-gemv-single-threaded.md` is the other half of the problem: even a
  cached dequant is slow if the matmul that consumes it runs on one core.