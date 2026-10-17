# CPU blockwise-FP8 stacked MoE does not run

- **Repo:** mistral.rs
- **Components:** `mistralrs-quant/src/blockwise_fp8/mod.rs`, `mistralrs-core/src/moe/experts/backends.rs`
- **Status:** **fixed on `cpu-decode-bandwidth`** by `56044c0e9`, all three. Kept as the record
  of what was wrong, since the fixes are part of the Kolibri-1 PR rather than a standalone one
  and a reviewer will want to know why `blockwise_fp8` changed.
- **Impact at time of writing:** every CPU MoE model loaded from an FP8 checkpoint

## Correction

An earlier revision of this file read "open, three independent hard failures". All three were
closed by one commit, and the file was not updated at the time.

| # | failure | fix |
|---|---|---|
| 1 | dequant kernel handed a rank-3 expert stack | expert axis folded before `fp8_blockwise_dequantize`, results stacked back (`blockwise_fp8/mod.rs`) |
| 2 | gather implemented only the flattened CUDA layout | non-`(rank 2, rank 3)` shapes delegated to `UnquantLinear::gather_forward_raw` rather than duplicated |
| 3 | non-LoRA stacked forward under-reshaped the down projection | `down_in` reshaped to `(batch, seq, top_k, inter)` before the gather (`backends.rs`) |

**Not re-verified end to end.** The fix is reasoned from the shapes and the three code sites;
the FP8 checkpoint has not been loaded since, because the only one available is 74 GB and the
host is shared. If this is being used as evidence for anything, run the forward once first.

## What

Three separate defects, each of which aborts the forward on CPU with a shape or dtype error.
They are not performance problems; there is no output at all.

### 1. The dequant kernel is handed a rank-3 stack

`BlockwiseFP8Linear::dequantize_w` passes `self.weight` straight to
`ops::fp8_blockwise_dequantize`, whose `CpuStorage` arm asserts the weight is rank 2:

```
if weight_l.dims().len() != 2 {
    candle_core::bail!("Expected weight to be rank 2");
}
```

`mistralrs-quant/src/blockwise_fp8/ops.rs:130`

Stacked experts are `[num_experts, out, in]`, so every FP8 MoE gather hits this:

```
Error: moe experts forward
Expected weight to be rank 2
```

### 2. The gather implements only the flattened CUDA layout

`BlockwiseFP8Linear::gather_forward_raw` opens with `indices.dims2()?` and branches on
`x.dim(1) == 1`. That is the layout `uses_flattened_gather` produces, which is
`device.is_cuda() || any projection is NVFP4`. On CPU it is handed
`(batch, seq, 1, 1, hidden)` with `(batch, seq, top_k)` indices and fails:

```
Error: moe experts forward
unexpected rank, expected: 2, got: 3 ([1, 37, 6])
```

`UnquantLinear::gather_forward_raw` in `mistralrs-quant/src/unquantized/mod.rs` already
implements every shape; the FP8 layer just does not delegate to it.

### 3. The non-LoRA stacked forward under-reshapes the down projection

`mistralrs-core/src/moe/experts/backends.rs`, `FastExpertsWeights::forward_gather`. The gate
and up gathers receive `xs` reshaped to rank 5, so the down gather needs `down_in` at rank 4:

```rust
self.fused_down_proj.gather_forward(&down_in, &indices)?
```

`forward_gather_lora`, twenty lines below, does exactly this reshape. Without it the down
projection receives a rank-3 input against rank-3 indices.

## Measured

Reached by adding `Kolibri1ForCausalLM` and pointing it at `/mnt/models/Aleph-Alpha/Kolibri-1`
(fp8, 74 GB, 384 experts, 50 layers). Each defect was hit in sequence on the same model.

## Fix

1. Fold the expert axis in `dequantize_w`: loop `0..num_experts`, dequantize each
   `weight.i(e)` / `weight_scale_inv.i(e)`, and `Tensor::stack` the results. Rank 2 is
   untouched.
2. In `gather_forward_raw`, when the shapes are not the flattened CUDA layout, dequantize and
   delegate to `UnquantLinear::gather_forward_raw` rather than growing a second dispatch.
3. Reshape `down_in` to `(batch, seq, top_k, inter)` before the down gather, matching the LoRA
   path.

## Related

- `../mistralrs/02-cpu-gguf-moe-dequant-every-call.md` is the same consumer once this works:
  the fallback dequantizes every expert on every forward.
- `../mistralrs/01-cpu-fp8-dequant-every-forward.md` covers the per-forward dequant on the
  safetensors path.

## Note on scope

Found while porting a model to CPU. All three are pre-existing and independent of that work,
and each blocks any CPU MoE model from an FP8 checkpoint, so they are worth filing separately
from the performance issues in this directory.