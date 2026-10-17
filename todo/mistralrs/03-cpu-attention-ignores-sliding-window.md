# CPU attention kernels ignore the sliding window

- **Repo:** mistral.rs
- **Component:** `mistralrs-core`, CPU attention backends
- **Status:** open, not started
- **Impact:** every sliding-window-attention model on CPU

## What

`SdpaParams` carries a `sliding_window`, and the CUDA flash-attention kernels use it. The CPU
kernels do not. Grepping `sliding` across `mistralrs-core/src/attention/backends/cpu/` finds
only two hits, both in test or benchmark scaffolding:

```
attention/backends/cpu/mod.rs:205:        sliding_window: None,
attention/backends/cpu/tests.rs:12:        sliding_window: None,
```

The CPU path instead always receives a materialized mask tensor and applies it per element:

`mistralrs-core/src/attention/mod.rs:284-293`

```rust
if let AttentionMask::Custom(mask_tensor) = mask {
    if q.device().is_cpu() {
        let q = q.transpose(1, 2)?;
        ...
        return run_flash_attn_cpu_for_dtype(&q, &k, &v, Some(mask_tensor), sdpa_params);
    }
    ...
}
```

and the per-element lookup happens in the decode inner loop, via `MaskInfo::value`:

`mistralrs-core/src/attention/backends/cpu/single_q.rs:334-339`

```rust
let mask_delta = ctx
    .mask
    ...
    .map(|mask| slope * mask.value(b_i, h_i, 0, kv_pos))
```

`mask.rs` resolves that by matching the mask's rank and computing a flat index, which is real
work in the innermost loop rather than a comparison against a precomputed bound.

## Impact

Two costs, both specific to sliding-window models on CPU:

1. **Mask memory.** `CausalMasker.make_causal_mask` builds the full `[q_len, kv_len]` mask for
   the window (`mistralrs-core/src/layers_masker.rs:194-228`). For a 513-token window over a
   32k context that is a mask two orders of magnitude larger than the window itself.
2. **Per-element work.** Every score in the decode loop pays a rank switch and an index
   computation before it can be masked.

Models affected today: gemma2, gemma3, qwen3, qwen3_moe, glm4, gpt_oss (the last two select
per-layer window sizes from `layer_types`), and any new SWA model including Kolibri-1.

Not benchmarked in isolation. The claim is read off the dispatch path; the per-element cost
follows from the call site but was not separately measured.

## Suggested direction

Pass the window into `CpuAttnCtx` and bounds-check arithmetically where the mask currently
does an index lookup, keeping the mask path for genuinely custom masks (bidirectional, vision).
`full.rs` and `single_q.rs` both construct `CpuAttnCtx`, so the change is contained to those
two constructors plus `mask.rs`.

Two things to be careful about:

- **This is a shared kernel.** It serves every architecture that uses CPU attention, so the
  gate has to be golden output per SWA architecture, not a single unit test. A wrong bound is
  silent: it produces plausible numbers on short sequences.
- **Decode already relies on the cache for correctness.** `make_causal_mask` returns
  `AttentionMask::None` when `tgt_len == 1`, which is only correct because `RotatingCache`
  drops out-of-window tokens itself (`mistralrs-core/src/kv_cache/rotating_cache.rs:253`).
  If the window ever becomes a kernel parameter instead, that invariant becomes load-bearing
  and should be asserted rather than assumed.

## Window semantics, for whoever implements this

The mask builder and the kernel must agree on whether `sliding_window` is a token count or a
left distance. `sliding_window_left` at `mistralrs-core/src/attention/mod.rs:70-72` converts
with `window - 1`, and `eager_attention_mask` (`:74-103`) treats the value as a count with
`key_idx <= query_pos - window`. The model this was found against, Kolibri-1, documents the
window as "512 preceding tokens plus the current token", which is a count of 513 and a left
distance of 512, and those two conventions differ by one. Gate on a prompt longer than the
window; a shorter one passes either way.