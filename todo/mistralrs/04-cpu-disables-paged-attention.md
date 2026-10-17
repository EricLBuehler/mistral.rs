# PagedAttention is disabled whenever CPU is in the device map

- **Repo:** mistral.rs
- **Component:** `mistralrs-core`, normal pipeline load
- **Status:** open, design decision
- **Impact:** CPU serving, not CPU inference

## What

`mistralrs-core/src/pipeline/normal.rs:961-967`

```rust
// TODO: PagedAttention is not supported with CPU for now.
// This check is not really necessary because `get_device_layers` should prevent it.
let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
if mapping_uses_cpu && paged_attn_config.is_some() {
    warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
    paged_attn_config = None;
}
```

The comment already says what the author thought of it. Two things make the current behavior
worse than the comment suggests:

1. It triggers on `any()` CPU in the map, not on a mix. A pure-CPU run with `--paged-attn`
   also has it disabled, and the warning text ("contains a mix of GPU and CPU") is then
   misleading.
2. It is a silent capability downgrade. A user who passes `--paged-attn` on CPU gets a warning
   and a different KV cache implementation, with no error, and the two have different memory
   behavior.

## Impact

Without paged attention on CPU, a CPU server loses:

- continuous batching, so throughput is bounded by one sequence per slot,
- prefix caching, because there is no block table to key on,
- bounded KV memory, since the non-paged `NormalCache` grows by `CACHE_GROW_SIZE` chunks and
  is not laid out for eviction.

This is the main thing between "CPU inference works" and "CPU is a serving target". It was not
attempted in this work.

## Why it is filed as a decision rather than a defect

Two reasonable directions, with different costs:

- **Implement a CPU paged-attention kernel.** Real work: the block table, the gather/scatter
  over blocks, and a CPU kernel that consumes them. The attention kernels here already have a
  decode path (`single_q.rs`) that is close in shape, but it consumes a dense contiguous KV
  slice, not blocks.
- **Keep the non-paged path on CPU and document the trade-off.** Nearly free, and honest as
  long as the warning says what actually happened.

Either is defensible. What is not defensible is the current state, where the warning is wrong
for the common case. **At minimum the message and the condition should match reality**: a
pure-CPU run should not be told it has a mix of GPU and CPU.

## Suggested first step

Split the condition and fix the message, independently of whether a kernel gets written:

```rust
let has_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
let has_accel = mapper.get_unique_devices().iter().any(|d| !d.is_cpu());
if has_cpu && paged_attn_config.is_some() {
    warn!(
        "PagedAttention has no CPU backend; using the non-paged KV cache{}.",
        if has_accel { " (device map is mixed)" } else { "" }
    );
    paged_attn_config = None;
}
```

That is a small, self-contained change that removes a misleading diagnostic, and it makes the
remaining gap legible to whoever picks up the kernel work.

## Related

- `../../docs/src/content/docs/reference/paged-attention` is the user-facing documentation for
  this feature and does not currently mention the CPU restriction, which is the other half of
  this issue.