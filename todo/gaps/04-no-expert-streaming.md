# No expert streaming, and the expert stack is materialized on CPU

- **Repo:** mistral.rs
- **Components:** `mistralrs-core/src/moe/experts/checkpoint.rs`, `mistralrs-core/src/moe/experts/backends.rs`
- **Status:** open, unrequested
- **Impact:** any MoE model whose weights do not fit in RAM

## What

There is no expert streaming, and on the text path there is no weight offloading of any
kind. `NormalLoadingMetadata` has no offload field, `device_map` assigns whole transformer
layers to devices, and every weight is resident after `Model::new`. The only `offloaded` flag
in the repository is a bool on the Flux diffusion blocks.

For a CPU deployment the consequence is worse than "no streaming", because the CPU path
*forces a full materialization of the expert stack* where an accelerator would not.

`StackedExpertWeights::from_checkpoint` builds `[num_experts, out, in]` per projection.
For per-expert-named checkpoints it does this per expert:

```rust
gate_up_experts.push(Tensor::cat(&[&gate, &up], 0)?);
down_experts.push(down);
...
Tensor::stack(&gate_up_experts, 0)?, Tensor::stack(&down_experts, 0)?
```

`Tensor::cat` and `Tensor::stack` allocate, so ~3.02 GB per layer of Kolibri-1 lands in
**anonymous** memory that the kernel cannot reclaim, on top of the mapped checkpoint.

### The mapped checkpoint is not the problem

`SafetensorBackend::load_name` delegates to `MmapedSafetensors::load`, which calls
`convert_slice` (`candle-core/src/safetensors.rs:115`):

```rust
if (data.as_ptr() as usize).is_multiple_of(size_in_bytes) {
    // SAFETY This is safe because we just checked that this was correctly aligned.
    let data: &[T] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
    Tensor::from_slice(data, shape, device)
```

So a correctly aligned tensor load is already a **zero-copy view** into the page cache,
which is file-backed and reclaimable. Verified against the shipped Kolibri-1 checkpoint:
all 2737 tensors in a shard have 8-byte-aligned `data_offsets`, and the data section starts
at an 8-byte boundary, so every BF16 and F32 tensor takes the view branch.

The stack is therefore the *only* thing converting reclaimable page cache into unreclaimable
anonymous memory.

## Measured

`/mnt/models/Aleph-Alpha/Kolibri-1-bf16`, 149 GB across 32 shards, 384 experts x 50 layers.
Resident set peaks past 255 GB and the process is OOM-killed on a 255 GB host:

```
149 GB mapped checkpoint + ~151 GB of Tensor::stack copies  >  255 GB
```

The fp8 checkpoint (74 GB) loads in 33 s and runs, which is how the golden comparison in
`kolibri1-support-plan.md` is done. The bf16 checkpoint needs either a ~300 GB host or this
change.

## Is anyone asking for it

No. Searches of the tracker for expert streaming, expert caching, expert residency, expert
offload and on-demand expert loading return nothing on point. The closest items:

| Issue | State | What it is |
|---|---|---|
| [#1992](https://github.com/EricLBuehler/mistral.rs/issues/1992) Dynamic RAM/VRAM Layer-Offload Balancing | closed | Layer-level device placement. Chooses how many layers go on GPU; it does not make individual experts evictable. |
| [#1550](https://github.com/EricLBuehler/mistral.rs/issues/1550) KV offload to VectorDB | open | KV cache, not experts. |
| [#751](https://github.com/EricLBuehler/mistral.rs/issues/751) CUDA OOM with a presumed "full" offload to CPU | closed | Phi-3.5-MoE, and a bug in CPU placement rather than a request. |
| [#713](https://github.com/EricLBuehler/mistral.rs/issues/713) Phi 3.5 MoE for low-memory systems | closed | No body. |

`docs/src/content/docs/reference/troubleshooting.md` answers load-time OOM with "add
`--quant 4`, otherwise `--quant 2` or split across GPUs", which is consistent with there
being nothing here to point at.

### Adjacent work already in flight

[#2380](https://github.com/EricLBuehler/mistral.rs/pull/2380) (`perf(gguf): sparse per-expert
MoE forward on CPU, no full dequantize`, open) is aimed at the same CPU MoE consumer and
overlaps `../mistralrs/02-cpu-gguf-moe-dequant-every-call.md` and
`../../todo/candle/03-aarch64-packed-moe-gemv-serial.md`. It reports ~200x on Qwen3.6-35B-A3B
Q3_K_M on x86_64, where `QTensor::indexed_gemv` returns `None` because the repacked per-expert
GEMV is aarch64-only; `../../todo/candle/03` is the aarch64 half of the same gap. Worth
tracking before writing anything here, since it touches the same call site.

## Suggested direction

Three levels, in increasing cost. The first is the one that matters on CPU, and it is not
streaming at all.

### L1. Do not stack experts on CPU (the high-value change)

Keep the per-expert tensors as the views they already are and gather by expert id:

- Hold `Vec<Tensor>` per projection rather than one stacked tensor, sourced from the same
  `experts.{i}.{gate,up,down}_proj` names `stacked_enk` already reads. No `cat`, no `stack`,
  so no anonymous allocation at all. The `gate`/`up` `cat` has to go too: it is 2.01 GB per
  layer of the 3.02.
- Gather with the expert index directly. `mistralrs-quant/src/moe_gather_cpu.rs` already does
  the arithmetic `w[(expert * out + row) * in ..]` on a contiguous stack; the per-expert form
  is the same expression against `per_expert[expert]`.
- Two gathers (gate, up) instead of one fused `gate_up` gather costs one extra pass over the
  activation, which is negligible next to the weight traffic, and is the price of not copying.
- Select this backend only when `layer_device.is_cpu()`. The stacked layout is what makes the
  CUDA and Metal batched expert GEMMs work and should not be disturbed.

Expected effect: ~151 GB of unreclaimable anonymous memory becomes reclaimable file-backed
pages, and the working set for decode is only the experts a token actually routes to. BF16
Kolibri-1 stops OOMing on a 255 GB host.

### L2. Build the stack incrementally under a byte budget

Keep the fused layout but materialize it layer by layer with a global cap, so a model that
is close to fitting degrades instead of dying. Needs an eviction point, which means knowing
when a layer is finished; the layer loop in `NormalModel::forward` is the natural boundary.
This is the shape that generalizes to multi-model serving, where several models share a host.

### L3. True streaming

Router-first ordering: the top-k is known before the expert GEMM, so only those experts need
to be resident. This is the version that matters on GPU, where the transfer is the cost. It
needs a bounded LRU, a residency layer that survives the rayon pool (the forward is
parallel, so residency cannot be a plain `&mut`), and a decision about whether prefetching
is worth it. llama.cpp and vLLM are both working here:
[llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) on
expert-aware SSD streaming and
[llama.cpp #20757](https://github.com/EricLBuehler/mistral.rs/issues/20757) on a two-tier
GPU+RAM expert cache.

On CPU with mmap already in place, L3 buys bounded memory that L1 largely gets for free from
the page cache. Do L1 first and L3 only if the measurements say the page-fault cost hurts.

## Where the numbers would come from

Resident set and decode throughput on
`/mnt/models/Aleph-Alpha/Kolibri-1-bf16`, sampled from `/proc/<pid>/{statm,status}` during
decode, against the current OOM. A regression in tok/s from faulting pages back in is the
thing to watch for, and is the reason the change should be CPU-only and opt-out.

## Related

- `../../todo/mistralrs/02-cpu-gguf-moe-dequant-every-call.md` - the same consumer dequantizing
  every expert per forward.
- `../../todo/candle/03-aarch64-packed-moe-gemv-serial.md` - the aarch64 repacked GEMV.
- `kolibri1-support-plan.md` section "What the goldens do not cover yet", which is where this
  OOM first showed up.

## Note on scope

Unrequested. The tracker has no expert-streaming ask, so this is written down as a gap and a
design rather than a proposal, and L1 should be discussed with a maintainer before anyone
spends time on it.