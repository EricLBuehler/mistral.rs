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

## What `../vllm.cpp` already has, and how much of it transfers

The sibling project has expert streaming for this exact model. It is worth knowing how much
of it is reusable before writing any of this, and the answer is: the mechanism is not
portable, the policy is small, and the CPU case should not need either.

| Piece | Lines | Where it is used | Portable? |
|---|---|---|---|
| `Kolibri1DequantCache` | 331 | `kolibri1_forward.cpp:184`, the **CPU** forward | Already device-free and CPU-proven, but it solves `../mistralrs/01`, not this gap |
| `Kolibri1TTExpertSlotPolicy` | ~120 | `kolibri1_tt.cpp` only | Host-side, device-free, portable as an LRU slot pool keyed by (layer, expert) |
| `kolibri1_tt_stream.cpp/h` | 526 | Tenstorrent path | No |

`kolibri1_tt_stream` is built around a **device** memory budget: FP8_E4M3 slot buffers on
the card, router readback from the device, a host-side logical-expert to slot remap, a fetch
executor that fills slots, a per-token stream bound asserted at runtime, and TT graph resets
driven by a slot fingerprint. None of that has a CPU counterpart, and none of it is the
problem here.

The slot policy is the piece that would port, and it is genuinely small: `Touch(layer,
expert)` returns a slot, evicting the layer's LRU when full; `SlotFor`, `ResidentCount`,
`SuggestSlotFor`; plus a `Fingerprint()` over the slot tables to invalidate memoized work.
Around 120 lines of logic with no device dependency.

But the reason to skip it is in that file's own motivation. It exists because a Tenstorrent
B2b has a small **device** budget and must keep only hot experts resident on the card. On
CPU the page cache is already an LRU residency manager, sized by the kernel, evicting clean
file-backed pages under pressure. Building a residency manager on top of that is what L1
exists to avoid.

So the honest read of "copy what is in vllm.cpp" is:

- Do **not** copy the streaming machinery. It is not portable and L1 removes the need.
- The slot policy is cheap to lift if L1 measures badly and a hard bound is still wanted.
- The dequant cache is the piece actually worth porting, and it belongs to
  `../mistralrs/01-cpu-fp8-dequant-every-forward.md`, where its CPU-proven status and its
  v1 corruption post-mortem are now recorded.

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

## The 32 GB question, and the answer that is cheaper than streaming

A 32 GB system cannot hold this model at any usable precision: the routed experts are 140.6 GB
in bf16 and 70.3 GB in fp8, out of 19200 (layer, expert) slots.

The resident set splits cleanly:

```
non-expert, bf16      4.85 GB   attention 3.17, shared expert 0.37, router 0.09,
                                  embeddings + lm_head 1.22
routed experts      140.62 GB bf16, 70.31 GB fp8
one expert             7.50 MB bf16
per token              2.56 GB   7 experts x 50 layers, 300 distinct (layer, expert) slots
```

So a 32 GB box with 26 GB usable has 21.15 GB for experts: 2887 bf16 slots of 19200 (15%),
or 5775 fp8 slots (30%).

**But the real answer is not to build a cache manager - it is that the page cache already is
one.** The GGUF is mmap'd, so if the MoE arm gathered only the selected experts out of the
mapping, the kernel would keep the hot ones resident and stream the cold ones from disk, for
free. That is what "expert streaming" means here, and it is a change of *call site*, not a new
subsystem.

**Which is `todo/mistralrs/02`.** The GGUF MoE arm today calls
`qtensor.dequantize(device)` and dequantises the whole expert stack on every forward. Scope
that to the selected experts at the gather - the same shape the reference implementation uses,
where the dequant is per expert at the gather and writes bf16 straight into the destination rows
- and the 32 GB case follows, along with the fp8 case, because the fp8 path has the same
shape of defect and the same fix.

### What the 32 GB case would cost

Compute is not the constraint: 3.46B active parameters is 6.9 GFLOP per token, about 5 ms at
1.4 TFLOP/s. The traffic is 2.56 GB per token, so the box is I/O bound by two orders of
magnitude.

| hit rate | from disk | at 2 GB/s NVMe |
|---|---|---|
| 30% | 1.79 GB/token | 0.90 s/token, 1.11 tok/s |
| 60% | 1.03 GB/token | 0.51 s/token, 1.95 tok/s |
| 90% | 0.26 GB/token | 0.13 s/token, 7.80 tok/s |

**The hit rate is the whole design, and it is not known.** It depends on how skewed the routing
distribution is: if a small set of experts carries most of the traffic, a 15% cache gets most of
the hits; if routing is uniform, it does not. This is measurable from the existing expert-id
dumps, and it should be measured before designing anything - a uniform-routing model and a
skewed-routing model want completely different caches.

### Two quantisation routes to the same box

- **No streaming, degraded quality.** llama.cpp has published `Kolibri-1-Q2_K.gguf` at 26.7 GiB,
  which fits a 32 GB box entirely resident. Our GGUF arm already handles `GgmlDType::Q2K`
  (`mistralrs-quant/src/gguf/archive.rs:1453`, and candle's `QTensor` covers `Q2K`), so this
  path is available today. `IQ2_XXS` at 21.2 GiB is a better fit but uses the IQ types we do
  not implement.
- **Streaming, full quality.** The 4.85 GB non-expert set plus a streamed expert cache, as
  above.

The first is a quantisation question and the second is a systems question. They are not
alternatives to each other so much as different products, and the first is much cheaper to
deliver because the artefact already exists.


## Measured: the expert working set is small enough that the page cache is the cache

From a 1000-step chained decode (300,000 selections, in the order the model touches them):

```
per layer, over the whole trace:
  layer   distinct used   top-6 share   top-20 share   for 90%
      0             26         54.9%          99.8%         14
      1             21         81.7%         100.0%          8
     25             56         68.1%          97.8%         12
     49             65         72.1%          94.7%         13

  experts per layer needed for 90% of that layer's selections:
    min 8, median 14, max 21   (of 384)
```

**Each layer only ever uses 21 to 71 of its 384 experts**, and the median layer needs 14 of
them to cover 90% of its selections. That is the whole design question, answered.

Simulated over the real access order:

| cache | slots | GB bf16 | GB fp8 | hit rate |
|---|---|---|---|---|
| flat LRU | 300 | 2.2 | 1.1 | 58.3% |
| **flat LRU** | **1000** | **7.3** | **3.7** | **98.9%** |
| flat LRU | 2887 | 21.1 | 10.6 | 99.2% |
| per-layer LRU | 16/layer | 5.9 | 2.9 | 98.3% |
| per-layer LRU | 24/layer | 8.8 | 4.4 | 98.9% |
| static top-24/layer, no eviction | 1200 | 8.8 | 4.4 | 97.6% |

**A flat LRU of 1000 slots - 7.3 GB in bf16, 3.7 GB in fp8 - gets 98.9%.** The working set is
that small because the hot set per layer is small, and the page cache is already a flat LRU.

### So the resident set on a 32 GB box

```
non-expert, bf16                    4.85 GB
hot experts (1000 slots, bf16)       7.30 GB
                                 ----------
                                  12.15 GB resident, on a 26 GB budget
```

and at a 98.9% hit rate the per-token disk traffic is 2.56 GB x 1.1% = **28 MB per token**,
not 2.56 GB. At 2 GB/s that is 0.014 s per token, so the box becomes **compute bound rather
than I/O bound** - 6.9 GFLOP per token is 4-17 ms on a laptop-class CPU.

**This makes the 32 GB case a call-site change rather than a subsystem.** mmap the GGUF, gather
only the selected experts out of the mapping, and let the kernel's own LRU do the caching,
because the working set is ~1000 slots and a 32 GB box's page cache holds it. There is no cache
manager to write, no eviction policy, no prefetch logic.

### Caveats on the measurement

- **One prompt, 1000 steps.** The pool sizes are per-layer and prompt-specific; a different
  domain may route differently. The per-layer LRU at 16-24 adapts, and the static top-24 does
  not, so the dynamic form is the safer design even though the static form scores well here.
- The prefill and the decode were traced with the same prompt, so the decode positions are
  the prompt's tokens at 1000 different context lengths. That is a fair sample of routing
  variety but not of distinct text.
- The measurement is of *our* router, which is now known to be f32-faithful, so the pool
  should match the reference's. It should be re-checked against the reference's own routing
  before the cache is sized from it.
