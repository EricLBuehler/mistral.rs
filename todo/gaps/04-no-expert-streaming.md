# No expert streaming, and the expert stack is materialized on CPU

- **Repo:** mistral.rs
- **Components:** `mistralrs-core/src/moe/experts/checkpoint.rs`, `mistralrs-core/src/moe/experts/backends.rs`
- **Status:** the strategy is settled and the *loader* is the last piece of it; the
  validation (a real 32 GB box) has not been done. What is stale in this file is marked
  stale below
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

### The mapped checkpoint IS the problem, and the reason is in candle

**This section was wrong and is corrected here.** It claimed that a correctly aligned tensor
load is "already a zero-copy view into the page cache, which is file-backed and reclaimable",
and that the stack is therefore the only thing converting reclaimable page cache into
unreclaimable anonymous memory.

The alignment check in candle's `convert_slice` is a red herring. It takes a slice of the
mmap'd bytes and then calls `Tensor::from_slice`, which goes to
`device.storage_from_slice` and then to `CpuStorage::from_slice` - and that is
`v.to_vec()`. **The tensor is copied.** The alignment check only chooses between two copies,
the aligned one and the unaligned one, and either way the mmap's pages are released and the
model ends up in anonymous memory.

Measured, loading the 149 GB bf16 checkpoint and sampling the process:

```
rss=142202688KB  anon=118888704KB  file_clean=23376256KB
```

119 GB of the 142 GB resident is anonymous. The whole model is copied into RAM at load, so
none of it is in reclaimable page cache, and the per-expert view path cannot help: the views
are views of copies.

**Which moves the blocker to the bottom of the stack.** The expert stacking is a real and
separate problem - it is another 70 GB on top of the copy - but it is second order. The first
order is that no safetensors model in this repository is ever in reclaimable page cache,
because candle's CPU storage has no zero-copy path.

**And the GGUF path is the one that does.** `mistralrs-quant/src/gguf/archive.rs:603` hands
candle a `Cow::Borrowed` slice of the mmap'd bytes, and `QStorage` holds it borrowed, so the
GGUF's quantised weights stay in the file's page cache and the kernel can reclaim them under
pressure. That is the difference between the two routes, and it is why the Q2_K GGUF is not
merely a cheaper artefact: it is the only one of the three whose weights can be streamed at
all.

**What the safetensors route would need** is a zero-copy CPU storage in candle - a
`CpuStorage` variant that borrows the mapping instead of copying it. That is an upstream
change, and it is worth an issue against candle rather than a workaround here.

### (the old claim, which was wrong) The mapped checkpoint is not the problem

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


## The plan, 2026-10-21

### The thesis, which the measurements already support

**The cache manager is the page cache.** The working set is about 1000 (layer, expert)
slots - 7.3 GB in bf16, 3.7 GB in fp8 - and a flat LRU of that size scores 98.9% over the
real access order. At that hit rate the per-token disk traffic is 2.56 GB x 1.1% = 28 MB, so
the box is compute bound rather than I/O bound. The kernel's own LRU over the file's
reclaimable pages is already a flat LRU of exactly that kind, so "expert streaming" here is
a property of the *call site* - gather the routed experts out of the mapping and let the
kernel keep the hot ones - and not a subsystem to build.

### Three routes, and where each one stands

| route | artefact | resident set | state |
|---|---|---|---|
| Q2_K GGUF | 26.67 GiB, already at `/tmp/opencode/kolibri1-q2k.gguf` | all of it | **works today** - the GGUF arm handles `Q2K`, the scoped gather is in, and the loader is zero-copy - so this needs a measurement and nothing else |
| bf16 safetensors | 149 GB | 4.85 GB + 7.3 GB of hot experts | **does not work** - the per-expert path avoids the stack, but candle copies the whole model out of the mapping at load, so the 149 GB is anonymous RAM whatever the expert layout is |
| Q2_K or a larger GGUF quant | 26.67 GiB and up | the file's reclaimable pages plus the hot experts | **the only route that can stream** - the GGUF loader hands candle a borrowed slice, so the quantised weights stay in page cache |
| fp8 safetensors | 74 GB | 4.85 GB + 3.7 GB of hot experts | **the loader still fuses** - `BlockwiseFP8Linear` holds a rank-3 fp8 weight, and the per-expert branch hands it *views*, so the fp8 checkpoint is the one route that still materialises |

The OOM quoted at the top of this file - "149 GB mapped + 151 GB of `Tensor::stack` copies" -
predates the per-expert path and is no longer what the bf16 route does. It should be
re-measured rather than quoted.

### Phase 1: validate, with no new code

The validation is a memory limit, and this host can make one exactly. cgroup v2 here has the
`memory` and `cpuset` controllers, so a child cgroup with `memory.max=32G`,
`memory.swap.max=0` and `cpuset.cpus=64-95` is a 32 GB box with no swap - which is the
target, since the box's own 256 GB swapfile would otherwise hide the behaviour being
measured. The limit is cgroup-local, so the reclaim it causes is charged to the cgroup and
not to whatever else is on the host, and excluding swap is what makes it a faithful
emulation rather than a slow one. The cpuset is there for a second reason: node 2 is a zone,
so the measurement does not contend with whatever else the host is doing.

- Decode `Kolibri-1-Q2_K.gguf` (26.67 GiB, `general.file_type = 10` = MOSTLY_Q2_K) under
  the limit. **It is already on the box, at `/tmp/opencode/kolibri1-q2k.gguf`** - the
  download is not needed, and this host's outbound network is throttled to about 0.4 MB/s
  anyway, which would have made it a nineteen-hour job. It should fit entirely, and the
  question is the rate.
- Then the interesting one, and it has to be a **GGUF** and not the safetensors checkpoint:
  the 149 GB bf16 checkpoint cannot be streamed at all, because candle copies it out of the
  mapping at load. A GGUF of 27 GB and up under a 16 GB limit is the real test - the file's
  pages are reclaimable and the hot experts stay resident.
- The bf16 route is the one worth proving, because the Q2_K route has no streaming in it at
  all - it fits, so it demonstrates nothing about the thesis.

### Measured: the RAM budget, with the swapfile

The box has 256 GB of swap on the same NVMe as the model files, and it is the backing store
for the copied model. The limits below are `memory.max` with `memory.swap.max` generous; the
runs are on node 2 with 16 threads at 3.4 GHz.

| route | RAM budget | decode | swap delta |
|---|---|---|---|
| bf16 safetensors, 149 GB | unmetered | 8.65 tok/s | - |
| | 32 GB | **4.90 tok/s** (median 5.13) | 18.2 GB out |
| | 16 GB | **2.49 tok/s** (median 3.34) | +0.9 GB (the 32 GB run had already paged the cold part out) |
| fp8 safetensors, 74 GB + a 70 GB stack | 32 GB | **0.86 tok/s** (median 1.02) | - |
| | 16 GB | **0.77 tok/s** (median 0.88) | - |

The working set is what makes this work: about 5.6 GB of the model is touched every step -
the non-expert weights and the routed experts - and the rest is cold and sits in swap. The
hot part stays resident because it is touched constantly, and the kernel's LRU over the
anonymous pages keeps it there.

**Two things this changes.**

- The quantisation does not buy throughput on this path, only footprint. The fp8 route is
  three times *slower* than bf16 at the same RAM budget, because the per-forward dequantisation
  dominates and the quantisation only makes the weights smaller. Its case is a box that cannot
  hold the bf16 model at all.
- **`vm.swappiness=0` must not be used for this codebase.** The box's guidance to set it for a
  bf16 run is right for a genuinely mmap'd model, whose pages should be reclaimed as page
  cache. Ours is copied out of the mapping, so it is anonymous, and with swappiness 0 it
  cannot be paged out: a 32 GB limit OOM-kills the load with the swapfile idle and
  `memory.swap.current` at zero. The default of 60 is the right one here.

### Which of the two can be fixed here, and which cannot

The distinction is not the expert layout. It is **whether the bytes can be held borrowed**.

`QStorage` holds a `Cow<'_, [u8]>` and `QStorage::from_data` takes one, so a quantised
tensor can be a view of the mapping - which is exactly what the GGUF path does at
`mistralrs-quant/src/gguf/archive.rs:603`. A plain `Tensor` cannot: `CpuStorage` owns its
`Vec`s, and `Tensor::from_slice` copies. So:

- **The GGUF route already streams.** Its quantised weights stay in the file's page cache
  and the kernel reclaims the cold ones. Nothing to fix; it needs a measurement.
- **The fp8 safetensors route can be fixed here**, in three pieces: read the file ourselves
  over an mmap and keep the quantised bytes borrowed, give the dequant kernel a byte-slice
  entry point so it can read those bytes without a `Tensor` owning them, and hold the experts
  per-expert so the gather dequantises the routed ones. Every byte in that checkpoint is
  quantised, so none of it needs a plain tensor to be borrowed.
- **The bf16 safetensors route cannot be fixed here.** Its weights are plain bf16 tensors,
  and a plain tensor can only own its storage, so the copy is unavoidable without a
  `CpuStorage` variant that borrows. That is candle's #3708, and the comment on it has the
  measurement: 119 GB of a 142 GB resident load is anonymous, and under a 32 GB cgroup with
  no swap the load is OOM-killed.

So the honest answer to "should we make bf16 and fp8 work like gguf" is: **fp8 yes, bf16 not
without candle, and the GGUF route is already the streaming one.**

### Phase 2: the fp8 loader, the only code

The fp8 route is the one that still materialises, and it is the same 70 GB copy in two
places. The per-expert fp8 weights and their scales are already loaded as views - the
distributed loader does exactly that, at `mistralrs-quant/src/distributed/layers.rs:2820`,
reading `gate_proj.weight` and `gate_proj.weight_scale_inv` per expert - and then it stacks
them:

```rust
let gate_fp8 = Tensor::stack(&gate_fp8_vec, 0)?;
let gate_scale = Tensor::stack(&gate_scale_vec, 0)?;
```

which is the 70 GB of anonymous memory the streaming design is about. The single-process
loader stacks them the same way, through `ExpertCheckpoint::stacked_proj`.

The fix is to keep the views and dequantise the routed experts at the gather, which is what
`PerExpertQuantizedLinear` already does for a stacked tensor - it needs to work over
per-expert views instead:

1. **`mistralrs-quant/src/per_expert_quantized.rs`**: let `PerExpertQuantizedLinear` hold
   per-expert fp8 weight and scale views as well as a stacked pair, and dequantise the routed
   experts from whichever form it has. The dequant and the remap are already written and
   already bit-identical to the whole-stack path. For the zero-copy form the dequant needs a
   byte-slice entry point - the dispatch already works on `&[F8E4M3]` and a `Layout`, so it is
   the `Tensor` wrapper that has to go.
2. **`mistralrs-core/src/moe/experts/backends.rs`**: in the per-expert branch, when the
   checkpoint is blockwise fp8, read the weight and scale views and build that instead of
   `PerExpertLinear`. The scale name is the quantised format's convention,
   `<projection>.weight_scale_inv`, and the per-expert scale shape is
   `ceil(out / block_rows) x ceil(in / block_cols)`.
3. **`mistralrs-quant/src/distributed/layers.rs`**: the same change for the distributed path,
   which stacks for the same reason.
4. **The test**: the per-expert fp8 gather must be bit-identical to the stacked fp8 gather, in
   the same way the GGUF one is.
5. **The measurement**: the fp8 route under a 16 GB cgroup.

The resident set afterwards is the file's pages, which are reclaimable, plus the hot experts
in dequantised form - 3.7 GB of anonymous memory for a thousand slots - plus the dense
layers' dequantised copies, which the dense-weight cache keeps at 4.85 GB. That is about
9 GB, which is what makes 16 GB plausible; the dense cache is the largest single item and it
is the one that can be turned off.



A per-expert fp8 linear: the fp8 weight per expert as views into the mapping, and the
scoped dequant that is already written (`scoped_expert_stack`, bit-identical to the
whole-stack path). The forward is done; this is the loader, and it is the mirror image of
the GGUF fix. It is the last piece for the 74 GB fp8 route, which is the route that would
otherwise be the most attractive one on a 32 GB box: 3.7 GB of hot experts and 28 MB per
token of disk traffic.

### Phase 3: re-validate the working set against the reference's routing

The 98.9% was measured over a 1000-step trace of *our* router on *one* prompt, and the
caveat at the end of this file says both. The routing data to check it against now exists
on all three sides - the device's per-layer selections, the transcription's and ours - for the
golden ids, so the hot sets can be compared directly. A 1000-step trace of the reference's
own routing would be better and is the one thing worth asking for.

### The risks, named

- **The page cache's LRU is not ours.** A long prefill touches every expert and evicts the
  hot set; it recovers, because the hot set is only about 1000 slots, but a prefill on a
  memory-starved box is the worst case and the thing to watch.
- **The cold reads are random, so it is latency and not bandwidth.** Seven experts per layer
  per token, seven concurrent reads at a time, the layers sequential.
- **A shared box is not a 32 GB box.** The file's pages compete with everything else on the
  host, so the 32 GB case assumes a dedicated machine.
- **The Q2_K route is a quality trade, not the same model.** It is the cheapest way to get a
  number, not a way to get Kolibri-1.
