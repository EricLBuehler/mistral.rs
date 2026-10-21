# CPU FP8 dequantizes the whole weight on every forward

- **Repo:** mistral.rs
- **Component:** `mistralrs-quant`, blockwise FP8 linear
- **Status:** **fixed and measured**. The scoped dequant is implemented and bit-exact, and it
  takes the fp8 decode from 70.8 s per token to 2.40 s - 29.5x, measured as a same-session
  A/B on the same build, the same weights, the same node and the same governor. The cost as
  last measured before the fix was 74 s per decode token on the Kolibri-1 fp8
  checkpoint, which is about 650x slower than the same model in bf16. This is not a
  second-order optimisation: it makes the fp8 path unusable for decode, and it is the
  single largest gap between this implementation and the vllm.cpp CPU tier.
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

## Measured: 74 seconds per token

Kolibri-1 fp8 checkpoint (74 GB, the original release), node 3, 32 threads, 3.4 GHz:

```
prefill 64 tokens in 120.13s   (0.53 tok/s)
decode step 0 (len 65) token 79 in 72.346s
decode step 1 (len 66) token 79 in 72.494s
decode step 2 (len 67) token 79 in 68.917s
decode step 3 (len 68) token 79 in 82.698s
mean TPOT 74.114 s  (0.01 tok/s)
```

Against the same model in bf16 on the same box:

| checkpoint | decode | per token |
|---|---|---|
| bf16, 149 GB | 9.11 tok/s | 0.110 s |
| fp8, 74 GB | 0.014 tok/s | 74.1 s |

**74 s per token is 74 GB of dequantisation at about 1 GB/s** - the whole checkpoint, every
forward, which is exactly what `dequantize_w` does.

### The MoE residency cache does not help

`MISTRALRS_MOE_CACHE_MB=16384` was the obvious thing to try, because that cache already exists
and is a byte-budgeted LRU over expert slabs. It makes no difference:

```
with    MISTRALRS_MOE_CACHE_MB=16384:  mean TPOT 74.114 s  (0.01 tok/s)
without:                               mean TPOT 71.8 s
```

The cache holds the routed slabs of the *quantised* path. The fp8 path's cost is elsewhere: it
re-dequantises the full checkpoint, which the cache does not cover.

### What the reference does instead

The vllm.cpp CPU tier, same fp8 checkpoint, same node, 32 threads:

| configuration | tok/s | per token |
|---|---|---|
| dequant cache off | 4.019 | 0.25 s |
| dequant cache on, 16 GiB budget | 9.066 | 0.110 s |

Their cache-off figure implies they dequantise only the *selected* experts - 7 of 384 per
layer, about 2.75 GB per token - and the 16 GiB cache removes even that. 0.25 s for 2.75 GB is
about 11 GB/s of dequantisation, which is a plausible rate for a blockwise fp8 kernel. Ours is
74 GB at 1 GB/s, so the gap is 27x in bytes and 11x in rate.

Two separable things are wrong on our side: the *scope* of the dequantisation (the whole
checkpoint rather than the selected experts) and its *rate*. The scope is the larger one and it
is the one that matters.

### The dequant body is fast enough. The scope is the problem.

`mistralrs-quant/examples/fp8_dequant_rate.rs` times `fp8_blockwise_dequantize` on one
expert slab in isolation, which separates the rate from the scope:

```
model.layers.20.mlp.experts.100.gate_proj.weight: [512, 2560] F8E4M3
scale_inv: [4, 20] F32
1310720 elements: 1310720 bytes in (fp8), 2621440 bytes out (bf16)
best of 5: 0.0005 s
  output rate : 5.72 GB/s of bf16
  input  rate : 2.86 GB/s of fp8
  at 2.75 GB of expert traffic per token that is 0.48 s per token
```

The reference implementation dequantises per expert at the gather and writes bf16 straight
into the destination rows, at about 11 GB/s of output. So the dequant body is the same order
as the reference's, about 2x slower, and **not** the 1 GB/s the end-to-end figure suggested.

**So the two defects are not equal in size:**

| | ours | reference | factor |
|---|---|---|---|
| bytes dequantised per token | 74 GB | 2.75 GB | 27x |
| dequant rate, bf16 out | 5.72 GB/s | ~11 GB/s | 2x |

The scope is 27x and the rate is 2x, and the scope is what turns a 0.5 s token into a 74 s
one. Fixing the scope alone would land us around 0.5-1 s per token; fixing the rate as well
would land us at the reference's 0.25 s.

**This also says where the fix goes: not into the dequant arithmetic, which is a few flops per
element and memory-bound, but into the call site.** The gather already knows which experts it
wants and the weights are already in per-expert slabs, so dequantising the selected experts at
the gather is a change of call site rather than of code shape. The vllm.cpp CPU tier does
exactly this, and keeps a byte-budgeted LRU over (layer, expert) entries on top of it.

### Why this is the biggest remaining item

This is not a throughput detail. It is the reason the fp8 checkpoint - a quarter the size of
the bf16 one, and therefore the version most people would actually want on CPU - is
unusable. It is also the one place where the gap to the vllm.cpp CPU tier is not a rounding
difference but a factor of 290.


## Swap would not fix this, and on this box it cannot even be enabled

Considered, because the fp8 path's 148 GB of per-forward allocation looks like a job for
swap. Three things rule it out.

**Swapfiles do not work on btrfs here.** Both filesystems that could hold one are btrfs
(`/mnt` and `/` are the same NVMe, `/home` is the other), and the kernel rejects a swapfile on
both:

```
$ swapon /tmp/.swaptest
swapon: /tmp/.swaptest: swapon failed: Invalid argument
```

including with `chattr +C`, which is btrfs's documented requirement for swapfile support.
Enabling swap here would mean partitioning the NVMe, which is not a change to make on a shared
device.

**zram is available** (`/sys/block/zram0`, with `zstd` among the algorithms, currently
unconfigured) and would work where a file cannot - it is RAM-backed and compressed, so it
costs CPU rather than the NVMe bandwidth the model reads need. It buys less than it looks:
bf16 does not compress, so 64 GB of zram holds perhaps 50-58 GB of the allocation, and it
consumes RAM that the page cache wants for the mmap'd weights.

**And it treats the wrong problem.** Swap only ever holds *anonymous* memory; file-backed
pages are dropped and re-read from the file, never swapped. So:

- For the **bf16 path**, which is mmap'd, the page cache already pages the model and swap
  adds nothing. This is why the bf16 path has never OOM-killed.
- For the **fp8 path**, the 148 GB is anonymous, so swap would work - and would turn the OOM
  kill into roughly a doubling of the 74 s step, at the cost of 148 GB of writes to the same
  NVMe that the model reads come from. On a shared box that trades our failure mode for
  everyone else's bandwidth.

**The fix that removes the need for swap is the same fix that unlocks the 32 GB case**: scope
the dequantisation to the selected experts at the gather, which is what `todo/02` says. The
measured expert working set is about 1000 slots, so the scoped path's peak anonymous memory
falls from 148 GB to roughly 7 GB and the OOM stops being possible.

If a safety net is wanted in the meantime, zram is the tool that works here, and it should be
announced and sized with the peers' agreement - it consumes RAM that the page cache wants, on
a box where two model-sized workloads are already close to the limit.


## Fixed: dequantise only the routed experts

`scoped_expert_stack` (`mistralrs-quant/src/blockwise_fp8/mod.rs`) resolves the experts a
gather names, dequantises only those, and remaps the indices onto the compacted stack. The
values are the ones the whole stack holds, so the gather computes the same thing.

The bit-exactness is the acceptance criterion, and it is checked in both shape regimes:
small enough that the gather falls back to a matmul, and large enough that the gather kernel
serves it. The two take different arithmetic, and the scoped path has to agree with the
whole-stack path in each. A second test checks that only the routed experts are dequantised.

This is the mirror image of the GGUF fix in `c58f341a7`, which was the same defect on the
other quantisation path.

**Not yet measured.** The speed needs the fp8 checkpoint loaded and the box quiet, and the
box is under peer-Ce's P150 decode bench, whose timings are the ones that would be
contaminated by a 74 GB memory-faulting load. The claim file records the wait.

**What the measurement should show.** The 74 s per token was 74 GB of dequantisation at
about 1 GB/s, and the summary's arithmetic puts about 13 s of that in the dequant and about
60 s in the allocation and faulting of the 148 GB dequantised stack per forward. The scoped
path dequantises `top_k` experts per layer instead of all of them, which is 1.6% of the
stack at decode, so both terms should collapse: the dequant to about 0.2 s and the
allocation to about a second. What remains should be the gather itself, over the routed
experts only.

**The measurement command**, for when the box is free:

```
KOLIBRI1_MODEL_DIR=/mnt/models/Aleph-Alpha/Kolibri-1 \
  cargo test --release -p mistralrs-core kolibri1_decode_benchmark -- --ignored --nocapture
```

`decode_prof` is enabled in that test, so the per-phase breakdown comes with it.


## Measured: 70.8 s per token becomes 2.40 s

Same session, same build, same weights, node 3, 16 threads, 3.4 GHz, `vm.swappiness=0`:

| path | mean TPOT | tok/s | steps |
|---|---|---|---|
| whole stack (the old path) | **70.810 s** | 0.014 | 4 |
| scoped (the new path) | **2.402 s** | 0.42 | 8 |

**29.5x.** The whole-stack number reproduces the 74.114 s recorded earlier, so the two are
comparable; the difference between them is the whole-stack dequantisation and nothing else.

The phase breakdown is where the change is:

| phase | whole stack | scoped | ratio |
|---|---|---|---|
| `routed` (the MoE experts) | 69506.86 ms (98.2%) | 1293.32 ms (53.9%) | **53.8x** |
| `attn_proj` | 1046.89 ms | 957.13 ms | 1.1x |
| `shared` | - | 130.83 ms | - |

No swap was used: SwapFree went from 249,924,352 kB to 249,945,664 kB across the baseline run.

**What is left, and it is the next item.** Two phases now dominate, and neither is the MoE
gather any more:

- `attn_proj`, 957 ms, 40% of the step. The attention and output projections are dense fp8
  layers, and they are dequantised whole on every forward even though the weights do not
  change. That is the same defect as the MoE one, on the layers that were not the point of
  the model. A dequantised-weight cache would turn it into a read of 2.6 GB per forward.
- `routed`, 1293 ms, 54% of the step. The experts are scoped now, but they are dequantised one
  expert at a time, 300 calls per forward, and each call carries the fixed cost of a candle
  custom op. The bytes are 393 MB of fp8 per forward, which is about 0.14 s of arithmetic
  against the 1.29 s measured, so most of it is per-call overhead. Dequantising a layer's six
  routed experts in one call instead of six would cut the calls per forward from 300 to 50.

The fp8 path at 2.40 s per token is now 20x slower than the same model in bf16 at 0.116 s, and
both of the phases above are the reason. That is the right shape for the comparison: fp8
buys the memory and costs the dequantisation, and the dequantisation is now confined to the
weights that actually change.

**A measurement trap that cost 2.7x and is worth recording.** With `RAYON_NUM_THREADS` unset
under `numactl --cpunodebind=3`, candle's pool under-uses the allowed cores and the same
bf16 benchmark reads 3.20 tok/s instead of 8.65. Every number on this box needs
`RAYON_NUM_THREADS` and `MISTRALRS_CPU_THREADS` set to the thread count, or the pools
disagree with each other. `mistralrs-quant/src/cpu_threads.rs` says so, and it is worth
believing it: the first bf16 measurement taken here was 2.9x off the recorded 9.11 for exactly
this reason, and the recorded numbers were right.
