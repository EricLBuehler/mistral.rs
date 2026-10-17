# Withdrawn: get_num_threads does not misreport under CPU pinning

- **Repo:** `huggingface/candle`
- **candle rev checked:** `12bb5d1` (upstream master, 2026-10-14) and pinned `66a8cf1`
- **Status:** withdrawn. The reported mechanism is not reachable on this host and the observed
  symptom has a different cause. Kept as a record so it is not re-investigated.
- **Superseded by:** `01-cpu-decode-gemv-single-threaded.md`

## The claim, as previously written

`default_num_threads` never consults the process affinity mask, because neither
`linux_env_cpu_mask` nor `linux_preferred_cpus` reads it. On a host where every CPU has
`cpu_capacity`, `linux_preferred_cpus` would return all 128 cores regardless of a `numactl`
pin. Proposed fix: fall back to `std::thread::available_parallelism()`.

## Why it is withdrawn

**The count is correct under pinning.** Measured on this host, 128 cores, pinned to one node:

```
unpinned                          numactl --cpunodebind=3
  affinity     0-127                affinity     96-127
  get_num_threads  128              get_num_threads  32
  available_parallelism 128        available_parallelism  32
```

`CANDLE_CPU_MASK=0-15` under the same pin gives 16, so that override path works too.

**The reason is a third branch the original analysis missed.** `linux_detect_preferred_cpus`
returns `None` when every CPU's capacity is within 10% of the maximum:

```rust
if max == 0 || min * 10 >= max * 9 { return None; }
```

This host reports `cpu_capacity = 1024` on all 128 CPUs, so it returns `None`, and
`default_num_threads` falls through to `num_cpus::get_physical()` - which *does* respect the
affinity mask. The branch that ignores affinity is unreachable here.

**The symptom was borrowed from 01.** The original evidence was a throughput and occupancy
table, not a thread count:

| configuration | decode tok/s | cores busy |
|---|---|---|
| `RAYON_NUM_THREADS` unset | 2.2-2.8 | 15.1 of 32 |
| `RAYON_NUM_THREADS=32` | 4.2-5.1 | 21.8 of 32 |

Both configurations report 32 threads in every pool. Idle cores during decode are what you get
when the decode GEMV does not use them, which is what `01` measures directly: at `m=1` gemm
runs 9x the CPU time with no bandwidth gain. Setting `RAYON_NUM_THREADS` helped because it
changed how decode work was partitioned elsewhere, not because it corrected a count.

The original file's own measurement already contradicted its claim - it printed
`get_num_threads = 32` next to `available_parallelism = Ok(32)` and read the agreement as the
bug being "confusing" rather than as a disproof.

## What survives

A narrow, untested case: on a host with heterogeneous `cpu_capacity` (Intel P-cores plus
E-cores, where the E-cores report a lower value), `linux_detect_preferred_cpus` returns
`Some(top tier)` and that list ignores the affinity mask entirely. Pinning to a subset that
excludes the fast cores would then yield a count for CPUs the process may not run on.

This host is homogeneous, so the case cannot be reproduced here and there is no measurement to
file. It is recorded in case someone hits it on mixed hardware.
