# Leaving `RAYON_NUM_THREADS` unset under `numactl` CPU pinning under-uses the allowed cores

- **Repo:** huggingface/candle
- **Pinned rev:** `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0)
- **Status:** open, workaround known
- **Impact:** anyone pinning to a subset of CPUs on Linux

## What

`candle_core::utils::get_num_threads()` does not respect a `numactl` / `taskset` CPU affinity
mask. `default_num_threads` (`candle-core/src/utils.rs:382-401`) is:

```rust
#[cfg(target_os = "linux")]
{
    linux_env_cpu_mask()          // reads CANDLE_CPU_MASK only
        .or_else(linux_preferred_cpus)   // reads /sys/.../cpu_capacity
        .map(<[_]>::len)
        .unwrap_or_else(num_cpus::get_physical)
}
```

Neither branch consults the calling process's actual affinity:

- `linux_env_cpu_mask` (`:450-458`) only reads the `CANDLE_CPU_MASK` env var.
- `linux_preferred_cpus` (`:444-447`, backed by `linux_detect_preferred_cpus` at `:503`)
  enumerates `/sys/devices/system/cpu/cpu*/cpu_capacity` and takes every CPU that has the
  file. On this host all 128 CPUs have `cpu_capacity`, so it returns 128 regardless of an
  affinity mask.

`linux_affinity_cpus` (`:468-474`) does the right thing, but it is gated behind
`CANDLE_CPU_AFFINITY` and is only used by `set_thread_affinity`, not by the thread count.

`std::thread::available_parallelism` does respect affinity and returns the correct number
here, which is why it is worth changing to rather than reimplementing.

## Measured

On the 4-node host, pinning to one node (32 of 128 cores):

```
$ numactl --cpunodebind=3 ...probe
candle get_num_threads     = 32
available_parallelism      = Ok(32)
```

So the observable count is correct, which is what makes the bug confusing. The measurable
effect is a discrepancy in work actually done. Same binary, same model, same pin, Qwen3-4B bf16
decode at depth 512:

| configuration | decode tok/s | cores busy during decode |
|---|---|---|
| `RAYON_NUM_THREADS` unset | 2.2-2.8 | 15.1 of 32 |
| `RAYON_NUM_THREADS=32` | 4.2-5.1 | 21.8 of 32 |

Both cases report 32 threads in every pool. The unpinned run still occupies only 15 of its 32
allowed cores, and loses about half its throughput.

## Why it matters

It is invisible in the common case: without pinning, the count is right and nobody notices. It
bites exactly when someone is trying to measure or constrain CPU behavior, which is when a
wrong number is most damaging, because the measurement is then wrong in a way that looks like
a property of the workload.

## Suggested direction

`default_num_threads` should fall back to `std::thread::available_parallelism()` before
`num_cpus::get_physical()`. That function already returns the affinity-aware count on Linux
and needs no new detection code.

The existing `CANDLE_CPU_MASK` and `CANDLE_CPU_AFFINITY` paths should keep priority, so
explicit configuration is unaffected.

## Workaround

Set `RAYON_NUM_THREADS` explicitly. `get_num_threads()` reads it first
(`candle-core/src/utils.rs:403-409`), so this pins every pool consistently and restores the
throughput, as the table above shows.

## Note

This was found from mistral.rs, where it made an otherwise-correct thread-pool default look
wrong. It is filed here because the fix belongs in candle: mistral.rs can set the variable but
cannot make candle's pools respect an affinity mask on its own.