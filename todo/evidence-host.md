# Evidence host

All measurements in this directory were taken on one machine. Numbers do not transfer to other
hosts without re-measuring, so each issue file says which host it used.

## Host

| Property | Value |
|---|---|
| CPU | Ampere-1a (Grace), 128 cores, SMT off (1 thread per core) |
| NUMA | 4 nodes, 32 cores each: node0 `0-31`, node1 `32-63`, node2 `64-95`, node3 `96-127` |
| ISA | `asimd`, `asimddp`, `bf16`, `flagm`, `i8mm` |
| Memory | 255 GiB, no swap |
| Governor | `schedutil` by default - **see below, this caps the host at 1.0 of 3.4 GHz** |
| Kernel | Linux 5.x, aarch64 |
| Rust | 1.99.0, `RUSTFLAGS` from the repo `.cargo/config.toml` (`-C target-cpu=native`) |
| mistral.rs | `3f2515e9b` (master at time of writing) |
| candle | pinned rev `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0) |

## Governor: the default is 1.0 of 3.4 GHz, and that is deliberate

`scaling_governor` is `schedutil` (cppc_cpufreq) and on this host **it never ramps**:

```
scaling_min_freq  1,000,000 kHz
scaling_cur_freq  1,000,000 kHz   <- idle, and under a 16-thread decode
scaling_max_freq  3,400,000 kHz
```

A 64-thread load across all four nodes held 1.0 GHz in 11 of 12 samples, and 1.21 GHz once.
So under the default policy this host runs at **29% of its clock**.

**The policy: `schedutil` is the normal operating state, and it stays that way.** The 1 GHz cap
is intentional and is what the box is for. `performance` is for top-speed measurement only,
and the switch is a deliberate, recorded act rather than a default.

```
sudo sh -c 'for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do echo performance > "$c"'
```

It is a runtime change and does not survive a reboot. **Put it back afterwards**, and record
which governor produced any number you publish.

What the difference is worth, for a Kolibri-1 decode at one node and 16 threads:

| governor | tok/s | `attn_proj` | `routed` |
|---|---|---|---|
| `schedutil` (1.0 GHz) | 7.14 | 49.6 ms | 62.4 ms |
| `performance` (3.4 GHz) | 9.56 | 36.6 ms | 46.8 ms |

Not 3.4x, so the kernels were partly memory-bound after all - the clock was a third of the
story. `candle/01` measures 14.4 GB/s on `performance` against the 16.9 GB/s recorded on
`schedutil`.

**Every throughput and bandwidth figure in this directory was taken at 1.0 GHz unless it says
otherwise.** Most were, because the default was in force and nobody had gone looking. That
includes the numbers the thread-count and `MIN_PARALLEL_OUTPUTS` choices were based on, so
those optima are all 1 GHz optima and should be re-measured before being relied on.

## Shared-host caveat

Nodes 0, 2 and 3 are shared with other users and, since 2026-10-20, with other agents working
the same model on the same box. **Every performance number here was taken pinned to NUMA node
3** with:

```
numactl --cpunodebind=3 --interleave=all
```

Numbers taken before that pinning are marked in the individual files as contaminated and are
used only for direction, never as a result. Two threads also had to be accounted for
separately: node 3 is shared with another workload at times, and the raw thread sweep
(/proc-based) is noted where it was used.

### Box claims: announce before you start

The box is shared with other agents, and the peer channel is not enough on its own - the other
agents poll the machine rather than read chat. So the claim state lives in a file:
**`/tmp/box-claims.txt`**, one line per active claim:

```
<agent>  <what>  <node>  <mem GB>  <governor>  <until local time>  <pid(s)>
```

Read it before starting anything long, and write a line before you start, with an ETA. A claim
that begins without a line is indistinguishable from a stuck process.

The rules, in the order they were learned:

1. **Announce before starting, with an ETA.** A 30-minute grid started without a line here
   held node 3 while a peer polled for 2h15m for a quiet window.
2. **Memory is the binding constraint, not the load.** 255 GB with no swap, and two
   model-sized workloads do not fit: 149 GB for a bf16 Kolibri-1 plus 74 GB for the fp8 one
   is 223 of 255, and a third allocation tips it into the OOM killer. Check `free -g` and the
   process table - the `oom_score` tells you who the kernel will take, and the load average
   does not.
3. **A cache drop is a machine-wide event.** `drop_caches` evicts every other agent's resident
   model pages and turns their measurement into a disk benchmark that nobody chose. It is also
   self-amplifying as a retry mechanism: dropping the cache makes the next run re-read the
   model from disk, so it is slower and more likely to fail, which drops the cache again.
4. **The governor is global.** `schedutil` is the normal operating state at 1.0 GHz;
   `performance` at 3.4 GHz is a deliberate, announced exception, and the last user restores
   `schedutil` and says so on the line.
5. **A cell that needed retries is not the same cell as one that passed first time.** Record
   it in the row.

### The box is shared with a peer agent, and that has its own hazards

The other agent is working Kolibri-1 in the vllm.cpp repo. Three things collide, and all three
have produced wrong numbers rather than just slow ones:

- **The governor is global.** Changing it affects a 78B llama.cpp run that may be in flight. A
  reference line measured across a governor change is neither number. Announce governor
  changes; do not flip it silently.
- **RAM is the binding constraint.** A Kolibri-1 forward needs ~150 GB resident, and a
  146 GB GGUF is already in page cache. Loading one evicts the other, and the OOM killer
  takes whoever is largest. Announce large reservations.
- **Untracked files in a shared repo.** Artifacts written into the vllm.cpp working tree on
  `main` can be swept into a commit by a `git add -A`. Keep additions on a branch.

Also: `available_parallelism` and `get_num_threads` both report the *mask* size, not the
schedulable cores, so `numactl --cpunodebind=3` plus `RAYON_NUM_THREADS=16` is the only
configuration whose thread count is unambiguous.

## Memory bandwidth reference

Streaming read, 8 GB working set, node 3, python threads:

| threads | aggregate read |
|---|---|
| 1 | 10.1 GB/s |
| 4 | 38.5 GB/s |
| 8 | 37.6 GB/s |
| 16 | 76.5 GB/s |
| 32 | 122.7 GB/s |

This is a **lower bound** on the node's ceiling, not the ceiling itself. The Python probe uses
32 separate allocations and 32 OS threads, and the later kernel measurements in
`mistralrs/` exceed 122.7 GB/s on the same node. Treat 122.7 as "at least this much".

## Model set

| Path | Arch | Format | Size |
|---|---|---|---|
| `/mnt/models/unsloth-qwen3.5-0.8B-gguf` | qwen35 | GGUF Q4_K_M | 508 MB |
| `~/.cache/huggingface/.../Qwen--Qwen3-4B` | qwen3 | safetensors bf16 | 4B |
| `/mnt/models/Qwen3.5-9B-hf` | qwen3_5 | safetensors bf16 | 9B, multimodal |
| `/mnt/models/unsloth-qwen3.6-35B-A3B-gguf` | qwen35moe | GGUF Q4_K_M | 21 GB |
| `/mnt/models/Aleph-Alpha/Kolibri-1` | kolibri1 | safetensors fp8 block | 74 GB |
| `/mnt/models/Aleph-Alpha/Kolibri-1-bf16` | kolibri1 | safetensors bf16 | 156 GB |

Qwen3-4B is the reference harness for the dense bf16 work: it is the smallest model that goes
through `UnquantLinear` on CPU, which is the path most safetensors checkpoints take.