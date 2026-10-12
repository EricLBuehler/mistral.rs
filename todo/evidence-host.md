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
| Kernel | Linux 5.x, aarch64 |
| Rust | 1.99.0, `RUSTFLAGS` from the repo `.cargo/config.toml` (`-C target-cpu=native`) |
| mistral.rs | `3f2515e9b` (master at time of writing) |
| candle | pinned rev `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0) |

## Shared-host caveat

Nodes 0, 1 and 2 were shared with other users for most of this work (load average 50-80).
**Every performance number here was taken pinned to NUMA node 3** with:

```
numactl --cpunodebind=3 --interleave=all
```

Numbers taken before that pinning are marked in the individual files as contaminated and are
used only for direction, never as a result. Two threads also had to be accounted for
separately: node 3 is shared with another workload at times, and the raw thread sweep
(/proc-based) is noted where it was used.

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