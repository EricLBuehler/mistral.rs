---
title: Run across multiple machines
description: Use the ring backend for distributed inference across hosts.
sidebar:
  order: 9
---

The ring backend is a distributed transport selected by `RING_CONFIG`. It is separate from [multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/), which uses `MISTRALRS_MN_*` variables and NCCL across all ranks.

Use this page when you explicitly want the ring backend.

## Build

The `ring` feature must be compiled in:

```bash
cargo install --path mistralrs-cli --features "cuda flash-attn ring"
```

If the binary is also built with `nccl`, set `MISTRALRS_NO_NCCL=1` when launching so `Comm::from_device` selects the ring backend.

## Configuration

The ring backend reads its configuration from a JSON file pointed to by the `RING_CONFIG` environment variable. Each participant has its own `RING_CONFIG` with rank-specific values.

Config shape:

```json
{
  "master_ip": "10.0.0.1",
  "master_port": 9000,
  "port": 9001,
  "right_port": 9002,
  "right_ip": "10.0.0.2",
  "rank": 0,
  "world_size": 3
}
```

Non-master ranks (`rank != 0`) must specify `master_ip`. The master rank (`rank = 0`) is reachable via `master_ip`.

## Environment

Ring backend selection is controlled by `RING_CONFIG`:

| Variable | Purpose |
|---|---|
| `RING_CONFIG` | Path to the per-rank ring JSON config. |
| `MISTRALRS_NO_NCCL=1` | Required only when the same binary also has `nccl` and you want to force ring. |

Full env var reference: [environment variables](/mistral.rs/reference/environment-variables/).

## Notes

The ring backend is Linux-only. For CUDA tensor parallelism on one machine, prefer [single-machine multi-GPU](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/). For CUDA tensor parallelism across machines, prefer [multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/).
