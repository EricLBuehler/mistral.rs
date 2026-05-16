---
title: Run across multiple machines
description: The ring backend for distributed inference across hosts.
sidebar:
  order: 6
---

When a model exceeds one machine's GPU memory, mistral.rs can split it across multiple hosts via a ring backend.

## Build

The `ring` feature must be compiled in:

```bash
cargo install --path mistralrs-cli --features "cuda flash-attn ring"
```

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

## Multi-node environment variables

Multi-node coordination is controlled through environment variables, not CLI flags:

| Variable | Purpose |
|---|---|
| `RING_CONFIG` | Path to the per-rank ring JSON config. |
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total world size across nodes. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Local TP size override on the node. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Number of worker nodes (set on head). |
| `MISTRALRS_MN_HEAD_PORT` | Head node port. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Head node address (set on workers). |
| `MISTRALRS_MN_WORKER_ID` | Worker node id. |
| `MISTRALRS_NO_NCCL=1` | Disable NCCL fallback. |

Full env var reference: [environment variables](/mistral.rs/reference/environment-variables/).

## Notes

The ring backend is Linux-only. For single-machine multi-GPU, prefer NCCL-based [tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/).
