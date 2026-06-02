---
title: Multi-node NCCL inference
description: Run tensor parallelism across machines while each node contributes local CUDA ranks.
sidebar:
  order: 8
---

Multi-node NCCL inference extends tensor parallelism across machines. Each node contributes one or more local CUDA ranks to one global NCCL communicator.

This is separate from the [ring backend](/mistral.rs/guides/perf/multi-machine-ring/). Multi-node NCCL uses `MISTRALRS_MN_*` variables and does not use `RING_CONFIG`.

## Build

Build the same CUDA+NCCL binary on every node:

```bash
cargo install mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

Every node should use the same model, dtype, quantization, and runtime arguments. Use `CUDA_VISIBLE_DEVICES` on each node to choose the local GPUs that participate.

## Environment

Use the same local TP size on every node. The global world size should be:

```text
global world size = local TP size * number of nodes
```

Common variables:

| Variable | Where | Purpose |
|---|---|---|
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | All nodes | Total NCCL ranks across all nodes. Presence of this variable enables multi-node mode. |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | All nodes | Number of local CUDA ranks contributed by each node. |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Head node | Number of worker nodes. |
| `MISTRALRS_MN_HEAD_PORT` | Head node | TCP port used to distribute the NCCL id to worker nodes. |
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Worker nodes | `host:port` of the head node. |
| `MISTRALRS_MN_WORKER_ID` | Worker nodes | Zero-based worker node id. |

Do not set `MISTRALRS_NO_NCCL`. Do not set `RING_CONFIG`.

## Example

Two nodes, four GPUs per node:

Head node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MISTRALRS_MN_GLOBAL_WORLD_SIZE=8 \
MISTRALRS_MN_LOCAL_WORLD_SIZE=4 \
MISTRALRS_MN_HEAD_NUM_WORKERS=1 \
MISTRALRS_MN_HEAD_PORT=9000 \
mistralrs serve -m Qwen/Qwen3-32B --quant 4
```

Worker node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MISTRALRS_MN_GLOBAL_WORLD_SIZE=8 \
MISTRALRS_MN_LOCAL_WORLD_SIZE=4 \
MISTRALRS_MN_WORKER_SERVER_ADDR=10.0.0.1:9000 \
MISTRALRS_MN_WORKER_ID=0 \
mistralrs serve -m Qwen/Qwen3-32B --quant 4
```

Send client requests to the head node.

## Notes

The head port must be reachable from every worker. NCCL must also be able to use the network interface between nodes; on multi-interface machines, set NCCL networking variables such as `NCCL_SOCKET_IFNAME` in the shell before launching.

Use the ring backend only when you intentionally want the `ring` transport. A binary built with both `nccl` and `ring` will prefer NCCL unless `MISTRALRS_NO_NCCL=1` is set.
