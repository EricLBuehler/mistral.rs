---
title: Run across multiple machines
description: The ring backend for distributed inference across hosts. For models that exceed any single box you have.
sidebar:
  order: 6
---

When a model exceeds the GPU memory of a single machine, mistral.rs can split it across hosts using the ring backend. Hosts are arranged in a logical ring, and activations pass around it sequentially per forward pass.

The ring backend is more sensitive to network latency than NCCL and is Linux-only. Use it when single-node inference is not feasible.

## When to use it

Use the ring backend when:

- Multiple GPU machines are connected by a fast network (10 GbE minimum, 100 GbE for throughput).
- The model exceeds single-node capacity.
- Machines are on the same subnet, or a network with sub-millisecond round-trip.

Do not use it when:

- One machine has multiple GPUs. Use [tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) instead.
- The network exceeds a few milliseconds round-trip. Per-token round-trips dominate.
- A hard latency guarantee is required. Distributed inference is throughput-oriented.

## Prerequisites

- Linux on every participating machine. Ring backend is Linux-only.
- A Rust build with the `ring` feature: `cargo install --path mistralrs-cli --features "cuda flash-attn ring"`.
- Full TCP reachability between all machines.
- Optionally a shared NFS or similar for the Hugging Face cache, to avoid per-machine downloads.

## Starting a ring

Pick one machine as coordinator:

```bash
mistralrs serve \
  --ring-coordinator 0.0.0.0:9999 \
  --ring-worker-count 3 \
  -m <large-model>
```

On each worker:

```bash
mistralrs serve \
  --ring-worker coordinator.example.com:9999 \
  -m <large-model>
```

The coordinator waits for all workers to join, shards the model across all participants (including itself), and serves on its normal HTTP port. Clients only talk to the coordinator.

## Sharding strategy

The ring splits by layer count by default. A 70-layer model across 3 workers places 24, 23, and 23 layers on the participants. Each layer's weights live only on its assigned machine.

For asymmetric machines, set per-machine layer counts in the ring config file. See the [topology guide](/mistral.rs/guides/perf/topology/).

## Performance expectations

For a well-tuned ring on a fast network:

- First-token latency is comparable to single-machine inference of the same model size.
- Subsequent tokens are slightly slower per token due to ring traversal. Overhead is measurable but acceptable at 100 GbE; more visible at 10 GbE.
- High-concurrency throughput is close to a single-machine equivalent because pipeline parallelism hides per-token cost.

The ring backend is for models that do not fit otherwise; it does not beat single-machine inference on workloads that do fit.

## Failure modes

A worker disconnect mid-inference kills the in-flight request. The coordinator refuses new requests until reconnection or restart. For HA deployments, run two rings behind a load balancer rather than making a single ring fault-tolerant.

Network partitions can cause hangs. `--ring-timeout` caps how long the coordinator waits on a slow worker; default 30 seconds. Raise it if legitimate long generations are timing out.

## What to read next

- [Tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) — single-machine multi-GPU.
- [Topology](/mistral.rs/guides/perf/topology/) — per-node layer assignment.
