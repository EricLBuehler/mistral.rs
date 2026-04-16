---
title: Run across multiple machines
description: The ring backend for distributed inference across hosts. For models that exceed any single box you have.
sidebar:
  order: 6
---

When a model does not fit even on one machine's worth of GPUs, mistral.rs can split it across several machines. The ring backend we use for this arranges the hosts in a logical ring and passes activations around it sequentially for each forward pass.

This is a sharper tool than the single-machine NCCL setup. It works when NCCL does not (models larger than one node), but it is also more sensitive to network latency, and it is Linux-only. Use it when the alternative is not being able to run the model at all.

## When to use it

The ring backend is right when:

- You have multiple machines with GPUs connected by a fast network (10 GbE at minimum, 100 GbE or better if you care about throughput).
- The model is too large for a single node.
- The machines are on the same subnet, or at least on a network where the round-trip is under a millisecond.

It is not the right tool when:

- You just have one machine with multiple GPUs. Use [tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) instead.
- Your network is higher-latency than a few milliseconds. Generation speed degrades fast when every token round-trips between hosts.
- You need a hard latency guarantee. Distributed inference is a throughput-oriented tool.

## Prerequisites

- Linux on every participating machine. The ring backend is Linux-only; it does not run on macOS or Windows.
- A Rust build with the `ring` feature enabled: `cargo install --path mistralrs-cli --features "cuda flash-attn ring"`.
- Each machine can reach every other machine over TCP.
- Ideally a shared NFS or similar for the Hugging Face cache, so each machine does not download the model separately. This is optional but saves a lot of time on first start.

## Starting a ring

Pick one machine to be the coordinator. On that machine:

```bash
mistralrs serve \
  --ring-coordinator 0.0.0.0:9999 \
  --ring-worker-count 3 \
  -m <large-model>
```

On each of the other machines (the workers):

```bash
mistralrs serve \
  --ring-worker coordinator.example.com:9999 \
  -m <large-model>
```

The coordinator waits until all workers have joined, then shards the model across them (including itself) and starts serving on the coordinator's normal HTTP port. Clients only ever talk to the coordinator.

## Sharding strategy

By default, the ring backend splits the model by layer count. A 70-layer model across 3 workers puts 24, 23, and 23 layers on the three participants. Each layer's weights exist only on the machine responsible for that layer.

If your machines have different GPU counts or different amounts of VRAM, you can set per-machine layer counts in the ring config file. See the [topology guide](/mistral.rs/guides/perf/topology/) for the config shape.

## Performance expectations

For a well-tuned ring with a fast network:

- First-token latency is roughly equivalent to single-machine inference of the same model size.
- Subsequent tokens are slightly slower per token because each one has to traverse the ring. The overhead is measurable but not catastrophic at 100 GbE; at 10 GbE it is more noticeable.
- Throughput at high concurrency is close to a single-machine equivalent, because the ring's pipeline parallelism hides some of the per-token cost.

In short: use it for workloads that do not fit otherwise, but do not expect it to beat single-machine inference on a workload that does fit.

## Failure modes

A worker disconnect mid-inference kills the in-flight request. The coordinator will refuse new requests until either the worker reconnects or you restart the ring. For high-availability deployments, you probably want redundancy at the model level (run two rings with a load balancer in front) rather than trying to make a single ring fault-tolerant.

Network partitions are more insidious. If two machines lose contact but not with the coordinator, generation can hang. The `--ring-timeout` flag sets a cap after which the coordinator gives up on a slow worker; the default is 30 seconds, which is usually fine, but worth raising if you see timeouts on legitimate long generations.

## What to read next

- [Tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) if everything fits on one machine.
- [Topology](/mistral.rs/guides/perf/topology/) for per-node layer assignment.
