# NCCL in mistral.rs

Mistral.rs supports distributed inference on CUDA with Tensor Parallelism via NCCL.

> Note: Multi-node support is coming! Distributed inference on Apple hardware is also being investigated.

Tensor Parallelism (TP) is automatically used to accelerate distributed inference when more than one CUDA GPUs are detected. The tensor parallelism size is always automatically set to the total number of GPUs.

TP splits the model into shards and benefits from fast single-node interconnects like NVLink. Both `normal` and `vision` models support tensor parallelism.

**Important**: The world size (total number of GPUs) must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, etc.). This is a requirement for optimal performance and correct operation of the distributed algorithms.

> Note: In mistral.rs, if NCCL is enabled, then automatic device mapping *will not* be used.

**Important**: To build for NCCL, be sure to add the `nccl` feature flag (for example: `--features nccl,cuda`).

See the following environment variables:

|Name|Function|Usage|
|--|--|--|
|`MISTRALRS_NO_NCCL=1`|Disable TP and NCCL|If the model does not fit on the available CUDA devices, disabling NCCL will re-enable automatic device mapping|

## Single-Node Support

Set the number of ranks using `MISTRALRS_MN_LOCAL_WORLD_SIZE`, e.g.,

```bash
MISTRALRS_MN_LOCAL_WORLD_SIZE=2 mistralrs serve -p 8000 -m Qwen/Qwen3-30B-A3B-Instruct-2507
```

where, if no `MISTRALRS_MN_LOCAL_WORLD_SIZE` env given, mistral.rs will split the model across all available devices.

## Multi-node support

```
# Head node:
MISTRALRS_MN_GLOBAL_WORLD_SIZE=32 MISTRALRS_MN_HEAD_NUM_WORKERS=1 MISTRALRS_MN_HEAD_PORT=<PORT> mistralrs run -m ...

# For the worker nodes:
MISTRALRS_MN_GLOBAL_WORLD_SIZE=32 MISTRALRS_MN_WORKER_ID=0 MISTRALRS_WORKER_SERVER_ADDR=<HEAD ADDR>:<PORT> mistralrs run -m ...
MISTRALRS_MN_GLOBAL_WORLD_SIZE=32 MISTRALRS_MN_WORKER_ID=1 MISTRALRS_WORKER_SERVER_ADDR=<HEAD ADDR>:<PORT> mistralrs run -m ...
MISTRALRS_MN_GLOBAL_WORLD_SIZE=32 MISTRALRS_MN_WORKER_ID=2 MISTRALRS_WORKER_SERVER_ADDR=<HEAD ADDR>:<PORT> mistralrs run -m ...
```

Multi-node support in mistral.rs divides the nodes into two groups: a "head" node, and multiple "worker" nodes. Head node choice is arbitrary.
For example, if a system has 8 nodes, there will be 1 "head" node, and 7 "worker" nodes. 

To enable multi-node, set the `MISTRALRS_MN_GLOBAL_WORLD_SIZE=<number>` environment variable to the total number of GPUs in all nodes, including "head" and "worker"s. **Note**: This number must be a power of 2.

It is recommended to use server mode with mistral.rs when in multi-node. **Currently, you must send requests to every node!**

The following environment variables must be set for each node:

**Head node:**

|Name|Function|Usage|
|--|--|--|
|`MISTRALRS_MN_HEAD_NUM_WORKERS=<number>`|The number of worker nodes which will be connected.|This should be the number of nodes in the system, minus 1 for the head node.|
|`MISTRALRS_MN_HEAD_PORT=<PORT>`|The port on which to communicate with the worker nodes.|Worker nodes will connect to this port via TCP sockets|

**Worker node:**

|Name|Function|Usage|
|--|--|--|
|`MISTRALRS_MN_WORKER_ID=<number>`|The 0-indexed worker ID for this worker node.|If there are 4 nodes (1 head, 3 workers), then the worker ids will be 0, 1, and 2|
|`MISTRALRS_MN_WORKER_SERVER_ADDR=<ADDR>:<PORT>`|The IP address and port to connect to the server.|This is used to establish communication with the head node.|
