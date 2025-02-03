# Distributed inference in mistral.rs

Mistral.rs supports distributed inference on CUDA with Tensor Parallelism via NCCL.

> Note: Multi-node support is coming! Distributed inference on Apple hardware is also being investigated.

Tensor Parallelism (TP) is automatically used to accelerate distributed inference when more than one CUDA GPUs are detected. The tensor parallelism size is always automatically set to the total number of GPUs.

TP splits the model into shards and benefits from fast single-node interconnects like NVLink (if the interconnects are a bottleneck, check out `MISTRALRS_PIPELINE_PARALLEL`).

> Note: In mistral.rs, if NCCL is enabled, then automatic device mapping *will not* be used.

See the following environment variables:

|Name|Function|Usage|
|--|--|--|
|`MISTRALRS_NO_NCCL=1`|Disable TP and NCCL|If the model does not fit on the available CUDA devices, disabling NCCL will re-enable automatic device mapping|
|`MISTRALRS_PIPELINE_PARALLEL=<number> (default: 1 = disabled)`|Parallelize the model along the layers in addition to the GPUs|Increasing this value is useful for tuning performance on a model-specific basis. It does not change the number of GPUs required, but can help when the single-node interconnects are a bottleneck.|
