# Advanced features

mistral.rs supports the following advanced features enabling faster inference:

- Distributed inference with NCCL
- FlashAttention V2/V3
- PagedAttention

:::{toctree}
:maxdepth: 1
:hidden:

distributed
flash_attention
paged_attention
:::

- <project:distributed.md>
  - Multi-GPU (CUDA) inference
  - Multi-node inference
- <project:flash_attention.md>
  - FlashAttention V2/V3
- <project:paged_attention.md>
  - PagedAttention