# Distributed inference in mistral.rs

Mistral.rs supports distributed inference with a few strategies
- [NCCL](NCCL.md) (recommended for CUDA)
- [Ring backend](RING.md) (supported on all devices)

**What backend is best?**
- **For CUDA-only system**: NCCL
- **Anything else**: Ring backend

The Ring backend is also **heterogenous**! This means that you can use the Ring backend on any set of multiple devices connected over TCP.
For example, you can connect 2 Metal systems, or 2 Metal and 1 CPU system with the Ring backend!