# FlashAttention in mistral.rs

Mistral.rs supports FlashAttention V2 and V3 on CUDA devices (V3 is only supported when CC >= 9.0).

> Note: If compiled with FlashAttention and [PagedAttention](PAGED_ATTENTION.md) is enabled, then FlashAttention will be used in tandem to accelerate
the prefill phase.

## GPU Architecture Compatibility

| Architecture | Compute Capability | Example GPUs | Feature Flag |
|--------------|-------------------|--------------|--------------|
| Ampere | 8.0, 8.6 | RTX 30*, A100, A40 | `--features flash-attn` |
| Ada Lovelace | 8.9 | RTX 40*, L40S | `--features flash-attn` |
| Hopper | 9.0 | H100, H800 | `--features flash-attn-v3` |
| Blackwell | 10.0, 12.0 | RTX 50* | `--features flash-attn` |

> Note: FlashAttention V2 and V3 are mutually exclusive
> Note: To use FlashAttention in the Python SDK, [compile from source](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md).
