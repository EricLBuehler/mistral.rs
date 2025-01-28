# FlashAttention in mistral.rs

Mistral.rs supports FlashAttention V2 and V3 on CUDA devices (V3 is only supported when CC >= 9.0).

> Note: If compiled with FlashAttention and [PagedAttention](PAGED_ATTENTION.md) is enabled, then FlashAttention will be used in tandem to accelerate
the prefill phase.

## Using FlashAttention V2/V3

To use FlashAttention V2/V3, compile with the following feature flags.

|FlashAttention|Feature flag|
|--|--|
|V2 (CC < 9.0)| `--features flash-attn` |
|V3 (CC >= 9.0)| `--features flash-attn-v3` |

> Note: FlashAttention V2 and V3 are mutually exclusive
> Note: To use FlashAttention in the Python API, [compile from source](../mistralrs-pyo3/README.md).
