# mistralrs Python SDK

`mistralrs` is the Python SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a blazing-fast LLM inference engine.

## Documentation

For full documentation, see:
- [Python SDK Documentation](https://ericlbuehler.github.io/mistral.rs/tutorials/03-python-sdk/)
- [Installation Guide](https://ericlbuehler.github.io/mistral.rs/guides/install/)

## Quick Install

```bash
pip install mistralrs        # CPU, or Metal on Apple Silicon
```

NVIDIA CUDA wheels ship as GitHub release assets because they vary by CUDA toolkit level
(`cuda128`, `cuda129`, `cuda131`) and GPU compute capability. See the Python SDK
installation guide for the `--find-links` command.
