---
title: Troubleshooting
description: Verified causes and fixes.
sidebar:
  order: 14
---

Before debugging setup issues, run `mistralrs doctor`. It reports detected hardware, compiled accelerator features, and Hugging Face connectivity.

For unlisted issues, file an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues) with a reproducer.

## Installation and build

### `mistralrs: command not found` after install

The binary is at `~/.cargo/bin/mistralrs`. The directory is added to `PATH` by `rustup`, but the change does not apply to the current shell. Open a new shell or run `source "$HOME/.cargo/env"`.

### Build fails with `flash-attn` feature enabled

Flash attention requires compute capability 8.0+. On older GPUs, drop `flash-attn` from features and rebuild with `cuda nccl cudnn` on Linux when NCCL is installed, or `cuda cudnn` otherwise.

### `mistralrs login` rejects the token

The token must start with `hf_`. The validation happens in `mistralrs login` before saving.

## Model loading

### Gated repository (Gemma, LLaMA, FLUX.1-dev, etc.)

Accept the license on the model's Hugging Face page, then save a token with `mistralrs login`. The token is stored at `~/.cache/huggingface/token` (or `$HF_HOME/token`).

### `Out of memory` on load

Add `--quant 4`. If still too large, try `--quant 2` or split across GPUs with `-n "0:N1;1:N2;..."`.

## Runtime

### Generation slower than expected

Verify accelerator features are compiled in with `mistralrs doctor`. If `cuda` is missing, the binary was built without GPU support.

For CUDA decode throughput, also check whether PagedAttention is active. FlashInfer paged decode and CUDA graphs are enabled by default for compatible CUDA paged decode paths.

### CUDA graphs do not appear to help

CUDA graphs apply to supported single-token decode steps only. They do not speed up prompt prefill. The first time a graph shape is seen, mistral.rs pays warmup and capture overhead; steady-state decode is the part that can improve.

If graph capture or replay fails, mistral.rs logs a warning and disables CUDA graphs for that loaded pipeline. Set `MISTRALRS_CUDA_GRAPHS=0` to compare with the normal CUDA path.

### Response cut off

`max_tokens` is most likely too low. Check `finish_reason`, `length` means the token limit; `stop` means a stop sequence matched.

## HTTP server

### `Connection refused` hitting localhost

Check the `Server listening on http://...` line in the server output to confirm host and port.

### CORS errors in a browser

The default allows any origin. Custom CORS configuration is only available programmatically through `MistralRsServerRouterBuilder`.

### `413 Payload Too Large`

The default body limit is 50 MB and is not configurable via the CLI. Configure programmatically through `MistralRsServerRouterBuilder`.

### UI does not load at `/ui`

The UI is on by default. Check that `--no-ui` was not passed at startup, and that no reverse proxy is rewriting `/ui`.

## Sessions

### Sessions disappear between requests

The session expired (30-minute idle TTL) or was evicted (128-session cap, LRU). Long-lived sessions need explicit export/import via `/v1/sessions/{id}`.

## Python SDK

### `from mistralrs import Runner` fails with `ImportError`

The wrong wheel was installed. Reinstall with the matching variant: `mistralrs-cuda` for NVIDIA, `mistralrs-metal` for Apple Silicon, `mistralrs` for CPU/MKL.

## Rust SDK

### `ModelBuilder::build()` requires a tokio runtime

The SDK requires a running tokio runtime. Use `#[tokio::main]` or create a runtime with `tokio::runtime::Runtime::new()`.
