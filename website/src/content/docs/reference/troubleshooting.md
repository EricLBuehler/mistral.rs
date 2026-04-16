---
title: Troubleshooting
description: Symptoms and fixes for common errors and surprises.
sidebar:
  order: 14
---

This page is keyed by symptom rather than cause. Find what is happening, read the entry, follow the fix. For anything not listed here, an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues) with enough detail to reproduce is the best path forward.

## Installation and build

### `mistralrs: command not found` after install

The binary is in `~/.cargo/bin/mistralrs` but that directory is not on your current shell's `PATH`. Either open a new shell, or run `source "$HOME/.cargo/env"` in your current one.

### `linker cc not found` or similar during cargo install

Your system is missing a C toolchain. On Ubuntu: `sudo apt install build-essential`. On Fedora: `sudo dnf install gcc`. On macOS: `xcode-select --install`.

### `failed to find CUDA ... Is the CUDA toolkit installed?`

Either the CUDA toolkit is not installed, or `nvcc` is not in `PATH`. Verify with `nvcc --version`. On Linux, installing via your distribution's package manager usually places it correctly. Set `CUDA_ROOT` before building if you have a non-standard install location.

### Build fails with `flash-attn` feature enabled

Flash attention needs compute capability 8.0 or higher. On older GPUs, drop `flash-attn` from your features and rebuild with just `cuda cudnn`.

### `rustc too old` during build

mistralrs needs Rust 1.88 or newer. `rustup update stable` fixes this.

## Model loading

### `Repository not found` or 404 from Hugging Face

Either the repo id is wrong, or the model is gated (requires accepting a license) and you have not accepted it yet. Gemma and LLaMA models are commonly gated. Visit the model page on huggingface.co, accept the license, then run `mistralrs login` to save your token.

### `Failed to find config.json` for a valid-looking repo

The repo exists but mistralrs could not find a model config. Either the repo is not a full model (maybe it only has a tokenizer), or you need to specify a specific file with `-f`.

### Load hangs for a long time on first run

This is normal. The first run downloads the model into your Hugging Face cache. Weights for a 7B model are roughly 14 GB in BF16; expect a few minutes on a typical connection. The `--logging` flag (or `RUST_LOG=info`) shows progress. Once downloaded, subsequent runs start in seconds.

### `Out of memory` immediately on load

The model does not fit in your GPU's VRAM at its native precision. Solutions:

- Add `--isq 4` for 4-bit quantization.
- If it still does not fit, try `--isq 2` or split across GPUs with tensor parallelism.
- If you cannot fit any quantization level, you need a smaller model or more hardware.

## Runtime

### Generation is very slow

Several possible causes. Work through them in order:

1. Check `mistralrs doctor`. Verify the accelerator features are compiled in. If you are using CUDA but `cuda` is not in the feature list, the binary was built without GPU support.
2. Verify the model is actually on GPU. `nvidia-smi` should show memory use corresponding to the loaded model.
3. Check if you are in debug mode. Always use `cargo build --release` or `cargo run --release` when benchmarking.
4. For very long contexts, paged attention helps noticeably. Make sure it is on (`mistralrs doctor` reports this too).
5. For batched workloads, flash attention helps. Make sure it is enabled if your GPU supports it.

### Output quality is lower than expected

A few common causes:

1. The chat template was auto-detected wrong. See the [chat templates guide](/mistral.rs/guides/customize/chat-templates/).
2. Sampling is too aggressive. Try lowering `temperature` (try 0.3 for code, 0.7 for chat) and removing penalties.
3. Quantization is too aggressive. Move up a bit width.
4. Prompt format is off. If you are passing a raw prompt string instead of a `messages` array, the chat template may not be applied.

### Response is cut off partway

Most likely `max_tokens` is set too low (default is often 256 in client libraries). Raise it.

Less commonly, the model hit an internal stop sequence. Check `finish_reason` in the response; `length` means you hit the token limit, `stop` means a stop sequence matched.

### Streaming stops emitting chunks partway through

Usually an OOM or an engine-level error. Check the server logs for an error line around the time the stream stopped.

## Tool calling

### Model does not call tools even when they are enabled

The model has to decide to use them. For questions that do not benefit from a tool (basic factual questions), it will not. If you believe a tool should be used, try rewording the prompt to make the case for it clearer.

If you want to force a specific tool, use `tool_choice: {"type": "function", "function": {"name": "..."}}` in the request.

### Tool calls succeed but the model ignores the result

Something is wrong with the tool output shape. Check that the tool is returning either a valid JSON string or a structured `ToolOutput`. Plain text that looks like an error message gets confused with actual errors.

### `--enable-code-execution` flag not recognized

The binary was built with `--no-default-features`. Rebuild with the `code-execution` feature, which is part of the default set.

## HTTP server

### `Connection refused` hitting localhost

Either the server is not running, or it is bound to a different interface. Check `mistralrs serve` output for the "Serving on ..." line that reports host and port.

### CORS errors in a browser

The request's origin is not in the server's allowed list. Either add it with `--allowed-origin` or remove the allowed-origin flag entirely to allow any origin (appropriate for development, not for production).

### `413 Payload Too Large`

The request exceeded the body size limit. Raise with `--max-body-limit <bytes>`. Default is 50 MB.

### UI does not load at `/ui`

Did you pass `--ui` at startup? The endpoint is only mounted when that flag is set.

## Sessions

### Sessions disappear between requests

Either the session expired (30-minute idle TTL) or you are hitting a different server instance than the one holding the state. For long-running sessions, export and re-import them (see [persist sessions](/mistral.rs/guides/agents/persist-sessions/)).

### `Session not found` on a GET that just worked

The session may have been evicted due to the 128-session cap. Import the exported version if you have one; otherwise restart fresh.

## Python SDK

### `from mistralrs import Runner` fails with `ImportError`

The wheel was not built with support for your Python version, or you pip-installed the wrong variant for your hardware. Reinstall with the matching variant: `mistralrs-cuda` for NVIDIA, `mistralrs-metal` for Apple Silicon, `mistralrs` for CPU or MKL.

### `Runner` construction takes forever

Model loading does. Put it in a one-time setup (or a Jupyter cell) rather than reconstructing it per request.

## Rust SDK

### `ModelBuilder::build()` deadlocks or hangs

Make sure your tokio runtime is `#[tokio::main]` or created with `tokio::runtime::Runtime`. The SDK requires a running tokio runtime; calling `build().await` from a non-tokio context hangs.

## Things that are not bugs

- Output differences between CUDA and Metal for the same model and seed. Floating-point ordering differs between accelerators; outputs will be close but not identical.
- Thinking tokens appearing in output for Qwen3. The model is designed to emit them; if you do not want to see them, filter them in your client code.
- First few tokens being slower than subsequent tokens. This is the time-to-first-token; it includes prompt processing. Throughput is measured on subsequent tokens.
