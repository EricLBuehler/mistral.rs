---
title: Troubleshooting
description: Symptoms and fixes for common errors and surprises.
sidebar:
  order: 14
---

Keyed by symptom. For unlisted issues, file an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues) with a reproducer.

## Installation and build

### `mistralrs: command not found` after install

The binary is at `~/.cargo/bin/mistralrs` but the directory is not on the current shell's `PATH`. Open a new shell or run `source "$HOME/.cargo/env"`.

### `linker cc not found` or similar during cargo install

Missing C toolchain. Ubuntu: `sudo apt install build-essential`. Fedora: `sudo dnf install gcc`. macOS: `xcode-select --install`.

### `failed to find CUDA ... Is the CUDA toolkit installed?`

CUDA toolkit not installed, or `nvcc` not on `PATH`. Verify with `nvcc --version`. On Linux, the distribution package manager is the typical install path. Set `CUDA_ROOT` for non-standard install locations.

### Build fails with `flash-attn` feature enabled

Flash attention requires compute capability 8.0+. On older GPUs, drop `flash-attn` and rebuild with `cuda cudnn`.

### `rustc too old` during build

mistral.rs requires Rust 1.88+. `rustup update stable` resolves this.

## Model loading

### `Repository not found` or 404 from Hugging Face

The repo id is wrong or the model is gated (license acceptance required) and not yet accepted. Gemma and LLaMA models are commonly gated. Visit the model page on huggingface.co, accept the license, then run `mistralrs login` to save the token.

### `Failed to find config.json` for a valid-looking repo

The repo exists but has no model config. Either the repo is not a full model (e.g., tokenizer-only), or a specific file must be passed via `-f`.

### Load hangs for a long time on first run

Normal. The first run downloads the model into the Hugging Face cache. A 7B model in BF16 is ~14 GB. `--logging` (or `RUST_LOG=info`) shows progress. Subsequent runs start in seconds.

### `Out of memory` immediately on load

The model does not fit in VRAM at native precision. Solutions:

- Add `--isq 4` for 4-bit quantization.
- If still too large, try `--isq 2` or split across GPUs with tensor parallelism.
- If no quantization level fits, use a smaller model or more hardware.

## Runtime

### Generation is very slow

In order:

1. Check `mistralrs doctor`. Verify accelerator features are compiled in. Missing `cuda` means GPU support was not built.
2. Verify the model is on GPU. `nvidia-smi` should show memory matching the loaded model.
3. Check for debug builds. Use `cargo build --release` or `cargo run --release` for benchmarks.
4. For long contexts, paged attention helps. Verify it is on (`mistralrs doctor` reports this).
5. For batched workloads, flash attention helps. Verify it is enabled when supported.

### Output quality is lower than expected

Common causes:

1. Wrong auto-detected chat template. See the [chat templates guide](/mistral.rs/guides/customize/chat-templates/).
2. Sampling too aggressive. Lower `temperature` (0.3 for code, 0.7 for chat) and remove penalties.
3. Quantization too aggressive. Move up a bit width.
4. Wrong prompt format. Raw prompt strings instead of a `messages` array may bypass the chat template.

### Response is cut off partway

Most likely `max_tokens` is too low (often 256 in client libraries). Raise it.

Less commonly, the model hit an internal stop sequence. Check `finish_reason` — `length` means the token limit; `stop` means a stop sequence matched.

### Streaming stops emitting chunks partway through

Usually OOM or an engine error. Check server logs around the stop time.

## Tool calling

### Model does not call tools even when enabled

The model decides. For questions that do not benefit from a tool, it will not call one. To force a specific tool, use `tool_choice: {"type": "function", "function": {"name": "..."}}`.

### Tool calls succeed but the model ignores the result

The tool output shape is wrong. Verify the tool returns valid JSON or a structured `ToolOutput`. Plain text resembling errors gets confused with actual errors.

### `--enable-code-execution` flag not recognized

The binary was built with `--no-default-features`. Rebuild with `code-execution` (in the default feature set).

## HTTP server

### `Connection refused` hitting localhost

Server not running, or bound to a different interface. Check `mistralrs serve` output for the "Serving on ..." line.

### CORS errors in a browser

The request's origin is not in the allowed list. Add it with `--allowed-origin`, or remove the flag entirely to allow any origin (development only).

### `413 Payload Too Large`

Body exceeded the size limit. Raise with `--max-body-limit <bytes>`. Default 50 MB.

### UI does not load at `/ui`

`--ui` was not passed at startup. The endpoint mounts only when the flag is present.

## Sessions

### Sessions disappear between requests

Session expired (30-minute idle TTL) or a different server instance is hitting. For long sessions, export and re-import (see [persist sessions](/mistral.rs/guides/agents/persist-sessions/)).

### `Session not found` on a GET that just worked

The session was likely evicted at the 128-session cap. Import an exported version if available; otherwise restart fresh.

## Python SDK

### `from mistralrs import Runner` fails with `ImportError`

The wheel was not built for the Python version, or the wrong variant was installed. Reinstall with the matching variant: `mistralrs-cuda` for NVIDIA, `mistralrs-metal` for Apple Silicon, `mistralrs` for CPU or MKL.

### `Runner` construction takes forever

Model loading is the slow part. Use one-time setup (or a Jupyter cell) instead of reconstructing per request.

## Rust SDK

### `ModelBuilder::build()` deadlocks or hangs

Verify the tokio runtime is `#[tokio::main]` or created with `tokio::runtime::Runtime`. The SDK requires a running tokio runtime; calling `build().await` from a non-tokio context hangs.

## Things that are not bugs

- Output differences between CUDA and Metal for the same model and seed. Floating-point ordering differs across accelerators; outputs are close but not identical.
- Thinking tokens appearing in Qwen3 output. The model emits them by design; filter in client code if undesired.
- First few tokens slower than subsequent tokens. Time-to-first-token includes prompt processing. Throughput is measured on subsequent tokens.
