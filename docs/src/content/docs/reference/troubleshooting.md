---
title: Troubleshooting
description: Verified causes and fixes.
---

Before debugging setup issues, run `mistralrs doctor`. It reports detected hardware, compiled accelerator features, and Hugging Face connectivity.

For unlisted issues, file an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues) with a reproducer.

## Installation and build

### `mistralrs: command not found` after install

On macOS and Linux, installer-managed binaries are exposed at `~/.local/bin/mistralrs`, backed by `~/.mistralrs/mistralrs`. Open a new terminal or run the PATH setup command printed by the installer. For POSIX shells, use `. "$HOME/.mistralrs/env"`. Manual Cargo installs use `~/.cargo/bin/mistralrs` and may require `. "$HOME/.cargo/env"`.

On Windows, the installer adds `%USERPROFILE%\.local\bin` to PATH for the current PowerShell session and future terminals. Open a new terminal if another application has an older PATH. If another copy takes precedence, use `command -v mistralrs` on macOS/Linux or `Get-Command mistralrs` in PowerShell to check which one runs.

### Installer download or build fails

The installer prints the failed command's complete diagnostics and the path to its saved log. Retry a failed download; it does not automatically switch to a source build. For live command output, rerun with `--verbose` on macOS/Linux or `-Verbose` in PowerShell. See [installer options](/reference/installer/#command-line-options) for the invocation syntax and [build prerequisites](/developer/from-source/#platform-prerequisites) for required development tools.

### Updating on Windows

Stop running `mistralrs` processes, then rerun the [PowerShell installer](/reference/installer/#guided-installation). Windows cannot replace an executable while it is running.

### Build fails with `flash-attn` feature enabled

Flash attention requires compute capability 8.0+ (see [hardware support](/reference/hardware-support/)). On older GPUs, drop `flash-attn` and rebuild:

- `cuda nccl cudnn` on Linux with NCCL installed.
- `cuda cudnn` otherwise.

## Model loading

### Gated repository (Gemma, LLaMA, FLUX.1-dev, etc.)

Accept the license on the model's Hugging Face page, then run `hf auth login` or set `HF_TOKEN` to a [read-only token](https://huggingface.co/settings/tokens) with access to that repository. mistral.rs reads the shared Hugging Face credentials. Check that you haven't passed `--token-source none`, which ignores saved and environment tokens. See [Hugging Face authentication](/guides/models/run-any-model/#hugging-face-authentication).

### `Out of memory` on load

Quantize the weights. `--quant 4` needs about a quarter of the memory of BF16, and `--quant auto` picks the highest bit width that fits your hardware. See [quantize a model](/guides/quantization/quantize-a-model/).

If you have several GPUs, you can also spread the model across them. See [distributed inference](/guides/perf/distributed-inference/).

## Runtime

### Generation slower than expected

Run `mistralrs doctor` and check the `Build features:` line. If `cuda` (or `metal` on a Mac) is missing, the binary was built without GPU support. Reinstall with the [installer](/reference/installer/) or rebuild with the right [Cargo features](/reference/cargo-features/).

On CUDA, also check that you haven't turned off [paged attention](/guides/perf/paged-attention/) with `--paged-attn off`. The fastest CUDA decode path, with FlashInfer kernels and CUDA graphs, uses paged attention.

### CUDA graphs don't seem to help

Turning graphs off with `MISTRALRS_CUDA_GRAPHS=0` makes little or no difference. Likely causes:

- **You're measuring prompt time.** Graphs speed up generating tokens, not reading the prompt, so time to first token doesn't change. Compare decode tokens per second instead.
- **The model is large.** Graphs cut the CPU cost of launching kernels, which is a big share of each token's time only on small models.
- **You're measuring during warmup.** A new batch size is recorded the first time it appears, which costs time. Measure once traffic is steady.
- **Graphs turned themselves off.** If recording fails, mistral.rs logs a warning and runs that model without graphs. Check the startup log.

See [CUDA graphs](/guides/perf/paged-attention/#cuda-graphs).

### Response cut off

`max_tokens` is most likely too low. Check `finish_reason`:

- `length` - token limit reached.
- `stop` - generation ended naturally (EOS token) or a configured stop token/string matched.

## HTTP server

### `Connection refused` hitting localhost

Check the `Server listening on http://...` line in the server output to confirm host and port.

### CORS errors in a browser

The default allows any origin, but only a fixed set of methods and headers. Check that your request uses those, listed in [CORS and request body limit](/reference/cli-toml-config/#cors-and-request-body-limit). Origins can only be changed by building the server with `MistralRsServerRouterBuilder`.

### `413 Payload Too Large`

The default body limit is 50 MiB, and there's no CLI flag for it. To raise it, build the server with `MistralRsServerRouterBuilder::with_max_body_limit`. See [CORS and request body limit](/reference/cli-toml-config/#cors-and-request-body-limit).

### UI does not load at `/ui`

The UI is on by default. Check that `--no-ui` was not passed at startup, and that no reverse proxy is rewriting `/ui`.

## Sessions

### Sessions disappear between requests

The session expired (30-minute idle TTL) or was evicted (128-session cap, LRU). Long-lived sessions need explicit export/import via `/v1/sessions/{id}`. See [persist sessions](/guides/agents/persist-sessions/).

## Python SDK

### `from mistralrs import Runner` fails with `ImportError`

The wrong wheel was installed. `pip install mistralrs` gives the CPU (Linux/Windows) or Metal (macOS) wheel; for NVIDIA, install a CUDA wheel from the release with `--find-links` and the `+cudaNNN.smNN` matching your driver and GPU. See [Python SDK getting started](/guides/python/getting-started/#installing).

## Rust SDK

### `ModelBuilder::build()` requires a tokio runtime

The SDK requires a running tokio runtime. Use `#[tokio::main]` or create a runtime with `tokio::runtime::Runtime::new()`.
