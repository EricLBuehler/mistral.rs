---
title: Your first model
description: Install mistral.rs, download a small language model, and have a conversation with it in your terminal. About five minutes.
sidebar:
  order: 1
---

## Installing

The install script detects your accelerator (NVIDIA GPU, Apple Silicon, Intel CPU with MKL, or none) and builds the binary with the matching feature flags.

Linux or macOS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

The binary is installed to `~/.cargo/bin/mistralrs`. The installer adds that directory to your `PATH`, but the change does not apply to the current shell. Start a new shell, or run `source "$HOME/.cargo/env"`. Then verify:

```bash
mistralrs --version
```

If the command prints a version, installation succeeded. To check detected hardware, compiled accelerator features, and Hugging Face connectivity, run:

```bash
mistralrs doctor
```

For "command not found" or missing-toolkit errors, see the per-platform [installation guides](/mistral.rs/guides/install/linux-cuda/).

## Running a model

[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) is used here. The native BF16 weights are about 8 GB and fit on a 12 GB GPU. The license does not require acceptance on Hugging Face. On smaller GPUs the download succeeds but the model will not fit at native precision; see [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) for quantization.

```bash
mistralrs run -m Qwen/Qwen3-4B
```

On first run the command:
- Downloads the weights and configuration from Hugging Face into the local cache
- Loads the model onto the detected accelerator

The model is ready when an empty prompt appears:

```
> 
```

Type a message and press Enter. The response streams a token at a time:

```
> What does Rust's ownership system actually buy you?
Rust's ownership model gives you memory safety without a garbage collector...
```

A few commands are available at the prompt: `/clear` resets the conversation, `/exit` (or `Ctrl+D`) quits, and `/help` lists the rest.

## Notes

The model loads at native precision (BF16 for Qwen3-4B), so the full weights must fit in GPU memory. For larger models that do not fit, use `--quant 4`: it prefers a prebuilt UQFF from `mistralrs-community` if one exists, otherwise applies ISQ at 4 bits. `--quant auto` benchmarks your hardware and picks. See [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) for the details.

`mistralrs` infers the model architecture, chat template, and target device from the Hugging Face repository. Every inferred choice can be overridden with a flag.

## Next steps

- [Serving a model as an API](/mistral.rs/tutorials/02-serve-an-api/): put the same model behind an OpenAI-compatible HTTP endpoint.
- [Using the Python SDK](/mistral.rs/tutorials/03-python-sdk/): embed a model in a Python program.
- [Quantizing a model](/mistral.rs/tutorials/06-quantize-a-model/): run larger models on the same hardware.
