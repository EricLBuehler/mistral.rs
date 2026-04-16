---
title: Your first model
description: Install mistralrs, download a small language model, and have a conversation with it in your terminal. About five minutes.
sidebar:
  order: 1
---

By the end of this page you will have `mistralrs` installed, a 4-billion-parameter language model loaded onto your GPU, and a chat session open in your terminal. Most of the time you spend will be waiting for weights to download.

This is the first tutorial, so it keeps things narrow. Quantization, HTTP serving, and the language SDKs each get their own page later on. For now, the goal is just to get a model running and send it a message.

## Installing

The install script looks at what kind of accelerator you have (an NVIDIA GPU, an Apple Silicon chip, an Intel CPU with MKL, or nothing special) and builds the binary with the matching features turned on. You do not need to pick feature flags yourself.

On Linux or macOS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

On Windows, from a PowerShell prompt:

```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

The script installs into `~/.cargo/bin/mistralrs`. That directory gets added to your `PATH` as part of the install, but your current shell will not pick that up automatically. Either start a fresh shell, or run `source "$HOME/.cargo/env"` in your current one. After that, confirm the binary is there:

```bash
mistralrs --version
```

If that prints a version number, you are ready to move on. If you get "command not found" or something more exotic like a missing CUDA toolkit, the per-platform [installation guides](/mistral.rs/guides/install/linux-cuda/) cover the usual causes. The rest of this tutorial assumes the script worked.

## Running a model

We are going to use [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). It is a good first model: the weights are around 8 GB in their native BF16 format, which means it loads comfortably on a 12 GB GPU, and Qwen's license is open enough that you will not be asked to sign anything on Hugging Face before downloading. If your GPU is smaller than 12 GB, the download will still work but the model will not fit at native precision. [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) covers how to shrink it.

```bash
mistralrs run -m Qwen/Qwen3-4B
```

The first time you run this command it has more work to do than you might expect. It has to fetch the weights from Hugging Face into your local cache, pick the correct chat template for Qwen3 (no config from you), and load the model onto whatever accelerator it finds. Budget a few minutes on a typical connection. Once the cache is populated, later runs start in seconds.

You will know the model is ready when you see an empty prompt like this:

```
> 
```

From here, type a message and press Enter. The response streams back a token at a time:

```
> What does Rust's ownership system actually buy you?
Rust's ownership model gives you memory safety without a garbage collector...
```

Follow-up questions work the way you would expect. The conversation history carries from turn to turn until you clear it or leave the session.

A handful of commands work at the prompt. The ones you will reach for most are `/clear` to throw away the conversation so far and start fresh, `/exit` (or `Ctrl+D`) to quit, and `/help` to see the rest. There are a few others for changing sampling on the fly that we will not get into here.

## Before you leave

Two things are worth understanding about what that single `mistralrs run` command actually did, because they will come up again in later tutorials.

First, the model loaded at its native precision, which for Qwen3-4B is BF16. That means the full weights had to fit in your GPU's memory. Qwen3-4B is small enough that this is fine on a laptop-class GPU. If you try something larger later (a 32B model, say) and run out of memory, the fix is to quantize the model at load time, and that is what [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) walks through.

Second, you never had to tell `mistralrs` what kind of model Qwen3 is, how its chat template works, or which device to use. The binary figured all of that out from the Hugging Face repository. This is the design that most of the CLI is built around: in the common case, `mistralrs run -m <repo>` should do the obvious thing. When it does not, every choice it makes can be overridden with a flag, but you rarely need them on day one.

## What to try next

If you want to keep going, the obvious next steps are:

- [Serving a model as an API](/mistral.rs/tutorials/02-serve-an-api/), which takes the same model you just ran and puts it behind an OpenAI-compatible HTTP endpoint.
- [Using the Python SDK](/mistral.rs/tutorials/03-python-app/), if you would rather embed a model in a Python program than talk to a running binary.
- [Quantizing a model](/mistral.rs/tutorials/06-quantize-a-model/), for when you want to run something bigger than 4B on the hardware you have.
