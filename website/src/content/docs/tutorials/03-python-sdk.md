---
title: Call a model from Python
description: Install the mistralrs Python package, load a model in-process, send a chat request, and stream tokens. About ten minutes.
sidebar:
  order: 3
---

[Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/) ran a model behind an HTTP server and talked to it using the OpenAI Python client. That is the right shape for production, but for notebooks, scripts, and experiments it is often simpler to skip the server and load the model directly inside your Python process.

The Python SDK is a thin wrapper around the same Rust engine the `mistralrs` binary uses. You get the same quantization, the same chat template auto-detection, and the same multimodal support, just called from Python instead of a command line. In this tutorial we will load Qwen3-4B, ask it a question, and stream a response token by token.

## Installing the right wheel

The Python package ships as a few differently named wheels on PyPI, one per accelerator. Pick the one that matches what you installed the CLI against:

```bash
pip install mistralrs             # CPU, or Intel CPU with MKL
pip install mistralrs-cuda        # NVIDIA GPUs
pip install mistralrs-metal       # Apple Silicon
pip install mistralrs-mkl         # Intel CPU, MKL wheel with symbols pinned
pip install mistralrs-accelerate  # macOS, Accelerate framework
```

You only need one. All of them expose the same `from mistralrs import ...` API at runtime; the difference is only in which backend was compiled in. If you are not sure what your hardware wants, the [install guide](/mistral.rs/guides/install/) has a decision table.

Python 3.10 or newer is required. The wheels are built for Linux, macOS (arm64), and Windows.

## Loading a model

Here is the smallest useful program. Save it as `hello.py`:

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[
            {"role": "user", "content": "In one sentence, what is Rust known for?"}
        ],
        max_tokens=256,
    )
)

print(response.choices[0].message.content)
```

Run it with `python hello.py`. The first execution will take a while because the weights have to download into your Hugging Face cache. Subsequent runs start quickly.

There are two objects worth understanding here before we move on.

`Runner` is the thing that owns the loaded model. Creating one is the expensive step; you generally want exactly one per process and you want it to live for as long as you are using the model. If you are working in a Jupyter notebook, put the `Runner` construction in its own cell so you are not reloading the weights every time you edit a prompt.

`Which` tells the runner what kind of model to load. `Which.Plain(model_id="...")` is the right choice for standard text models like Qwen3. There are other variants for multimodal models (`Which.MultimodalPlain`), quantized checkpoints in GGUF format (`Which.GGUF`), LoRA adapters (`Which.Lora`), and so on. For this tutorial `Which.Plain` is all you need.

The `in_situ_quant="4"` argument is the Python equivalent of the CLI's `--isq 4`. It quantizes weights to 4 bits at load time so the unquantized version never has to fit in memory. Leave it off if you have enough VRAM and want full precision.

## Streaming tokens

Set `stream=True` on the request and the runner returns an iterator of chunks instead of a single response:

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

stream = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Write me a haiku about ownership."}],
        max_tokens=128,
        stream=True,
    )
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

Each chunk is a `ChatCompletionChunkResponse` with the same shape that OpenAI's streaming protocol uses. The interesting field is `choices[0].delta.content`, which carries one incremental piece of the assistant reply. When the model is done you will get a chunk whose `finish_reason` is set and whose `delta.content` is `None`, which is why the example checks that `delta` is truthy before printing.

## Before you leave

A handful of things to know as you build from here.

The Runner keeps the model in memory for the lifetime of the Python process. You can send as many requests against it as you want, sequentially or from separate threads, and each one reuses the loaded weights. If you find yourself wanting to swap models, create a new Runner; the old one will release its GPU memory when it goes out of scope.

Chat history is not automatic. Every call to `send_chat_completion_request` is independent, so if you want multi-turn conversation you build the `messages` list yourself, appending the user's new question and the assistant's previous reply before each new request. That is also how the OpenAI client works, so it should feel familiar if you have used that API.

For everything else the Python SDK exposes, including embedding requests, speech synthesis, image generation, and the full multimodal request shape, the [Python reference](/mistral.rs/reference/python-api/) has the exhaustive surface.

## What to try next

- [Tutorial 4](/mistral.rs/tutorials/04-rust-sdk/) shows the equivalent flow from Rust.
- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) layers tool calling, web search, and code execution on top of a running model.
- The [Python SDK guides](/mistral.rs/guides/python/streaming/) cover streaming with async, multimodal input, and persistent agent sessions.
