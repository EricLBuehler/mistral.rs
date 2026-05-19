---
title: Call a model from Python
description: Install the mistralrs Python package, load a model in-process, send a chat request, and stream tokens. About ten minutes.
sidebar:
  order: 3
---

The Python SDK loads the model in-process and wraps the same Rust engine that backs the `mistralrs` binary.

## Installing the right wheel

The Python package ships as one wheel per accelerator. Install the one matching your hardware:

```bash
pip install mistralrs             # CPU, or Intel CPU with MKL
pip install mistralrs-cuda        # NVIDIA GPUs
pip install mistralrs-metal       # Apple Silicon
pip install mistralrs-mkl         # Intel CPU, MKL wheel with symbols pinned
pip install mistralrs-accelerate  # macOS, Accelerate framework
```

Install only one. All wheels expose the same `from mistralrs import ...` API; they differ only in the compiled backend.

Python 3.10 or newer is required. Wheels are built for Linux, macOS (arm64), and Windows.

## Loading a model

Save as `hello.py`:

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

Run with `python hello.py`. The first run downloads the weights into the Hugging Face cache.

[`Runner`](/mistral.rs/reference/python/runner/) owns the loaded model. Construction loads the weights; reuse one `Runner` for the lifetime of the process to avoid reloading.

[`Which`](/mistral.rs/reference/python/which/) selects the model loader. `Which.Plain(model_id="...")` is correct for standard text models. Other variants exist for multimodal models (`Which.MultimodalPlain`), GGUF checkpoints (`Which.GGUF`), and LoRA adapters (`Which.Lora`).

`in_situ_quant="4"` is the equivalent of the CLI's `--isq 4`. It quantizes weights to 4 bits at load time. Omit it for full precision.

## Streaming tokens

Set `stream=True` to receive an iterator of chunks instead of a single response:

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

Each chunk is a `ChatCompletionChunkResponse` with the OpenAI streaming shape. `choices[0].delta.content` carries one incremental piece of the reply. The terminating chunk has `finish_reason` set and `delta.content == None`, which is why the example checks `delta` before printing.

## Notes

The Runner keeps the model in memory for the process lifetime. Requests can be sent sequentially or from multiple threads, all reusing the loaded weights. To swap models, construct a new Runner; the old one releases GPU memory when it goes out of scope.

Chat history is not tracked. Each call to `send_chat_completion_request` is independent. Multi-turn conversation requires assembling the `messages` list manually, appending each new user question and prior assistant reply.

The full Python surface (embeddings, speech, image generation, multimodal requests) is documented in the [Python reference](/mistral.rs/reference/python/).

## Next steps

- [Tutorial 4](/mistral.rs/tutorials/04-rust-sdk/): the equivalent flow from Rust.
- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/): add tool calling, web search, and code execution.
- The [Python SDK guides](/mistral.rs/guides/python/streaming/) cover async streaming, multimodal input, and persistent agent sessions.
