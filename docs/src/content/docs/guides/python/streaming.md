---
title: Stream tokens from Python
description: Iterate over tokens as they generate. Blocking and async patterns for working with streaming responses in Python.
sidebar:
  order: 1
---

Streaming displays output as it generates rather than after the full response. The Python SDK exposes streaming as a plain iterator usable from sync and async code.

## Blocking streaming

The simplest pattern is a for loop:

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(Which.Plain(model_id="Qwen/Qwen3-4B"), in_situ_quant="4")

stream = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Explain prime numbers."}],
        max_tokens=256,
        stream=True,
    )
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

Each chunk's `delta.content` is a string (one step of new output) or `None`. `None` appears at stream start (role set, no text yet) and end (finish reason emitted).

## Async streaming

The SDK does not expose a native async iterator. Wrap the synchronous iterator in an executor:

```python
import asyncio
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(Which.Plain(model_id="Qwen/Qwen3-4B"), in_situ_quant="4")

async def stream_response(prompt: str):
    stream = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="Qwen/Qwen3-4B",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
    )

    loop = asyncio.get_event_loop()
    while True:
        chunk = await loop.run_in_executor(None, next, stream, None)
        if chunk is None:
            break
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Consume with `async for`:

```python
async def main():
    async for delta in stream_response("Write a haiku."):
        print(delta, end="", flush=True)
```

## Streaming into a web framework

For FastAPI, the same pattern works as a response generator:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
runner = Runner(Which.Plain(model_id="Qwen/Qwen3-4B"), in_situ_quant="4")

@app.get("/stream")
async def stream(prompt: str):
    def iter():
        s = runner.send_chat_completion_request(
            ChatCompletionRequest(
                model="Qwen/Qwen3-4B",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        )
        for chunk in s:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(iter(), media_type="text/event-stream")
```

For production, run mistralrs as an HTTP server (see [Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/)) and call it with the OpenAI Python client rather than loading the model in the web app process. The HTTP server's streaming is more robust under load.

## Catching errors during streaming

Streaming can fail mid-response: out of memory, generation failure, client disconnect. Wrap the loop:

```python
try:
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
except Exception as e:
    print(f"\n\nStream ended: {e}", file=sys.stderr)
```

The engine flushes generated content before raising. Partial output is preserved.
