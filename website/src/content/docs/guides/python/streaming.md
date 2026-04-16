---
title: Stream tokens from Python
description: Iterate over tokens as they generate. Blocking and async patterns for working with streaming responses in Python.
sidebar:
  order: 1
---

Streaming lets you show the user a response as it generates rather than waiting for the whole thing. The Python SDK's streaming is exposed as a plain iterator, which makes it easy to use from both regular scripts and async applications.

## Blocking streaming

The simplest pattern is a for loop over the stream:

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

Each chunk's `delta.content` is either a string (one step of new output) or `None`. The `None` values appear at the start of the stream (when the role is set but no text is generated yet) and at the end (when the finish reason is emitted). Checking for truthiness skips them cleanly.

## Async streaming

For async applications, wrap the synchronous iterator in an executor. The SDK does not expose a native async iterator yet, but running the blocking loop in a thread works fine:

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

    # Run the blocking iterator in a thread, yielding chunks back to asyncio.
    loop = asyncio.get_event_loop()
    while True:
        chunk = await loop.run_in_executor(None, next, stream, None)
        if chunk is None:
            break
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Then consume it with `async for`:

```python
async def main():
    async for delta in stream_response("Write a haiku."):
        print(delta, end="", flush=True)
```

## Streaming into a web framework

For FastAPI or similar, the same pattern works as a response generator:

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

For a production server, consider running mistralrs as an HTTP server (see [Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/)) and calling it with the OpenAI Python client rather than loading the model inside your web app process. The HTTP server's streaming is more robust under load.

## Catching errors during streaming

Streaming can fail mid-response: out of memory, a subsequent token cannot be generated, or the client disconnects. Wrap the loop in a try/except so your code handles partial responses gracefully:

```python
try:
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
except Exception as e:
    print(f"\n\nStream ended: {e}", file=sys.stderr)
```

The mistralrs engine flushes whatever has been generated so far before raising. You will not lose partial output.
