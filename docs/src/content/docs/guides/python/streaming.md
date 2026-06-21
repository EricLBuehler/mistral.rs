---
title: Stream tokens from Python
description: Async iteration, FastAPI integration, and mid-stream error handling for Python streaming responses.
---

This guide covers consuming a streaming response from async code, from web frameworks, and handling failures mid-stream. The basics (setting `stream=True` to get a synchronous iterator of chunks) are in [getting started](/mistral.rs/guides/python/getting-started/#streaming-tokens).

## Async streaming

The SDK does not expose a native async iterator. Wrap the synchronous iterator in an executor:

```python
import asyncio
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(Which.Plain(model_id="Qwen/Qwen3-4B"))

async def stream_response(prompt: str):
    stream = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="default",
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
from mistralrs import Runner, Which, ChatCompletionRequest

app = FastAPI()
runner = Runner(Which.Plain(model_id="Qwen/Qwen3-4B"))

@app.get("/stream")
async def stream(prompt: str):
    def iter():
        s = runner.send_chat_completion_request(
            ChatCompletionRequest(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        )
        for chunk in s:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(iter(), media_type="text/plain")
```

For production, run mistralrs as an HTTP server and call it with the OpenAI Python client rather than loading the model in the web app process. The HTTP server's streaming is more robust under load; see the [OpenAI-compatible API guide](/mistral.rs/guides/serve/openai-compatible-apis/).

## Catching errors during streaming

Streaming can fail mid-response: out of memory, generation failure, validation errors. The iterator raises `ValueError` with the engine's error message as its next item. Chunks already yielded are unaffected, so partial output survives:

```python
import sys

try:
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
except ValueError as e:
    print(f"\n\nStream ended: {e}", file=sys.stderr)
```

The iterator ends after the chunk whose choices all carry a `finish_reason`.

When server-side tools run during generation (web search, code execution, shell, MCP tools), the chunk iterator skips the engine's tool-progress events transparently; you only receive content chunks. To observe tool progress, use the [agentic runtime](/mistral.rs/guides/agents/agentic-runtime/) event stream instead.
