---
title: Build an agent
description: Turn on tool calling, web search, and code execution so the model can take actions, not just reply. Watch the agentic loop work end to end. About fifteen minutes.
sidebar:
  order: 5
---

So far every request in these tutorials has been a single round trip: you send a prompt, the model replies, and that is the end of the exchange. That works fine for most things you would use a chat model for, but it leaves the interesting part of the last few years of LLM development on the table. Modern models know how to call tools. Given a description of what a tool does and access to invoke it, they will decide on their own when to reach for one and how to use the result.

mistral.rs ships a server-side implementation of this loop. When you turn it on, the server sees the model ask for a tool, runs that tool, feeds the result back to the model, and keeps going until the model produces a normal reply. All of this happens inside a single HTTP request, so your client just sees the final answer. In this tutorial we will turn on two of the built-in tools (web search and Python code execution), watch them work from the web UI, and then look at the structured data the server returns so you can build on top of it from code.

We will use Qwen3-4B throughout. Nothing here is specific to Qwen; the same flags and response shapes work against any model that supports tool calling, which is most of the ones mistral.rs ships today.

## Starting the server with agents enabled

Both features are off by default because each one carries a cost (network access for search, a Python subprocess for code execution). Turning them on is a pair of flags:

```bash
mistralrs serve --ui \
  --enable-search \
  --enable-code-execution \
  -m Qwen/Qwen3-4B
```

`--enable-search` lets the model issue web search queries through the built-in search tool. `--enable-code-execution` lets it run Python in an isolated subprocess that persists across calls within a session. Code execution is compiled in by default, so unless you built `mistralrs` with `--no-default-features` the flag should just work.

Open `http://localhost:1234/ui` in a browser once the server is ready.

## Watching it work in the UI

Paste this into the chat box:

```
What is the current population of Tokyo, and what fraction of Japan's total population does that represent? Show your working.
```

The response takes longer than a normal chat reply because the model is now running a loop. What you should see is:

1. A collapsed block labeled with a search icon appears, showing the query Qwen decided to run. Click it open and you get the URLs and snippets the server retrieved.
2. A second block appears for a Python execution. Open it and you see the exact code the model asked to run, followed by stdout.
3. Sometimes a third or fourth block for follow-up searches or calculations.
4. Finally a normal-looking reply that cites the numbers and explains the arithmetic.

This is the agentic loop. Everything between your question and the final reply happened inside a single HTTP request. From the client's point of view, this was just a chat completion; from the server's point of view, several rounds of "model asks, server executes, server feeds the result back" played out in between.

The UI is rendering structured events that the server sends as part of the response. You do not need the UI to access them, and in practice for production you would be parsing the same events from your own code. That is the next thing to look at.

## What the API returns

Stop the UI and hit the endpoint directly:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the square root of 99856? Verify by squaring the result."}
    ]
  }'
```

If everything is set up correctly, the model will decide to run a Python block to do the math rather than guessing. The response body will look roughly like this, with the usual `choices` array plus an extra `agentic_tool_calls` field:

```json
{
  "id": "...",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "316. Squaring 316 gives 99856, which matches."
      },
      "finish_reason": "stop"
    }
  ],
  "agentic_tool_calls": [
    {
      "round": 0,
      "name": "execute_python",
      "arguments": "{\"code\":\"import math\\nprint(math.sqrt(99856))\\nprint(316**2)\"}",
      "result_content": "316.0\n99856\n",
      "result_images_base64": []
    }
  ],
  "usage": { "..." : "..." }
}
```

The `choices` array carries what the model said in the end; that part is OpenAI-compatible and any existing client will read it correctly. The `agentic_tool_calls` field is an extension specific to mistral.rs. Each entry in the array records one round of the loop: which tool was invoked, the arguments the model passed in, what the server got back. If a tool produced images (for example matplotlib plots from a code block), they show up as base64 strings in `result_images_base64`.

If you prefer to stream the response, the loop emits progress events as Server-Sent Events with type `agentic_tool_call_progress`. Each event has a `phase` of either `"calling"` (right before the tool runs) or `"complete"` (with the result), which is how the UI animates the collapsible blocks. The full schema for these events lives in the [HTTP API reference](/mistral.rs/reference/http-api/).

## Before you leave

A few things that trip people up on their first agent.

The model has to want to call a tool. Enabling the flags does not force it to search or execute code on every question. It gives the model the option, along with a description of what each tool is for, and the model decides. Asking "what is 2 + 2" will still just produce "4", because the model does not need a calculator for that. Asking for the factorial of 37, on the other hand, usually triggers a code block.

Code execution is stateful within a session. A subsequent request that reuses the same session id picks up the same Python subprocess, so variables defined earlier are still available. When you do not pass a session id, a new one is created for you and returned in the response. This is what makes multi-turn agents practical: the model can build up a computation incrementally without having to re-run every previous step. The [persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/) goes into this in detail.

The tools you get with the two flags we used are the built-in ones. If you want to expose your own tools (a calendar API, a vector search over your internal docs, a shell), you have two options: implement them as MCP servers and connect mistral.rs to them as a client, or register tool callbacks directly through the Rust or Python SDK. Both are covered in the [agent guides](/mistral.rs/guides/agents/).

## What to try next

- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) returns to the basics: how to fit a bigger model on your GPU.
- [The MCP client guide](/mistral.rs/guides/agents/connect-mcp-server/) walks through connecting mistral.rs to a third-party MCP server as a source of tools.
- [The persistent sessions guide](/mistral.rs/guides/agents/persist-sessions/) covers how to build longer-running agents that keep state across separate requests.
