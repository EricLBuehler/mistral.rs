---
title: The agentic loop
description: Why tool calling runs server-side in mistralrs, what that costs, and when you would want to opt out.
sidebar:
  order: 2
---

Tool calling has a standard interaction shape across providers: the model emits a structured request, something else runs the tool, and the model sees the result on the next turn. The design question is where that "something else" runs. The traditional answer is the client.

mistral.rs runs it server-side. The tool loop runs inside one HTTP request: the model calls a tool, the server runs it, feeds the result back, and the model continues until producing a regular reply. Clients see one chat completion that took longer than usual.

This page covers the rationale and the implications.

## The status quo

The classic pattern:

1. Client sends a prompt with tool definitions.
2. Server returns a response containing a tool call.
3. Client parses the call, runs the tool locally, sends a follow-up request with the result.
4. Server sees the result and produces the final answer.

Fine for one tool call. For multi-step plans (search, summarize, search again, execute code, explain), every round is a separate HTTP request. The client manages conversation state, retries, rate limits, and latency.

The KV cache from the previous turn is usually gone by the time the next request arrives. The model re-reads the entire conversation including all intermediate tool results, from scratch.

For an eight-tool-call plan, this is a noticeable latency penalty. It also puts complexity in a place that has nothing to do with the user's task.

## What we do instead

Server-side loop. With tools enabled:

1. The server runs inference.
2. On a tool call, the server runs the tool (built-in search, Python subprocess, MCP connection — whichever applies).
3. Tool result is appended to the in-memory message history.
4. Server resumes inference from where it left off.
5. Loop until a regular reply or the round cap.

All inside one HTTP request. The KV cache stays alive across rounds because nothing has released it. The client receives the final response plus an audit log of intermediate steps.

## Benefits

**Latency.** Keeping the KV cache alive across rounds saves prompt-processing on every round past the first. For an eight-round plan, this is meaningful. Per-round cost reduces to tool execution plus a short inference step instead of full prompt reprocessing.

**Simpler clients.** No client-side loop required. The same HTTP request shape works for plain chat and tool-calling chat; only the response payload changes.

**Consistency.** Built-in tools (search, code execution, MCP) are wired in once on the server. Every client benefits without per-SDK reimplementation.

**Tool reliability.** On tool failure, the server has full context to decide what to tell the model. It can retry, supply a structured error message, or bail out and return a partial response. Client-side loops have to make these decisions with less information.

## Costs

**Less control for clients wanting to intercept tool calls.** Applications requiring tool-call inspection (audit, transform, reject) need a different shape. Classic client-side tool calling is still supported: pass tools in the request, the server returns the tool call without executing it. The client handles the loop.

**Longer-running requests.** A request firing many tool calls can take seconds or minutes. Synchronous clients expecting fast HTTP responses are awkward here. Streaming responses give progress events; the Responses API supports async background work.

**Server-side session state.** Multi-turn conversation state lives on the server, managed via LRU cache. The server is stateful in a way a pure request-response server is not. Stateless deployments should use the classic client-side loop.

## When to use which

Server-side loop (default):

- Standard chat interfaces.
- Agentic applications where tool calls are incidental to the task.
- Any use of built-in tools (search, code execution, MCP) without interposition.

Client-side loop:

- Applications needing to see or modify every tool call.
- Migrations from existing OpenAI-based code with an existing loop.
- Stateless server deployments.

The engine supports both. The choice is whether to enable server-side tools (`--enable-search`, etc.) or pass `tools` per request and handle calls in the client.

## A design consequence: tool rounds versus round trips

In the classic model, agent plan rounds correspond to HTTP round trips. In mistral.rs, rounds are internal to one request. Limit reasoning changes accordingly.

The relevant cap is tool rounds per request, default 10 (`--max-tool-rounds`). HTTP rate limits are a separate layer.

## The fallback path

When the model invokes a nonexistent tool or a tool fails unrecoverably, the server does not error out. It injects a structured error message into the history and continues the loop. The model usually recognizes the failure and tries something else.

This is hard to get right in client-side loops because the client must synthesize error messages the model understands. Doing it in the engine keeps the error format consistent across tool types.
