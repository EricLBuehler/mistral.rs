---
title: The agentic loop
description: Why tool calling runs server-side in mistralrs, what that costs, and when you would want to opt out.
sidebar:
  order: 2
---

Every major language model provider has settled on the same interaction shape for tool calling: the model emits a structured request to call a tool, something else runs the tool, and the model sees the result on the next turn. The interesting design question is where that "something else" lives. Traditionally the answer has been "in the client."

mistralrs does it server-side instead. The tool loop runs inside a single HTTP request: the model calls a tool, the server runs the tool, feeds the result back, and the model keeps going until it produces a regular reply. From the client's point of view, this was just a chat completion that happened to take a while.

This page is about why we made that choice and what it implies for how you build on top.

## The status quo

The classic pattern looks like this:

1. Client sends a prompt with tool definitions.
2. Server returns a response that includes a tool call.
3. Client parses the tool call, runs the tool locally, sends a follow-up request with the result.
4. Server sees the result, produces the final answer.

That is fine for a single tool call. For multi-step plans (search, summarize, search again, execute code, explain), every round is a separate HTTP request. The client has to manage the conversation state, handle retries, respect rate limits, and reason about latency.

More importantly: the model's state on the server has to be re-hydrated on every request. The KV cache from the previous turn is usually gone by the time the next one arrives. The model re-reads the entire conversation, including all the intermediate tool results, from scratch.

For a plan that takes eight tool calls to finish, this is a noticeable latency penalty. It is also a weird place for complexity to live, because the client is doing bookkeeping that has nothing to do with the task the user cares about.

## What we do instead

Server-side loop. When a request comes in with tools enabled:

1. The server runs inference.
2. If the model emits a tool call, the server runs the tool (using whatever resolver applies: the built-in search tool, the Python subprocess, an MCP connection).
3. The tool result gets appended to the in-memory message history.
4. The server runs inference again from where it left off.
5. Loop until the model produces a regular reply or the round cap is hit.

All of this happens inside the same HTTP request. The KV cache is alive across tool rounds, because nothing has released it. The client sends one request and gets back the final response along with an audit log of what happened in between.

## What this buys you

**Latency.** Keeping the KV cache alive across tool rounds saves the prompt-processing work on every round past the first. For an eight-round plan, that is a meaningful speedup. The per-round overhead is tool execution time and a short inference step; prompt processing from scratch would dominate otherwise.

**Simpler clients.** You do not have to implement the tool loop in your client code. The same HTTP request shape that does plain chat does tool-calling chat; only the response payload changes.

**Consistency.** Built-in tools (search, code execution, MCP) are wired in once, on the server, and every client benefits. You do not re-implement them per language SDK.

**Tool reliability.** When a tool fails, the server has full context to decide what to tell the model. It can retry, supply an error message in a structured form the model understands, or bail out and return a useful partial response. A client-side loop has to make those decisions too, but often with less information.

## What this costs

**Less control for clients that want to intercept tool calls.** If your application logic needs to see every tool call before it runs (to audit, transform, or reject), the server-side loop is not the right shape. For those cases, we still support classic client-side tool calling: pass tools in the request, and the server returns the tool call without executing it. You handle the loop.

**Longer-running requests.** A request that fires off several tool calls can take many seconds or minutes. For synchronous clients that expect fast HTTP responses, this is awkward. Streaming responses mitigate it (you get progress events as tools run), and the Responses API surface is designed for async background work if that is what you need.

**Session state on the server.** Because the server remembers what has happened in a multi-turn conversation, it has to manage that memory. We do, with an LRU cache, but it means the server is stateful in a way a pure request-response server is not. For deployments that want zero server-side state, the classic client-side loop sidesteps this.

## When to use which

Server-side loop (the default):

- Normal chat interfaces where the user sends a question and waits for an answer.
- Agentic applications where the tool calls are incidental to the task.
- Anything using the built-in tools (search, code execution, MCP) where there is no reason to interpose.

Client-side loop:

- Applications that need to see or modify every tool call.
- Migrations from existing OpenAI-based code where the loop already exists.
- Deployments that want the server to be stateless.

The engine supports both. The difference is whether you enable the server-side tools with `--enable-search` and friends, or whether you pass `tools` per request and handle the calls yourself.

## A design consequence: tool rounds versus round trips

In the classic model, "rounds" of an agent plan correspond to HTTP round trips. In the mistralrs model, rounds are internal to a single request. This changes how you reason about limits.

Instead of rate limiting by request count, the relevant limit for agents is tool rounds per request, which we cap at 10 by default (configurable via `--max-tool-rounds`). That is the knob for preventing a runaway loop; HTTP rate limits are a separate layer.

## The fallback path

If the model tries to use a tool that does not exist, or the tool fails in a way that cannot be recovered, the server does not just error out. It injects a message into the history describing the problem in a structured form, and continues the loop. The model usually recognizes the failure and tries something else.

This is one of the things that is hard to get right in a client-side loop, because the client has to synthesize error messages the model will understand. Doing it in the engine means we can keep the error format consistent across tool types.
