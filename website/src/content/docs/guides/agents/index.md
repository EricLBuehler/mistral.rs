---
title: Build agents
description: Tool calling, code execution, web search, MCP, and persistent sessions. The pieces that turn a chat model into something that takes action.
---

Agent-shaped applications are what models are really good at now, but they require the engine to do a few things that plain chat does not: run a tool loop, manage session state, execute external code safely, and talk to other systems that expose tool APIs. mistral.rs ships all of those as first-class features.

These guides cover the pieces one at a time. [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) is the end-to-end walkthrough if you have not done it yet.

## What is in this section

- [Tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/): the OpenAI-compatible tool protocol and how it works in mistralrs.
- [Enable code execution](/mistral.rs/guides/agents/enable-code-execution/): turn on the Python executor, understand session lifecycle, isolate failures.
- [Web search](/mistral.rs/guides/agents/web-search/): configure the built-in search tool.
- [Connect to an MCP server](/mistral.rs/guides/agents/connect-mcp-server/): act as a client to external tool providers.
- [Expose mistralrs as an MCP server](/mistral.rs/guides/agents/expose-as-mcp/): let other tools call your models.
- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/): keep agent state across separate HTTP requests.
- [Configure the tool loop](/mistral.rs/guides/agents/configure-tool-loop/): round caps, dispatch URLs, default behavior for new requests.

For the design rationale (why server-side tool loops, how session splicing works), see the [explanation section](/mistral.rs/explanation/agentic-loop/).
