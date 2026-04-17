---
title: Build agents
description: Tool calling, code execution, web search, MCP, and persistent sessions. The pieces that turn a chat model into something that takes action.
---

Agent applications need a tool loop, session state, safe external code execution, and connections to systems exposing tool APIs. mistral.rs ships these as first-class features.

These guides cover each piece. [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) is the end-to-end walkthrough.

## What is in this section

- [Tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/) — the OpenAI-compatible tool protocol and its mistral.rs implementation.
- [Enable code execution](/mistral.rs/guides/agents/enable-code-execution/) — Python executor, session lifecycle, isolation.
- [Web search](/mistral.rs/guides/agents/web-search/) — built-in search tool configuration.
- [Connect to an MCP server](/mistral.rs/guides/agents/connect-mcp-server/) — act as a client to external tool providers.
- [Expose mistralrs as an MCP server](/mistral.rs/guides/agents/expose-as-mcp/) — let other tools call your models.
- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/) — agent state across separate HTTP requests.
- [Configure the tool loop](/mistral.rs/guides/agents/configure-tool-loop/) — round caps, dispatch URLs, default behavior.

For design rationale (server-side tool loops, session splicing), see the [explanation section](/mistral.rs/explanation/agentic-loop/).
