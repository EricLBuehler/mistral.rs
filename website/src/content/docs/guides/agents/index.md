---
title: Build agents
description: Tool calling, code execution, web search, MCP, generated media, and persistent sessions. The pieces that turn a chat model into something that takes action.
---

mistral.rs supports server-side tool calling, code execution, web search, MCP client, generated media, and persistent sessions. [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) is the end-to-end walkthrough.

## Contents

- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/): model output, tool progress, generated media, and sessions as one local runtime surface.
- [Tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/): the OpenAI-compatible tool protocol and its mistral.rs implementation.
- [Enable code execution](/mistral.rs/guides/agents/enable-code-execution/): Python executor, session lifecycle, isolation.
- [Web search](/mistral.rs/guides/agents/web-search/): built-in search tool configuration.
- [Connect to an MCP server](/mistral.rs/guides/agents/connect-mcp-server/): act as a client to external tool providers.
- [Expose mistralrs as an MCP server](/mistral.rs/guides/agents/expose-as-mcp/): let other tools call your models.
- [Persist sessions](/mistral.rs/guides/agents/persist-sessions/): agent state across separate HTTP requests.
- [Configure the tool loop](/mistral.rs/guides/agents/configure-tool-loop/): round caps, dispatch URLs, default behavior.

For design rationale (server-side tool loops, session splicing), see the [explanation section](/mistral.rs/explanation/agentic-loop/).
