---
title: Agents & tools
description: Tool calling, built-in tools, permissions, sessions, and MCP.
---

mistral.rs turns a chat model into something that takes action: it can run the entire tool loop server-side, execute Python and web searches locally, and stream model text, tool progress, files, and media from one request. The agent system has three layers: the tool protocol (how a model requests external work), tool execution (built-in code execution, web search, MCP tools, callbacks, or external dispatch), and the app runtime (streaming output, tool progress, files, generated media, and sessions as one local surface). Start with [tool calling](/mistral.rs/guides/agents/tool-calling-basics/) for the protocol, or jump to [build an agent](/mistral.rs/guides/agents/build-an-agent/) for the end-to-end walkthrough.
