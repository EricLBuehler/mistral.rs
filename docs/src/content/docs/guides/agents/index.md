---
title: Agents & tools
description: Tool calling, built-in tools, OpenAI-compatible Skills, permissions, sessions, and MCP.
---

mistral.rs can run the entire tool-calling loop server-side, execute Python, shell commands, and web searches locally, use OpenAI-compatible Skills, accept OpenAI-compatible file inputs, and stream model text, tool progress, files, and media from one request.

The agent system has three layers:

- **Tool protocol** - how a model requests external work.
- **Tool execution** - built-in code execution, shell, web search, OpenAI-compatible Skills, [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) tools, callbacks, or external dispatch.
- **App runtime** - streaming output, tool progress, files, generated media, and sessions as one local surface.

Where to start:

- [Tool calling](/mistral.rs/guides/agents/tool-calling-basics/) for the protocol.
- [Build an agent](/mistral.rs/guides/agents/build-an-agent/) for the end-to-end walkthrough.
- [OpenAI-compatible Skills](/mistral.rs/guides/agents/skills/) for uploaded Skills and local SDK mounts.
- [OpenAI-compatible file inputs](/mistral.rs/guides/agents/file-inputs/) for uploaded, inline, and URL-backed request files.
- [Shell execution](/mistral.rs/guides/agents/enable-shell/) for command execution, working directories, and sandboxing.
