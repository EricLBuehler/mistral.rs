---
title: Agents & tools
description: Tool calling, built-in tools, OpenAI-compatible Skills, permissions, sessions, and MCP.
---

mistral.rs can run a complete agent on your own machine. Turn on tools when you start the server, and the model can search the web, run Python, run shell commands, follow Skills, read files you attach, and call your own tools. mistral.rs runs each tool the model asks for, feeds the result back, and lets the model continue until it has an answer, all inside one request. Your app gets the model's text, every tool step, and any files the tools wrote, streamed or in one response. Sessions keep the tool history and the Python interpreter between requests.

The quickest way to try it is to turn on every built-in tool and open the web UI at `http://localhost:1234/ui`:

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

## Where to start

| To | Read |
|---|---|
| Build a working agent, step by step | [Build an agent](/guides/agents/build-an-agent/) |
| Define your own tools, force or limit tool use, and set round caps | [Tool calling](/guides/agents/tool-calling-basics/) |
| Let the model search the web | [Web search](/guides/agents/web-search/) |
| Let it run Python and return charts and files | [Code execution](/guides/agents/enable-code-execution/) |
| Let it run command-line programs | [Shell execution](/guides/agents/enable-shell/) |
| Give it packaged instructions and scripts for a task | [Skills](/guides/agents/skills/) |
| Attach CSVs, documents, or code to a request | [File inputs](/guides/agents/file-inputs/) |
| Use tools from an MCP (Model Context Protocol) server | [Connect to an MCP server](/guides/agents/connect-mcp-server/) |
| Let other agents call your model over MCP | [Expose mistral.rs as an MCP server](/guides/agents/expose-as-mcp/) |
| Require approval before a tool runs | [Permissions and approvals](/guides/agents/permissions-and-approvals/) |
| Keep agent state across requests | [Sessions](/guides/agents/persist-sessions/) |
| Show tool progress, files, and approvals in your own app | [Agentic runtime for apps](/guides/agents/agentic-runtime/) |
