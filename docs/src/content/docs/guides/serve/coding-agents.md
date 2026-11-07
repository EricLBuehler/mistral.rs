---
title: Use with coding agents
description: Run Claude Code, Codex, and other coding agents on a local model.
---

mistral.rs works with coding agents such as Claude Code and Codex. Serve a model, point your agent at it once, then use the agent in your projects as usual.

Agents fill the context window quickly, so use a model that's good at agentic coding and has a long window. The server reports each model's limit as `max_model_len` in `GET /v1/models`; [context length](/guides/perf/context-length/) explains how to raise it.

## How it works

Two programs run on your machine:

1. **mistral.rs** serves the model.
2. **Your agent** sends prompts to the server, decides what to do, and works on your files.

## Start the server

```bash
mistralrs serve -m Qwen/Qwen3-Coder-Next --quant 4
```

Leave it running. The agents below connect to `localhost:1234`. Check that the model is up, and note its `max_model_len`:

```bash
curl http://localhost:1234/v1/models
```

## Claude Code

Install Claude Code:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Add this to `~/.claude/settings.json`, with `CLAUDE_CODE_MAX_CONTEXT_TOKENS` set to your server's `max_model_len`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:1234",
    "ANTHROPIC_AUTH_TOKEN": "mistralrs",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "default",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "default",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "default",
    "CLAUDE_CODE_SUBAGENT_MODEL": "default",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072"
  }
}
```

Run it in your project directory:

```bash
cd /path/to/project
claude
```

Claude Code picks different models for different jobs, such as background tasks and subagents, so the snippet maps all of them to the local model. The base URL has no `/v1`; Claude Code adds the API path itself.

## Codex

Install Codex:

```bash
npm install -g @openai/codex
```

Add this to `~/.codex/config.toml`, with `model_context_window` set to your server's `max_model_len`:

```toml
model = "default"
model_provider = "mistralrs"
model_context_window = 131072

[model_providers.mistralrs]
name = "mistral.rs"
base_url = "http://localhost:1234/v1"
wire_api = "responses"
```

Run it in your project directory:

```bash
cd /path/to/project
codex
```

To switch between mistral.rs and OpenAI instead of always using the local model, put the snippet in `~/.codex/mistralrs.config.toml` and run `codex --profile mistralrs`.

## Other agents

Point any agent that speaks the OpenAI API at `http://localhost:1234/v1`, or the Anthropic API at `http://localhost:1234`, and use `default` as the model. With [several models loaded](/guides/serve/multiple-models/), use an id from `GET /v1/models` instead.

## Troubleshooting

| Problem | Fix |
|---|---|
| Claude Code gets 404s | Remove `/v1` from `ANTHROPIC_BASE_URL`. |
| Codex gets 404s | Keep `/v1` at the end of `base_url`. |
| The agent reports that the prompt is too long | The conversation outgrew `max_model_len`. Match the agent's context setting to it, or give the server a [longer context](/guides/perf/context-length/). |
| Long prompts time out | Raise Claude Code's `API_TIMEOUT_MS` or Codex's `stream_idle_timeout_ms`. |
| The server is behind a proxy that checks keys | Set the key in `ANTHROPIC_AUTH_TOKEN` for Claude Code, or add `env_key = "MY_KEY_VAR"` to the Codex provider. mistral.rs itself doesn't check keys. |
