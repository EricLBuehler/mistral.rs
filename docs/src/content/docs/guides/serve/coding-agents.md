---
title: Use Codex and Claude Code
description: Configure coding agents to use a local mistral.rs server.
sidebar:
  order: 7
---

mistral.rs can back coding-agent clients through the compatibility APIs those
clients already speak.

| Client | API surface | Base URL to configure |
|---|---|---|
| Codex | OpenAI Responses | `http://localhost:1234/v1` |
| Claude Code | Anthropic Messages | `http://localhost:1234` |

Use `default` for a single-model server. With multi-model serving, use a model
id exactly as it appears in `GET /v1/models`.

## Start mistral.rs

Start a model that is tuned for tool use and code:

```bash
mistralrs serve -p 1234 -m Qwen/Qwen3-Coder-Next
```

Check that the server is reachable:

```bash
curl http://localhost:1234/v1/models
```

For Codex and Claude Code, let the coding agent own normal file edits, shell
commands, and repository inspection. Enable mistral.rs server-side agent tools
only when you deliberately want web search or Python code execution to happen
inside the mistral.rs server as well.

## Codex

Codex uses the OpenAI Responses wire API for custom providers. Put provider
configuration in your user-level `~/.codex/config.toml`; Codex ignores provider
configuration in project-local `.codex/config.toml` files.

```toml
model = "default"
model_provider = "mistralrs"
model_context_window = 32768
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false

[model_providers.mistralrs]
name = "mistral.rs"
base_url = "http://localhost:1234/v1"
wire_api = "responses"
request_max_retries = 1
stream_max_retries = 0
stream_idle_timeout_ms = 300000

[profiles.mistralrs]
model = "default"
model_provider = "mistralrs"
```

Then launch Codex with the `mistralrs` profile:

```bash
codex --profile mistralrs
```

If a reverse proxy enforces authentication, add `env_key = "MISTRALRS_API_KEY"`
under `[model_providers.mistralrs]` and export that variable before launching
Codex. The local mistral.rs server itself does not validate API keys.

Codex tool calls arrive through `/v1/responses` as Responses function tools.
mistral.rs routes them through the same tool-calling path used by Chat
Completions. For direct Responses examples, see the
[OpenAI Responses API guide](/mistral.rs/guides/serve/openai-responses-api/).

## Claude Code

Claude Code should use the Anthropic-compatible root URL, without `/v1`.
Claude Code appends `/v1/messages` and `/v1/messages/count_tokens` itself.

For a persistent local setup, add this to `~/.claude/settings.json` or to a
project-local `.claude/settings.local.json`:

```json
{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:1234",
    "ANTHROPIC_API_KEY": "not-used",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "default",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "default",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "mistral.rs default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Local model served by mistral.rs",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

Or set the same values for one shell session:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_API_KEY=not-used
export ANTHROPIC_MODEL=sonnet
export ANTHROPIC_DEFAULT_SONNET_MODEL=default
export ANTHROPIC_DEFAULT_OPUS_MODEL=default
export ANTHROPIC_DEFAULT_HAIKU_MODEL=default
export ANTHROPIC_CUSTOM_MODEL_OPTION=default
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

Mapping the Sonnet, Opus, and Haiku defaults to `default` makes the `sonnet`
setting and Claude Code background or planning calls use the single loaded
mistral.rs model. If you serve several models, map each Claude Code default to
the local model id you want for that role.

Claude Code client tools arrive as Anthropic tool definitions and later
`tool_result` content blocks. mistral.rs translates these to its internal tool
format. For direct Anthropic examples, see the
[Anthropic Messages API guide](/mistral.rs/guides/serve/anthropic-messages-api/).

## Server-side agent tools

The coding clients already provide their own editing and shell tools. Server-side
mistral.rs tools are separate:

| Feature | How it reaches mistral.rs |
|---|---|
| Client-side tool use | Codex Responses function tools or Claude Code Anthropic tools |
| Server web search | `web_search_options` or Anthropic `web_search_*` server tools |
| Server code execution | `enable_code_execution` or Anthropic `code_execution_*` server tools |

Server-side web search and code execution require the corresponding mistral.rs
agentic runtime flags at server startup. Use them when you want the model server
to perform web search or Python execution independently of the coding client's
own terminal tools.

## Examples

Copyable config snippets live in `examples/server/`:

| File | What it shows |
|---|---|
| `codex_config.toml` | User-level Codex provider config for `/v1/responses`. |
| `claude_code_settings.json` | Claude Code settings for `/v1/messages`. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Codex returns 404 | Include `/v1` in the Codex provider `base_url`. |
| Claude Code returns 404 | Remove `/v1` from `ANTHROPIC_BASE_URL`. |
| The client requests an Anthropic model id | Use `default`, or map Claude Code default model env vars to your local ids. |
| A remote server accepts any key | Put authentication in a reverse proxy. mistral.rs does not validate compatibility API keys. |
| Claude Code sends beta fields your proxy rejects | Set `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` in Claude Code. |
| Long streams time out | Raise Codex `stream_idle_timeout_ms` or Claude Code `API_TIMEOUT_MS`. |

## Upstream references

- [Codex configuration reference](https://developers.openai.com/codex/config-reference)
- [Claude Code environment variables](https://code.claude.com/docs/en/env-vars)
- [Claude Code LLM gateway configuration](https://code.claude.com/docs/en/llm-gateway)
