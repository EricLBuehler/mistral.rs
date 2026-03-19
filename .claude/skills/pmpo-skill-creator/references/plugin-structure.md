# Claude Code Plugin Structure Reference

Reference for generating Claude Code marketplace-compatible plugins.

## Plugin Directory Structure

```
plugin-name/
├── .claude-plugin/        # Metadata directory — ONLY plugin.json goes here
│   └── plugin.json        # Plugin manifest (required for marketplace)
├── commands/              # Slash commands (simple markdown files)
├── agents/                # Agent definitions (markdown)
├── skills/                # Agent Skills (SKILL.md directories)
│   ├── skill-one/
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   └── references/
│   └── skill-two/
│       └── SKILL.md
├── hooks/                 # Lifecycle hooks
│   └── hooks.json
├── .mcp.json              # MCP server configuration (optional)
├── .lsp.json              # Language server configuration (optional)
└── README.md              # Documentation (for humans, NOT loaded by agents)
```

> **Critical**: Commands, agents, skills, and hooks go at the plugin ROOT, NOT inside `.claude-plugin/`.

## plugin.json Manifest

```json
{
  "name": "plugin-name",
  "description": "What this plugin does",
  "version": "1.0.0"
}
```

Fields:
- `name` — Plugin namespace, becomes prefix for slash commands (`/plugin-name:command`)
- `description` — Human-readable description
- `version` — Semver version string

## Skills in Plugins

Each skill is a directory with a `SKILL.md` inside the `skills/` directory:

```
skills/
├── code-review/
│   └── SKILL.md
└── pdf-processor/
    ├── SKILL.md
    └── scripts/
```

Skills are automatically discovered when the plugin is installed.

## Hooks

`hooks/hooks.json` defines lifecycle hooks:

```json
{
  "hooks": [
    {
      "event": "SubagentStop",
      "match_agent": "agent_name",
      "steps": [
        {
          "type": "command",
          "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/hook-script.sh"
        }
      ]
    },
    {
      "event": "Stop",
      "steps": [
        {
          "type": "command", 
          "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/finalize.sh"
        }
      ]
    }
  ]
}
```

### `${CLAUDE_PLUGIN_ROOT}` Variable

Use this variable in hook commands and MCP configs to reference files within the plugin's installation directory. This is necessary because plugins are copied to a cache location when installed.

### Hook Events

| Event | Trigger |
|-------|---------|
| `PostToolUse` | After a tool is used |
| `SubagentStop` | When a subagent completes |
| `Stop` | When the main session ends |

## Marketplace Distribution

### marketplace.json

```json
{
  "marketplace_version": "1.0.0",
  "name": "Marketplace Name",
  "plugins": [
    {
      "name": "plugin-name",
      "version": "1.0.0",
      "description": "What this plugin does",
      "path": "plugins/plugin-name"
    }
  ]
}
```

### Hosting

- GitHub (recommended): Push to repo, users install with `/plugin marketplace add <url>`
- Users refresh with `/plugin marketplace update`

## Agent Definitions

`agents/*.md` files define subagents with custom system prompts:

```markdown
# Agent Name

Role description and system prompt for the agent.

## Instructions
- What this agent does
- Its constraints
- Its tools
```

## Installation

```bash
# From GitHub
claude-code /plugin install <github-url>

# Local development
claude-code --plugin-dir /path/to/plugin
```
