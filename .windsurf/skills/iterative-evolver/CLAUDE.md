# Claude Code Development Guide

This file provides guidance for AI assistants working **on** this repository (developing, modifying, debugging). For the skill's functionality, see `SKILL.md`. For project overview, see `README.md`.

## Architecture

The skill follows PMPO (Prometheus Meta-Prompting Orchestration) adapted for iterative evolution:
- **Phase controllers** in `prompts/` drive each loop phase
- **State management** via pluggable providers (see `references/state-management.md`)
- **Domain adapters** in `references/domain/` provide domain-specific knowledge
- **Schemas** in `references/schemas/` define the output and provider contracts
- **Subagents** in `agents/` specialize in individual phases
- **Subskills** in `skills/` provide slash command entry points
- **Hooks** in `hooks/` trigger state persistence and workflow dispatch at lifecycle events

## Key Files

| File | Role |
| --- | --- |
| `SKILL.md` | Canonical skill definition — source of truth for behavior |
| `prompts/meta-controller.md` | Orchestration entry point — provider resolution, iteration, error handling |
| `prompts/persist.md` | Phase 6 — provider-agnostic state persistence |
| `hooks/hooks.json` | Lifecycle hooks — per-phase checkpoints, workflow dispatch, finalization |
| `references/state-management.md` | State provider architecture and lifecycle |
| `references/workflow-integration.md` | Workflow trigger semantics and action types |
| `.claude-plugin/plugin.json` | Plugin manifest — skill/agent/hook/MCP registration |
| `.mcp.json` | MCP server configuration (Tavily, sequential-thinking) |

## Development Guidelines

### Modifying Phase Controllers

Each prompt in `prompts/` follows a consistent structure:
1. Role/purpose section
2. Objectives
3. Inputs
4. Process (numbered steps)
5. Output contract (YAML)
6. Rules section
7. Degree of freedom
8. Example

When modifying, preserve all sections and update cross-references.

### Adding a New Domain Adapter

1. Create `references/domain/<name>.md` with domain-specific knowledge
2. Update the routing table in `prompts/meta-controller.md`
3. Update the domain table in root `SKILL.md`
4. Update the domain table in `README.md`
5. Add domain-specific assessment criteria to `prompts/assess.md`

### Adding a New Subagent

1. Create `agents/<name>.md` with YAML frontmatter (`name`, `description`, `allowed_tools`)
2. Define a focused system prompt in the body
3. Reference existing phase controllers and schemas
4. Plugin auto-discovers agents from the `agents/` directory

### Adding a New Subskill

1. Create `skills/<command-name>/SKILL.md` with YAML frontmatter
2. Define setup, user input parsing, and default behavior
3. Add the skill path to `.claude-plugin/plugin.json`

### Adding a New State Provider

1. Create a provider config JSON following `references/schemas/state-provider.schema.json`
2. If custom type: create a provider script that accepts commands: `init`, `get`, `save`, `checkpoint`, `finalize`, `list`
3. Update `references/state-management.md` with provider documentation
4. If MCP-based: add server config to `.mcp.json`

### Adding Workflow Triggers

1. Define triggers following `references/schemas/workflow-trigger.schema.json`
2. Register triggers in one of three places:
   - `workflow_triggers` array in `evolution_state.json` (per-evolution)
   - `triggers.json` in the evolution's state directory (per-project)
   - `default_triggers` in provider config (global)
3. Supported action types: `command`, `webhook`, `mcp_tool`, `workflow_file`
4. See `references/workflow-integration.md` for variable substitution and conditions

### Modifying Hooks

Hooks are defined in `hooks/hooks.json`. Supported events:
- `PostToolUse` — Runs after file write operations (state validation)
- `SubagentStop` — Runs when a subagent completes (checkpoint + dispatch per phase)
- `Stop` — Runs when the main session ends (finalize + cycle_complete dispatch)

Hook scripts live in `scripts/` and must exit 0 (success) or 2 (feedback to agent).

### Key Scripts

| Script | Purpose |
| --- | --- |
| `scripts/state-resolve-provider.sh` | 6-tier provider resolution waterfall |
| `scripts/state-init.sh` | Initialize or resume named evolution |
| `scripts/state-checkpoint.sh` | Mid-session state snapshot |
| `scripts/state-finalize.sh` | Archive end-state to history |
| `scripts/workflow-dispatch.sh` | Match and fire workflow triggers |
| `scripts/validate-state.sh` | Validate evolution_state.json integrity |
| `scripts/finalize-session.sh` | Session cleanup |
| `scripts/log-reflection.sh` | Log reflection output |

## Testing

```bash
# Validate YAML frontmatter in all skills
for f in skills/*/SKILL.md; do head -5 "$f" | grep -q "^---" && echo "✅ $f" || echo "❌ $f"; done

# Check file reference integrity
grep -roh 'references/[a-zA-Z0-9/_.-]*' prompts/ | sort -u | while read f; do [ -e "$f" ] && echo "✅ $f" || echo "❌ $f"; done

# Validate agent frontmatter
for f in agents/*.md; do head -5 "$f" | grep -q "^---" && echo "✅ $f" || echo "❌ $f"; done

# Validate all JSON files
for f in references/schemas/*.json .claude-plugin/*.json hooks/*.json .mcp.json; do
  python3 -c "import json; json.load(open('$f')); print(f'✅ $f')" 2>/dev/null || echo "❌ $f"
done

# Verify all scripts are executable
for f in scripts/*.sh; do [ -x "$f" ] && echo "✅ $f" || echo "❌ $f"; done

# Test provider resolution (should fall back to filesystem)
bash scripts/state-resolve-provider.sh
```

## Key Design Principles

> **Domain-Agnostic Core**: The PMPO loop (meta-controller + phase controllers) must NEVER contain domain-specific logic. All domain knowledge lives in `references/domain/`. If you find yourself adding software-specific or business-specific code to a prompt, put it in a domain adapter instead.

> **Provider-Agnostic State**: The state management layer must NEVER assume a specific backend. All state operations go through the state provider interface. If you find yourself writing directly to `evolution_state.json`, use the state provider scripts instead.

> **Non-Blocking Triggers**: Workflow triggers must NEVER halt the evolution cycle. They are fire-and-forget with logging. If a trigger fails, log the error and continue.
