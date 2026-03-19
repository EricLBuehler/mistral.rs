# Claude Code Development Guide

This file provides guidance for AI assistants working **on** this repository (developing, modifying, debugging). For the skill's functionality, see `SKILL.md`. For project overview, see `README.md`.

## Architecture

The skill follows PMPO (Prometheus Meta-Prompting Orchestration):
- **Phase controllers** in `prompts/` drive each loop phase
- **Domain adapters** in `references/domain/` provide domain-specific knowledge
- **Schemas** in `references/schemas/` define the output contracts
- **Subagents** in `agents/` specialize in individual phases
- **Templates** in `assets/templates/` provide output scaffolding
- **State scripts** in `scripts/` manage named artifact lifecycle

## Key Files

| File | Role |
| --- | --- |
| `SKILL.md` | Canonical skill definition — source of truth for behavior |
| `prompts/meta-controller.md` | Orchestration entry point — routing, iteration, error handling |
| `prompts/persist.md` | Phase 5 — state persistence procedures |
| `hooks/hooks.json` | Lifecycle hooks — validation, logging, cleanup, checkpoints |
| `references/content-types.md` | Content type taxonomy — direct vs meta-prompt |
| `.claude-plugin/plugin.json` | Plugin manifest — skill/agent/hook/MCP registration |
| `.mcp.json` | e2b sandbox MCP server configuration |

## Development Guidelines

### Modifying Phase Controllers

Each prompt in `prompts/` follows a consistent structure:
1. Purpose/objective section
2. Procedure/steps section
3. Output contract (YAML example)
4. Rules section
5. Examples section (added in v1.1.0)

When modifying, preserve all sections and update cross-references.

### Adding a New Domain

1. Create `references/domain/<name>.md` with domain-specific knowledge
2. Add a template in `assets/templates/` if the domain needs output scaffolding
3. Update the routing table in `prompts/meta-controller.md`
4. Update the domain table in root `SKILL.md`
5. Create `skills/refine-<name>/SKILL.md` as a slash command
6. Add the new skill path to `.claude-plugin/plugin.json`

### Adding a New Content Type

1. Add the type to `references/schemas/content-type.schema.json` enum
2. Document the type in `references/content-types.md`
3. Map to a domain adapter in `prompts/meta-controller.md` Content Type Routing table
4. Update detection heuristics in `prompts/specify.md` Content Type Detection section
5. If `meta:*`, ensure `domain/meta-prompt.md` covers the platform-specific knowledge

### Adding a New Subagent

1. Create `agents/<name>.md` with YAML frontmatter (`name`, `description`, `allowed_tools`)
2. Define a focused system prompt in the body
3. Reference existing phase controllers and schemas
4. Plugin auto-discovers agents from the `agents/` directory

### Modifying Hooks

Hooks are defined in `hooks/hooks.json`. Supported events:
- `PostToolUse` — Runs after file write operations
- `SubagentStop` — Runs when a subagent completes (includes phase checkpoint + dispatch)
- `Stop` — Runs when the main session ends (includes finalize + complete dispatch)

Hook scripts live in `scripts/` and must exit 0 (success) or 2 (feedback to agent).

## State Provider Development

### How Providers Work

The system uses a 6-tier waterfall (`scripts/state-resolve-provider.sh`) to identify the active state backend:

| Tier | Source | Config File |
|------|--------|-------------|
| 1 | `$REFINER_PROVIDER_CONFIG` env var | Any path |
| 2 | Project-local | `.refiner-provider.json` |
| 3 | Global | `~/.refiner/provider.json` |
| 4 | MCP tool probe | Auto-detected |
| 5 | Agent memory probe | Auto-detected |
| 6 | Filesystem fallback | Always available |

### Implementing a Custom Provider

Create a custom provider by defining three commands in `state-provider.schema.json`:

```json
{
  "provider_type": "custom",
  "config": {
    "init_command": "my-provider init --name $ARTIFACT_NAME",
    "save_command": "my-provider save --name $ARTIFACT_NAME --state $STATE_FILE",
    "load_command": "my-provider load --name $ARTIFACT_NAME"
  }
}
```

## Workflow Trigger Authoring

Configure triggers in the refinement state to fire on lifecycle events:

```json
{
  "event": "on_refinement_complete",
  "action": {
    "type": "command",
    "target": "echo 'Refinement complete for ${artifact_name}'"
  }
}
```

Supported action types: `command`, `webhook`, `mcp_tool`, `workflow_file`.

Variable substitution uses `${event_field}` syntax (see `references/schemas/hook-event.schema.json`).

## Content Type System

### Direct vs Meta

- **`direct:*`** — Refiner produces the final artifact (`.tsx`, `.html`, `.svg`, etc.)
- **`meta:*`** — Refiner produces a *prompt* that drives another system (DALL-E, Sora, Claude, etc.)

### Evaluation Strategy

| Strategy | Used By | Focus |
|----------|---------|-------|
| `output_inspection` | `direct:*` | Evaluate generated files |
| `prompt_quality` | `meta:*` | Evaluate prompt clarity and specificity |
| `test_execution` | `meta:*` (test gen on) | Generate test output, evaluate both |

See `references/content-types.md` for the full taxonomy and detection heuristics.

## Script Reference

| Script | Purpose |
|--------|---------|
| `state-resolve-provider.sh` | 6-tier provider resolution |
| `state-init.sh` | Init/resume named artifact state |
| `state-checkpoint.sh` | Mid-phase state snapshot |
| `state-finalize.sh` | Archive and finalize refinement |
| `workflow-dispatch.sh` | Event-driven trigger dispatcher |
| `validate-manifest.sh` | Manifest schema validation |
| `post-execute-check.sh` | Post-execute file verification |
| `log-reflection.sh` | Reflection logging |
| `finalize-session.sh` | Session cleanup |
| `validate-constraints.sh` | Constraint validation |

## Design Principles

1. **Provider-agnostic** — All state operations go through the resolved provider; never hardcode filesystem paths in prompts
2. **Non-blocking triggers** — Workflow dispatches are fire-and-forget; never halt the PMPO loop
3. **Content-type-driven** — Evaluation strategy adapts to content type; `meta:*` evaluates prompts, `direct:*` evaluates output
4. **Schema-first** — All state structures have JSON schemas; validate before persisting

## Testing

```bash
# Run marketplace validation
bash scripts/validate-marketplace.sh

# Validate YAML frontmatter in all skills
for f in skills/*/SKILL.md; do head -5 "$f" | grep -q "^---" && echo "✅ $f" || echo "❌ $f"; done

# Check file reference integrity
grep -roh 'references/[a-zA-Z0-9/_.-]*' prompts/ | sort -u | while read f; do [ -e "$f" ] && echo "✅ $f" || echo "❌ $f"; done

# Validate all JSON schemas
for f in references/schemas/*.json; do python3 -c "import json; json.load(open('$f'))" && echo "✅ $f" || echo "❌ $f"; done

# Test provider resolution
bash scripts/state-resolve-provider.sh

# Test state init
bash scripts/state-init.sh test-artifact content direct:content

# Verify no hardcoded brand references remain
grep -r "sediment://" . --include="*.md"  # Should return nothing
```
