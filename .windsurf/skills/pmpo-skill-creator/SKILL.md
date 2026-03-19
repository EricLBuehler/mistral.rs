---
name: pmpo-skill-creator
description: >
  Use this skill to create, clone, or extend Agent Skills using PMPO orchestration.
  Generates complete, production-ready skills with agentskills.io spec compliance,
  Claude Code plugin/marketplace support, OpenCode tools compatibility, state management,
  lifecycle hooks, and workflow triggers. Supports three modes: create (from scratch),
  clone (adapt existing skill to new domain), and extend (add capabilities to existing skill).
allowed-tools: code_interpreter file_system
---

# PMPO Skill Creator

A PMPO-driven generator for production-ready Agent Skills. Creates complete skill packages with full lifecycle support — state management, domain adapters, phase controllers, hooks, schemas, sub-skills, and multi-platform distribution.

## Creation Modes

### `create` — New skill from intent

Generates a complete skill from a description of what it should do. Produces all files needed for PMPO orchestration, agentskills.io compliance, and multi-agent deployment.

### `clone` — Adapt existing skill

Copies the structure of an existing skill (like iterative-evolver or artifact-refiner) and adapts it for a new domain. Preserves architectural patterns while replacing domain-specific content.

### `extend` — Add capabilities

Adds new domain adapters, phases, sub-skills, or platform support to an existing skill without breaking current functionality.

## What Gets Generated

Every generated skill includes:

| Component | Files | Purpose |
|-----------|-------|---------|
| Core | `SKILL.md`, `CLAUDE.md`, `AGENTS.md` | Skill manifest, dev guide, contributor guide |
| Prompts | `prompts/*.md` | PMPO phase controllers (specify, plan, execute, reflect, persist) |
| Agents | `agents/*.md` | Specialized subagent definitions |
| References | `references/*.md` | Domain adapters, theory docs, architecture refs |
| Schemas | `references/schemas/*.json` | JSON schema contracts for state and output |
| Scripts | `scripts/*.sh` | State lifecycle management (init, checkpoint, finalize, dispatch) |
| Sub-skills | `skills/*/SKILL.md` | Slash command entry points |
| Hooks | `hooks/hooks.json` | Lifecycle hooks with checkpoint + dispatch |
| Plugin | `.claude-plugin/plugin.json` | Claude Code marketplace manifest |
| Templates | `assets/templates/*` | Output scaffolding for generated artifacts |

## Inputs

```yaml
skill_name: string       # Required — name for the new skill
intent: string           # What the skill should do
mode: string             # create | clone | extend
source_skill: string     # Required for clone/extend — path to source skill
target_platforms: array   # Which platforms to support (default: all)
  # Options: agentskills-io, claude-code, opencode, cursor, gemini-cli
tools_directory: string   # Optional — custom tools output directory (e.g., .opencode/tools/)
domain: string           # Optional — primary domain for the skill
```

## Outputs

```yaml
generated_skill:
  path: string           # Directory containing the complete skill
  file_count: integer    # Total files generated
  platforms: array       # Platforms supported
  validation: object     # Spec compliance results
```

## Artifact-Refiner Integration (Optional)

When the artifact-refiner skill is available, the creator can delegate markdown and prompt content generation to it using `content_type: meta:agent-prompt`. This produces iteratively refined prompt text. If artifact-refiner is not available, the creator generates content directly.

## Execution Model (PMPO Loop)

### Startup Protocol

1. **Resolve Provider**: `scripts/state-resolve-provider.sh`
2. **Init/Resume State**: `scripts/state-init.sh <skill_name> [mode]`
3. **Mode Routing**: Route to create, clone, or extend workflow

### Phase Loop

1. **Specify** (`prompts/specify.md`) — Clarify intent, classify mode, analyze source skill
2. **Plan** (`prompts/plan.md`) — Design architecture, map files, plan domain adapters
3. **Execute** (`prompts/execute.md`) — Generate all files using templates + AI
4. **Reflect** (`prompts/reflect.md`) — Validate against agentskills.io spec, check completeness
5. **Persist** (`prompts/persist.md`) — Write validated state and generated files
6. **Loop or Terminate** — Continue if validation fails, stop if spec-compliant

### Phase Hooks

After each phase: checkpoint (`state-checkpoint.sh`) + dispatch (`workflow-dispatch.sh`)

## Required Tools

- `code_interpreter` or e2b MCP sandbox — JSON schema validation, YAML parsing
- `file_system` — Read source skills, write generated files

## Persistent State Files

- `.creator/skills/<name>/state.json` — Named creation state
- `.creator/registry.json` — Skill creation registry
- `dist/<skill-name>/` — Generated skill output directory

## Quality Standards

Generated skills MUST:
1. Pass agentskills.io SKILL.md validation (required frontmatter, <500 lines)
2. Include all JSON schemas referenced in prompts
3. Have executable scripts with proper shebang lines
4. Include cross-reference integrity (no dangling file references)
5. Support at least agentskills.io + Claude Code plugin format
6. Include state management lifecycle (init, checkpoint, finalize)
7. Include hooks with per-phase checkpoint + dispatch

## Exemplar Skills

The creator uses these as structural templates:

| Skill | Location | Files | Domains |
|-------|----------|-------|---------|
| iterative-evolver | UAR `.agent/skills/` | 48 | 8 (software, business, product, etc.) |
| artifact-refiner | standalone repo | 79 | 7 (logo, ui, code, meta-prompt, etc.) |

Both demonstrate the complete feature set: PMPO loop, named state, providers, hooks, workflow triggers, sub-skills, domain adapters, JSON schemas.

## Quick Start

- `/create-skill` — Create a new skill from scratch
- `/clone-skill` — Clone an existing skill for a new domain
- `/extend-skill` — Add capabilities to an existing skill
- `/validate-skill` — Validate a skill against the agentskills.io spec
