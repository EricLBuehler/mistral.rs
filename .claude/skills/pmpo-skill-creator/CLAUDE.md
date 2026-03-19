# Claude Code Development Guide

This file provides guidance for AI assistants working **on** this repository (developing, modifying, debugging). For the skill's functionality, see `SKILL.md`.

## Architecture

The skill follows PMPO (Prometheus Meta-Prompting Orchestration):
- **Phase controllers** in `prompts/` drive each loop phase
- **Agents** in `agents/` specialize in architecture, generation, and validation
- **References** in `references/` provide spec knowledge and exemplar analysis
- **Schemas** in `references/schemas/` define state and output contracts
- **Templates** in `assets/templates/` scaffold generated files
- **Scripts** in `scripts/` manage creation state lifecycle

## Key Files

| File | Role |
|------|------|
| `SKILL.md` | Canonical skill definition — source of truth |
| `prompts/meta-controller.md` | Orchestration entry point — mode routing, PMPO loop |
| `references/agentskills-spec.md` | Condensed agentskills.io specification |
| `references/plugin-structure.md` | Claude Code plugin format reference |
| `references/exemplar-skills.md` | Structural analysis of evolver + refiner |
| `scripts/validate-skill.sh` | Generated skill validation |

## Development Guidelines

### Modifying Phase Controllers

Each prompt in `prompts/` follows a consistent structure:
1. Purpose/objective section
2. Procedure/steps section
3. Output contract (YAML example)
4. Rules section

When modifying, preserve all sections and update cross-references.

### Adding a New Template

1. Create the template in `assets/templates/` with `{{variable}}` injection points
2. Document the template variables in the file header
3. Reference the template in `prompts/execute.md`
4. Add the template to the file inventory in `references/exemplar-skills.md`

### Adding Platform Support

1. Add platform-specific generation logic to `references/platform-adapters.md`
2. Update `prompts/execute.md` to generate platform files
3. Update `prompts/reflect.md` to validate platform-specific output
4. Add the platform to the `target_platforms` enum in `references/schemas/skill-spec.schema.json`

### Mode-Specific Development

| Mode | Key Files | Special Behavior |
|------|-----------|-----------------|
| `create` | All templates, all phases | Full generation from scratch |
| `clone` | `prompts/specify.md` (source analysis) | Reads source skill, adapts structure |
| `extend` | `prompts/specify.md` (gap analysis) | Non-destructive additions only |

## Template System

Templates use `{{variable}}` syntax for injection. Variables are resolved during Execute:

| Variable | Source | Example |
|----------|--------|---------|
| `{{skill_name}}` | Specify output | `document-refiner` |
| `{{skill_description}}` | Specify output | `Refine document artifacts...` |
| `{{domain_adapters}}` | Plan output | List of domain .md files |
| `{{phase_list}}` | Plan output | `specify, plan, execute, reflect, persist` |
| `{{allowed_tools}}` | Specify output | `code_interpreter file_system` |

## Script Reference

| Script | Purpose |
|--------|---------|
| `state-resolve-provider.sh` | 6-tier provider resolution |
| `state-init.sh` | Init/resume creation state |
| `state-checkpoint.sh` | Mid-phase snapshot |
| `state-finalize.sh` | Archive completed creation |
| `workflow-dispatch.sh` | Event-driven trigger dispatcher |
| `validate-skill.sh` | Full skill validation suite |

## Testing

```bash
# Validate all JSON schemas
for f in references/schemas/*.json; do python3 -c "import json; json.load(open('$f'))" && echo "✅ $f" || echo "❌ $f"; done

# Check script permissions
for f in scripts/*.sh; do [ -x "$f" ] && echo "✅ $f" || echo "❌ $f"; done

# Cross-reference integrity
grep -roh 'references/[a-zA-Z0-9/_.-]*' prompts/ | sort -u | while read f; do [ -e "$f" ] && echo "✅ $f" || echo "❌ $f"; done

# Validate template variables
grep -roh '{{[a-zA-Z_]*}}' assets/templates/ | sort -u

# Test creation state init
bash scripts/state-init.sh test-skill create
```
