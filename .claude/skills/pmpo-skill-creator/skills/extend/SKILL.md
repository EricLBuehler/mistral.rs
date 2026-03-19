---
name: extend-skill
description: Add new capabilities to an existing Agent Skill without breaking current functionality. Non-destructive additions only.
---

# Extend Skill

Add new domain adapters, phases, sub-skills, schemas, or platform support to an existing skill. All additions are non-destructive — existing files are only modified to add references.

## Usage

```
/extend-skill
```

## What You'll Be Asked

1. **Target skill** — Path to the skill to extend
2. **Extension type** — What to add (domain, sub-skill, schema, platform, etc.)
3. **Extension details** — Specifics of what to add

## Extension Types

| Type | What Gets Added |
|------|----------------|
| **Domain** | New `references/domain/<name>.md` + reference in SKILL.md |
| **Sub-skill** | New `skills/<command>/SKILL.md` + reference in SKILL.md |
| **Schema** | New `references/schemas/<name>.schema.json` |
| **Platform** | New platform output files (plugin.json, tools/, etc.) |
| **Agent** | New `agents/<name>.md` + reference in SKILL.md |
| **Phase** | New `prompts/<phase>.md` + update meta-controller |

## Rules

- **Never delete** existing files
- **Never rename** existing files
- **Only modify** existing files to add references to new components
- Validate that extensions don't conflict with existing components

## Example

```
/extend-skill
> Target: .agent/skills/iterative-evolver
> Type: domain
> Details: Add a "devops" domain adapter for infrastructure evolution
```
