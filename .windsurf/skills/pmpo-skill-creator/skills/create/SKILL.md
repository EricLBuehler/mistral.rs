---
name: create-skill
description: Create a new Agent Skill from scratch using PMPO orchestration. Generates a complete skill package with all required files.
---

# Create Skill

Generate a complete, production-ready Agent Skill from a description of what it should do.

## Usage

```
/create-skill
```

## What You'll Be Asked

1. **Skill name** — What to call the skill (kebab-case)
2. **Intent** — What the skill should do
3. **Complexity** — Simple (3-5 files), Standard (15-25), or Full (30-50+)
4. **Platforms** — Which platforms to support (agentskills-io, claude-code, opencode)
5. **Tools directory** — Custom path for OpenCode tools (optional)

## What Gets Generated

A complete skill directory in `dist/<skill-name>/` with:
- Core files (SKILL.md, CLAUDE.md, AGENTS.md)
- PMPO phase controllers
- Agent definitions
- Reference materials
- JSON schemas
- State management scripts (if standard/full)
- Lifecycle hooks (if standard/full)
- Platform-specific outputs

## Example

```
/create-skill
> Name: code-reviewer
> Intent: Review code changes for quality, security, and best practices
> Complexity: standard
> Platforms: agentskills-io, claude-code
```
