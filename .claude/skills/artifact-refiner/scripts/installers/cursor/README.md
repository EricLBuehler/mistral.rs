# Cursor / Cursor Agent

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install for current project (default)
./artifact-refiner-skill/scripts/installers/cursor/install.sh --project

# Or install globally
./artifact-refiner-skill/scripts/installers/cursor/install.sh --global
```

## What It Does

Cursor uses `.mdc` rule files (not skill directories). This installer:

1. **Generates** `.cursor/rules/artifact-refiner.mdc` — a rule file describing when and how to use the skill
2. **Symlinks** `.cursor/skills/artifact-refiner → <repo>` — direct access to the full skill

| Scope | Rule File | Skill Symlink |
|---|---|---|
| Project | `.cursor/rules/artifact-refiner.mdc` | `.cursor/skills/artifact-refiner → <repo>` |
| Global | `~/.cursor/rules/artifact-refiner.mdc` | `~/.cursor/skills/artifact-refiner → <repo>` |

## How Cursor Rules Work

Cursor rules are `.mdc` (Markdown Cursor) files in `.cursor/rules/`. Each file has YAML frontmatter with:

- `description` — when the rule should apply
- `globs` — file patterns to trigger on (optional)
- `alwaysApply` — whether to always include in context

The generated rule is set to `alwaysApply: false` (Agent Select mode), meaning Cursor's agent decides when to load it based on the task description.

## Verify

```bash
cat .cursor/rules/artifact-refiner.mdc
ls -la .cursor/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/cursor/install.sh --uninstall
```

## Requirements

- Cursor IDE (0.48+)
- Git, Bash
