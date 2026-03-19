# Antigravity (Google DeepMind)

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally
./artifact-refiner-skill/scripts/installers/antigravity/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/antigravity/install.sh --project
```

## What It Does

Creates a symbolic link from Antigravity's skills directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.gemini/antigravity/skills/artifact-refiner → <repo>` |
| Project | `.agent/skills/artifact-refiner → <repo>` |

## How Antigravity Discovers Skills

Antigravity is Google's agentic coding platform. It auto-discovers skills that contain a `SKILL.md` file with YAML frontmatter (name, description, etc.). Skills are loaded on-demand based on task context.

Project-level skills go in `.agent/skills/`. Antigravity also supports workflows (`.agent/workflows/`) and rules (`.agent/rules/`).

## Verify

```bash
ls -la ~/.gemini/antigravity/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/antigravity/install.sh --uninstall
```

## Requirements

- Google Antigravity
- Git, Bash
