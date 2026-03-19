# Codex CLI (OpenAI)

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally (recommended)
./artifact-refiner-skill/scripts/installers/codex/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/codex/install.sh --project
```

## What It Does

Creates a symbolic link from Codex's skills directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.codex/skills/artifact-refiner → <repo>` |
| Project | `.codex/skills/artifact-refiner → <repo>` |

## How Codex Discovers Skills

Codex CLI (OpenAI) adopted the Agent Skills open standard (same `SKILL.md` format as Claude Code). Skills in `~/.codex/skills/` are auto-discovered and available across all sessions.

Codex also reads `AGENTS.md` files for project-level instructions (these cascade from root to CWD).

## Verify

```bash
ls -la ~/.codex/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/codex/install.sh --uninstall
```

## Requirements

- Codex CLI (v0.27+)
- Git, Bash
