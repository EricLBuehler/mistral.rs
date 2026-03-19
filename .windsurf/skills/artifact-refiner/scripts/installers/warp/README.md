# Warp Terminal

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install for current project (default)
./artifact-refiner-skill/scripts/installers/warp/install.sh --project

# Or install global skill symlink
./artifact-refiner-skill/scripts/installers/warp/install.sh --global
```

## What It Does

### Project Install (default)

1. **Symlinks** `.warp/skills/artifact-refiner → <repo>` for direct skill access
2. **Appends** an artifact-refiner section to `AGENTS.md` (creates if needed)

### Global Install

Creates `~/.warp/skills/artifact-refiner → <repo>` for access across projects.

> **Note**: Warp's Global Rules are managed via the Warp Drive UI. Use `--global` to create a skill symlink, then reference it from a Global Rule.

## How Warp Discovers Rules

Warp uses `AGENTS.md` as its default project rules file. It also recognizes:

- `WARP.md`, `CLAUDE.md`, `GEMINI.md`
- `.cursorrules`, `.clinerules`, `.windsurfrules`
- `.github/copilot-instructions.md`

Rules precedence:
1. Subdirectory `AGENTS.md`
2. Root `AGENTS.md`
3. Global Rules (Warp Drive)

## Verify

```bash
ls -la .warp/skills/artifact-refiner/SKILL.md
grep artifact-refiner AGENTS.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/warp/install.sh --uninstall
```

## Requirements

- Warp terminal (v2.0+)
- Git, Bash
