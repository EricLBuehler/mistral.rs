# Zed IDE

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install for current project (default)
./artifact-refiner-skill/scripts/installers/zed/install.sh --project

# Or install globally
./artifact-refiner-skill/scripts/installers/zed/install.sh --global
```

## What It Does

### Project Install (default)

1. **Symlinks** `.zed/skills/artifact-refiner → <repo>` for direct skill access
2. **Appends** an artifact-refiner section to `.rules` (creates if needed)

### Global Install

Creates `~/.config/zed/skills/artifact-refiner → <repo>`, then reference it from the Rules Library.

## How Zed Discovers Rules

Zed reads a `.rules` file at the worktree root. It also recognizes (first match wins):

- `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`
- `.github/copilot-instructions.md`
- `AGENT.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`

Zed also has a **Rules Library** (Agent Panel → `...` → `Rules...`) for managing global rules. Rules marked as "Default" are included in every agent thread.

## Using @rule in Threads

You can reference specific rules on demand in the Agent Panel:

```
@rule artifact-refiner
```

Or include the full skill file:

```
/file .zed/skills/artifact-refiner/SKILL.md
```

## Verify

```bash
ls -la .zed/skills/artifact-refiner/SKILL.md
cat .rules  # should contain artifact-refiner section
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/zed/install.sh --uninstall
```

## Requirements

- Zed IDE (v0.170+)
- Git, Bash
