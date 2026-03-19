# Windsurf (Codeium)

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install for current project (default)
./artifact-refiner-skill/scripts/installers/windsurf/install.sh --project

# Or install globally
./artifact-refiner-skill/scripts/installers/windsurf/install.sh --global
```

## What It Does

Windsurf uses `.windsurf/rules/` for Cascade agent rules. This installer:

1. **Generates** `.windsurf/rules/artifact-refiner.md` — a rule file for the Cascade agent
2. **Symlinks** `.windsurf/skills/artifact-refiner → <repo>` — direct access to the full skill

| Scope | Rule File | Skill Symlink |
|---|---|---|
| Project | `.windsurf/rules/artifact-refiner.md` | `.windsurf/skills/artifact-refiner → <repo>` |
| Global | `~/.codeium/windsurf/rules/artifact-refiner.md` | `~/.codeium/windsurf/skills/artifact-refiner → <repo>` |

## How Windsurf Rules Work

Windsurf's Cascade agent reads rule files from `.windsurf/rules/`. Rules provide project-specific context that overrides global rules. The legacy `.windsurfrules` file in the project root is also supported but deprecated.

Global rules are stored at `~/.codeium/windsurf/memories/global_rules.md` or managed via Windsurf Settings → Memories and Rules.

## Verify

```bash
cat .windsurf/rules/artifact-refiner.md
ls -la .windsurf/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/windsurf/install.sh --uninstall
```

## Requirements

- Windsurf IDE
- Git, Bash
