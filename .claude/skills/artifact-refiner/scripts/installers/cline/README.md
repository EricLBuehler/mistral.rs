# Cline / Cline CLI

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally (default)
./artifact-refiner-skill/scripts/installers/cline/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/cline/install.sh --project
```

## What It Does

Creates a symbolic link from Cline's skills/rules directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.cline/skills/artifact-refiner → <repo>` |
| Project | `.clinerules/artifact-refiner → <repo>` |

## Prerequisites

> **Important**: Skills are experimental in Cline and must be enabled manually.

1. Go to **Settings → Features → Enable Skills**
2. The "Skills" tab will appear under the rules/workflows panel

## How Cline Discovers Skills

Cline uses the Agent Skills spec (`SKILL.md` with YAML frontmatter). Skills are discoverable but only fully loaded when relevant to your request — they don't overload the context.

Older `.clinerules` files/directories still work for backward compatibility. The preferred structure is:

```
.clinerules/
├── artifact-refiner/      ← symlink to cloned repo
│   └── SKILL.md
└── other-skill/
    └── SKILL.md
```

## Cline CLI

For the Cline CLI, custom config directories are supported:

```bash
cline --config /path/to/custom/config "your task"
```

Skills in `~/.cline/skills/` are discoverable via the Skills tab.

## Verify

```bash
ls -la ~/.cline/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/cline/install.sh --uninstall
```

## Requirements

- Cline VS Code extension (v3.49+) or Cline CLI
- Skills feature enabled in settings
- Git, Bash
