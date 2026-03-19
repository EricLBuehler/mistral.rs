# OpenCode

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally
./artifact-refiner-skill/scripts/installers/opencode/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/opencode/install.sh --project
```

## What It Does

Creates a symbolic link from OpenCode's skills directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.config/opencode/skills/artifact-refiner → <repo>` |
| Project | `.opencode/skills/artifact-refiner → <repo>` |

## How OpenCode Discovers Skills

OpenCode searches multiple locations for skills:

- `.opencode/skills/<name>/SKILL.md` (project)
- `~/.config/opencode/skills/<name>/SKILL.md` (global)
- `.claude/skills/<name>/SKILL.md` (Claude-compatible)
- `.agents/skills/<name>/SKILL.md` (agents-compatible)

Skills are loaded on-demand via the native `skill` tool. Access permissions can be configured in `opencode.json`.

## Custom Config Directory

You can also use `OPENCODE_CONFIG_DIR`:

```bash
export OPENCODE_CONFIG_DIR=/path/to/my/config
# Skills will be discovered in $OPENCODE_CONFIG_DIR/skills/
```

## Verify

```bash
ls -la ~/.config/opencode/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/opencode/install.sh --uninstall
```

## Requirements

- OpenCode CLI
- Git, Bash
