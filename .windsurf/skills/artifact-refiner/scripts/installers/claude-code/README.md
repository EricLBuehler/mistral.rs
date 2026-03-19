# Claude Code / Claude Desktop

## Installation

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally (available in all projects)
./artifact-refiner-skill/scripts/installers/claude-code/install.sh --global

# Or install for current project only
./artifact-refiner-skill/scripts/installers/claude-code/install.sh --project
```

## What It Does

Creates a symbolic link from your Claude Code skills directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.claude/skills/artifact-refiner → <repo>` |
| Project | `.claude/skills/artifact-refiner → <repo>` |

## Alternative: Plugin Marketplace

If the skill is published to the Claude plugin marketplace:

```bash
claude plugin install artifact-refiner
```

Or from a local directory:

```bash
/plugin add /path/to/artifact-refiner-skill
```

## Verify

```bash
ls -la ~/.claude/skills/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/claude-code/install.sh --uninstall
```

## How Claude Discovers Skills

Claude Code auto-discovers skills in `~/.claude/skills/` (global) and `.claude/skills/` (project). Each skill must have a `SKILL.md` file with YAML frontmatter. Claude reads this to understand when and how to invoke the skill.

The artifact-refiner's slash commands (`/refine-logo`, `/refine-ui`, etc.) are implemented as nested skills in the `skills/` subdirectory.

## Requirements

- Claude Code or Claude Desktop
- Git (for cloning the repo)
- Bash shell
