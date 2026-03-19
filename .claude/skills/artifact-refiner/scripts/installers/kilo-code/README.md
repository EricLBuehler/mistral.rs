# Kilo Code

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally (default)
./artifact-refiner-skill/scripts/installers/kilo-code/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/kilo-code/install.sh --project
```

## What It Does

Creates a symbolic link from Kilo Code's rules directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.kilocode/rules/artifact-refiner → <repo>` |
| Project | `.kilocode/rules/artifact-refiner → <repo>` |

## How Kilo Code Discovers Rules

Kilo Code reads `.md` files from:

- `.kilocode/rules/` — project-level rules (all modes)
- `.kilocode/rules-${mode}/` — mode-specific rules
- `~/.kilocode/rules/` — global rules

Rules can also be managed through Kilo Code's built-in UI (Rules tab).

## Mode-Specific Rules

For mode-specific integration, create a directory named `.kilocode/rules-code/`:

```bash
mkdir -p .kilocode/rules-code
ln -s /path/to/artifact-refiner-skill .kilocode/rules-code/artifact-refiner
```

## Verify

```bash
ls -la ~/.kilocode/rules/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/kilo-code/install.sh --uninstall
```

## Requirements

- Kilo Code VS Code extension
- Git, Bash
