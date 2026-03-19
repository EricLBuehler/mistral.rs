# Roo Code

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install globally (default)
./artifact-refiner-skill/scripts/installers/roo-code/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/roo-code/install.sh --project
```

## What It Does

Creates a symbolic link from Roo Code's rules directory to the cloned repo:

| Scope | Target Path |
|---|---|
| Global | `~/.roo/rules/artifact-refiner → <repo>` |
| Project | `.roo/rules/artifact-refiner → <repo>` |

## How Roo Code Discovers Rules

Roo Code reads `.md` files recursively from:

- `.roo/rules/` — workspace-wide rules (all modes)
- `.roo/rules-{mode-slug}/` — mode-specific rules
- `~/.roo/rules/` — global rules (all projects)

Files are loaded alphabetically and appended to the mode's instructions. The skill's `SKILL.md` and prompt files will be automatically discovered.

## Custom Modes

You can create a Roo mode specifically for artifact refinement:

1. Open Roo Code panel → Prompts (book icon) → "+" button
2. Name: `Artifact Refiner`, Slug: `artifact-refiner`
3. Role: "Expert artifact refinement agent using PMPO methodology"
4. Custom Instructions: `.roo/rules-artifact-refiner/`

Then create `.roo/rules-artifact-refiner/` → symlink to the repo.

## Verify

```bash
ls -la ~/.roo/rules/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/roo-code/install.sh --uninstall
```

## Requirements

- Roo Code VS Code extension
- Git, Bash
