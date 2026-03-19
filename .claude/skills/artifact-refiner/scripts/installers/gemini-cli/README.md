# Gemini CLI

## Installation

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git

# Install as a linked extension (default)
./artifact-refiner-skill/scripts/installers/gemini-cli/install.sh --global

# Or install for current project
./artifact-refiner-skill/scripts/installers/gemini-cli/install.sh --project
```

## What It Does

Gemini CLI uses an **extension** system. This installer:

- **If `gemini` CLI is available**: Runs `gemini extensions link <repo>` to register the skill as a linked extension
- **If `gemini` CLI is not found**: Creates a manual symlink in `~/.gemini/extensions/`

| Scope | Approach |
|---|---|
| Global | `gemini extensions link` or `~/.gemini/extensions/artifact-refiner → <repo>` |
| Project | `.gemini/skills/artifact-refiner → <repo>` |

## How Gemini CLI Extensions Work

Extensions are the primary mechanism to extend Gemini CLI. An extension can bundle:

- **MCP servers** — custom tools
- **Custom commands** — slash commands
- **GEMINI.md** — persistent context for the model
- **Agent Skills** — SKILL.md files

The `gemini extensions link` command creates a symlink from the extensions directory to your development directory. Changes are reflected immediately without reinstalling.

## Managing Extensions

```bash
# List installed extensions
gemini extensions list

# Update extensions
gemini extensions update artifact-refiner

# Disable temporarily
gemini extensions disable artifact-refiner

# Re-enable
gemini extensions enable artifact-refiner
```

## Verify

```bash
gemini extensions list
# Or manually:
ls -la ~/.gemini/extensions/artifact-refiner/SKILL.md
```

## Uninstall

```bash
./artifact-refiner-skill/scripts/installers/gemini-cli/install.sh --uninstall
```

## Requirements

- Gemini CLI (`npm install -g @anthropic/gemini-cli` or via npx)
- Git, Bash
