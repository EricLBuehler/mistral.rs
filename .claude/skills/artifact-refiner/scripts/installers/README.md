# Artifact Refiner — Multi-Environment Installers

One-command installation for 12 AI coding environments, using symbolic links to the cloned repository.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/GQAdonis/artifact-refiner-skill.git
cd artifact-refiner-skill

# 2. Run the installer for your environment
scripts/installers/<environment>/install.sh --global
```

## Supported Environments

| Environment | Installer | Install Type | Global Path | Project Path |
|---|---|---|---|---|
| [Claude Code](claude-code/) | `claude-code/install.sh` | Symlink (skill dir) | `~/.claude/skills/` | `.claude/skills/` |
| [Codex CLI](codex/) | `codex/install.sh` | Symlink (skill dir) | `~/.codex/skills/` | `.codex/skills/` |
| [OpenCode](opencode/) | `opencode/install.sh` | Symlink (skill dir) | `~/.config/opencode/skills/` | `.opencode/skills/` |
| [Antigravity](antigravity/) | `antigravity/install.sh` | Symlink (skill dir) | `~/.gemini/antigravity/skills/` | `.agent/skills/` |
| [Cursor](cursor/) | `cursor/install.sh` | .mdc rule + symlink | `~/.cursor/rules/` | `.cursor/rules/` |
| [Gemini CLI](gemini-cli/) | `gemini-cli/install.sh` | Extension link | `~/.gemini/extensions/` | `.gemini/skills/` |
| [Windsurf](windsurf/) | `windsurf/install.sh` | .md rule + symlink | `~/.codeium/windsurf/rules/` | `.windsurf/rules/` |
| [Roo Code](roo-code/) | `roo-code/install.sh` | Symlink (rules dir) | `~/.roo/rules/` | `.roo/rules/` |
| [Cline](cline/) | `cline/install.sh` | Symlink (skill dir) | `~/.cline/skills/` | `.clinerules/` |
| [Kilo Code](kilo-code/) | `kilo-code/install.sh` | Symlink (rules dir) | `~/.kilocode/rules/` | `.kilocode/rules/` |
| [Warp](warp/) | `warp/install.sh` | AGENTS.md + symlink | `~/.warp/skills/` | `.warp/skills/` |
| [Zed](zed/) | `zed/install.sh` | .rules + symlink | `~/.config/zed/skills/` | `.zed/skills/` |

## Common Flags

All installers support:

| Flag | Description |
|---|---|
| `--global` | Install to user-level directory (default for most) |
| `--project` | Install to current project directory |
| `--uninstall` | Remove the symlink / generated files |
| `--help` | Show usage information |

## How It Works

Each installer creates a **symbolic link** from the environment's expected skill/rules directory to the cloned repository. This means:

- ✅ **One copy of the code** — no duplication
- ✅ **Always up-to-date** — `git pull` updates all environments
- ✅ **Idempotent** — safe to run multiple times
- ✅ **Clean uninstall** — `--uninstall` removes only what was created

### Special Cases

- **Cursor** and **Windsurf** use rule files (`.mdc`/`.md`) rather than skill directories. The installer generates an appropriate rule file *and* creates a symlink for full skill access.
- **Gemini CLI** uses its built-in `gemini extensions link` command when available, falling back to a manual symlink otherwise.
- **Cline** requires Skills to be explicitly enabled in settings before they're discoverable.
- **Warp** uses `AGENTS.md` for project rules; the installer appends a section to it and creates a skill symlink.
- **Zed** uses `.rules` files at the worktree root; the installer appends a section and creates a skill symlink.

## Directory Structure

```
scripts/installers/
├── README.md          ← You are here
├── common.sh          ← Shared helpers (colors, symlink ops, flag parsing)
├── antigravity/       ← Google Antigravity
├── claude-code/       ← Claude Code / Claude Desktop
├── cline/             ← Cline / Cline CLI
├── codex/             ← OpenAI Codex CLI
├── cursor/            ← Cursor Agent
├── gemini-cli/        ← Google Gemini CLI
├── kilo-code/         ← Kilo Code
├── opencode/          ← OpenCode
├── roo-code/          ← Roo Code
├── warp/              ← Warp Terminal
├── windsurf/          ← Windsurf (Codeium)
└── zed/               ← Zed IDE
```

Each subdirectory contains `install.sh` + `README.md`.

## Requirements

- **Git** — for cloning the repository
- **Bash** — all scripts are POSIX-compatible bash
- **The target environment** — whichever editor/CLI you're installing for

## Author

Built by [Travis James](https://travisjames.ai) as part of the Artifact Refiner skill project.
