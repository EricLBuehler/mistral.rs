# Slash Command Formats by Tool

This document describes how each AI tool implements slash commands,
and how `kbd-generate-commands.sh` maps the canonical `workflows.yaml`
definition to each tool's native format.

---

## Overview: The Cross-Tool Standard

KBD uses a single canonical YAML source (`workflows/workflows.yaml`) and a
bash generator script (`scripts/kbd-generate-commands.sh`) to produce
tool-specific command files. All tools share the same underlying pattern:
**a Markdown file whose name becomes the slash command name**.

Run the generator after any change to `workflows.yaml`:
```bash
bash .agent/skills/kbd-process-orchestrator/scripts/kbd-generate-commands.sh
```

Or target a specific tool:
```bash
bash .agent/skills/kbd-process-orchestrator/scripts/kbd-generate-commands.sh --tool=claude
```

---

## Tool Reference

### Antigravity (Gemini AI)

**Directory**: `.agent/workflows/`
**Invocation**: `/kbd-assess` (slash command shown in Antigravity skill list)
**Format**:
```markdown
---
description: Short description shown in slash command picker
---

[workflow instructions]
```
**Arguments**: Passed via `$ARGUMENTS` or accessed from the Antigravity conversation context.
**Notes**: Antigravity reads the `description` from SKILL.md frontmatter for the parent skill,
and reads per-command content from `.agent/workflows/<name>.md`.

---

### Claude Code

**Directory**: `.claude/commands/` (project-scoped, team shareable)
**Global home**: `~/.claude/commands/` (personal/cross-project)
**Invocation**: `/kbd-assess` typed in Claude Code chat
**Format**:
```markdown
---
description: Short description shown in command picker
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch
---

[workflow instructions]
Arguments available as $ARGUMENTS or $1, $2, ...
Bash commands prefixed with ! are executed directly.
```
**Arguments**: `$ARGUMENTS` (all text after command), `$1`, `$2`, etc.
**Notes**: Supports `!bash-command` execution inline. Team-shareable via git.

---

### Cursor Agent

**Directory**: `.cursor/commands/`
**Invocation**: Type `/` in Cursor Agent chat → select command from autocomplete
**Format**:
```markdown
# /command-name

Description of what this command does.

[workflow instructions]
Arguments: $ARGUMENTS
```
**Arguments**: Passed inline. No special frontmatter required.
**Notes**: Simple markdown. Cursor reads the file content as the full agent prompt.
Can reference other files with `@filename` syntax.

---

### Cline

**Directory**: `.clinerules/workflows/`
**Invocation**: Type `/kbd-assess.md` in Cline chat (note: `.md` suffix required in invocation)
**Format**:
```markdown
# workflow-name

Description.

[workflow instructions]
```
**Arguments**: Supported via template substitution in newer Cline versions.
**Notes**: The `.clinerules/` directory also holds general behavior rules (`.clinerules` file
or individual `.md` files). Workflows are separate from rules.
**Important**: Cline invocation uses the filename WITH `.md` extension: `/kbd-assess.md`

---

### OpenCode

**Directory**: `.opencode/commands/` (project-local)
**Global home**: `~/.config/opencode/commands/` (cross-project)
**Invocation**: `/kbd-assess` in OpenCode terminal UI
**Format**:
```markdown
---
description: Short description
---

[workflow instructions]
Arguments: $NAME or $1, $2, ...
Shell output: !shell-command result is included
File refs: @filename included inline
```
**Arguments**: Named (`$ARGUMENTS`) or positional (`$1`). Shell output via `!command`.
**Notes**: OpenCode also supports JSON-based command definitions in `opencode.jsonc`.
The Markdown approach is simpler and used by the KBD generator.

---

### Windsurf Cascade

**Directory**: `.windsurf/workflows/`
**Invocation**: `/name-of-workflow` in Windsurf Cascade panel (auto-generated from filename)
**Format**:
```markdown
# workflow-name

Description.

[step-by-step workflow instructions]

Cascade follows these steps:
1. Step one
2. Step two
```
**Arguments**: Not natively supported via placeholders; include instructions to
Cascade about how to handle user-provided arguments in the workflow body.
**Notes**: Windsurf also has `.windsurf/rules/` for always-on behaviors. Workflows
are invocable on-demand. Team-shareable via git. Wave 8 (2025) added full workflow support.

---

## Command Scope Matrix

| Command | Antigravity | Claude Code | Cursor | Cline | OpenCode | Windsurf |
|---------|-------------|-------------|--------|-------|----------|----------|
| `/kbd-init` | ✅ | ✅ | ✅ | ✅ (`/kbd-init.md`) | ✅ | ✅ |
| `/kbd-assess` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-plan` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-execute` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-reflect` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-status` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-new-phase` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/kbd-full-phase` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Keeping Commands in Sync

Commands are generated from `workflows/workflows.yaml`.
**Never edit the generated tool-directory files directly** — they will be overwritten.

To update commands:
1. Modify `workflows/workflows.yaml` (add/change commands)
2. Optionally modify `workflows/templates/<name>.md` (change command body)
3. Run: `bash .agent/skills/kbd-process-orchestrator/scripts/kbd-generate-commands.sh`
4. Commit all generated files

The `/kbd-init` command also runs the generator automatically during project setup.
