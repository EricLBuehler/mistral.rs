# Platform Adapters Reference

How to generate skills compatible with different AI agent platforms. All platforms start from the agentskills.io base format, then add platform-specific files.

## Universal Base: agentskills.io

Every generated skill MUST include:
```
skill-name/
└── SKILL.md     # Valid frontmatter + instructions
```

This is the universal format recognized by all compliant platforms.

## Claude Code

### Additional Files
```
skill-name/
├── .claude-plugin/
│   └── plugin.json        # Marketplace manifest
├── agents/                # Subagent definitions
│   └── *.md
├── hooks/
│   └── hooks.json         # Lifecycle hooks
├── skills/                # Nested sub-skills
│   └── <command>/SKILL.md
└── commands/              # Simple slash commands
    └── *.md
```

### Key Details
- Plugin namespace: `name` in plugin.json becomes command prefix
- Hooks use `${CLAUDE_PLUGIN_ROOT}` for script paths
- Skills auto-discovered from `skills/` directory
- See `references/plugin-structure.md` for complete format

## OpenCode

### Additional Files
```
skill-name/
└── <tools_directory>/     # Default: .opencode/tools/
    └── <tool-name>.ts
```

### Tool Definition Format
```typescript
import { tool } from "@opencode/tool";
import { z } from "zod";

export default tool({
  name: "<tool-name>",
  description: "<what the tool does>",
  parameters: z.object({
    input: z.string().describe("Input description"),
  }),
  execute: async (args) => {
    // Tool logic
    return { result: "output" };
  }
});
```

### Key Details
- Tools defined in TypeScript using `tool()` helper with Zod schemas
- Place in `.opencode/tools/` for project-scope or global tools directory
- OpenCode auto-installs deps via `bun install` if `package.json` present
- The `tools_directory` is configurable — set in `skill_spec.tools_directory`

## Cursor

### Recognition
- Cursor recognizes standard agentskills.io `SKILL.md` format
- No additional platform-specific files needed
- Skills placed in workspace are auto-discovered

## Gemini CLI

### Extensions Framework
- Gemini CLI supports custom slash commands via `.toml` files
- Extensions can reference SKILL.md-based skills
- MCP server integration for tool access

### Key Details
- Extensions defined in `~/.gemini/extensions/`
- Can wrap SKILL.md-based skills with Gemini-specific configuration

## Roo Code

### Recognition
- Supports custom modes via markdown definitions
- Can consume SKILL.md-based skills as modes
- MCP server integration for extensibility

## cowork CLI (Universal Installer)

All skills can be installed across ALL platforms using the `cowork` CLI:

```bash
# Install from GitHub
cowork install user/repo

# Install to specific agents
cowork install user/repo -a claude-code -a cursor -a opencode

# Install as plugin (preserves full structure)
cowork install user/repo --plugin

# Reinstall (remove + install fresh)
cowork install user/repo --reinstall

# Install locally to project
cowork install user/repo --local
```

### Supported Agents (cowork)
`amp`, `antigravity`, `claude-code`, `clawdbot`, `codex`, `cursor`, `droid`, `gemini-cli`, `github-copilot`, `goose`, `kilo`, `kiro-cli`, `opencode`, `roo`, `trae`, `windsurf`

## Generation Strategy

When generating for multiple platforms:

1. **Always generate**: agentskills.io base (`SKILL.md`)
2. **Add claude-code if requested**: `plugin.json`, `hooks/`, `skills/`
3. **Add opencode if requested**: `<tools_directory>/*.ts`
4. **Others**: Use agentskills.io base — cowork handles distribution

The `tools_directory` path can be specified by the user. Default is `.opencode/tools/` but can be any path (e.g., `tools/`, `.tools/`, etc.).
