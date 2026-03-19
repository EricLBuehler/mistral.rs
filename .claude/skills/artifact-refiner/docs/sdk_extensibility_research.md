# Agent Runner SDK Extensibility Research

## Research Question

Would a TypeScript + Python SDK for the iterative-evolver provide value? How do agent runners allow users to register custom tools, and what integration points exist?

---

## Agent Runner Landscape

### MCP is the Universal Protocol

**Every major agent runner now supports MCP** as its primary extensibility mechanism. This is the single most important finding — MCP has won the "plugin protocol" war for AI coding agents:

| Runner | MCP Support | Extra Extensibility | Language |
|--------|-------------|---------------------|----------|
| **OpenCode** | ✅ Full | **Custom Tools** (Zod, inline) + **Plugins** (hooks) | TS/JS |
| **Claude Code** | ✅ Full | Desktop Extensions (`.mcpb` bundles) | Any |
| **Codex CLI** | ✅ Full | `~/.codex/config.toml` providers | Any |
| **Gemini CLI** | ✅ Full | **Extensions** framework + slash commands | Any |
| **Cursor** | ✅ Full | Settings UI for server config | Any |
| **Roo Code** | ✅ Full | Custom modes + Agent Maestro REST bridge | Any |
| **Gemini Code Assist** | ✅ Migrating | Legacy Tool Calling → MCP by March 2026 | Any |

---

## Deep Dive: OpenCode's Custom Tools (Most Relevant Model)

OpenCode provides the **strongest typed tool model** — exactly what our SDK would replicate:

### How It Works

1. **Tool files** go in `.opencode/tools/` (project) or `~/.config/opencode/tools/` (global)
2. **The `tool()` helper** provides type-safe definitions with Zod schemas:
   ```typescript
   import { tool } from "opencode"
   
   export default tool({
     description: "Initialize evolution state",
     args: z => ({ name: z.string(), domain: z.string() }),
     execute: async (args, context) => {
       // Implementation
       return { state: { ... } }
     }
   })
   ```
3. **Auto-dependency install**: Add a `package.json` → OpenCode runs `bun install` at startup
4. **Multi-language**: Tool _definitions_ are TypeScript but can invoke any language via child process
5. **Context injection**: Tools receive `context.directory` and `context.worktree`
6. **Plugins**: Can define hooks (`PostToolUse`, lifecycle events) + tools in a single module

### What Makes This Powerful

- **No MCP server boilerplate** — tools are just functions with schemas
- **Inline, typed** — IDE autocomplete, compile-time validation
- **Auto-installed** — Zero manual setup for consumers of a tool package
- **Composable** — Plugins can combine tools + hooks in one module

---

## Deep Dive: Other Runners

### Claude Code
- Extensibility = **MCP servers** (`.mcpb` bundles for Desktop, `claude mcp add` for CLI)
- No equivalent to OpenCode's inline `tool()` — you must build a full MCP server
- Supports both stdio and HTTP transports

### Codex CLI
- MCP via `~/.codex/config.toml` — servers configured under `[mcp_servers]` section
- No inline tool definition — MCP only
- Strong on model provider customization but minimal on tool extensibility

### Gemini CLI
- **Extensions framework** (announced Oct 2025) — most similar to OpenCode's model
- `.toml`-based slash commands + MCP prompts
- Extensions from Google partners (Dynatrace, Elastic, Shopify, Snyk, Stripe)
- Function calling API for programmatic tool registration

### Cursor
- MCP servers via Settings → MCP panel
- Can write MCP servers in any language (stdio or HTTP)
- No inline tool SDK — requires full MCP server implementation

### Roo Code
- MCP + custom modes (Code, Architect, Debug, Ask, + user-defined)
- Agent Maestro REST bridge for headless/CI use
- Extensibility is primarily through MCP and mode configuration

---

## SDK Recommendation

### Verdict: **Yes — an SDK would be highly valuable**

The research reveals a clear gap in the ecosystem:

```
┌────────────────────────────────────┐
│   MCP Protocol (Universal)         │  ← Every runner supports this
│   ┌────────────────────────────┐   │
│   │ Evolver MCP Server         │   │  ← Works everywhere, but verbose
│   └────────────────────────────┘   │
├────────────────────────────────────┤
│   Typed Tool SDK (OpenCode-style)  │  ← Only OpenCode has this today
│   ┌────────────────────────────┐   │
│   │ @evolver/sdk (TS)          │   │  ← Low-friction, typed, inline
│   │ evolver-sdk (Python)       │   │
│   └────────────────────────────┘   │
└────────────────────────────────────┘
```

### Two-Tier Strategy

#### Tier 1: MCP Server (Universal Compatibility)
A standalone MCP server that any agent runner can connect to:
- Exposes `evolution_init`, `evolution_checkpoint`, `evolution_finalize`, `evolution_list`, `workflow_dispatch` as MCP tools
- Works with Claude Code, Codex, Cursor, Roo Code, Gemini CLI — zero SDK needed
- Users configure it like any other MCP server

#### Tier 2: Typed SDK Packages (Developer Experience)

**TypeScript** (`@evolver/sdk` on npm):
```typescript
import { StateProvider, WorkflowTrigger, HookEvent } from "@evolver/sdk"

// Implement a custom state provider
const myProvider: StateProvider = {
  init: async (name, domain) => { ... },
  checkpoint: async (name, event) => { ... },
  finalize: async (name) => { ... },
  list: async () => { ... },
  save: async (name, state) => { ... },
}

// OpenCode custom tool definition
export default tool({
  description: "Custom evolution state provider",
  args: z => ({ command: z.enum(["init", "checkpoint", "finalize"]) }),
  execute: async (args) => myProvider[args.command]()
})
```

**Python** (`evolver-sdk` on PyPI):
```python
from evolver_sdk import StateProvider, WorkflowTrigger, HookEvent

class MyProvider(StateProvider):
    async def init(self, name: str, domain: str) -> EvolutionState: ...
    async def checkpoint(self, name: str, event: HookEvent) -> str: ...
    async def finalize(self, name: str) -> str: ...
```

### What the SDKs Provide

| Feature | TypeScript | Python |
|---------|-----------|--------|
| `StateProvider` interface | ✅ | ✅ |
| `WorkflowTrigger` types | ✅ | ✅ |
| `HookEvent` schema | ✅ | ✅ |
| `EvolutionState` model | ✅ | ✅ |
| Zod schema exports | ✅ | N/A (Pydantic) |
| OpenCode `tool()` wrappers | ✅ | N/A |
| MCP server builder | ✅ | ✅ |
| CLI integration helpers | ✅ | ✅ |

### Runner Compatibility Matrix

| Runner | MCP Server | TS SDK | Python SDK |
|--------|-----------|--------|------------|
| OpenCode | ✅ | ✅ Custom tools + plugins | ⚡ Via subprocess |
| Claude Code | ✅ | ⚡ As MCP server | ⚡ As MCP server |
| Codex CLI | ✅ | ⚡ As MCP server | ⚡ As MCP server |
| Gemini CLI | ✅ | ✅ As extension | ⚡ As MCP server |
| Cursor | ✅ | ⚡ As MCP server | ⚡ As MCP server |
| Roo Code | ✅ | ⚡ As MCP server | ⚡ As MCP server |

> ✅ = native integration, ⚡ = works via protocol bridge

### Why This Matters for Iterative Evolver

1. **OpenCode users** get first-class inline tools — `npm install @evolver/sdk`, add to `.opencode/tools/`, done
2. **All other runners** get an MCP server — `npx @evolver/mcp-server` or configure in settings
3. **Python users** get Pydantic models + async providers — can implement custom backends
4. **Custom providers** become typed contracts — no guessing at interfaces
5. **Workflow triggers** become structured — type-safe action definitions
6. **Cross-runner portability** — same provider works everywhere via MCP layer

---

## Sources

- [OpenCode Custom Tools](https://opencode.ai/docs/custom-tools/) — tool() helper, Zod schemas, auto-deps
- [OpenCode Plugins](https://opencode.ai/docs/plugins/) — hooks, event lifecycle, npm integration
- [Claude Desktop Extensions](https://www.anthropic.com/engineering/desktop-extensions) — .mcpb bundles
- [Codex CLI Config](https://developers.openai.com/codex/config-advanced/) — TOML MCP setup
- [Gemini CLI Extensions](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-cli-extensions/) — extension framework
- [Cursor MCP Integration](https://medium.com/@rajakrishna/supercharging-cursor-ai-ide-with-mcp-99ceb3d1e16d)
- [Roo Code](https://github.com/RooCodeInc/Roo-Code) — custom modes + MCP
- [Gemini Code Assist MCP Migration](https://www.digitalapplied.com/blog/google-gemini-code-assist-agent-mode-guide)
