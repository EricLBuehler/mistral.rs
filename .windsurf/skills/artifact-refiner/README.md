# Artifact Refiner

A PMPO-driven, iterative artifact refinement engine for logos, UI components, content, images, code, meta-prompts, and A2UI specifications. Built as a Claude Code plugin and [Agent Skills](https://agentskills.io) compliant skill with cross-session state management and multi-agent deployment support.

## Quick Start

Use domain-specific slash commands:

```
/refine-logo     — Logo and brand system refinement
/refine-ui       — React/HTML UI component refinement
/refine-content  — Content/Markdown refinement
/refine-image    — Image artifact refinement
/refine-a2ui     — A2UI specification refinement
/refine-status   — Check current refinement progress
/refine-validate — Run validation checks on current state
```

Or invoke the general skill with a free-form request describing what you want to refine.

## Installation

### Via cowork (Recommended)

Install across all supported agents in one command:

```bash
cowork install GQAdonis/artifact-refiner-skill
```

This deploys to up to 15 agents (Claude Code, Cursor, OpenCode, Gemini CLI, Windsurf, Roo Code, etc.).

### As a Claude Plugin

```bash
claude plugin install artifact-refiner
```

### Per-Agent Installers

Individual installer scripts are available for each supported agent:

```bash
# Examples
bash scripts/installers/claude-code/install.sh
bash scripts/installers/cursor/install.sh
bash scripts/installers/opencode/install.sh
bash scripts/installers/gemini-cli/install.sh
```

See [`scripts/installers/README.md`](scripts/installers/README.md) for the full list of 12 supported agents.

### Manual

Clone the repository into your Claude Code plugins directory:

```bash
git clone https://github.com/GQAdonis/artifact-refiner-skill.git ~/.claude/plugins/artifact-refiner
```

### e2b Sandbox (Optional)

For sandboxed code execution, set your e2b API key:

```bash
export E2B_API_KEY="your-key-here"
```

The skill works without e2b by falling back to the built-in `code_interpreter`.

### Browser Preview Tooling (Optional, for UI/A2UI)

Install local preview tooling for TSX compilation and browser screenshots:

```bash
npm install
```

This enables:
- `node scripts/compile-tsx-preview.mjs` (TSX → browser preview bundle)
- `node scripts/render-preview.mjs` (preview render + screenshot + diagnostics)

## How It Works

The skill uses **PMPO** (Prometheus Meta-Prompting Orchestration) — a structured, iterative refinement loop:

1. **Specify** — Transform intent into structured specification with constraints
2. **Plan** — Convert specification into executable strategy
3. **Execute** — Apply transformations via AI + deterministic tools
4. **Reflect** — Evaluate outputs against constraints
5. **Persist** — Write validated state to disk via state provider
6. **Loop or Terminate** — Continue if constraints unsatisfied, stop if converged

### Startup Protocol

Before the loop begins, the skill executes a 3-step startup:

1. **Resolve Provider** — `scripts/state-resolve-provider.sh` (6-tier waterfall)
2. **Init/Resume State** — `scripts/state-init.sh <artifact_name>` (creates new, resumes active, or seeds from prior cycle)
3. **Detect Content Type** — Classify the artifact as direct or meta type

### Phase Hooks

After each phase: `state-checkpoint.sh` + `workflow-dispatch.sh`

### Cycle Finalization

On terminate: `state-finalize.sh` archives to history + dispatches `on_refinement_complete`

For the full PMPO methodology, see [`references/pmpo-theory.md`](references/pmpo-theory.md).

## Content Types

Artifacts are classified by content type, which determines how they're produced and evaluated. See [`references/content-types.md`](references/content-types.md) for the full taxonomy.

### Direct Types — Output IS the Artifact

| Type | Output | Evaluation |
|------|--------|------------|
| `direct:svg` | SVG markup | Render + visual inspection |
| `direct:react` | `.tsx` / `.jsx` components | Render + visual inspection |
| `direct:html` | HTML/HTMX markup | Render + visual inspection |
| `direct:content` | Reports, specs, docs | Structure, tone, completeness |
| `direct:code` | Source files in any language | Lint, test, format |

### Meta Types — Output is a Prompt

| Type | Output | Evaluation |
|------|--------|------------|
| `meta:image-prompt` | Image generation prompts | Prompt quality analysis |
| `meta:video-prompt` | Video generation prompts | Prompt quality + test generation |
| `meta:agent-prompt` | Agent/skill instruction prompts | Prompt quality + structural analysis |

## Named Artifacts & Cross-Session State

Artifacts can be **named** for cross-session persistence. Use `artifact_name` to identify an artifact that persists beyond a single conversation:

```
Refine the "nexaflow-brand" logo to use a darker palette
```

The skill will:
1. Look up `nexaflow-brand` in the state registry
2. Resume from the last checkpoint if active, or seed a new cycle from the finalized state
3. Carry forward all constraints, decisions, and refinement history

### State Provider

State is persisted via a configurable provider, resolved through a 6-tier waterfall:

1. Environment variable (`REFINER_PROVIDER_CONFIG`)
2. Project-local config (`.refiner/provider.json`)
3. Global config (`~/.config/artifact-refiner/provider.json`)
4. MCP state server (if connected)
5. Memory tool (if available)
6. Filesystem fallback (always available)

See [`references/schemas/state-provider.schema.json`](references/schemas/state-provider.schema.json) for provider configuration.

## Workflow Triggers

Lifecycle events can fire workflow triggers:

| Event | When |
|-------|------|
| `on_phase_complete` | After any PMPO phase completes |
| `on_iteration_complete` | After a full Specify→Persist loop |
| `on_refinement_complete` | When refinement terminates |
| `on_regression` | When Reflect detects quality regression |

See [`references/schemas/workflow-trigger.schema.json`](references/schemas/workflow-trigger.schema.json) for trigger configuration.

## Domain Adapters

| Domain | Reference | Description |
|--------|-----------|-------------|
| Logo | [`domain/logo.md`](references/domain/logo.md) | SVG logos, brand identity systems |
| UI | [`domain/ui.md`](references/domain/ui.md) | React/HTML components with browser preview |
| A2UI | [`domain/a2ui.md`](references/domain/a2ui.md) | A2UI specification refinement |
| Image | [`domain/image.md`](references/domain/image.md) | Image artifacts and generation prompts |
| Content | [`domain/content.md`](references/domain/content.md) | Blog posts, reports, documentation |
| Code | [`domain/code.md`](references/domain/code.md) | Source files — lint, test, format |
| Meta-prompt | [`domain/meta-prompt.md`](references/domain/meta-prompt.md) | Prompts for generation, agent instructions |

## Directory Structure

```
artifact-refiner/
├── SKILL.md                    # Canonical skill definition (Agent Skills spec)
├── CLAUDE.md                   # AI assistant development guide
├── AGENTS.md                   # Contributor guide
├── agents/                     # PMPO phase subagents
│   ├── pmpo-specifier.md
│   ├── pmpo-planner.md
│   ├── pmpo-executor.md
│   ├── pmpo-reflector.md
│   └── artifact-validator.md
├── assets/
│   ├── templates/              # HTML/React templates for output generation
│   └── vendor/                 # Optional local runtime assets (e.g., htmx.min.js)
├── examples/                   # Complete walkthrough examples
│   ├── logo-refinement/
│   ├── content-refinement/
│   └── ui-preview-refinement/
├── hooks/hooks.json            # Lifecycle hooks (per-phase checkpoint + dispatch)
├── prompts/                    # PMPO phase controllers
│   ├── meta-controller.md      # Orchestration entry point
│   ├── specify.md              # Intent → structured specification
│   ├── plan.md                 # Specification → execution strategy
│   ├── execute.md              # Strategy → artifact output
│   ├── reflect.md              # Output → quality evaluation
│   └── persist.md              # State → provider-agnostic persistence
├── references/
│   ├── pmpo-theory.md          # PMPO orchestration methodology
│   ├── content-types.md        # Content type taxonomy & routing
│   ├── domain/                 # Domain adapters
│   │   ├── logo.md, ui.md, a2ui.md, image.md, content.md
│   │   ├── code.md             # Code artifacts (lint, test, format)
│   │   └── meta-prompt.md      # Meta-prompt refinement
│   └── schemas/                # JSON Schema contracts
│       ├── artifact-manifest.schema.json
│       ├── constraints.schema.json
│       ├── content-type.schema.json
│       ├── hook-event.schema.json
│       ├── refinement-state.schema.json
│       ├── state-provider.schema.json
│       └── workflow-trigger.schema.json
├── scripts/
│   ├── state-resolve-provider.sh   # 6-tier provider resolution
│   ├── state-init.sh               # Init/resume named state
│   ├── state-checkpoint.sh         # Mid-phase snapshot
│   ├── state-finalize.sh           # Archive completed refinement
│   ├── workflow-dispatch.sh        # Event-driven trigger dispatcher
│   ├── compile-tsx-preview.mjs     # TSX → browser preview bundle
│   ├── render-preview.mjs          # Preview render + screenshot
│   ├── lib/preview-utils.mjs       # Preview utilities
│   ├── validate-constraints.sh     # Constraint validation
│   ├── validate-manifest.sh        # Manifest validation
│   └── installers/                 # Per-agent installation scripts (12 agents)
├── skills/                     # Slash command skills
│   ├── refine-logo/
│   ├── refine-ui/
│   ├── refine-content/
│   ├── refine-image/
│   ├── refine-a2ui/
│   ├── refine-status/
│   └── refine-validate/
├── .claude-plugin/             # Plugin manifest
│   ├── plugin.json
│   └── marketplace.json
└── .mcp.json                   # e2b sandbox MCP server config
```

## Persistent State Files

- `artifact_manifest.json` — Output contract (validated against schema)
- `constraints.json` — Active constraint definitions
- `refinement_log.md` — Iteration history and decisions
- `decisions.md` — Convergence rationale
- `dist/` — Generated artifact outputs
- `.refiner/artifacts/<name>/state.json` — Named artifact state (cross-session)
- `.refiner/registry.json` — Artifact creation registry

## Browser Preview Output Contract

For `ui` and `a2ui` artifacts, preview evidence is persisted under:

- `dist/previews/<artifact-id>/preview.html`
- `dist/previews/<artifact-id>/screenshot.png`
- `dist/previews/<artifact-id>/preview-report.json`

Preview metadata is stored in `artifact_manifest.json` under `preview.runs`, including runtime source (`local` or `network`) and render status.

## Examples

See the [`examples/`](examples/) directory for complete walkthroughs:

- **[Logo Refinement](examples/logo-refinement/)** — NexaFlow brand system from spec to final manifest
- **[Content Refinement](examples/content-refinement/)** — Blog post from rough draft to polished HTML
- **[UI Preview Refinement](examples/ui-preview-refinement/)** — Browser preview HTML + screenshot + diagnostics report

## Try It

```
> /refine-logo Create a modern logo for "AcmeAPI" using navy blue and gold
```

The skill will:
1. Resolve state provider and init named state
2. Detect content type (`direct:svg`)
3. Generate a structured specification with brand constraints
4. Plan an SVG → PNG → showcase pipeline
5. Execute using image generation + code interpreter
6. Reflect on constraint satisfaction
7. Checkpoint state and dispatch lifecycle events
8. Iterate until all constraints are met
9. Finalize and output a complete brand system in `dist/`

## Author

**Travis James** — [travisjames.ai](https://travisjames.ai)

## License

MIT — See [LICENSE](LICENSE) for details.
