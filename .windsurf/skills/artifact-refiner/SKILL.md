---
name: artifact-refiner
version: "1.1.0"
description: >
  Use this skill when creating or iteratively refining named artifacts (logos, UI components,
  A2UI specifications, images, code, content, or meta-prompts) using structured PMPO orchestration with
  explicit constraints, deterministic execution, persistent artifact state, and cross-session retrieval.
authors:
  - "Travis James"
tools:
  - code_interpreter
  - file_system
  - image_generation
  - browser_renderer
triggers:
  keywords:
    - refine
    - artifact
    - logo
    - ui component
    - iterate
    - pmpo
    - improve artifact
    - refine image
    - create artifact
    - refine logo
    - refine content
    - refine code
    - a2ui
  semantic: >
    Refine, improve, or iteratively create a named artifact such as a logo, UI component,
    image, code, or content using the PMPO orchestration loop.
---

# Artifact Refiner

A PMPO-driven, artifact-centric refinement engine capable of creating and iteratively improving artifacts across multiple domains using AI reasoning and deterministic code execution. Supports both direct artifact output and meta-prompt refinement for generating prompts that drive other processes.

## Supported Artifact Domains

- **Logos & brand systems** — SVG/PNG variants, wordmarks, icons, showcase pages
- **React / HTML UI concepts** — Component hierarchies, design tokens, accessibility
- **A2UI specifications** — Structural integrity, schema compliance, normalization
- **Image assets** — Composition, brand colors, resolution, format conversion
- **Content artifacts** — Markdown/HTML structure, tone, heading normalization
- **Code artifacts** — Source files in any language, lint, test, format
- **Meta-prompts** — Prompts for image/video generation, agent instructions, workflow orchestration

## Core Principles

1. **Artifact-centric** — State persisted to disk, never in conversational context
2. **Tool-augmented** — Uses code interpreter / e2b sandbox for deterministic transformations
3. **Constraint-driven** — Structured constraints with severity levels drive convergence
4. **Iterative** — Explicit convergence rules and maximum iteration guards
5. **PMPO meta-loop** — Specify → Plan → Execute → Reflect → Persist → Loop/Terminate
6. **Named & persistent** — Artifacts retrieved by name across sessions
7. **Content-type aware** — Direct output vs meta-prompt refinement with distinct evaluation strategies

## Named Artifacts

Every refinement session requires an `artifact_name` — a human-readable key for cross-session retrieval.

```yaml
artifact_name: "acme-dashboard"  # Unique retrieval key
```

### Lifecycle

- **New**: `state-init.sh acme-dashboard ui direct:react` → Creates fresh state with UUID
- **Resume**: If active state exists → Resumes from last checkpoint
- **Continue**: If finalized → Creates new cycle seeded from prior state
- **List**: Check `.refiner/registry.json` for all known artifacts

## Content Types

Artifacts are classified by content type, which determines how they're produced and evaluated. See `references/content-types.md` for full taxonomy.

### Direct Types — Output IS the Artifact

| Type | Output | Evaluation |
|------|--------|------------|
| `direct:react` | `.tsx` / `.jsx` components | Render + visual inspection |
| `direct:html` | HTML/HTMX markup | Render + visual inspection |
| `direct:content` | Reports, specs, docs | Structure, tone, completeness |
| `direct:image` | SVG/PNG/WebP files | Visual quality, dimensions |
| `direct:code` | Source files (any lang) | Syntax, tests, lint |

### Meta Types — Output is a Prompt That DRIVES Another Process

| Type | Output | Evaluation |
|------|--------|------------|
| `meta:image-prompt` | Image generation prompt | Prompt clarity, platform fit |
| `meta:video-prompt` | Video generation prompt | Temporal coherence |
| `meta:agent-prompt` | System + user prompt pair | Instruction clarity, guardrails |
| `meta:workflow` | Orchestration instructions | Completeness, error handling |
| `meta:composite` | Mixed bundle | Per-component |

## State Provider

State is managed through a tiered provider system. The startup protocol resolves the active provider:

1. `REFINER_PROVIDER_CONFIG` (environment variable)
2. `.refiner-provider.json` (project-local)
3. `~/.refiner/provider.json` (global)
4. MCP state tool (probe)
5. Agent memory (probe)
6. Filesystem fallback (always available)

See `references/schemas/state-provider.schema.json` for provider configuration.

## Workflow Triggers

Lifecycle events can fire workflow triggers defined in the refinement state:

| Event | When |
|-------|------|
| `on_phase_complete` | After any PMPO phase completes |
| `on_iteration_complete` | After a full Specify→Persist loop |
| `on_refinement_complete` | When refinement terminates |
| `on_regression` | When Reflect detects quality regression |
| `on_approval_required` | When human approval gate is reached |

See `references/schemas/workflow-trigger.schema.json` for trigger definitions.

## Execution Model (PMPO Loop)

The skill follows the Prometheus Meta-Prompting Orchestration loop. For full theory, see `references/pmpo-theory.md`.

### Startup Protocol

1. **Resolve Provider**: `scripts/state-resolve-provider.sh`
2. **Init/Resume State**: `scripts/state-init.sh <artifact_name> [type] [content_type]`
3. **Detect Content Type**: Classify via heuristics (see `prompts/specify.md`)

### Phase Loop

1. **Specify** (`prompts/specify.md`) — Transform intent into structured specification
2. **Plan** (`prompts/plan.md`) — Convert specification into executable strategy
3. **Execute** (`prompts/execute.md`) — Apply transformations via AI + deterministic tools
4. **Reflect** (`prompts/reflect.md`) — Evaluate outputs against constraints
5. **Persist** (`prompts/persist.md`) — Write validated state to disk
6. **Loop or Terminate** — Continue if constraints unsatisfied, stop if converged

### Phase Hooks

After each phase: checkpoint (`state-checkpoint.sh`) + dispatch (`workflow-dispatch.sh`)

### Cycle Finalization

On terminate: `state-finalize.sh` archives to history + dispatches `on_refinement_complete`

## Required Tools

- `code_interpreter` or e2b MCP sandbox (`mcp__e2b-sandbox__run_python_code`)
- `file_system` — Read/write artifact files

### Optional Tools

- `image_generation` — For logo/image domains
- `browser_renderer` — For UI/A2UI preview rendering
- Local fallback scripts — `node scripts/compile-tsx-preview.mjs` and `node scripts/render-preview.mjs`

## Inputs

```yaml
artifact_name: string  # Required — cross-session retrieval key
artifact_type: string  # logo | ui | a2ui | image | content | code | meta-prompt
content_type: string   # direct:react | direct:html | meta:image-prompt | etc.
constraints: array     # See references/schemas/constraints.schema.json
target_state:
  description: object  # Desired end state
current_state: optional object  # Existing artifact to refine
```

## Outputs

```yaml
refined_artifact: object
artifact_manifest: object  # See references/schemas/artifact-manifest.schema.json
refinement_log: string
generated_files: array     # Written to dist/
preview_artifacts: optional array  # dist/previews/<artifact-id>/*
refinement_state: object   # Persisted to .refiner/artifacts/<name>/state.json
```

## Persistent State Files

The skill creates and maintains these files — **state must never rely on conversational context**:

- `artifact_manifest.json` — Output contract (validated against schema)
- `constraints.json` — Active constraint definitions
- `refinement_log.md` — Iteration history and decisions
- `decisions.md` — Convergence rationale
- `dist/` — Generated artifact outputs
- `dist/previews/` — Browser preview HTML, screenshots, and diagnostics (UI/A2UI)
- `.refiner/artifacts/<name>/state.json` — Named refinement state
- `.refiner/registry.json` — Artifact registry for cross-session lookup

## Deterministic Execution Rule

Before performing transformations, determine:

> **Does this refinement require deterministic computation?**

- **YES** → Generate minimal executable code → Execute via code interpreter or e2b sandbox → Validate file outputs → Update manifest
- **NO** → Perform AI-only refinement

For `ui` and `a2ui`, deterministic execution includes:
1. TSX preview compilation (when applicable)
2. Browser preview rendering
3. Screenshot + preview diagnostics capture
4. Manifest preview metadata update

## Termination Conditions

Refinement ends when:

- No blocking constraint violations remain
- All required artifact outputs exist in `dist/`
- Required preview evidence exists in `dist/previews/` for `ui`/`a2ui` runs
- Manifest validates against `references/schemas/artifact-manifest.schema.json`
- Further improvements fall below threshold
- Maximum iterations (5) reached

## Failure Handling

- Tool execution errors → Log, retry (max 2 retries), then degrade gracefully
- Missing files → Detect and regenerate
- Infinite refinement → Prevented via `max_iterations` guard in meta-controller
- State provider failure → Fall back to filesystem

## Domain Adapters

Domain-specific refinement knowledge lives in `references/domain/`:

| Domain | Reference | Template |
|--------|-----------|----------|
| Logo | `references/domain/logo.md` | `assets/templates/logo-showcase.template.html` |
| UI | `references/domain/ui.md` | `assets/templates/react-components-shadcn-ui-template.tsx` |
| A2UI | `references/domain/a2ui.md` | `assets/templates/a2ui-preview-template.html` |
| Image | `references/domain/image.md` | — |
| Content | `references/domain/content.md` | `assets/templates/content-report.template.html` |
| Code | `references/domain/code.md` | — |
| Meta-Prompt | `references/domain/meta-prompt.md` | — |

## Quick Start

Use domain-specific slash commands for focused refinement:

- `/refine-logo` — Logo and brand system refinement
- `/refine-ui` — React/HTML UI component refinement
- `/refine-content` — Content/Markdown refinement
- `/refine-image` — Image artifact refinement
- `/refine-a2ui` — A2UI specification refinement
- `/refine-status` — Check current refinement progress
- `/refine-validate` — Run validation checks on current state
