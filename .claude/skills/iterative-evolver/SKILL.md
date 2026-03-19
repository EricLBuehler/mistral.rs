---
name: iterative-evolver
description: >
  Use this skill for any iterative evolution scenario — assessing current state,
  analyzing the landscape, planning improvements, executing changes, and reflecting
  on results. Works across any domain: software, business, product, research,
  content, operations, compliance, or any area requiring iterative improvement
  against goals.
allowed-tools: file_system web_search code_interpreter browser tavily sequential_thinking memory
---

# Iterative Evolver

A PMPO-driven, domain-agnostic iterative evolution engine. Assesses current state against goals, analyzes the competitive/external landscape, plans improvements, executes changes, and reflects on results — across any domain.

## Named Evolutions

Every evolution cycle has a user-defined **evolution name** — a human-friendly identifier for cross-session retrieval:

```
/evolve "uar-api-improvement"
/evolve-status "q1-sales-strategy"
```

The name is the primary key. State is loaded by name at the start of every session, enabling:
- Cross-session continuity
- End-state → start-state chaining across cycles
- Mid-session crash recovery via checkpoints

## state provider

State is persisted through a **state provider** abstraction. The skill automatically resolves the best available provider:

| Priority | Provider | When Used |
|----------|----------|-----------|
| 1 | User config | `$EVOLVER_PROVIDER_CONFIG` env var or `.evolver-provider.json` |
| 2 | MCP tool | A dedicated state MCP server is available |
| 3 | Agent memory | Memory MCP server (e.g., `surreal_memory`) is available |
| 4 | Filesystem | Always available — `.evolver/` directory (default) |

See `references/state-management.md` for full provider architecture.

### Filesystem Provider (Default)

```
.evolver/
  registry.json                  # Maps evolution_name → state path
  evolutions/
    {evolution_name}/
      state.json                 # Current evolution state
      checkpoints/               # Mid-session snapshots
      history/                   # Finalized iteration archives
```

## Workflow Triggers

External workflows can be triggered at lifecycle events without modifying the skill:

```json
{
  "event": "on_cycle_complete",
  "action": { "type": "command", "target": "cargo test" }
}
```

See `references/workflow-integration.md` for trigger semantics.

## Supported Evolution Domains

- **Software** — Codebase health, architecture, dependencies, spec compliance
- **Business** — Market positioning, competitive intelligence, strategic planning
- **Product** — UX audit, design systems, feature prioritization, accessibility
- **Research** — Literature review, gap analysis, hypothesis formulation, publication strategy
- **Content** — Content audit, SEO, editorial strategy, audience engagement
- **Operations** — Process optimization, bottleneck analysis, KPI improvement
- **Compliance** — Regulatory audit, standards mapping, remediation planning
- **Generic** — Any domain not listed above — the skill adapts

## Core Principles

1. **Goal-centric** — All evolution tracked against explicit goals and target states
2. **Domain-agnostic** — PMPO loop is invariant; domain adapters supply specialization
3. **Landscape-aware** — Real-time external analysis via web research
4. **Constraint-driven** — Structured constraints with severity levels drive convergence
5. **Provider-agnostic** — State persisted through pluggable providers (files, memory, MCP, custom)
6. **Hook-extensible** — Lifecycle hooks trigger external workflows without skill modification
7. **PMPO meta-loop** — Assess → Analyze → Plan → Execute → Reflect → Loop/Terminate

## Execution Model (PMPO Loop)

The skill follows the Prometheus Meta-Prompting Orchestration loop adapted for iterative evolution. For full theory, see `references/pmpo-theory.md`.

### Startup
1. **Resolve provider** — Determine state backend (`scripts/state-resolve-provider.sh`)
2. **Init/resume state** — Load or create named evolution (`scripts/state-init.sh`)
3. **Load domain adapter** — Route to `references/domain/{domain}.md`

### Loop
1. **Assess** (`prompts/assess.md`) — Evaluate current state against goals
2. **Analyze** (`prompts/analyze.md`) — Scan external landscape, identify opportunities and threats
3. **Plan** (`prompts/plan.md`) — Create prioritized improvement plan
4. **Execute** (`prompts/execute.md`) — Apply changes using domain-appropriate tools
5. **Reflect** (`prompts/reflect.md`) — Measure movement, compare before/after, decide next steps
6. **Persist** (`prompts/persist.md`) — Write validated state via state provider
7. **Loop or Terminate** — Continue if goals unsatisfied, stop if converged

After each phase: **checkpoint** + **dispatch workflow triggers**.

## Required Tools

- `file_system` — Read/write state files, reports, plans

### Optional Tools

- `web_search` or `tavily` — External landscape research
- `code_interpreter` — For domains requiring deterministic computation
- `browser` — For UI/web-based assessment
- `sequential_thinking` — For complex multi-step reasoning
- `memory` — For cross-session knowledge persistence (also serves as state provider)

## Inputs

```yaml
evolution_name: string       # Human-friendly name for cross-session retrieval
evolution_domain: string     # software | business | product | research | content | operations | compliance | generic
goals: array                 # What are we trying to achieve?
  - description: string
    priority: high | medium | low
    measurable_criteria: optional string
constraints: optional array  # Boundaries and requirements
target_state:
  description: string        # Where do we want to be?
current_state: optional object  # Where are we now? (auto-assessed if omitted)
context:
  project_path: optional string  # For software domains
  documents: optional array      # Reference documents
  prior_assessments: optional array  # Previous evolution cycle reports
workflow_triggers: optional array    # External workflows to fire at lifecycle events
```

## Outputs

```yaml
assessment_report: object       # Structured assessment of current state
analysis_report: object         # Landscape analysis with findings
improvement_plan: object        # Prioritized action items
execution_results: object       # What was done, what succeeded/failed
reflection_report: object       # Delta analysis, goal tracking, lessons
evolution_state: object         # Full state manifest (validated against schema)
generated_reports: array        # Written to output directory
```

## Persistent State

State must never rely on conversational context. The state provider handles persistence:

- `evolution_state` — Full state manifest with name, provider, triggers, checkpoints
- `assessment` — Latest assessment results
- `analysis` — Latest landscape analysis
- `plan` — Current improvement plan
- `evolution_log` — Iteration history and decisions
- `decisions` — Convergence rationale and approval records
- `reports/` — Generated assessment and reflection reports
- `checkpoints/` — Mid-session snapshots for crash recovery
- `history/` — Finalized iteration archives

## Termination Conditions

Evolution cycle ends when:

- All high-priority goals are satisfied
- No blocking constraints are violated
- Target state alignment exceeds threshold (configurable, default 90%)
- Further improvements fall below impact threshold
- Maximum iterations reached (configurable, default 5)
- User explicitly terminates

On termination, state is **finalized** and archived. Re-running with the same `evolution_name` starts a new cycle seeded from the finalized end-state.

## Failure Handling

- State provider unavailable → Fall back to filesystem
- Tool execution errors → Log, retry (max 2), degrade gracefully
- External research fails → Proceed with cached/prior data, flag gap
- Assessment data incomplete → Identify gaps, request user input
- Infinite iteration → Prevented by `max_iterations` guard
- Workflow trigger fails → Log error, continue (triggers are non-blocking)

## Domain Adapters

Domain-specific evolution knowledge lives in `references/domain/`:

| Domain | Reference |
|--------|-----------|
| Software | `references/domain/software.md` |
| Business | `references/domain/business.md` |
| Product | `references/domain/product.md` |
| Research | `references/domain/research.md` |
| Content | `references/domain/content.md` |
| Operations | `references/domain/operations.md` |
| Compliance | `references/domain/compliance.md` |
| Generic | `references/domain/generic.md` |

## Quick Start

Use domain-specific slash commands or the universal entry point:

- `/evolve` — Full evolution cycle (auto-detects or prompts for domain)
- `/evolve-assess` — Assess current state only
- `/evolve-analyze` — Landscape analysis only
- `/evolve-plan` — Create improvement plan from existing data
- `/evolve-execute` — Execute an existing plan
- `/evolve-status` — Check current evolution cycle status
- `/evolve-report` — Generate report from current state
