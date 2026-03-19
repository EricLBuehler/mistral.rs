# Iterative Evolver

A PMPO-driven, domain-agnostic iterative evolution engine for AI agents.

## What It Does

The Iterative Evolver automates the cycle every professional uses (consciously or not) when improving anything:

1. **Assess** — Where am I now relative to my goals?
2. **Analyze** — What's happening in the wider world that affects my goals?
3. **Plan** — What should I improve, in what order?
4. **Execute** — Make the improvements
5. **Reflect** — Did it work? What's different? Do I iterate again?

This cycle works identically whether you're evolving a software codebase, a business strategy, a research agenda, or a content program.

## Quick Start

```
/evolve
```

The skill auto-detects your domain or asks. For targeted work:

| Command | What It Does |
|---------|-------------|
| `/evolve` | Full 5-phase cycle |
| `/evolve-assess` | Assess current state against goals |
| `/evolve-analyze` | Research external landscape |
| `/evolve-plan` | Create improvement plan |
| `/evolve-execute` | Execute existing plan |
| `/evolve-status` | Check cycle progress |
| `/evolve-report` | Generate assessment report |

## Example Usage

### Software Evolution
> "Assess our API against the latest A2A protocol spec, analyze what competitors are doing, and plan our next improvements."

### Business Strategy
> "Assess our market position in the fintech space, analyze emerging competitors, and create a strategic improvement plan."

### Research
> "Assess our literature coverage on transformer architectures, analyze recent publications we might be missing, and plan our next research directions."

### Content
> "Assess our blog's SEO performance, analyze competitor content strategies, and plan our editorial calendar improvements."

## How It Works

The skill uses **Prometheus Meta-Prompting Orchestration (PMPO)** — a structured cognitive architecture that:

- **Persists all state to disk** — survives context windows, session resets, and model swaps
- **Uses domain adapters** — specialized knowledge modules for 8 domains (with a generic fallback)
- **Separates thinking from doing** — AI reasons; tools execute deterministically  
- **Enforces convergence** — structured constraints prevent infinite iteration
- **Supports human gates** — configurable pause points for review and approval

## Supported Domains

| Domain | What Gets Assessed |
|--------|-------------------|
| Software | Build health, tests, lint, spec compliance, architecture |
| Business | Market position, revenue, competitive landscape, strategy |
| Product | UX quality, design system, accessibility, feature coverage |
| Research | Literature coverage, methodology rigor, publication impact |
| Content | SEO, engagement, freshness, editorial quality |
| Operations | Process efficiency, bottlenecks, KPIs, change management |
| Compliance | Regulatory coverage, standards gaps, remediation status |
| Generic | Adapts to any domain using universal assessment frameworks |

## Architecture

```
SKILL.md                    ← Skill definition (you are here)
prompts/
  meta-controller.md        ← PMPO orchestrator
  assess.md                 ← Phase 1: Assessment
  analyze.md                ← Phase 2: Landscape analysis
  plan.md                   ← Phase 3: Improvement planning
  execute.md                ← Phase 4: Execution
  reflect.md                ← Phase 5: Reflection
  persist.md                ← State persistence
agents/
  assessor.md               ← Assessment specialist
  analyst.md                ← Landscape research specialist
  planner.md                ← Planning specialist
  executor.md               ← Execution specialist
  reflector.md              ← Reflection specialist
references/
  domain/                   ← Domain-specific knowledge adapters
  schemas/                  ← Output contract schemas
  pmpo-theory.md            ← PMPO theory reference
skills/                     ← Slash command subskills
hooks/                      ← Lifecycle hooks
scripts/                    ← Validation and utility scripts
```

## License

MIT
