# PMPO Theory Reference

Prometheus Meta-Prompting Orchestration (PMPO) is a structured loop for AI-driven creation and evolution. This reference defines the theory for inclusion in generated skills.

## Core Principle

Complex AI tasks benefit from structured phases with clear contracts between them. PMPO decomposes work into ordered phases, each with a specialized controller and agent, producing validated outputs that feed the next phase.

## Phase Architecture

### Standard 5-Phase Loop

```
Specify → Plan → Execute → Reflect → Persist
   ↑                                    │
   └────────── Loop (if needed) ←───────┘
```

| Phase | Purpose | Agent Type |
|-------|---------|------------|
| **Specify** | Transform intent into structured spec | Analyst/Architect |
| **Plan** | Design approach, map resources, set constraints | Architect/Planner |
| **Execute** | Carry out the plan, produce output | Generator/Executor |
| **Reflect** | Validate output, score quality, decide loop/terminate | Validator/Reflector |
| **Persist** | Durably save state and output | Generator/Persister |

### Alternative Phase Configurations

Skills may use subsets or variations:

| Configuration | Phases | Use Case |
|--------------|--------|----------|
| Full PMPO | S → P → E → R → Persist | Complex creation/evolution |
| Assess-Plan-Execute | A → P → E → R → Persist | Landscape-aware evolution (evolver) |
| Specify-Execute | S → E → R → Persist | Simpler refinement |
| Execute-Reflect | E → R → Persist | Single-shot with validation |

## State Management

### Named State

Every PMPO session has a named identifier for cross-session persistence:
- **Skill creation**: `creation_id` + `skill_name`
- **Evolution**: `evolution_id` + `evolution_name`
- **Refinement**: `refinement_id` + `artifact_name`

### State Lifecycle

```
Init → [Phase Loop] → Checkpoint* → Finalize
```

- **Init**: Create new or resume active state
- **Checkpoint**: Mid-session snapshot after each phase
- **Finalize**: Archive completed state, update registry

### Provider Abstraction

State can be persisted via:
1. Filesystem (always available as fallback)
2. Memory MCP tools
3. Custom MCP servers
4. Database backends

Resolution order: env → project config → global config → MCP → memory → filesystem.

## Hooks and Lifecycle Events

### Phase Hooks

```
Phase Start → Agent Work → Phase Complete → Checkpoint → Dispatch
```

After each phase:
1. **Checkpoint**: Snapshot current state for crash recovery
2. **Dispatch**: Fire lifecycle events for external integration

### Event Types

| Event | When | Payload |
|-------|------|---------|
| `on_phase_complete` | After any phase | Phase name, iteration |
| `on_creation_complete` | After successful termination | Full state |
| `on_regression` | Reflect detects quality drop | Delta details |
| `on_approval_required` | Human gate needed | Constraint details |

## Loop Control

### Convergence Criteria

The Reflect phase evaluates whether to loop or terminate:

| Signal | Action |
|--------|--------|
| All checks pass, quality ≥ threshold | **Terminate** |
| Fixable issues in output | **Loop** → Execute |
| Structural issues in design | **Loop** → Plan |
| Max iterations reached | **Terminate** with warnings |

### Iteration Limits

Default: 3-5 iterations maximum. Prevents infinite loops while allowing iterative improvement.

## Domain Adapters

Domain adapters customize PMPO phases for specific contexts:
- Define domain-specific evaluation criteria
- Provide domain vocabulary and examples
- Set quality thresholds for the domain
- Route tool usage for domain needs

Examples: software, business, content, compliance, code, meta-prompt, logo, ui.

## Inter-Phase Contracts

Every phase produces a structured output (typically YAML or JSON) that feeds the next phase. These contracts are defined in phase controller prompts and validated by schemas.

```
Specify → skill_spec
Plan → skill_plan
Execute → execution_result
Reflect → reflection
Persist → persist_result
```
