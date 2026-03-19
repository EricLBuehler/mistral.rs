# KBD Integration: iterative-evolver

KBD invokes `iterative-evolver` as the engine for the **Assess** and the
**Assess → Reflect** inner loop at the global phase level.

**Global skill location**: `.agent/skills/iterative-evolver/SKILL.md`
**Entry command**: `/evolve` or `/evolve-assess`
**State backend**: `.evolver/evolutions/<evolution-name>/state.json`

---

## When KBD Invokes iterative-evolver

| KBD Phase | Evolver Role | Entry Point |
|-----------|-------------|-------------|
| **Assess** | Deep codebase + spec gap analysis with domain context | `/evolve-assess` |
| **Full phase loop** | Run the complete Assess → Analyze → Plan → Reflect cycle | `/evolve` |
| **Reflect** | Generate a structured reflection with before/after delta | `/evolve-report` |

> **Note**: KBD's `/kbd-assess` is a lighter, waypoint-aware wrapper.
> When a phase requires deep multi-dimensional analysis or cross-session
> continuity, delegate to `iterative-evolver` with `evolution_domain: software`.

---

## How to Invoke (KBD → Evolver Contract)

```yaml
# Pass this to /evolve when starting a new KBD phase assessment
evolution_name: "<project-name>-<phase-name>"   # e.g. "hotseaters-phase-2-sales-module"
evolution_domain: software
goals:
  - description: "<phase goal 1>"
    priority: high
  - description: "<phase goal 2>"
    priority: medium
constraints:
  # Import from .kbd-orchestrator/constraints.md blocking constraints
target_state:
  description: "<one paragraph describing the completed phase outcome>"
context:
  project_path: "<absolute path to project root>"
  documents:
    - "AGENTS.md"
    - ".kbd-orchestrator/phases/<phase>/plan.md"
  prior_assessments:
    - ".kbd-orchestrator/phases/<prev-phase>/reflection.md"
workflow_triggers:
  - event: on_cycle_complete
    action:
      type: command
      target: "echo '[kbd] evolver cycle complete — update .kbd-orchestrator/phases/<phase>/assessment.md'"
```

---

## What KBD Reads Back

After `/evolve-assess` or `/evolve-report`, KBD reads:
- `.evolver/evolutions/<evolution-name>/state.json` → `assessment.report`
- `.evolver/evolutions/<evolution-name>/reports/` → latest report file

KBD then copies or symlinks the report to:
`.kbd-orchestrator/phases/<phase>/assessment.md`

---

## Cross-Session Resumption

The `evolution_name` is the resume key. KBD stores it in `project.json`:

```json
{
  "evolver_evolution_name": "<project-name>-<phase-name>"
}
```

If the evolver session was interrupted, re-running `/evolve "<evolution-name>"`
resumes from the last checkpoint automatically.

---

## When NOT to Use

- For small phases with < 3 changes: use `/kbd-assess` directly (lightweight)
- When the phase is well-understood: skip evolver, write `assessment.md` manually
- `iterative-evolver` is most valuable for the first assessment of a major new phase
