# KBD Integration: artifact-refiner

KBD invokes `artifact-refiner` as the **Quality Assurance engine** for each
OpenSpec change's implementation artifacts. It enforces constraints, validates
code quality, and iteratively refines until all blocking constraints pass.

**Global skill location**: `.agent/skills/artifact-refiner/SKILL.md`
**Entry command**: `/refine-validate` or `/refine-code`
**State backend**: `.refiner/artifacts/<artifact-name>/state.json`

---

## When KBD Invokes artifact-refiner

| KBD Phase | Refiner Role | Entry Point |
|-----------|-------------|-------------|
| **Execute** (per-change QA) | Validate and refine a completed change's code artifacts | `/refine-code` |
| **Execute** (per-change verification) | Validate constraints without refinement | `/refine-validate` |
| **Reflect** (constraint audit) | Check for remaining violations across all changes | `/refine-validate` |

KBD invokes refiner **per completed change**, after the executing tool marks
the change `DONE` in `progress.json` and before `/opsx:verify` or archiving.

---

## Artifact Lifecycle in KBD Context

```
change DONE in progress.json
  → /refine-code "<change-id>" (artifact-refiner)
      → checks blocking constraints from .kbd-orchestrator/constraints.md
      → iterates until constraints pass or max_iterations reached
      → writes refinement_log.md to .refiner/artifacts/<change-id>/
  → /opsx:verify (if OpenSpec)
  → /opsx:archive
  → progress.json updated: completed_by = "artifact-refiner"
```

---

## How to Invoke (KBD → Refiner Contract)

```yaml
# Pass this to /refine-code for a completed KBD change
artifact_name: "<change-id>"            # e.g. "change-007-complete-team-invitations"
artifact_type: code
content_type: direct:code
constraints:
  # Import directly from .kbd-orchestrator/constraints.md
  # Copy blocking constraints as the constraint list
target_state:
  description: >
    All blocking constraints in .kbd-orchestrator/constraints.md resolved.
    Build passes. Lint clean. No forbidden patterns.
workflow_triggers:
  - event: on_iteration_complete
    action:
      type: command
      target: "<build_health_command from .kbd-orchestrator/project.json>"
  - event: on_refinement_complete
    action:
      type: command
      target: "echo '[kbd] artifact-refiner complete for <change-id> — proceed to /opsx:verify'"
```

---

## What KBD Reads Back

After refinement, KBD checks:
- `.refiner/artifacts/<change-id>/refinement_log.md` — pass/fail history
- Blocking constraint status — all PASS required before archiving
- If any constraint FAIL remains → mark change `BLOCKED` in `progress.json`

---

## KBD Constraint Wiring

The refiner uses KBD's own constraint file, eliminating duplication:

```
.kbd-orchestrator/constraints.md  ──feeds──▶  artifact-refiner constraint list
```

Never define constraints independently in the refiner invocation. Always source
them from `.kbd-orchestrator/constraints.md`.

---

## When NOT to Use

- For trivial 1-file changes with low risk: run the constraint check commands manually
- When the project has no `.kbd-orchestrator/constraints.md`: use the generic
  constraint template from `references/constraints.md`
- `artifact-refiner` is most valuable for changes with 3+ files or complex TypeScript
