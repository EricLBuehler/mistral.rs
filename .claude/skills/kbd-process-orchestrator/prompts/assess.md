# KBD Process Orchestrator — Assess Phase

You are executing the **Assess** phase of the KBD lifecycle for the **current project**.

> **IMPORTANT**: Do NOT hard-code a project name. Read the project name from
> `.kbd-orchestrator/project.json` → `name`, or infer it from `AGENTS.md`,
> `CLAUDE.md`, `README.md`, or their equivalent.

## Goal

Produce a structured assessment of the current codebase against the active phase
goals. This is a fact-finding operation — not planning.

Also reconcile any cross-tool work completed since the last session by reading
`.kbd-orchestrator/phases/<phase>/progress.json`.

## Inputs Available to You

- **Project identity**: `.kbd-orchestrator/project.json` (or inferred)
- **Project rules**: `AGENTS.md` and/or `CLAUDE.md`
- **Canonical specs**: `openspec/specs/*.md` (if OpenSpec), or spec files defined
  in `.kbd-orchestrator/project.json` → `spec_paths`
- **Active changes**: `openspec/changes/` (if OpenSpec), or `.kbd-orchestrator/changes/`
- **Cross-tool progress**: `.kbd-orchestrator/phases/<phase>/progress.json`
- **Prior phase context**: `.kbd-orchestrator/phases/<prev-phase>/reflection.md` (if exists)
- **Source tree**: project source directories (inferred from stack)

## Assessment Dimensions

### 1. Implementation Completeness

For each domain area derived from the project specs:
- What features/entities exist in the codebase?
- What is fully implemented vs. stubbed vs. missing?
- What TODOs or placeholder comments remain?
- Map each area to: **DONE | PARTIAL | STUB | MISSING**

### 2. Spec Alignment

For each canonical spec file:
- Does the implementation match what the spec describes?
- What is specified but not implemented?
- What is implemented but not specified (unplanned scope creep)?

### 3. Cross-Tool Progress (New)

Read `progress.json` and report:
- Changes completed by other tools since last assessment
- Changes currently `IN_PROGRESS` by another tool
- Any blockers reported by executing tools

### 4. Build Health

Adapt to the project stack:
- **TypeScript/Next.js**: Does `pnpm run build` / `tsc --noEmit` pass?
- **Rust**: Does `cargo check --workspace` pass?
- **Python**: Does `mypy` or `ruff check` pass?
- **Any**: Are there failing tests? Known lint violations?

Report: **PASS | FAIL | UNKNOWN**

### 5. Constraint Compliance

Read `AGENTS.md` "Never Do" section and `.kbd-orchestrator/constraints.md` if present.
Check for known violations (import patterns, type safety, forbidden APIs, etc.)

### 6. Test Coverage

- Are tests present for new features?
- What percentage of new surface area is covered?

## Output Format

```
ASSESSMENT: <phase-name>
Project: <project-name>
Date: <ISO date>
Codebase baseline: <one sentence>
Cross-tool progress: <N changes done by other tools | none>

IMPLEMENTATION STATUS
- <area-1>: [DONE|PARTIAL|STUB|MISSING] — <detail>
- <area-2>: [DONE|PARTIAL|STUB|MISSING] — <detail>
(repeat for each spec area)

CROSS-TOOL PROGRESS
- <change-id>: <status> (by <tool>) — <last task or note>
(or: NONE — no cross-tool activity recorded)

SPEC GAP SUMMARY
- <gap 1>: <detail>
- <gap 2>: <detail>

BUILD HEALTH
- build check: [PASS|FAIL|UNKNOWN] — <command used>
- known violations: <list or NONE>
- test coverage: [FULL|PARTIAL|MINIMAL|NONE]

CONSTRAINT CHECK
- AGENTS.md violations: [NONE | <violations>]
- constraints.md violations: [NONE | N/A | <violations>]

GOAL PROGRESS
For each phase goal: [MET|PARTIAL|NOT MET] — <reason>

ASSESSMENT COMPLETE
```

Write output to `.kbd-orchestrator/phases/<phase-name>/assessment.md`.

After writing, update `.kbd-orchestrator/phases/<phase>/progress.json`:
- Set `assessment_complete: true`
- Set `last_updated_by` to the executing tool name
- Set `last_updated` to current ISO timestamp
