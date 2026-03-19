# KBD Process Orchestrator — Plan Phase

You are executing the **Plan** phase of the KBD lifecycle for the **current project**.

> **IMPORTANT**: Do NOT hard-code project names, technology stacks, or
> domain areas. Derive these from project context files.

## Goal

Produce a prioritized, ordered list of changes to implement during this phase.
Each change must be a discrete, independently implementable, verifiable feature slice.

Also determine whether OpenSpec is available and emit the appropriate commands.

## Inputs

- **Project identity**: `.kbd-orchestrator/project.json` or inferred
- **Phase assessment**: `.kbd-orchestrator/phases/<phase-name>/assessment.md`
- **Project rules**: `AGENTS.md`, `CLAUDE.md`
- **Canonical specs**: `openspec/specs/*.md` (if OpenSpec), or project spec files
- **Cross-tool progress**: `.kbd-orchestrator/phases/<phase>/progress.json`

## OpenSpec Detection

Check if `openspec/` directory exists at the project root.
- **YES** → use OpenSpec changes; emit `/opsx:new <change-id>` commands
- **NO** → use native KBD changes; emit `mkdir .kbd-orchestrator/changes/<id>` instructions

## Planning Rules

1. **One change = one vertical slice** — each change should cover the feature
   end-to-end (data layer, business logic, API, UI if applicable). Never create
   purely horizontal changes (e.g., "add types everywhere").

2. **Order by dependency** — if change B depends on change A, list A first.

3. **Order by customer value** — prefer changes that unlock visible capability
   over internal refactors.

4. **Keep changes implementable in one agent session** — if an area is too large,
   split into multiple changes.

5. **Assign an execution agent** — for each change, recommend the best tool:
   - Complex multi-file features, UI pages → **Antigravity** or **Claude Code**
   - Architecture decisions → **Roo Code (Architect mode)**
   - Focused implementation → **Roo Code (Code mode)** or **Codex**
   - Quick targeted edits → **OpenCode**, **Kilo Code**, or **Cline**
   - Parallel isolated tasks → **Codex** (via git worktrees) or **Cursor Agent**
   - Human review required → **Manual**

6. **Estimate complexity** — use S (< 1 hour), M (1–4 hours), L (4–8 hours)
   as a rough guide for a skilled AI agent, not for a human.

## Priority Rules (project-derived)

Read the project's AGENTS.md or CLAUDE.md for priority guidance. In the absence
of explicit priorities, apply:
1. Foundation / blocking dependencies first
2. User-facing features over internal tooling
3. Security and data integrity over convenience features
4. Revenue-enabling features over operational improvements

## Output Format

```
PLAN: <phase-name>
Project: <project-name>
Date: <ISO date>
OpenSpec available: YES | NO
Changes to implement: <count>

CHANGE LIST (ordered)
1. <change-id>: <one-line description>
   - Scope: <layers affected, e.g., ui | api | db | all>
   - Depends on: NONE | <change-id>
   - Recommended agent: <tool from registry>
   - Est. complexity: S | M | L
   - Customer value: HIGH | MEDIUM | LOW
   - Details: <2-3 sentences describing what to build>

2. ...

EXECUTION ROUND ORDER
Round 1 (parallel): <change-ids with no dependencies>
Round 2 (parallel): <change-ids depending on Round 1>
...

COMMANDS TO RUN
<if OpenSpec>:
/opsx:new <change-id-1>
/opsx:new <change-id-2>

<if no OpenSpec>:
mkdir -p .kbd-orchestrator/changes/<change-id-1>
# Create .kbd-orchestrator/changes/<change-id-1>/change.md

PLAN COMPLETE
```

Write output to `.kbd-orchestrator/phases/<phase-name>/plan.md`.

After writing, refresh the waypoint:
- Update `.kbd-orchestrator/current-waypoint.json` → `next_pending_change` = first change ID
- Set `exact_next_command` to the first `/opsx:new` or change creation command
- Update `.kbd-orchestrator/current-waypoint.md` with the same data
