# KBD Process Orchestrator — Reflect Phase

You are executing the **Reflect** phase of the KBD lifecycle for the **current project**.

> **IMPORTANT**: Do NOT hard-code project names, tech stacks, or crate structure.
> Derive all project-specific details from context files.

## Goal

Generate a complete phase reflection report that:
1. Measures goal achievement honestly, including cross-tool contributions
2. Surfaces technical debt introduced across all executing tools
3. Captures lessons for the knowledge base
4. Proposes the focus areas for the next phase

## Inputs

- **Project identity**: `.kbd-orchestrator/project.json` or inferred
- **Phase goals**: from `assessment.md` and `plan.md`
- **Assessment**: `.kbd-orchestrator/phases/<phase>/assessment.md`
- **Plan**: `.kbd-orchestrator/phases/<phase>/plan.md`
- **Cross-tool progress**: `.kbd-orchestrator/phases/<phase>/progress.json`
- **Archived changes**:
  - If OpenSpec: `openspec/changes/archive/<date>-<id>/` directories
  - If native KBD: `.kbd-orchestrator/changes/archive/<date>-<id>/` directories
- **Refinement logs** (if artifact-refiner was used): `.refiner/artifacts/`
- **AGENTS.md** — architectural rules to check integrity against

## Prerequisites

Before running reflect, verify all changes in this phase are complete:
- `progress.json` shows all changes as `DONE`
- If OpenSpec: all changes show `/opsx:verify` and `/opsx:archive` complete
- If native KBD: all change directories have been moved to `archive/`

If any changes are `BLOCKED`, note them explicitly and proceed with reflection
on what was completed.

## Reflection Dimensions

### 1. Goal Achievement

For each stated phase goal: **MET | PARTIAL | NOT MET**, with an honest reason.
Credit completed work regardless of which tool executed it.
Calculate overall completion percentage.

### 2. What Was Delivered

List all changes that were implemented and archived, noting which tool executed each.
Format: `- <change-id>` — <description> (by: <tool>)

### 3. Technical Debt Introduced

List any shortcuts, stubs, TODOs, or known violations deferred from this phase.
Be specific — mention file paths where known. Note which tool introduced the debt.

### 4. Architecture Integrity

Check against `AGENTS.md` "Never Do" section and `.kbd-orchestrator/constraints.md`:
- Were any "Never Do" rules violated?
- Are known constraint violations present?
- What technical patterns were broken?

### 5. Lessons Learned

Concrete, reusable learnings from this phase, especially around multi-tool coordination:
- What worked well between tools?
- What caused confusion or rework?
- What should the next phase do differently?

Format as bullet points suitable for adding to a Knowledge Item (KI).

### 6. Cross-Tool Coordination Review (New)

Assess how well the multi-tool workflow functioned:
- Were `progress.json` updates made reliably?
- Were there any gaps where state was lost between tools?
- What handoff notes worked well? What was unclear?
- Recommendations for improving the cross-tool protocol.

### 7. Next Phase Recommendations

Based on what was completed and what remains:
- What should the next phase focus on?
- What incomplete work should become high-priority changes in the next phase?
- What architectural decisions need human review before proceeding?

## Output Format

Write to `.kbd-orchestrator/phases/<phase-name>/reflection.md`:

```markdown
# Phase Reflection: <phase-name>
**Project:** <project-name>
**Date:** <ISO date>
**Phase completion:** <N>%
**Changes completed:** <N> / <total>

## Goals
| Goal | Status | Notes |
|------|--------|-------|
| <goal> | MET/PARTIAL/NOT MET | <honest reason> |

## Delivered Changes
- `<change-id>` — <description> (by: <tool>)

## Technical Debt
- <specific debt item with file path or location>
- (NONE if clean)

## Architecture Integrity
- AGENTS.md violations: NONE | <violations found>
- Constraint violations: NONE | N/A | <specific violations>

## Cross-Tool Coordination Notes
- Progress tracking: RELIABLE | GAPS FOUND — <detail>
- Handoff quality: CLEAR | UNCLEAR — <detail>
- Recommendations: <what to improve>

## Lessons Learned
- <lesson>

## Next Phase Focus
<recommended next phase name and top 3 priority areas>

## Context for Next Phase
Use this file as prior context for the next `/kbd-assess` invocation.
```

After writing, advance the waypoint to the next phase:
- Set `.kbd-orchestrator/current-waypoint.json` → `active_phase` = next phase name
- Set `next_pending_change` = null (plan not yet created)
- Set `exact_next_command` = `/kbd-assess <next-phase-name>`
- Commit: `git add .kbd-orchestrator && git commit -m "kbd: phase reflect complete — <phase-name>"`
