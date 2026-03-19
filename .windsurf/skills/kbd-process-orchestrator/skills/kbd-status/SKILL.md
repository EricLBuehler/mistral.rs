---
name: kbd-status
description: >
  Show current KBD process status — active phase, change inventory, goal
  completion, and next recommended action. Reads progress.json to surface
  work completed by all tools (Antigravity, Roo, Cursor, Cline, Codex, etc.)
---

# /kbd-status

Show current KBD process state for the active project.

## What this does

Reads orchestrator state and progress ledger, then produces a complete status
summary including cross-tool work visibility.

Output includes:
- Project name and active phase
- OpenSpec changes: active / in-progress / archived (if OpenSpec available)
- Native KBD changes: status from `progress.json`
- Goal completion: MET | PARTIAL | NOT MET per goal
- Last tool to update state and when
- Waypoint-guided next recommended action

## How to invoke

1. **Discover project identity** — read `.kbd-orchestrator/project.json` or infer
2. **Read waypoint** — `.kbd-orchestrator/current-waypoint.json`
3. **Read progress** — `.kbd-orchestrator/phases/<phase>/progress.json`
4. **Read phase artifacts** — `assessment.md`, `plan.md`, `execution.md`
5. **If OpenSpec**: read `openspec/changes/` active + `openspec/changes/archive/`
6. **Print status table**

## Output Example

```
KBD STATUS — <Project Name>
Active phase: phase-2-sales-module
Last updated by: roo-code (2026-03-12T04:30:00Z)

Goals:
  [✅] Sidebar user profile wired to real data
  [🔄] Team invitations email flow (IN_PROGRESS — started by antigravity)
  [⬜] Clients page full implementation
  [⬜] Deals 4-step wizard

Changes:
  DONE:        change-006-sidebar-user-profile (completed by: antigravity)
  IN_PROGRESS: change-007-complete-team-invitations (4/8 tasks, started by: antigravity)
  PENDING:     change-008-clients-page
  PENDING:     change-009-deals-page

Next action (from waypoint): /opsx:apply change-007-complete-team-invitations
```

## Examples

```
/kbd-status                   # current project + active phase
/kbd-status phase-1-foundation # status of a specific phase
```
