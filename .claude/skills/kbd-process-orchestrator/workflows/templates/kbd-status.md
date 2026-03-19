# /kbd-status

Show the current KBD process state.

## Instructions

1. Read `.kbd-orchestrator/project.json` → project name, active_phase, focus_project_path.
2. Read `.kbd-orchestrator/current-waypoint.json` → next recommended action.
3. Read `.kbd-orchestrator/phases/<active_phase>/progress.json` → change statuses,
   cross-tool activity (started_by, completed_by), blockers.
4. Read `.kbd-orchestrator/phases/<active_phase>/plan.md` → total change list.
5. If progress.json doesn't exist yet: show "Not started — run /kbd-assess first."
6. Display a formatted status report:
   - Project name and focus path
   - Active phase
   - Change inventory: PENDING / IN_PROGRESS / DONE / BLOCKED counts
   - Per-change status table with last_task_completed and next_task_pending
   - Cross-tool activity (which tool is working on what)
   - Any blockers with fallback_command
   - Next recommended action from current-waypoint.json
7. No files are written — status is read-only.
