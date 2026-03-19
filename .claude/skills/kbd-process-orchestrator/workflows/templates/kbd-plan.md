# /kbd-plan

Generate a prioritized, ordered change list for the current KBD phase.

## Instructions

1. Read `.kbd-orchestrator/project.json` → `active_phase`, `openspec_available`.
2. Read `.kbd-orchestrator/phases/<active_phase>/assessment.md`.
3. Read the full plan prompt from `.agent/skills/kbd-process-orchestrator/prompts/plan.md`.
4. Read `.kbd-orchestrator/phases/<active_phase>/progress.json` to skip already-completed changes.
5. If `openspec_available: true`: generate changes as OpenSpec proposals in `openspec/changes/`.
   If `openspec_available: false`: generate changes as `.kbd-orchestrator/changes/<id>/change.md`.
6. Assign each change a recommended executing agent from the tool registry.
7. Write the ordered plan to `.kbd-orchestrator/phases/<active_phase>/plan.md`.
8. Update `.kbd-orchestrator/current-waypoint.json` to point to `kbd-execute`.

If a phase name is provided as an argument (`$ARGUMENTS`), use it instead of `active_phase`.
