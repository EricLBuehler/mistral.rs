# /kbd-assess

Assess the codebase against the active phase goals.

## Instructions

1. Read `.kbd-orchestrator/project.json` to get `active_phase`, `focus_project_path`,
   and `workspace.folders` (reference folders for spec context).
2. Read `.kbd-orchestrator/phases/<active_phase>/plan.md` if it exists.
3. Read the full assess prompt from `.agent/skills/kbd-process-orchestrator/prompts/assess.md`.
4. Read `.kbd-orchestrator/phases/<active_phase>/progress.json` to reconcile
   any cross-tool progress already completed.
5. Execute the assessment across all dimensions defined in the prompt.
6. Write output to `.kbd-orchestrator/phases/<active_phase>/assessment.md`.
7. Update `.kbd-orchestrator/current-waypoint.json` to point to `kbd-plan`.

If a phase name is provided as an argument (`$ARGUMENTS`), use it instead of `active_phase`.
