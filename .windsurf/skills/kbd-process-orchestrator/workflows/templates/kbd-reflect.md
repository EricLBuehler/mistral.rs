# /kbd-reflect

Generate a phase reflection report and seed the next phase.

## Instructions

1. Read `.kbd-orchestrator/project.json` → `active_phase`.
2. Read `.kbd-orchestrator/phases/<active_phase>/progress.json`.
3. Read `.kbd-orchestrator/phases/<active_phase>/plan.md`.
4. Read the full reflect prompt: `.agent/skills/kbd-process-orchestrator/prompts/reflect.md`.
5. Calculate goal completion % from progress.json (tasks_done / tasks_total across all changes).
6. Assess cross-tool coordination effectiveness from progress.json `started_by`/`completed_by`.
7. Check if pmpo-skill-creator should be invoked for structural KBD improvements identified.
8. Write `reflection.md` to `.kbd-orchestrator/phases/<active_phase>/reflection.md`.
9. Generate next-phase seed in `reflection.md` under "Next Phase Seed" section.
10. Update `.kbd-orchestrator/current-waypoint.json`:
    - If next phase exists → point to `kbd-assess` for that phase
    - If no next phase → point to `kbd-new-phase`
11. Commit all `.kbd-orchestrator/` state.

If a phase name is provided as an argument (`$ARGUMENTS`), use it instead of `active_phase`.
