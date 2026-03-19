# /kbd-init

Initialize KBD for the current project. Auto-discovers project identity, stack,
constraints, and workspace structure. Generates `.kbd-orchestrator/project.json`
and `.kbd-orchestrator/constraints.md`.

## Instructions

Read `.agent/skills/kbd-process-orchestrator/skills/kbd-init/SKILL.md` and
execute the `/kbd-init` workflow.

Run the 7-step discovery algorithm:
1. Project name & description
2. Technology stack detection
3. Build / test / lint commands
4. Spec paths
5. Constraints (from AGENTS.md "Never Do")
6. VSCode workspace discovery (.code-workspace)
7. Agent preferences

Generate `.kbd-orchestrator/project.json` from `references/schemas/project.template.json`.
Generate `.kbd-orchestrator/constraints.md` from `references/constraints.md`.

If an argument is provided (`$ARGUMENTS`), use it as the project name override.

After init, print the KBD init summary showing focus project, reference folders,
and next commands to run.
