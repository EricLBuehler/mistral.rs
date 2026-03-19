---
name: kbd-plan
description: >
  Create a prioritized, ordered change list for the current project phase.
  Project-agnostic: reads assessment and project constraints to produce
  an ordered change list. Emits OpenSpec commands if OpenSpec is available;
  falls back to native KBD change files if not.
---

# /kbd-plan

Run the **Plan** phase of the KBD lifecycle for any project.

## What this does

Reads `.kbd-orchestrator/phases/<phase-name>/assessment.md` and produces an
ordered list of changes to implement this phase. Refreshes the waypoint so
every tool knows the exact next step.

Output:
- `.kbd-orchestrator/phases/<phase-name>/plan.md` — ordered change list
- `.kbd-orchestrator/current-waypoint.md` and `current-waypoint.json` — refreshed

## How to invoke

1. **Discover project identity** — read `.kbd-orchestrator/project.json` or infer
2. **Confirm the active phase** — from argument or waypoint
3. **Load assessment** — from `.kbd-orchestrator/phases/<phase>/assessment.md`
4. **Read project constraints** — from `AGENTS.md` and project spec files
5. **Follow the plan protocol** in `../prompts/plan.md`
6. **Write plan.md** with ordered change list and recommended agent per change
7. **If OpenSpec is available**: emit `/opsx:new <change-id>` commands
8. **If OpenSpec is NOT available**: emit directory creation instructions for
   `.kbd-orchestrator/changes/<change-id>/change.md`
9. **Refresh waypoint** files (`current-waypoint.md` and `current-waypoint.json`)

## Examples

```
/kbd-plan                                # uses active waypoint phase
/kbd-plan phase-1-foundation             # explicit phase name
```
