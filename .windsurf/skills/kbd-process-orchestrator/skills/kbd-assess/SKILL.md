---
name: kbd-assess
description: >
  Assess the current project codebase against the active phase goals.
  Project-agnostic: reads AGENTS.md, spec files, and codebase to produce
  a structured assessment. Works for any technology stack or language.
---

# /kbd-assess

Run the **Assess** phase of the KBD lifecycle for any project.

## What this does

Inspects the current codebase and produces a structured gap report against the
active phase's goals. Output written to
`.kbd-orchestrator/phases/<phase-name>/assessment.md`.

Also reads `progress.json` to incorporate any work completed by other tools
(Roo, Cursor, Cline, Codex, etc.) since the last session.

## How to invoke

1. **Discover project identity** — read `.kbd-orchestrator/project.json` or infer
   from `AGENTS.md`, `CLAUDE.md`, `README.md`, `package.json`, `Cargo.toml`, etc.
2. **Confirm the active phase** — from argument or `.kbd-orchestrator/current-waypoint.json`
3. **Resume from progress** — read `.kbd-orchestrator/phases/<phase>/progress.json`
   to account for cross-tool work done
4. **Load specs** — read `openspec/specs/*.md` if OpenSpec is available,
   otherwise read the canonical spec files defined in `.kbd-orchestrator/project.json`
5. **Inspect the codebase** — scan feature directories, components, routes, etc.
6. **Follow the assess protocol** in `../prompts/assess.md`
7. **Write assessment file** to `.kbd-orchestrator/phases/<phase>/assessment.md`
8. **Update progress.json** with `assessment_complete: true`

## Examples

```
/kbd-assess                              # uses active waypoint phase
/kbd-assess phase-1-foundation           # explicit phase name
/kbd-assess phase-2-sales-module         # for a new project phase
```
