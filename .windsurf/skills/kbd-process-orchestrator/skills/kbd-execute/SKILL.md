---
name: kbd-execute
description: >
  Select an execution backend for the active KBD phase, write canonical phase
  execution state, dispatch the phase to the appropriate tool or OpenSpec, and
  maintain KBD as the source of truth. Supports multi-tool handoff via
  progress.json protocol.
---

# /kbd-execute

Run the **Execute** phase of the KBD lifecycle.

## What this does

Reads `.kbd-orchestrator/phases/<phase-name>/plan.md`, selects the best
execution backend (tool or OpenSpec), writes `execution.md`, and dispatches the
phase while keeping KBD as the source of truth.

Also refreshes `.kbd-orchestrator/current-waypoint.json` so any AI tool can
resume cleanly.

## How to invoke

1. **Discover project identity** — read `.kbd-orchestrator/project.json` or infer
2. **Confirm the active phase** — from argument or waypoint
3. **Load waypoint** — `.kbd-orchestrator/current-waypoint.json` first when it exists
4. **Load assessment and plan** for the phase
5. **Follow the execute protocol** in `../prompts/execute.md`
6. **Write `execution.md`** with selected backend + dispatch contract
7. **Refresh waypoint** files
8. **Initialize `progress.json`** for the phase if it doesn't exist
9. **Dispatch** to selected backend or mark phase execution-ready

## Backend Types

| Backend | When to use |
|---------|------------|
| `openspec` | OpenSpec available; spec-backed traceability required |
| `native-tool` | Tool has explicit planning, inspectable progress |
| `hybrid` | Native tool for decomposition, OpenSpec for spec execution |
| `manual` | Human-only operation; no automation possible |

## Examples

```
/kbd-execute                             # uses active waypoint phase
/kbd-execute phase-2-sales-module        # explicit phase name
/kbd-execute phase-2-sales-module roo   # dispatch to Roo Code specifically
```
