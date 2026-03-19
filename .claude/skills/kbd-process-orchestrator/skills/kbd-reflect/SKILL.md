---
name: kbd-reflect
description: >
  Generate the phase reflection report after all changes in the phase are
  complete. Seeds the next phase's planning inputs. Project-agnostic:
  reads progress.json, archived changes, and constraint violations.
---

# /kbd-reflect

Run the **Reflect** phase of the KBD lifecycle.

## What this does

Generates `.kbd-orchestrator/phases/<phase-name>/reflection.md` summarizing:

- Goal achievement percentage (MET / PARTIAL / NOT MET per goal)
- Delivered changes (from progress.json and archive)
- Technical debt introduced
- Lessons captured for knowledge base
- Recommended focus for next phase

## Prerequisites

All changes for this phase must be:
- Completed (status `DONE` in `progress.json`)
- If OpenSpec: verified (`/opsx:verify`) and archived (`/opsx:archive`)
- If native KBD: moved to `.kbd-orchestrator/changes/archive/<date>-<id>/`

## How to invoke

1. **Discover project identity**
2. **Confirm the active phase** — from argument or waypoint
3. **Read `progress.json`** — incorporate work done by all tools
4. **Load all change data** — from `openspec/changes/archive/` if OpenSpec,
   or `.kbd-orchestrator/changes/archive/` if native KBD
5. **Follow the reflect protocol** in `../prompts/reflect.md`
6. **Write reflection** to `.kbd-orchestrator/phases/<phase>/reflection.md`
7. **Advance the waypoint** to the next phase
8. **Trigger**: `echo '[kbd] Reflection complete — advance to next phase with /kbd-new-phase'`

## Examples

```
/kbd-reflect                             # uses active waypoint phase
/kbd-reflect phase-1-foundation          # explicit phase name
```
