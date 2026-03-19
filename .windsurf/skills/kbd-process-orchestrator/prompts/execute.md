# KBD Process Orchestrator — Execute Phase

You are executing the **Execute** phase of the KBD lifecycle for the **current project**.

> **IMPORTANT**: Do NOT hard-code project names, technology stacks, or tool preferences.
> Derive project identity and constraints from context files.

## Goal

Select the best execution backend for the active phase, write a canonical KBD
execution artifact, dispatch the phase to the appropriate tool(s), and preserve
KBD as the single source of truth for execution state.

This is an orchestration step. You select, delegate, and coordinate — you do
not necessarily execute all tasks yourself.

## Inputs Available to You

- `.kbd-orchestrator/current-waypoint.json` (highest priority re-entry point)
- `.kbd-orchestrator/phases/<phase>/assessment.md`
- `.kbd-orchestrator/phases/<phase>/plan.md`
- `.kbd-orchestrator/phases/<phase>/progress.json`
- `AGENTS.md` and `CLAUDE.md`
- OpenSpec changes (`openspec/changes/`) if available
- `.kbd-orchestrator/constraints.md` if present

## Backend Selection

### Tool Registry (from SKILL.md)

| Backend ID | Tool | Best For |
|------------|------|----------|
| `antigravity` | Antigravity | Complex multi-file features, planning, browser verification |
| `roo-architect` | Roo Code (Architect) | Architecture decisions, system design |
| `roo-code` | Roo Code (Code) | Focused bounded implementation |
| `cursor-agent` | Cursor Agent | Multi-file refactoring, parallel subagent tasks |
| `claude-code` | Claude Code CLI | Large architectural changes |
| `codex` | OpenAI Codex | Parallel isolated tasks via git worktrees |
| `cline` | Cline | Terminal-first agentic workflows |
| `kilo-code` | Kilo Code | Targeted file edits |
| `windsurf` | Windsurf Cascade | Autonomous multi-step sessions |
| `opencode` | OpenCode | Quick targeted edits and patches |
| `openspec` | OpenSpec | Spec-backed changes with traceability |
| `hybrid` | Multiple | Combination: native for decomp, OpenSpec for QA |
| `manual` | Human | Operations requiring judgment or external tools |

### Selection Rules

**Use `openspec` when:**
- OpenSpec directory exists at project root
- The phase needs spec-backed traceability
- Native backend would be too opaque for verification

**Use a specific tool backend when:**
- The change is well-bounded and the tool has explicit progress tracking
- You are dispatching a specific agent to a specific change (not the whole phase)
- The task matches the tool's strengths (see registry above)

**Use `hybrid` when:**
- Native tool useful for decomposition; OpenSpec for canonical task execution
- Multiple tools need to cooperate on different changes within the same phase

**Use `manual` when:**
- Human judgment is required (e.g., business decisions, external account setup)
- No AI tool can fully automate the operation

### OpenSpec Fallback Rule

If the selected non-OpenSpec backend:
- Cannot produce inspectable progress
- Cannot keep scope bounded to the phase
- Becomes blocked by missing structure

→ Fall back to `openspec` and document why.

## Required Output

Write `.kbd-orchestrator/phases/<phase-name>/execution.md`:

```md
EXECUTION: <phase-name>
Project: <project-name>
Date: <ISO date>
Selected backend: <backend-id from registry>
Dispatched to: <specific tool or SELF for Antigravity>
Backend rationale: <why this backend was selected>
Backend entrypoint: <skill command, tool mode, CLI command, or manual process>
OpenSpec available: YES | NO
Source plan: .kbd-orchestrator/phases/<phase-name>/plan.md

EXECUTION SCOPE
- <change-id>: <one-line description>

DISPATCH CONTRACTS
For each change assigned to a non-self tool:
- <change-id> → <tool>
  Entry: <exact prompt or command to give the tool>
  Progress file: .kbd-orchestrator/phases/<phase>/progress.json
  Handoff: Report completion by updating progress.json and committing

APPROVAL GATES
- <gate or NONE>

FALLBACK CONDITIONS
- <condition that triggers fallback to openspec>

VERIFICATION REQUIREMENTS
- <build/test command specific to this project>

PROGRESS LEDGER
- [PENDING|IN_PROGRESS|DONE|BLOCKED] <change-id> — <tool>

OUTPUTS
- <artifact or NONE>

BLOCKERS
- <blocker or NONE>

REFLECTION HANDOFF
- <what kbd-reflect should consume from this phase>

EXECUTION READY
```

Also initialize `.kbd-orchestrator/phases/<phase>/progress.json` if it doesn't
exist (use the schema in `references/schemas/progress.schema.json`).

Also refresh the KBD waypoint files:
- `.kbd-orchestrator/current-waypoint.md`
- `.kbd-orchestrator/current-waypoint.json`

## Dispatch Protocol

### If dispatching to a non-self tool (Roo, Cursor, Cline, etc.)

Produce a **Tool Handoff Note** embedded in `execution.md` under each change:

```
HANDOFF NOTE for <tool>:
1. Read .kbd-orchestrator/current-waypoint.json
2. Read the change spec: [openspec path | .kbd-orchestrator/changes/<id>/change.md]
3. On start: update progress.json status → IN_PROGRESS, started_by → <tool>
4. On each task done: increment tasks_done, commit progress.json to git
5. On completion: status → DONE, completed_by → <tool>; run /opsx:verify + /opsx:archive if OpenSpec
6. On blocker: status → BLOCKED, add to blockers array, commit
```

### If backend = `openspec` and self-executing

1. Select or create the relevant OpenSpec change (already created in kbd-plan)
2. Treat OpenSpec tasks as the working execution surface
3. Sync progress back into `progress.json` after each task
4. Refresh the current waypoint so a later session resumes cleanly

## Questions the Execute Phase Must Answer

1. What backend / tool is selected for each change?
2. Why is it selected?
3. What artifact is canonical for execution progress?
4. What conditions force fallback to OpenSpec?
5. What evidence marks each change complete?
6. What data must be handed to `kbd-reflect`?

## Completion Condition

Execute phase is complete when `execution.md` exists, all changes have backend
assignments and handoff notes, `progress.json` is initialized, and the waypoint
files are refreshed.
