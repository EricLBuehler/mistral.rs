# KBD Cross-Tool Handoff Guide

This guide defines how to properly hand off a KBD change between AI tools.
Any tool that starts, continues, or completes a KBD change MUST follow this protocol.

---

## Why File-Based State?

KBD uses `.kbd-orchestrator/` as a **universal coordination contract** because:
- Every AI tool can read and write files
- Git provides version history + merge conflict detection
- No special runtime or message bus required
- Works across IDE extensions, CLI tools, cloud agents, and local agents

This is the same pattern used by Codex (git worktrees), Cursor (Background Agents API),
and Cline (Focus Chain) — but generalized for cross-tool compatibility.

---

## Before Starting a Change

```bash
# 1. Pull latest state from git (if repo is shared)
git pull

# 2. Read the current waypoint
cat .kbd-orchestrator/current-waypoint.json

# 3. Read the change spec
# If OpenSpec:
cat openspec/changes/<change-id>/tasks.md

# If native KBD:
cat .kbd-orchestrator/changes/<change-id>/change.md

# 4. Read progress.json
cat .kbd-orchestrator/phases/<phase>/progress.json
```

Then update `progress.json`:
```json
{
  "changes": {
    "<change-id>": {
      "status": "IN_PROGRESS",
      "started_by": "<your-tool-name>",
      "started_at": "<ISO timestamp>",
      "next_task_pending": "<first task description>"
    }
  }
}
```

Commit: `git add .kbd-orchestrator && git commit -m "kbd: start [<tool>] <change-id>"`

---

## During Execution (on each task completion)

```json
{
  "changes": {
    "<change-id>": {
      "tasks_done": N,
      "last_task_completed": "<task just done>",
      "next_task_pending": "<next task>"
    }
  }
}
```

Commit after every few tasks:
`git add .kbd-orchestrator && git commit -m "kbd: progress [<tool>] <change-id> task N/M"`

---

## On Change Completion

```json
{
  "last_updated_by": "<tool>",
  "last_updated": "<ISO timestamp>",
  "changes": {
    "<change-id>": {
      "status": "DONE",
      "tasks_done": N,
      "tasks_total": N,
      "completed_by": "<tool>",
      "completed_at": "<ISO timestamp>"
    }
  }
}
```

Then:
- **If OpenSpec**: run `cat openspec/changes/<id>/.openspec.yaml` to find status, then `/opsx:verify` and `/opsx:archive`
- **If native KBD**: `mv .kbd-orchestrator/changes/<id>/ .kbd-orchestrator/changes/archive/<date>-<id>/`

Update waypoint: advance `last_completed_change` and `next_pending_change`.

Final commit: `git add .kbd-orchestrator && git commit -m "kbd: complete [<tool>] <change-id>"`

Echo: `echo '[kbd] Change complete — run /kbd-status to see next action'`

---

## On Blocker

```json
{
  "changes": {
    "<change-id>": {
      "status": "BLOCKED",
      "blockers": ["<specific description of what is blocking>"]
    }
  }
}
```

Update waypoint `fallback_command` to describe what a human or another tool should do.

Commit: `git add .kbd-orchestrator && git commit -m "kbd: blocked [<tool>] <change-id>"`

---

## Tool-Specific Entry Points

### Antigravity
- Reads SKILL.md skills via slash commands
- Progress updates happen during task execution via write_to_file tool

### Roo Code (Architect mode)
- Feed the following in the Roo Architect prompt:
  ```
  Read .kbd-orchestrator/current-waypoint.json and openspec/changes/<id>/tasks.md.
  Execute the tasks. Update .kbd-orchestrator/phases/<phase>/progress.json on each task.
  ```

### Roo Code (Code mode)
- Feed each task individually as a focused Code mode request
- Update progress.json between tasks

### Cursor Agent
- Provide the waypoint + change spec in the Cursor Agent context
- Cursor subagents can work on individual tasks in parallel
- Each subagent updates progress.json on completion

### Codex (OpenAI)
- Codex works on isolated git worktrees
- Each change runs in its own worktree
- On completion, PR/merge updates the main branch
- progress.json must be committed in the worktree and merged back

### Cline (Plan/Act mode)
- Plan mode: read assessment.md + change spec → produce task list
- Act mode: execute tasks, update progress.json per task
- Focus Chain: inject `current-waypoint.json` + change spec into context

### Windsurf Cascade
- Provide waypoint + change spec in the Cascade session
- Cascade's Flow shared workspace tracks progress within the session
- Commit progress.json at session checkpoints

### OpenCode / Kilo Code
- Use for targeted single-file or few-file tasks within a change
- Update progress.json manually after the targeted edit

---

## Conflict Resolution

If two tools update `progress.json` concurrently and create a merge conflict:

1. **Never auto-resolve to "ours" or "theirs"** — always merge manually
2. Take the **latest `tasks_done` count** (higher number wins)
3. Take the **most recent `last_task_completed`** (by `last_updated` timestamp)
4. Merge `blockers` arrays (union, no deduplication required)
5. Status priority: `BLOCKED` > `IN_PROGRESS` > `DONE` > `PENDING`

---

## What Tools Should NOT Do

- Do NOT delete or replace `progress.json` wholesale — always merging-update
- Do NOT reset `tasks_done` counter
- Do NOT change `started_by` once set
- Do NOT skip the git commit after status changes
- Do NOT modify `.kbd-orchestrator/phases/<phase>/plan.md` during execution
  (that is read-only once created; changes go in a new phase plan)
