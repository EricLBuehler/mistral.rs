---
name: kbd-process-orchestrator
version: "2.0.0"
description: >
  The Universal Knowledge-Based Development (KBD) process orchestrator.
  Drives the full iterative PMPO lifecycle for ANY project —
  Assess → Analyze → Plan → Execute (backend dispatch) → Reflect — at every
  granularity level: global phases, OpenSpec changes, and artifact-level QA.
  Implements the process defined in TJ-KBD-UNIVERSAL-001.
  Coordinates execution across multiple AI tools (Antigravity, Roo Code,
  Cursor Agent, Claude Code, Codex, Cline, Kilo Code, Windsurf, OpenCode)
  using .kbd-orchestrator/ as the shared, file-based source of truth.
authors:
  - "Prometheus AGS"
allowed-tools: file_system web_search code_interpreter sequential_thinking memory
triggers:
  keywords:
    - kbd
    - phase
    - openspec
    - planning phase
    - reflect phase
    - kbd-plan
    - kbd-assess
    - kbd-execute
    - kbd-reflect
    - kbd-status
    - iterative evolver
  semantic: >
    Orchestrate a project phase, run the KBD assessment loop,
    create or manage OpenSpec changes, invoke artifact-refiner QA,
    or generate a phase reflection report.
---

# KBD Process Orchestrator

The universal process orchestrator for any software project. Implements the
Knowledge-Based Development lifecycle defined in **TJ-KBD-UNIVERSAL-001** using
PMPO orchestration at three nested levels: phase, change, and artifact.

This skill is **project-agnostic**. It derives project identity from context
(AGENTS.md, CLAUDE.md, README.md, package.json, Cargo.toml, pyproject.toml,
or explicit prompt arguments). Do not hard-code project names into this skill.

---

## Project Context Discovery

On every invocation, before acting, KBD MUST:

1. **Identify the project name** — read, in priority order:
   - Explicit argument (`/kbd-plan my-project`)
   - `.kbd-orchestrator/project.json` if it exists
   - `AGENTS.md` header or `CLAUDE.md` header
   - `README.md` first heading
   - `package.json` → `name`, `Cargo.toml` → `[package] name`, `pyproject.toml` → `name`

2. **Identify project-specific constraints** — read `AGENTS.md` and the
   project's canonical spec files (OpenSpec `openspec/specs/*.md`, or
   equivalent spec directory defined in `.kbd-orchestrator/project.json`).

3. **Derive the technology stack** — identify from lock files / config files
   to tailor the Build Health and Test Coverage assessment dimensions.

---

## The Three Levels

### Level 1 — Global Phase (this skill)
Assess → Analyze → Plan → Execute (backend selection + dispatch) → Reflect
KBD owns canonical phase state and delegates execution to OpenSpec, a native
planner backend, or a designated AI tool.

### Level 2 — OpenSpec Change (inner loop)
`/opsx:new` → `/opsx:apply` → `/opsx:verify` → `/opsx:archive`
Delegates QA to `artifact-refiner` when available.

### Level 3 — Artifact QA (innermost)
`artifact-refiner` Specify → Plan → Execute → Reflect → Persist

---

## Multi-Tool Coordination Architecture

KBD uses **file-based state** in `.kbd-orchestrator/` as the universal
coordination contract. Any AI tool that can read and write files can participate
in the KBD process, regardless of its internal planning mechanism.

### State Files (Source of Truth)

| File | Written by | Read by | Purpose |
|------|-----------|---------|---------|
| `.kbd-orchestrator/current-waypoint.json` | Any orchestrator | All tools | Resume contract — exact next step |
| `.kbd-orchestrator/current-waypoint.md` | Any orchestrator | All tools | Human-readable waypoint summary |
| `.kbd-orchestrator/phases/<phase>/assessment.md` | kbd-assess | kbd-plan | Gap analysis output |
| `.kbd-orchestrator/phases/<phase>/plan.md` | kbd-plan | kbd-execute | Ordered change list |
| `.kbd-orchestrator/phases/<phase>/execution.md` | kbd-execute | All tools | Backend dispatch contract |
| `.kbd-orchestrator/phases/<phase>/progress.json` | Any executing tool | kbd-status | Live task progress ledger |
| `.kbd-orchestrator/phases/<phase>/reflection.md` | kbd-reflect | Next phase | Phase retrospective |
| `.kbd-orchestrator/project.json` | Initial setup | All tools | Project identity + config |

### progress.json Protocol

Every AI tool that executes a KBD change MUST update `progress.json` on start
and on completion of each task. This is how KBD stays synchronized across tool
boundaries.

```json
{
  "phase": "<phase-name>",
  "last_updated": "<ISO 8601 timestamp>",
  "last_updated_by": "<tool-name: antigravity|roo|cursor|codex|cline|opencode|windsurf|human>",
  "changes": {
    "<change-id>": {
      "status": "PENDING|IN_PROGRESS|DONE|BLOCKED|SKIPPED",
      "tasks_total": 0,
      "tasks_done": 0,
      "last_task_completed": "<task description or null>",
      "next_task_pending": "<task description or null>",
      "started_by": "<tool-name>",
      "completed_by": "<tool-name or null>",
      "blockers": []
    }
  }
}
```

### Tool Registry

KBD recognizes the following execution agents. Each has a recommended usage pattern:

| Tool | Best For | Entry Point |
|------|----------|-------------|
| **Antigravity** | Complex multi-file features, planning, browser verification | `SKILL.md` slash commands |
| **Roo Code (Architect mode)** | Architecture decisions, system-level planning | Roo Architect mode prompt |
| **Roo Code (Code mode)** | Focused implementation of bounded tasks | Roo Code mode prompt |
| **Cursor Agent** | Multi-file refactoring, parallel subagent tasks | Cursor Agent mode |
| **Claude Code** | Large architectural changes, CLI-driven execution | `claude` CLI |
| **Codex (OpenAI)** | Parallel isolated tasks via git worktrees | OpenAI Codex app or CLI |
| **Cline** | Terminal-first agentic workflows with Plan/Act mode | Cline VSCode extension |
| **Kilo Code** | Targeted code edits in VSCode | Kilo Code extension |
| **Windsurf Cascade** | Autonomous multi-step tasks with shared session | Windsurf Cascade panel |
| **OpenCode** | Quick targeted edits and file patches | OpenCode CLI/extension |
| **Human** | Decisions requiring judgment, external tool operations | Manual |

---

## Knowledge Stack

| Layer | Sources |
|-------|---------|
| Project Identity | `.kbd-orchestrator/project.json`, `AGENTS.md`, `CLAUDE.md`, `README.md` |
| Spec Baselines | `openspec/specs/*.md` (if OpenSpec), or project spec directory |
| Change Specs | `openspec/changes/<id>/*.md` (if OpenSpec), or change directories |
| Execution State | `.kbd-orchestrator/phases/<phase>/` artifacts |
| Progress | `.kbd-orchestrator/phases/<phase>/progress.json` |

---

## Integration Layer

KBD delegates specialized work to **4 global skills**. These skills are NOT
copied into this skill's directory — they live in the global `.agent/skills/`
directory and are invoked by reference. This preserves a single source of truth
for each skill, ensures updates propagate automatically, and avoids maintenance split.

Full integration contracts are defined in `references/integrations/`:

| Global Skill | KBD Phase | Role | Integration Guide |
|-------------|-----------|------|-------------------|
| **iterative-evolver** | Assess | Deep codebase + spec gap analysis with cross-session continuity | `references/integrations/iterative-evolver.md` |
| **artifact-refiner** | Execute (per-change QA) | Constraint-driven code quality gate before archiving each change | `references/integrations/artifact-refiner.md` |
| **bdd-testing** | Execute (verification) | Behavioral verification gate — BDD scenarios must pass before DONE | `references/integrations/bdd-testing.md` |
| **pmpo-skill-creator** | Reflect (meta) | KBD self-improvement — extend kbd with new sub-skills discovered during reflection | `references/integrations/pmpo-skill-creator.md` |

### Why Global, Not Copies?

Each of these skills has its own:
- **PMPO loop** with independent phase states
- **Named file-backed state** (`.evolver/`, `.refiner/`, `.creator/`)
- **Entry commands** that are already registered globally

Copying them would require maintaining two versions, break the single-source-of-truth
principle, and make it impossible to share state between KBD and non-KBD invocations
of the same skill. The integration adapters in `references/integrations/` specify
the exact invocation contract (what to pass, what to read back) without duplicating
any logic.

### Integration Timing in the KBD Loop

```
ASSESS
 └─ /kbd-assess (lightweight, waypoint-aware)
    └─ if deep analysis needed: /evolve-assess "<project>-<phase>" ← iterative-evolver

PLAN
 └─ /kbd-plan (generates change list)

EXECUTE (per change)
 └─ <executing tool> implements the change
    └─ /bdd-testing: write feature file + step defs   ← bdd-testing
    └─ <tool runs the feature>: pnpm test:bdd
    └─ /artifact-refiner: constraint QA gate          ← artifact-refiner
    └─ /opsx:archive or native archive

REFLECT
 └─ /kbd-reflect (phase retrospective)
    └─ if structural gap found: /extend-skill or /create-skill  ← pmpo-skill-creator
```

---

## Global Phase Map

Phases are project-defined. KBD does not impose a fixed phase sequence.
The orchestrating agent reads `.kbd-orchestrator/phases/` to discover
existing phases and their status.

A typical phase progression for software projects:

| Phase | Pattern | Status |
|-------|---------|--------|
| Phase 0 | Baseline & KBD Setup | Recommend first |
| Phase 1 | Foundation / Core Architecture | High priority |
| Phase N | Feature Modules (iterative) | Per roadmap |
| Phase Final | Production Hardening | Final |

---

## Execution Model (PMPO Loop)

### Startup — always do this first

1. **Discover project identity** — follow Project Context Discovery above
2. **Load waypoint** — read `.kbd-orchestrator/current-waypoint.json` as the
   preferred resume contract before inferring the next action
3. **Load phase context** — identify active phase, load existing phase artifacts
4. **Load domain knowledge** — read `AGENTS.md`, spec files
5. **Check progress.json** — reconcile what any other tool has done since last session

### Loop

1. **Assess** (`prompts/assess.md`) — inspect repo, reconcile with spec, surface gaps
2. **Analyze** — identify highest-leverage missing features, prioritize
3. **Plan** (`prompts/plan.md`) — produce ordered list of changes for this phase
4. **Execute** (`prompts/execute.md`) — select backend, write `execution.md`, dispatch
5. **Reflect** (`prompts/reflect.md`) — run evolver report, capture lessons, seed next phase
6. **Persist** — write phase state, refresh waypoint, commit

After each phase: checkpoint + dispatch workflow triggers.

---

## OpenSpec Availability

OpenSpec is **optional**. KBD adapts:

### When OpenSpec IS available (`openspec/` directory exists)
- Use `/opsx:new` to create structured changes with proposal → design → tasks
- Progress tracked in `openspec/changes/<id>/tasks.md`
- Archiving via `/opsx:archive` feeds the reflection phase

### When OpenSpec is NOT available
- Use KBD's built-in change management via `.kbd-orchestrator/changes/<id>/`
- Create `change.md` (same structure as OpenSpec proposal + tasks combined)
- Track task status with `[ ]` / `[/]` / `[x]` in `change.md`
- Archive by moving to `.kbd-orchestrator/changes/archive/<date>-<id>/`

KBD **never** requires OpenSpec. The `execution.md` format accommodates both.

---

## Wayfinding State

KBD maintains a resumable return point for the current phase.

- Canonical files:
  - `.kbd-orchestrator/current-waypoint.md`
  - `.kbd-orchestrator/current-waypoint.json`
- Minimum fields:
  - `active_phase` — current phase name
  - `backend` — selected execution backend
  - `last_completed_change` — last archived/completed change ID
  - `next_pending_change` — next change to start
  - `preferred_re_entry_skill` — which skill to invoke on next session
  - `exact_next_command` — the exact `/opsx:new`, `/kbd-execute`, etc.
  - `fallback_command` — what to do if primary command fails

When the waypoint exists, any AI tool should consult it before deriving
status from broader phase discovery.

---

## Cross-Tool Reporting Protocol

When an AI tool (Roo, Cursor, Cline, Codex, etc.) is dispatched to execute a
KBD change, it MUST follow this protocol:

### On Start of a Change
1. Read `.kbd-orchestrator/current-waypoint.json`
2. Read the change spec (OpenSpec or `.kbd-orchestrator/changes/<id>/change.md`)
3. Update `progress.json`: set status → `IN_PROGRESS`, `started_by` → `<tool-name>`
4. Update waypoint: `last_updated_by` → `<tool-name>`

### During Execution (on each task completion)
1. Update `progress.json`: increment `tasks_done`, update `last_task_completed` and `next_task_pending`
2. Commit the progress file to git: `git add .kbd-orchestrator && git commit -m "kbd: progress update [<tool>] <change-id> task N/M"`

### On Change Completion
1. Update `progress.json`: set status → `DONE`, `completed_by` → `<tool-name>`
2. If OpenSpec: run `/opsx:verify` then `/opsx:archive`
3. If native KBD: move change to `.kbd-orchestrator/changes/archive/<date>-<id>/`
4. Update waypoint: advance `last_completed_change` and `next_pending_change`
5. Commit all state: `git add .kbd-orchestrator && git commit -m "kbd: change complete [<tool>] <change-id>"`
6. **Echo the KBD hook**: `echo '[kbd] Change complete — run /kbd-assess or /kbd-reflect as appropriate'`

### On Blocker
1. Update `progress.json`: set status → `BLOCKED`, add to `blockers` array
2. Update waypoint: set `fallback_command` to describe the blocker
3. Commit: `git add .kbd-orchestrator && git commit -m "kbd: blocked [<tool>] <change-id>"`

---

## Blocking Constraints (Project-Derived)

Unlike the previous version, KBD does not hard-code Rust/DocuMind constraints.
Project constraints are defined in:

1. `AGENTS.md` — "Never Do" and code style rules
2. `.kbd-orchestrator/constraints.md` — project-specific blocking/warning rules

The executing tool MUST read these files and apply constraints when verifying work.

---

## Required Tools
- `file_system` — Read/write spec files, phase reports, progress ledger
- `sequential_thinking` — Multi-step phase planning and gap analysis

## Optional Tools
- `web_search` / `tavily` — External research during Analyze phase
- `code_interpreter` — Run build/test commands during QA
- `memory` — Cross-session persistence for phase state

---

## Quick Start Commands

### First use in a new project
```
/kbd-init               # Auto-discover project and generate .kbd-orchestrator/project.json
/kbd-new-phase <name>   # Start the first phase
/kbd-assess             # Run the first assessment
```

> **IMPORTANT — project.json is GENERATED, not shipped.**
> `.kbd-orchestrator/project.json` is always created by `/kbd-init` using auto-discovery.
> It lives in the project repository, not in this skill directory.
> The skill only ships the template: `references/schemas/project.template.json`.
> Never commit project-specific values into the skill files.

### Ongoing workflow
- `/kbd-init [--force] [--dry-run]` — Initialize or re-initialize project context
- `/kbd-assess [phase-name]` — Assess current codebase against active phase goals
- `/kbd-plan [phase-name]` — Create prioritized change list for current phase
- `/kbd-execute [phase-name]` — Select execution backend and dispatch phase
- `/kbd-reflect [phase-name]` — Generate phase reflection report + seed next phase
- `/kbd-status` — Show current phase, change inventory, and waypoint-guided next action
- `/kbd-new-phase <name> [goals...]` — Start a new named phase with goals
- `/kbd-full-phase <name>` — Run full Assess → Plan → Execute → Reflect cycle

See `references/domain/kbd.md` for the generic KBD philosophy reference.
See `references/cross-tool-handoff.md` for the multi-tool coordination protocol.
See `prompts/` for the detailed phase execution protocols.
