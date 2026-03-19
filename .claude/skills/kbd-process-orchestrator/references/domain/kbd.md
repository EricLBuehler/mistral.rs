# KBD Philosophy & Universal Reference

This file documents the KBD (Knowledge-Based Development) philosophy.
It is used during the Assess and Plan phases to ground decisions.

Read this file on every KBD session regardless of which project is active.

---

## What is KBD?

**Knowledge-Based Development (KBD)** is an iterative, PMPO-driven software
development process designed to keep AI tools coherent, aligned, and productive
across multi-session, multi-tool, multi-phase projects.

KBD solves three core problems:
1. **Session amnesia** — AI tools lose context between sessions
2. **Tool fragmentation** — different AI tools each have their own planning state
3. **Spec drift** — implementation diverges from documented intent over time

---

## Core Principles

### 1. File-Based State is Universal
`.kbd-orchestrator/` is the single source of truth. Any tool that can read/write
files can participate in KBD — no special runtime, message bus, or integration
required.

### 2. Waypoint-First Resumption
On every session start, load `.kbd-orchestrator/current-waypoint.json` before
doing anything else. This file tells you exactly where to resume.

### 3. KBD is the Orchestrator, Tools are Workers
KBD owns phase state. AI tools (Roo, Cursor, Codex, etc.) are execution agents.
They report back to KBD by updating `progress.json` and committing it.

### 4. OpenSpec is Optional Enhancement
OpenSpec provides structured change management with proposal → design → tasks.
If OpenSpec is not present, KBD uses `.kbd-orchestrator/changes/<id>/change.md`
with the same structure. KBD never depends on OpenSpec to function.

### 5. Progress is Committed to Git
All KBD state changes are committed to git. This provides:
- Cross-tool visibility (git pull gives any tool the latest state)
- History and auditability
- Conflict detection (merge conflicts surface concurrent edits)

### 6. Assess Before Plan, Plan Before Execute, Execute Before Reflect
The PMPO loop is not optional. Skipping Assess means you plan without facts.
Skipping Plan means you execute without direction. Skipping Reflect means you
lose lessons and fail to seed the next phase.

---

## The PMPO Lifecycle

```
┌──────────────────────────────────────────────────────┐
│                KBD Phase Lifecycle                    │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐        │
│  │ ASSESS  │───▶│  PLAN   │───▶│ EXECUTE  │        │
│  └─────────┘    └─────────┘    └────┬─────┘        │
│       ▲                             │               │
│       │                             ▼               │
│       │                        (tools do work)      │
│       │                        progress.json ──git──┤
│       │                             │               │
│       │                             ▼               │
│       │                        ┌─────────┐         │
│       └────────────────────────│ REFLECT │         │
│                                └─────────┘         │
│                                      │              │
│                                      ▼              │
│                              Next Phase Seed         │
└──────────────────────────────────────────────────────┘
```

---

## Phase Naming Convention

Phases should be named descriptively with a prefix:
```
phase-0-baseline
phase-1-foundation
phase-2-<module-name>
phase-3-<module-name>
phase-N-production-hardening
```

---

## Project Context Discovery Order

KBD reads project identity from, in priority order:
1. Explicit argument (e.g., `/kbd-plan hotseaters phase-2`)
2. `.kbd-orchestrator/project.json` → `name`
3. `AGENTS.md` — first H1 or project name mention
4. `CLAUDE.md` — first H1 or project name mention
5. `README.md` — first H1
6. `package.json` → `name`
7. `Cargo.toml` → `[package] name`
8. `pyproject.toml` → `name`
9. Directory name of the repository root

---

## Cross-Tool Handoff Protocol Summary

See `references/cross-tool-handoff.md` for the full protocol.

Quick reference:
1. Any tool starting work → update `progress.json` status = IN_PROGRESS, commit
2. Any tool completing a task → update `tasks_done`, commit
3. Any tool completing a change → status = DONE, archive, advance waypoint, commit
4. Any tool hitting a blocker → status = BLOCKED, update `fallback_command`, commit
5. Next session starts → read `current-waypoint.json` first, then `progress.json`
