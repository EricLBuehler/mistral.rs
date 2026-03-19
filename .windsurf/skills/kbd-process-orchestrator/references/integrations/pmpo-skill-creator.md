# KBD Integration: pmpo-skill-creator

KBD invokes `pmpo-skill-creator` when the **KBD process itself needs to evolve**
— either to extend the orchestrator with new sub-skills, create project-specific
skill variants, or generate new skills discovered during a phase reflection.

**Global skill location**: `.agent/skills/pmpo-skill-creator/SKILL.md`
**Entry commands**: `/create-skill`, `/clone-skill`, `/extend-skill`, `/validate-skill`
**State backend**: `.creator/skills/<name>/state.json`

---

## When KBD Invokes pmpo-skill-creator

| KBD Phase | Creator Role | Entry Point |
|-----------|-------------|-------------|
| **Reflect** (meta-improvement) | Reflect surfaces need for new KBD sub-skill or domain adapter | `/create-skill` or `/extend-skill` |
| **Plan** (tooling gap) | Phase plan requires tooling that doesn't exist as a skill | `/create-skill` |
| **Post-init** | `/kbd-init` surfaces that a project needs a custom constraint skill | `/clone-skill` |
| **Any phase** | Validate the kbd-process-orchestrator itself against agentskills.io spec | `/validate-skill` |

---

## Key Usage Patterns

### 1. Extending kbd-process-orchestrator with a new sub-skill

When a phase reflection (`reflection.md`) identifies a recurring orchestration
pattern that should be codified:

```
/extend-skill
  source_skill: .agent/skills/kbd-process-orchestrator
  intent: "Add /kbd-audit sub-skill that scans .kbd-orchestrator/ for state consistency"
  mode: extend
```

This adds a new directory to `skills/kbd-audit/SKILL.md` without touching existing files.

### 2. Creating a project-specific domain adapter for iterative-evolver

```
/clone-skill
  source_skill: .agent/skills/iterative-evolver
  intent: "Create a saas-product domain adapter for iterative-evolver"
  domain: saas-product
  mode: clone
```

### 3. Validating the kbd skill itself

Run periodically to ensure the skill remains spec-compliant as it evolves:

```
/validate-skill
  skill_path: .agent/skills/kbd-process-orchestrator
```

---

## What KBD Reads Back

After skill creation/extension, KBD:
- Verifies the new sub-skill `SKILL.md` exists and passes `/validate-skill`
- Updates `SKILL.md` Quick Start if a new slash command was added
- Commits the new skill files: `git add .agent/skills/ && git commit -m "kbd: add <skill-name> sub-skill"`

---

## Self-Improvement Protocol

`pmpo-skill-creator` is the mechanism by which KBD improves itself. The Reflect
phase should explicitly ask:

> "Should any recurring pattern in this phase be codified as a new KBD sub-skill?"

If yes → invoke `pmpo-skill-creator` in extend mode → new sub-skill → validate →
commit → document in phase reflection.

---

## When NOT to Use

- For minor updates to existing skill files: edit directly, no need for creator
- For simple `SKILL.md` documentation fixes: edit directly
- `pmpo-skill-creator` is for **structural additions** (new sub-skills, domain adapters,
  new hooks) — not for content edits
