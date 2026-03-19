---
name: evolve
description: >
  Run a full iterative evolution cycle — Assess, Analyze, Plan, Execute, and
  Reflect. Auto-detects domain or prompts for it. Use for any subject that
  needs iterative improvement against goals.
---

# Evolve

Run the full PMPO iterative evolution cycle.

## Setup

1. Determine `evolution_domain` — auto-detect from context or ask user
2. Load domain adapter from `references/domain/<domain>.md`
3. Start the PMPO loop via `prompts/meta-controller.md`

## User Input

The user will provide: $ARGUMENTS

Parse the arguments for:
- What they want to evolve (the subject)
- What their goals are
- What domain this falls into (if stated)
- Any constraints or boundaries
- Whether they have prior assessments to build on

If goals are not explicitly stated, ask for them before proceeding.

## Default Behavior

- `max_iterations: 5`
- `approval_required: true` (pause after Reflect for human review)
- Domain auto-detection enabled
- State persisted to working directory
