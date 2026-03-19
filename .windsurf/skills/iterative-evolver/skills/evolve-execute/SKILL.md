---
name: evolve-execute
description: >
  Run only the Execute phase — carry out a previously created improvement plan.
  Requires an existing plan.json.
---

# Evolve Execute

Run only the PMPO Execute phase. Carries out the actions in an existing improvement plan.

## Setup

1. Load `plan.json` (required)
2. Load `assessment.json` and `analysis.json` for context
3. Load domain adapter from `references/domain/<domain>.md`
4. Run only `prompts/execute.md`

## Prerequisites

A plan must exist. If not, suggest running `/evolve-plan` first.

## User Input

The user will provide: $ARGUMENTS

Parse for:
- Specific phases or actions to execute (optional — defaults to all)
- Whether to skip approval gates

## Output

- Execution results updated in `evolution_state.json`
- Execution log appended to `evolution_log.md`
- Created/modified files as specified in the plan
