---
name: evolve-assess
description: >
  Run only the Assess phase — evaluate current state against goals.
  Produces a structured assessment without creating plans or making changes.
---

# Evolve Assess

Run only the PMPO Assess phase. Useful for getting a snapshot of where you stand.

## Setup

1. Determine `evolution_domain` from context or user input
2. Load domain adapter from `references/domain/<domain>.md`
3. Run only `prompts/assess.md`

## User Input

The user will provide: $ARGUMENTS

Parse for:
- Subject to assess
- Goals to assess against
- Domain (if stated)

## Output

- `assessment.json` — Structured assessment
- Assessment summary printed to user
