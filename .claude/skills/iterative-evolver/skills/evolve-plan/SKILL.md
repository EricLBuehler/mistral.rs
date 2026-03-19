---
name: evolve-plan
description: >
  Run only the Plan phase — create a prioritized improvement plan. Requires
  an existing assessment and optionally an analysis.
---

# Evolve Plan

Run only the PMPO Plan phase. Creates an actionable improvement plan from existing assessment and analysis data.

## Setup

1. Load `assessment.json` (required)
2. Load `analysis.json` (optional, enhances planning)
3. Load domain adapter from `references/domain/<domain>.md`
4. Run only `prompts/plan.md`

## Prerequisites

An assessment must exist. If not, suggest running `/evolve-assess` first.

## User Input

The user will provide: $ARGUMENTS

Parse for:
- Specific focus areas or constraints
- Effort/time budget
- Priority preferences

## Output

- `plan.json` — Structured improvement plan
- Plan summary printed to user
