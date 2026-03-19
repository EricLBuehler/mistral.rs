---
name: evolve-analyze
description: >
  Run only the Analyze phase — research the external landscape for competitors,
  trends, opportunities, and threats. Requires an existing assessment or goals.
---

# Evolve Analyze

Run only the PMPO Analyze phase. Useful for landscape research and competitive intelligence.

## Setup

1. Determine `evolution_domain` from context or user input
2. Load domain adapter from `references/domain/<domain>.md`
3. Load `assessment.json` if it exists (for context)
4. Run only `prompts/analyze.md`

## User Input

The user will provide: $ARGUMENTS

Parse for:
- Subject/domain to analyze
- Specific competitors or benchmarks to investigate
- Goals (if no prior assessment exists)

## Output

- `analysis.json` — Structured landscape analysis
- Analysis summary printed to user
