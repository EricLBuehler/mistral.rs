---
name: evolve-report
description: >
  Generate an evolution report from current state data. Produces a human-readable
  markdown report summarizing the full evolution cycle.
---

# Evolve Report

Generate a comprehensive evolution report from existing state data.

## Setup

1. Load all state files: `evolution_state.json`, `assessment.json`, `analysis.json`, `plan.json`
2. Load `evolution_log.md` and `decisions.md` for history

## Output

Generate a markdown report to `reports/evolution-report-<timestamp>.md` including:
- Executive summary
- Goals and current alignment
- Assessment findings
- Landscape analysis highlights
- Improvement plan overview
- Execution results (if available)
- Iteration history
- Lessons learned
- Recommended next steps

If minimal state exists, generate a partial report with available data and note gaps.
