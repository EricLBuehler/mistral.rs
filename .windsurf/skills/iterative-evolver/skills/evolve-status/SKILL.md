---
name: evolve-status
description: >
  Check the current status of an evolution cycle. Shows iteration count,
  goal alignment, phase completion, and convergence status.
---

# Evolve Status

Check the current evolution cycle status.

## Setup

1. Load `evolution_state.json` if it exists
2. Display current status

## Output

Display a summary including:
- Evolution ID and domain
- Current iteration / max iterations
- Goal alignment percentage
- Phases completed in current iteration
- Convergence status
- Last updated timestamp

If no evolution state exists, report "No active evolution cycle. Use `/evolve` to start one."
