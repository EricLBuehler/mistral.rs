#!/usr/bin/env bash
# log-reflection.sh — Logs reflection output after reflector subagent completes
# Exit 0 = OK, Exit 2 = feedback to agent

set -euo pipefail

STATE_FILE="evolution_state.json"
LOG_FILE="evolution_log.md"
DECISIONS_FILE="decisions.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check if state file exists
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# Extract reflection data and log it
python3 -c "
import json

state = json.load(open('$STATE_FILE'))
reflection = state.get('latest_reflection', {})
convergence = reflection.get('convergence', {})
iteration = state.get('current_iteration', '?')
domain = state.get('domain', 'unknown')

# Summary line for log
decision = convergence.get('decision', 'unknown')
rationale = convergence.get('rationale', 'No rationale provided')
alignment = convergence.get('target_alignment', '?')

summary = f'Iteration {iteration} ({domain}): {decision} — alignment {alignment}% — {rationale}'
print(f'📊 {summary}')
" 2>/dev/null || true

exit 0
