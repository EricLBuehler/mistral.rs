#!/usr/bin/env bash
# validate-state.sh — Validates evolution_state.json after file writes
# Exit 0 = OK, Exit 2 = feedback to agent

set -euo pipefail

STATE_FILE="evolution_state.json"

# Only validate if state file exists
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# Check valid JSON
if ! python3 -c "import json; json.load(open('$STATE_FILE'))" 2>/dev/null; then
  echo "⚠️  evolution_state.json is not valid JSON"
  exit 2
fi

# Check required fields
REQUIRED_FIELDS='["evolution_id", "domain", "current_iteration", "convergence_status"]'
MISSING=$(python3 -c "
import json, sys
state = json.load(open('$STATE_FILE'))
required = $REQUIRED_FIELDS
missing = [f for f in required if f not in state]
if missing:
    print(f'Missing fields: {missing}')
    sys.exit(1)
" 2>&1) || {
  echo "⚠️  $MISSING"
  exit 2
}

exit 0
