#!/usr/bin/env bash
# finalize-session.sh — Runs when the evolution session ends
# Ensures state is consistent and logs session end

set -euo pipefail

STATE_FILE="evolution_state.json"
LOG_FILE="evolution_log.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Log session end
if [ -f "$LOG_FILE" ]; then
  echo "" >> "$LOG_FILE"
  echo "---" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
  echo "## Session Ended — $TIMESTAMP" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
fi

# Update state timestamp if exists
if [ -f "$STATE_FILE" ]; then
  python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['updated_at'] = '$TIMESTAMP'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
" 2>/dev/null || true
fi

exit 0
