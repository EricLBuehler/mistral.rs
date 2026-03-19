#!/usr/bin/env bash
set -euo pipefail

# Create a mid-session checkpoint snapshot
# Usage: state-checkpoint.sh <skill_name> <phase_name>

SKILL_NAME="${1:?Usage: state-checkpoint.sh <skill_name> <phase_name>}"
PHASE_NAME="${2:?Usage: state-checkpoint.sh <skill_name> <phase_name>}"

STATE_DIR=".creator/skills/${SKILL_NAME}"
STATE_FILE="${STATE_DIR}/state.json"
CHECKPOINT_DIR="${STATE_DIR}/checkpoints"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "ERROR: No state file found for '${SKILL_NAME}'" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CHECKPOINT_FILE="${CHECKPOINT_DIR}/${PHASE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).json"

# Copy current state as checkpoint
cp "$STATE_FILE" "$CHECKPOINT_FILE"

# Update main state with checkpoint record
python3 -c "
import json, sys
from datetime import datetime, timezone
state = json.load(open('${STATE_FILE}'))
state['updated_at'] = '${NOW}'
if '${PHASE_NAME}' not in state.get('phases_completed', []):
    state.setdefault('phases_completed', []).append('${PHASE_NAME}')
state.setdefault('checkpoints', []).append({
    'phase': '${PHASE_NAME}',
    'iteration': state.get('current_iteration', 0),
    'timestamp': '${NOW}',
    'quality_score': state.get('quality_score')
})
with open('${STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
json.dump(state, sys.stdout, indent=2)
"
