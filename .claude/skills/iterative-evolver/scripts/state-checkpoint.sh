#!/usr/bin/env bash
# state-checkpoint.sh — Filesystem provider: snapshot mid-session state
# Usage: state-checkpoint.sh <evolution_name> [phase] [event_type]
# Output: checkpoint_id to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

EVOLUTION_NAME="${1:?Usage: state-checkpoint.sh <evolution_name> [phase] [event_type]}"
PHASE="${2:-unknown}"
EVENT_TYPE="${3:-checkpoint}"
STATE_DIR="${EVOLVER_STATE_DIR:-.evolver}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CHECKPOINT_ID="${PHASE}-$(date -u +"%Y%m%dT%H%M%S")"

EVOLUTION_DIR="${STATE_DIR}/evolutions/${EVOLUTION_NAME}"
STATE_FILE="${EVOLUTION_DIR}/state.json"
CHECKPOINT_DIR="${EVOLUTION_DIR}/checkpoints"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/${CHECKPOINT_ID}.json"

# Verify state exists
if [ ! -f "$STATE_FILE" ]; then
  echo "❌ No state found for evolution: ${EVOLUTION_NAME}" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

# Create checkpoint: copy current state + add checkpoint metadata
python3 -c "
import json

state = json.load(open('$STATE_FILE'))

checkpoint = {
    'checkpoint_id': '$CHECKPOINT_ID',
    'event_type': '$EVENT_TYPE',
    'phase': '$PHASE',
    'timestamp': '$TIMESTAMP',
    'state_snapshot': state
}

json.dump(checkpoint, open('$CHECKPOINT_FILE', 'w'), indent=2)

# Update state with checkpoint reference
checkpoints = state.get('checkpoints', [])
checkpoints.append({
    'checkpoint_id': '$CHECKPOINT_ID',
    'phase': '$PHASE',
    'event_type': '$EVENT_TYPE',
    'timestamp': '$TIMESTAMP'
})
state['checkpoints'] = checkpoints
state['updated_at'] = '$TIMESTAMP'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)

print('$CHECKPOINT_ID')
"

exit 0
