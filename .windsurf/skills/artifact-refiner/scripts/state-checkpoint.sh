#!/usr/bin/env bash
# state-checkpoint.sh — Filesystem provider: snapshot mid-session state
# Usage: state-checkpoint.sh <artifact_name> [phase] [event_type]
# Output: checkpoint_id to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

ARTIFACT_NAME="${1:?Usage: state-checkpoint.sh <artifact_name> [phase] [event_type]}"
PHASE="${2:-unknown}"
EVENT_TYPE="${3:-checkpoint_created}"
STATE_DIR=".refiner"
ARTIFACT_DIR="$STATE_DIR/artifacts/$ARTIFACT_NAME"
STATE_FILE="$ARTIFACT_DIR/state.json"

if [ ! -f "$STATE_FILE" ]; then
  echo "ERROR: No state file for artifact '$ARTIFACT_NAME'" >&2
  exit 1
fi

CHECKPOINT_ID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Save checkpoint snapshot
CHECKPOINT_DIR="$ARTIFACT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
cp "$STATE_FILE" "$CHECKPOINT_DIR/${CHECKPOINT_ID}.json"

# Update state with checkpoint reference
python3 -c "
import json
from datetime import datetime, timezone
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state.setdefault('checkpoints', []).append({
    'checkpoint_id': '$CHECKPOINT_ID',
    'phase': '$PHASE',
    'iteration': state.get('current_iteration', 0),
    'timestamp': '$NOW'
})
state['updated_at'] = '$NOW'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"

echo "$CHECKPOINT_ID"
