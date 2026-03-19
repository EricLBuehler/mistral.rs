#!/usr/bin/env bash
# state-init.sh — Filesystem provider: initialize or resume artifact refinement state
# Usage: state-init.sh <artifact_name> [artifact_type] [content_type]
# Output: JSON state to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

ARTIFACT_NAME="${1:?Usage: state-init.sh <artifact_name> [artifact_type] [content_type]}"
ARTIFACT_TYPE="${2:-content}"
CONTENT_TYPE="${3:-direct:content}"
STATE_DIR=".refiner"
ARTIFACT_DIR="$STATE_DIR/artifacts/$ARTIFACT_NAME"
STATE_FILE="$ARTIFACT_DIR/state.json"
REGISTRY_FILE="$STATE_DIR/registry.json"

# Ensure directories
mkdir -p "$ARTIFACT_DIR"
mkdir -p "$STATE_DIR"

# Resume existing active state
if [ -f "$STATE_FILE" ]; then
  EXISTING_STATUS=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('convergence_status', 'unknown'))" 2>/dev/null || echo "unknown")
  if [ "$EXISTING_STATUS" = "running" ]; then
    # Resume — update timestamp
    python3 -c "
import json, sys
from datetime import datetime, timezone
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state['updated_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print(json.dumps(state, indent=2))
"
    exit 0
  fi

  # Finalized prior cycle — seed new cycle
  PRIOR_ID=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('refinement_id', ''))" 2>/dev/null || echo "")
fi

# Generate new state
REFINEMENT_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

python3 -c "
import json
state = {
    'refinement_id': '$REFINEMENT_ID',
    'artifact_name': '$ARTIFACT_NAME',
    'artifact_type': '$ARTIFACT_TYPE',
    'content_type': '$CONTENT_TYPE',
    'started_at': '$NOW',
    'updated_at': '$NOW',
    'current_iteration': 0,
    'max_iterations': 5,
    'goals': [],
    'constraints': [],
    'phases_completed': [],
    'convergence_status': 'running',
    'iteration_history': [],
    'workflow_triggers': [],
    'checkpoints': [],
    'prior_cycle_id': '${PRIOR_ID:-}' or None
}
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print(json.dumps(state, indent=2))
"

# Update registry
if [ -f "$REGISTRY_FILE" ]; then
  python3 -c "
import json
with open('$REGISTRY_FILE', 'r') as f:
    registry = json.load(f)
registry.setdefault('artifacts', {})
registry['artifacts']['$ARTIFACT_NAME'] = {
    'path': '$ARTIFACT_DIR',
    'artifact_type': '$ARTIFACT_TYPE',
    'content_type': '$CONTENT_TYPE',
    'updated_at': '$NOW'
}
with open('$REGISTRY_FILE', 'w') as f:
    json.dump(registry, f, indent=2)
"
else
  python3 -c "
import json
registry = {
    'artifacts': {
        '$ARTIFACT_NAME': {
            'path': '$ARTIFACT_DIR',
            'artifact_type': '$ARTIFACT_TYPE',
            'content_type': '$CONTENT_TYPE',
            'updated_at': '$NOW'
        }
    }
}
with open('$REGISTRY_FILE', 'w') as f:
    json.dump(registry, f, indent=2)
"
fi
