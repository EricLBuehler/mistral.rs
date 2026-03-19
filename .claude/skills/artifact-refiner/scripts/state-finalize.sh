#!/usr/bin/env bash
# state-finalize.sh — Filesystem provider: mark end state and archive
# Usage: state-finalize.sh <artifact_name>
# Output: archived path to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

ARTIFACT_NAME="${1:?Usage: state-finalize.sh <artifact_name>}"
STATE_DIR=".refiner"
ARTIFACT_DIR="$STATE_DIR/artifacts/$ARTIFACT_NAME"
STATE_FILE="$ARTIFACT_DIR/state.json"
HISTORY_DIR="$STATE_DIR/history"

if [ ! -f "$STATE_FILE" ]; then
  echo "ERROR: No state file for artifact '$ARTIFACT_NAME'" >&2
  exit 1
fi

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TIMESTAMP_SLUG=$(echo "$NOW" | sed 's/[:]/-/g' | sed 's/T/_/')

# Update convergence status and finalized timestamp
python3 -c "
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
if state.get('convergence_status') == 'running':
    state['convergence_status'] = 'converged'
state['finalized_at'] = '$NOW'
state['updated_at'] = '$NOW'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"

# Archive to history
ARCHIVE_DIR="$HISTORY_DIR/$ARTIFACT_NAME/$TIMESTAMP_SLUG"
mkdir -p "$ARCHIVE_DIR"
cp -r "$ARTIFACT_DIR/"* "$ARCHIVE_DIR/" 2>/dev/null || true

# Update registry
REGISTRY_FILE="$STATE_DIR/registry.json"
if [ -f "$REGISTRY_FILE" ]; then
  python3 -c "
import json
with open('$REGISTRY_FILE', 'r') as f:
    registry = json.load(f)
if '$ARTIFACT_NAME' in registry.get('artifacts', {}):
    registry['artifacts']['$ARTIFACT_NAME']['finalized_at'] = '$NOW'
    registry['artifacts']['$ARTIFACT_NAME']['updated_at'] = '$NOW'
with open('$REGISTRY_FILE', 'w') as f:
    json.dump(registry, f, indent=2)
"
fi

echo "$ARCHIVE_DIR"
