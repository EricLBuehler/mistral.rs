#!/usr/bin/env bash
# state-finalize.sh — Filesystem provider: mark end state and archive
# Usage: state-finalize.sh <evolution_name>
# Output: archived path to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

EVOLUTION_NAME="${1:?Usage: state-finalize.sh <evolution_name>}"
STATE_DIR="${EVOLVER_STATE_DIR:-.evolver}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

EVOLUTION_DIR="${STATE_DIR}/evolutions/${EVOLUTION_NAME}"
STATE_FILE="${EVOLUTION_DIR}/state.json"
HISTORY_DIR="${EVOLUTION_DIR}/history"

# Verify state exists
if [ ! -f "$STATE_FILE" ]; then
  echo "❌ No state found for evolution: ${EVOLUTION_NAME}" >&2
  exit 1
fi

mkdir -p "$HISTORY_DIR"

# Finalize: update convergence status and archive
python3 -c "
import json

state = json.load(open('$STATE_FILE'))
iteration = state.get('current_iteration', 0)

# Archive current state to history
archive_path = '${HISTORY_DIR}/iteration-${iteration}.json' if iteration > 0 else '${HISTORY_DIR}/iteration-0.json'
archive_path = f'${HISTORY_DIR}/iteration-{iteration}.json'

archive = {
    'finalized_at': '$TIMESTAMP',
    'iteration': iteration,
    'convergence_status': state.get('convergence_status', 'terminated'),
    'state': state
}
json.dump(archive, open(archive_path, 'w'), indent=2)

# Update state to mark as finalized
if state.get('convergence_status') == 'running':
    state['convergence_status'] = 'terminated'
state['updated_at'] = '$TIMESTAMP'
state['finalized_at'] = '$TIMESTAMP'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)

print(archive_path)
"

# Update registry
REGISTRY_FILE="${STATE_DIR}/registry.json"
if [ -f "$REGISTRY_FILE" ]; then
  python3 -c "
import json
registry = json.load(open('$REGISTRY_FILE'))
if '$EVOLUTION_NAME' in registry.get('evolutions', {}):
    registry['evolutions']['$EVOLUTION_NAME']['status'] = 'finalized'
    registry['evolutions']['$EVOLUTION_NAME']['updated_at'] = '$TIMESTAMP'
    json.dump(registry, open('$REGISTRY_FILE', 'w'), indent=2)
" 2>/dev/null
fi

exit 0
