#!/usr/bin/env bash
# state-init.sh — Filesystem provider: initialize or resume evolution state
# Usage: state-init.sh <evolution_name> [domain] [goals_json]
# Output: JSON state to stdout
# Exit 0 = OK, Exit 1 = error

set -euo pipefail

EVOLUTION_NAME="${1:?Usage: state-init.sh <evolution_name> [domain] [goals_json]}"
DOMAIN="${2:-generic}"
GOALS_JSON="${3:-[]}"
STATE_DIR="${EVOLVER_STATE_DIR:-.evolver}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Ensure state directory structure
EVOLUTION_DIR="${STATE_DIR}/evolutions/${EVOLUTION_NAME}"
mkdir -p "${EVOLUTION_DIR}/checkpoints"
mkdir -p "${EVOLUTION_DIR}/history"

REGISTRY_FILE="${STATE_DIR}/registry.json"
STATE_FILE="${EVOLUTION_DIR}/state.json"

# Initialize registry if missing
if [ ! -f "$REGISTRY_FILE" ]; then
  echo '{"evolutions": {}}' > "$REGISTRY_FILE"
fi

# Check for existing state
if [ -f "$STATE_FILE" ]; then
  # Resume: load existing state
  EXISTING_STATUS=$(python3 -c "
import json
state = json.load(open('$STATE_FILE'))
print(state.get('convergence_status', 'running'))
")

  if [ "$EXISTING_STATUS" = "converged" ] || [ "$EXISTING_STATUS" = "max_iterations" ] || [ "$EXISTING_STATUS" = "terminated" ]; then
    # Prior cycle finalized — archive and create new cycle from end state
    ITERATION=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('current_iteration', 0))")
    cp "$STATE_FILE" "${EVOLUTION_DIR}/history/iteration-${ITERATION}.json"
    
    # Create new state seeded from the finalized state
    python3 -c "
import json, uuid
state = json.load(open('$STATE_FILE'))
state['evolution_id'] = str(uuid.uuid4())
state['current_iteration'] = 0
state['convergence_status'] = 'running'
state['phases_completed'] = []
state['updated_at'] = '$TIMESTAMP'
state['prior_cycle_id'] = state.get('evolution_id', '')
state.pop('latest_execution', None)
state.pop('latest_reflection', None)
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
print(json.dumps(state, indent=2))
"
  else
    # Active state — resume
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['updated_at'] = '$TIMESTAMP'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
print(json.dumps(state, indent=2))
"
  fi
else
  # New evolution — create fresh state
  python3 -c "
import json, uuid
state = {
    'evolution_id': str(uuid.uuid4()),
    'evolution_name': '$EVOLUTION_NAME',
    'domain': '$DOMAIN',
    'started_at': '$TIMESTAMP',
    'updated_at': '$TIMESTAMP',
    'current_iteration': 0,
    'max_iterations': 5,
    'goals': json.loads('$GOALS_JSON') if '$GOALS_JSON' != '[]' else [],
    'phases_completed': [],
    'convergence_status': 'running',
    'iteration_history': [],
    'workflow_triggers': [],
    'checkpoints': []
}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
print(json.dumps(state, indent=2))
"
fi

# Update registry
python3 -c "
import json
registry = json.load(open('$REGISTRY_FILE'))
registry['evolutions']['$EVOLUTION_NAME'] = {
    'path': '$EVOLUTION_DIR',
    'domain': '$DOMAIN',
    'updated_at': '$TIMESTAMP'
}
json.dump(registry, open('$REGISTRY_FILE', 'w'), indent=2)
" 2>/dev/null

exit 0
