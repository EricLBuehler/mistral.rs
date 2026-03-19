#!/usr/bin/env bash
set -euo pipefail

# Finalize a completed skill creation session
# Usage: state-finalize.sh <skill_name>

SKILL_NAME="${1:?Usage: state-finalize.sh <skill_name>}"

STATE_DIR=".creator/skills/${SKILL_NAME}"
STATE_FILE="${STATE_DIR}/state.json"
REGISTRY_FILE=".creator/registry.json"
HISTORY_DIR="${STATE_DIR}/history"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "ERROR: No state file found for '${SKILL_NAME}'" >&2
  exit 1
fi

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p "$HISTORY_DIR"

# Archive current state to history
CREATION_ID=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['creation_id'])")
cp "$STATE_FILE" "${HISTORY_DIR}/${CREATION_ID}.json"

# Update state to finalized
python3 -c "
import json
from datetime import datetime, timezone
state = json.load(open('${STATE_FILE}'))
state['convergence_status'] = 'converged'
state['current_phase'] = 'finalized'
state['updated_at'] = '${NOW}'
with open('${STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
"

# Update registry
python3 -c "
import json
registry = json.load(open('${REGISTRY_FILE}'))
if '${SKILL_NAME}' in registry.get('skills', {}):
    registry['skills']['${SKILL_NAME}']['status'] = 'converged'
    registry['skills']['${SKILL_NAME}']['updated_at'] = '${NOW}'
    with open('${REGISTRY_FILE}', 'w') as f:
        json.dump(registry, f, indent=2)
"

echo "Finalized creation session for '${SKILL_NAME}' (ID: ${CREATION_ID})"
echo "History archived to: ${HISTORY_DIR}/${CREATION_ID}.json"
