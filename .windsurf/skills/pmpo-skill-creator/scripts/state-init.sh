#!/usr/bin/env bash
set -euo pipefail

# Initialize or resume skill creation state
# Usage: state-init.sh <skill_name> <mode> [source_skill]
# mode: create | clone | extend

SKILL_NAME="${1:?Usage: state-init.sh <skill_name> <mode> [source_skill]}"
MODE="${2:?Usage: state-init.sh <skill_name> <mode> [source_skill]}"
SOURCE_SKILL="${3:-}"

STATE_DIR=".creator/skills/${SKILL_NAME}"
STATE_FILE="${STATE_DIR}/state.json"
REGISTRY_FILE=".creator/registry.json"

# Ensure directories exist
mkdir -p "$STATE_DIR"
mkdir -p ".creator"

# Initialize registry if absent
if [[ ! -f "$REGISTRY_FILE" ]]; then
  echo '{"skills":{}}' > "$REGISTRY_FILE"
fi

# Check for existing state
if [[ -f "$STATE_FILE" ]]; then
  STATUS=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['convergence_status'])" 2>/dev/null || echo "unknown")
  
  case "$STATUS" in
    running)
      # Resume active session
      echo ""
      echo "Resuming active creation session for '${SKILL_NAME}'"
      python3 -c "
import json, sys
from datetime import datetime, timezone
state = json.load(open('${STATE_FILE}'))
state['updated_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(state, sys.stdout, indent=2)
"
      exit 0
      ;;
    converged|failed)
      # Seed new cycle from finalized state
      PRIOR_ID=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['creation_id'])" 2>/dev/null || echo "")
      echo ""
      echo "Starting new cycle for '${SKILL_NAME}' (prior: ${PRIOR_ID})"
      ;;
    *)
      echo ""
      echo "Unknown state status '${STATUS}', starting fresh"
      PRIOR_ID=""
      ;;
  esac
else
  PRIOR_ID=""
fi

# Generate new creation ID
CREATION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Determine source_skill JSON value
if [[ -n "$SOURCE_SKILL" ]]; then
  SOURCE_JSON="\"${SOURCE_SKILL}\""
else
  SOURCE_JSON="null"
fi

# Determine prior_creation_id JSON value
if [[ -n "${PRIOR_ID:-}" ]]; then
  PRIOR_JSON="\"${PRIOR_ID}\""
else
  PRIOR_JSON="null"
fi

# Create initial state
cat > "$STATE_FILE" << EOF
{
  "creation_id": "${CREATION_ID}",
  "skill_name": "${SKILL_NAME}",
  "mode": "${MODE}",
  "source_skill": ${SOURCE_JSON},
  "current_phase": "specify",
  "started_at": "${NOW}",
  "updated_at": "${NOW}",
  "current_iteration": 0,
  "max_iterations": 3,
  "complexity_tier": null,
  "target_platforms": [],
  "files_generated": [],
  "files_validated": [],
  "quality_score": null,
  "convergence_status": "running",
  "phases_completed": [],
  "checkpoints": [],
  "reflection_history": [],
  "prior_creation_id": ${PRIOR_JSON}
}
EOF

# Update registry
python3 -c "
import json
from datetime import datetime, timezone
registry = json.load(open('${REGISTRY_FILE}'))
registry['skills']['${SKILL_NAME}'] = {
    'creation_id': '${CREATION_ID}',
    'mode': '${MODE}',
    'status': 'active',
    'iterations': 0,
    'created_at': '${NOW}',
    'updated_at': '${NOW}',
    'output_path': 'dist/${SKILL_NAME}/'
}
with open('${REGISTRY_FILE}', 'w') as f:
    json.dump(registry, f, indent=2)
"

echo ""
cat "$STATE_FILE"
