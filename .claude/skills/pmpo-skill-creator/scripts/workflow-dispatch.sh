#!/usr/bin/env bash
set -euo pipefail

# Dispatch lifecycle events for workflow triggers
# Usage: workflow-dispatch.sh <event> [payload...]
# Events: on_phase_complete, on_creation_complete, on_regression, on_approval_required

EVENT="${1:?Usage: workflow-dispatch.sh <event> [payload...]}"
shift
PAYLOAD="${*:-}"

TRIGGER_FILE=".creator/workflow-triggers.json"

# If no trigger file configured, silently exit (non-blocking)
if [[ ! -f "$TRIGGER_FILE" ]]; then
  exit 0
fi

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Read triggers and match against current event
python3 << 'PYEOF'
import json, subprocess, os, sys

event = os.environ.get("EVENT", sys.argv[1] if len(sys.argv) > 1 else "")
payload = os.environ.get("PAYLOAD", "")
trigger_file = os.environ.get("TRIGGER_FILE", ".creator/workflow-triggers.json")

try:
    with open(trigger_file) as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)

triggers = config.get("triggers", [])
matched = 0

for trigger in triggers:
    trigger_event = trigger.get("event", "")
    if trigger_event != event:
        continue
    
    # Check condition if present
    condition = trigger.get("condition", "")
    if condition:
        # Simple variable substitution
        condition = condition.replace("${event}", event)
        condition = condition.replace("${payload}", payload)
    
    action = trigger.get("action", {})
    action_type = action.get("type", "")
    
    if action_type == "command":
        cmd = action.get("command", "")
        cmd = cmd.replace("${event}", event)
        cmd = cmd.replace("${payload}", payload)
        # Fire and forget (non-blocking)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        matched += 1
    
    elif action_type == "webhook":
        url = action.get("url", "")
        data = json.dumps({"event": event, "payload": payload, "timestamp": os.environ.get("NOW", "")})
        subprocess.Popen(
            ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json", "-d", data, url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        matched += 1

if matched > 0:
    print(f"Dispatched {matched} trigger(s) for event '{event}'")
PYEOF
