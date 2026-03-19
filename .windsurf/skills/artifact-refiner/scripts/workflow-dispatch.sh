#!/usr/bin/env bash
# workflow-dispatch.sh — Dispatches workflow triggers for a lifecycle event
# Usage: workflow-dispatch.sh <artifact_name> <event_type> [phase]
# Reads triggers from state, matches by event, evaluates conditions, fires actions (non-blocking)

set -euo pipefail

ARTIFACT_NAME="${1:?Usage: workflow-dispatch.sh <artifact_name> <event_type> [phase]}"
EVENT_TYPE="${2:?Usage: workflow-dispatch.sh <artifact_name> <event_type> [phase]}"
PHASE="${3:-}"
STATE_DIR=".refiner"
STATE_FILE="$STATE_DIR/artifacts/$ARTIFACT_NAME/state.json"
LOG_FILE="$STATE_DIR/artifacts/$ARTIFACT_NAME/workflow.log"

if [ ! -f "$STATE_FILE" ]; then
  echo "WARN: No state file for '$ARTIFACT_NAME', skipping dispatch" >&2
  exit 0
fi

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build event payload
EVENT_PAYLOAD=$(python3 -c "
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
event = {
    'event_type': '$EVENT_TYPE',
    'artifact_name': '$ARTIFACT_NAME',
    'refinement_id': state.get('refinement_id', ''),
    'artifact_type': state.get('artifact_type', ''),
    'content_type': state.get('content_type', ''),
    'phase': '$PHASE' or None,
    'iteration': state.get('current_iteration', 0),
    'timestamp': '$NOW',
    'metadata': {}
}
print(json.dumps(event))
")

# Extract and match triggers
python3 << 'DISPATCH_SCRIPT'
import json
import subprocess
import os
import sys

state_file = os.environ.get("STATE_FILE", "") or "$STATE_FILE"
event_type = "$EVENT_TYPE"
phase = "$PHASE"
log_file = "$LOG_FILE"
event_payload = json.loads('$EVENT_PAYLOAD')
now = "$NOW"

try:
    with open(state_file, 'r') as f:
        state = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)

triggers = state.get('workflow_triggers', [])
if not triggers:
    sys.exit(0)

matched = 0
for trigger in triggers:
    if trigger.get('event') != event_type:
        continue

    # Evaluate condition if present
    condition = trigger.get('condition', '')
    if condition:
        try:
            ctx = {
                'phase': phase,
                'artifact_type': state.get('artifact_type', ''),
                'content_type': state.get('content_type', ''),
                'iteration': state.get('current_iteration', 0),
                'artifact_name': state.get('artifact_name', ''),
                'domain': state.get('artifact_type', ''),
            }
            if not eval(condition, {"__builtins__": {}}, ctx):
                continue
        except Exception as e:
            log_entry = f"[{now}] WARN: condition eval failed for trigger: {e}\n"
            with open(log_file, 'a') as lf:
                lf.write(log_entry)
            continue

    action = trigger.get('action', {})
    action_type = action.get('type', '')
    target = action.get('target', '')
    timeout = action.get('timeout_ms', 30000)

    # Variable substitution in target
    for key, val in event_payload.items():
        if isinstance(val, str):
            target = target.replace(f'${{{key}}}', val)
        elif val is not None:
            target = target.replace(f'${{{key}}}', str(val))

    log_entry = f"[{now}] DISPATCH: event={event_type} action={action_type} target={target}\n"
    with open(log_file, 'a') as lf:
        lf.write(log_entry)

    if action_type == 'command':
        try:
            subprocess.Popen(
                target,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            matched += 1
        except Exception as e:
            err_entry = f"[{now}] ERROR: command failed: {e}\n"
            with open(log_file, 'a') as lf:
                lf.write(err_entry)

    elif action_type == 'webhook':
        try:
            import urllib.request
            req = urllib.request.Request(
                target,
                data=json.dumps(event_payload).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=timeout // 1000)
            matched += 1
        except Exception as e:
            err_entry = f"[{now}] ERROR: webhook failed: {e}\n"
            with open(log_file, 'a') as lf:
                lf.write(err_entry)

    elif action_type in ('mcp_tool', 'workflow_file'):
        log_entry = f"[{now}] INFO: {action_type} '{target}' queued for agent execution\n"
        with open(log_file, 'a') as lf:
            lf.write(log_entry)
        matched += 1

print(f"Dispatched {matched} trigger(s) for event '{event_type}'")
DISPATCH_SCRIPT
