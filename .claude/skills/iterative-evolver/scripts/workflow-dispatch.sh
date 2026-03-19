#!/usr/bin/env bash
# workflow-dispatch.sh — Dispatches workflow triggers for a lifecycle event
# Usage: workflow-dispatch.sh <evolution_name> <event_type> [phase]
# Reads triggers from state, matches by event, evaluates conditions, dispatches actions.
# Triggers are fire-and-forget — failures are logged but never halt evolution.
# Exit 0 = always (triggers are non-blocking)

set -euo pipefail

EVOLUTION_NAME="${1:?Usage: workflow-dispatch.sh <evolution_name> <event_type> [phase]}"
EVENT_TYPE="${2:?Usage: workflow-dispatch.sh <evolution_name> <event_type> [phase]}"
PHASE="${3:-}"
STATE_DIR="${EVOLVER_STATE_DIR:-.evolver}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

EVOLUTION_DIR="${STATE_DIR}/evolutions/${EVOLUTION_NAME}"
STATE_FILE="${EVOLUTION_DIR}/state.json"
TRIGGERS_FILE="${EVOLUTION_DIR}/triggers.json"
LOG_FILE="evolution_log.md"

# Verify state exists
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# Collect triggers from state file and standalone triggers file
python3 << 'DISPATCH_SCRIPT'
import json, subprocess, sys, os

evolution_name = os.environ.get("EVOLUTION_NAME", sys.argv[1] if len(sys.argv) > 1 else "")
event_type = os.environ.get("EVENT_TYPE", sys.argv[2] if len(sys.argv) > 2 else "")
phase = os.environ.get("PHASE", sys.argv[3] if len(sys.argv) > 3 else "")
state_file = os.environ.get("STATE_FILE", "")
triggers_file = os.environ.get("TRIGGERS_FILE", "")
log_file = os.environ.get("LOG_FILE", "evolution_log.md")
timestamp = os.environ.get("TIMESTAMP", "")

# Load state
try:
    state = json.load(open(state_file))
except:
    sys.exit(0)

# Collect triggers from all sources
triggers = []

# Source 1: evolution state
triggers.extend(state.get("workflow_triggers", []))

# Source 2: standalone triggers file
try:
    trigger_doc = json.load(open(triggers_file))
    triggers.extend(trigger_doc.get("triggers", []))
except:
    pass

# Source 3: provider default triggers
provider_config = os.environ.get("EVOLVER_PROVIDER_CONFIG", "")
if provider_config and os.path.isfile(provider_config):
    try:
        provider = json.load(open(provider_config))
        triggers.extend(provider.get("default_triggers", []))
    except:
        pass

if not triggers:
    sys.exit(0)

# Build context for condition evaluation and variable substitution
context = {
    "evolution_name": evolution_name,
    "evolution_id": state.get("evolution_id", ""),
    "domain": state.get("domain", "generic"),
    "iteration": state.get("current_iteration", 0),
    "phase": phase,
    "alignment": state.get("latest_reflection", {}).get("convergence", {}).get("target_alignment", 0),
    "convergence_status": state.get("convergence_status", "running"),
    "timestamp": timestamp,
}

def evaluate_condition(condition, ctx):
    """Evaluate a simple condition expression against context."""
    if not condition:
        return True
    try:
        # Replace context variables
        expr = condition
        for key, val in ctx.items():
            if isinstance(val, str):
                expr = expr.replace(key, repr(val))
            else:
                expr = expr.replace(key, str(val))
        return eval(expr, {"__builtins__": {}}, {})
    except:
        return False

def substitute_vars(text, ctx):
    """Replace ${variable} with context values."""
    if not isinstance(text, str):
        return text
    result = text
    for key, val in ctx.items():
        result = result.replace(f"${{{key}}}", str(val))
    return result

def substitute_dict(obj, ctx):
    """Recursively substitute variables in a dict."""
    if isinstance(obj, str):
        return substitute_vars(obj, ctx)
    elif isinstance(obj, dict):
        return {k: substitute_dict(v, ctx) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_dict(v, ctx) for v in obj]
    return obj

dispatched = 0
for trigger in triggers:
    # Match event type
    if trigger.get("event") != f"on_{event_type}" and trigger.get("event") != event_type:
        continue
    
    # Evaluate condition
    if not evaluate_condition(trigger.get("condition"), context):
        continue
    
    action = trigger.get("action", {})
    action_type = action.get("type", "")
    target = substitute_vars(action.get("target", ""), context)
    inputs = substitute_dict(action.get("inputs", {}), context)
    timeout = action.get("timeout", 30)
    trigger_id = trigger.get("id", "unnamed")
    
    try:
        if action_type == "command":
            env = {**os.environ, **inputs.get("env", {})}
            result = subprocess.run(
                target, shell=True, capture_output=True, text=True,
                timeout=timeout, env=env
            )
            status = "✅" if result.returncode == 0 else f"❌ exit={result.returncode}"
            print(f"  {status} trigger '{trigger_id}': {target}")
        
        elif action_type == "webhook":
            import urllib.request
            data = json.dumps({**context, **inputs}).encode()
            headers = inputs.get("headers", {})
            headers["Content-Type"] = "application/json"
            req = urllib.request.Request(target, data=data, headers=headers, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                print(f"  ✅ trigger '{trigger_id}': webhook {resp.status}")
            except Exception as e:
                print(f"  ❌ trigger '{trigger_id}': webhook failed: {e}")
        
        elif action_type == "mcp_tool":
            # MCP tools are dispatched by the agent, not by shell scripts
            # Log the intent so the agent can pick it up
            print(f"  📋 trigger '{trigger_id}': MCP tool '{target}' queued (agent dispatch)")
        
        elif action_type == "workflow_file":
            if os.path.isfile(target):
                print(f"  📋 trigger '{trigger_id}': workflow '{target}' queued (agent dispatch)")
            else:
                print(f"  ❌ trigger '{trigger_id}': workflow file not found: {target}")
        
        dispatched += 1
    except subprocess.TimeoutExpired:
        print(f"  ⏱️ trigger '{trigger_id}': timed out after {timeout}s")
    except Exception as e:
        print(f"  ❌ trigger '{trigger_id}': {e}")

if dispatched > 0:
    print(f"\n📨 Dispatched {dispatched} trigger(s) for event '{event_type}'")

DISPATCH_SCRIPT

exit 0
