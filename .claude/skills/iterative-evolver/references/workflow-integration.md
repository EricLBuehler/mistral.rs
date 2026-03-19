# Workflow Integration

The Iterative Evolver supports **workflow triggers** — external actions that fire automatically at lifecycle events. This enables infinite flexibility without modifying the skill itself.

---

## Trigger Events

Triggers fire at specific points in the evolution lifecycle:

| Event | When | Typical Use |
|-------|------|-------------|
| `on_phase_start` | Before a PMPO phase begins | Pre-flight checks, resource allocation |
| `on_phase_complete` | After a PMPO phase finishes | Checkpoint state, notify stakeholders |
| `on_iteration_complete` | After Reflect + Persist in one iteration | Progress reports, CI/CD triggers |
| `on_cycle_complete` | When evolution terminates (converged or max) | Final reports, deployments, notifications |
| `on_regression` | When Reflect detects a regression | Alerts, rollback triggers |
| `on_approval_required` | When human approval gate is reached | Notifications to approvers |

---

## Trigger Definition

Triggers are defined in `workflow_triggers` within the evolution state or in a standalone `triggers.json` in the evolution's state directory.

```json
{
  "triggers": [
    {
      "id": "notify-slack-on-complete",
      "event": "on_cycle_complete",
      "condition": null,
      "action": {
        "type": "webhook",
        "target": "https://hooks.slack.com/services/...",
        "inputs": {
          "text": "Evolution '${evolution_name}' completed. Alignment: ${alignment}%"
        },
        "timeout": 10
      }
    },
    {
      "id": "run-tests-after-execute",
      "event": "on_phase_complete",
      "condition": "phase == 'execute' && domain == 'software'",
      "action": {
        "type": "command",
        "target": "cargo test",
        "inputs": {},
        "timeout": 300
      }
    }
  ]
}
```

See `references/schemas/workflow-trigger.schema.json` for the full schema.

---

## Action Types

### `command`
Executes a shell command. The command runs in the evolution's working directory.

```json
{
  "type": "command",
  "target": "npm run test",
  "inputs": {
    "env": { "CI": "true" }
  },
  "timeout": 120
}
```

### `webhook`
Sends an HTTP POST to a URL. The hook event payload is sent as the body.

```json
{
  "type": "webhook",
  "target": "https://example.com/hooks/evolution",
  "inputs": {
    "headers": { "Authorization": "Bearer ${WEBHOOK_TOKEN}" }
  },
  "timeout": 30
}
```

### `mcp_tool`
Calls an MCP tool by name. Inputs are passed as tool parameters.

```json
{
  "type": "mcp_tool",
  "target": "create_entity",
  "inputs": {
    "name": "${evolution_name}_result",
    "entity_type": "EvolutionResult",
    "observations": ["Completed iteration ${iteration}"]
  },
  "timeout": 30
}
```

### `workflow_file`
Executes a workflow definition file (YAML or Markdown with structured steps).

```json
{
  "type": "workflow_file",
  "target": "./workflows/post-evolution.yml",
  "inputs": {
    "evolution_name": "${evolution_name}",
    "domain": "${domain}"
  },
  "timeout": 600
}
```

---

## Variable Substitution

Trigger inputs support variable substitution using `${variable}` syntax:

| Variable | Value |
|----------|-------|
| `${evolution_name}` | The current evolution name |
| `${evolution_id}` | The internal UUID |
| `${domain}` | The evolution domain |
| `${iteration}` | Current iteration number |
| `${phase}` | Current/just-completed phase |
| `${alignment}` | Current goal alignment percentage |
| `${convergence_status}` | Current convergence status |
| `${timestamp}` | ISO 8601 timestamp |

---

## Conditional Triggers

The `condition` field accepts simple expressions:

```
phase == 'execute'
domain == 'software'
iteration > 1
alignment >= 90
phase == 'reflect' && domain == 'software'
```

Supported operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `&&`, `||`

If `condition` is null or empty, the trigger fires unconditionally for its event.

---

## Trigger Registration

Triggers can be registered in three places (checked in order):

1. **Evolution state** — `workflow_triggers` array in `evolution_state.json`
2. **State directory** — `triggers.json` in the evolution's state directory
3. **Provider config** — `default_triggers` in the provider configuration

This allows per-evolution, per-project, and global triggers.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Trigger times out | Log warning, continue evolution (triggers are non-blocking) |
| Trigger command fails | Log error with exit code, continue evolution |
| Webhook returns non-2xx | Log error with status code, continue evolution |
| MCP tool fails | Log error, continue evolution |
| Condition parse error | Log warning, skip trigger, continue evolution |

Triggers NEVER block or halt the evolution cycle. They are fire-and-forget with logging.

---

## Dispatch Protocol

The dispatch script (`scripts/workflow-dispatch.sh`) handles trigger execution:

1. Read hook event payload from stdin
2. Load triggers from state → state dir → provider config
3. Match triggers by event type
4. Evaluate conditions
5. Substitute variables
6. Execute matching actions
7. Log results to `evolution_log.md`
