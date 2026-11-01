---
title: Agent approvals
description: "Request and decision types for agent action approval callbacks."
sidebar:
  order: 10
---
## `AgentToolMetadata`

Describes the tool behind an agent action that is waiting for approval.

| Field | Type | Description |
| --- | --- | --- |
| `source` | `AgentToolSource` | Where the tool comes from: built in, registered by your app, an MCP server, or an external dispatch URL. |
| `kind` | `AgentToolKind` | What the tool does, such as code execution, shell, or web search. |
| `label` | `str` | Display name, such as `"Python code"`, `"Shell command"`, or the tool's own name. |


## `AgentToolApproval`

An agent action waiting for approval. Your `ChatCompletionRequest.agent_approval_callback` receives one.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `approval_id` | `str` | required | ID of this approval request. |
| `session_id` | `str` | required | Agent session the action belongs to. |
| `round` | `int` | required | Tool-call round within the request. |
| `tool` | `AgentToolMetadata` | required | The tool the model wants to run. |
| `arguments_json` | `str` | required | The tool call's arguments, as a JSON string. |
| `code` | `str \| None` | `None` | The `code` argument when it is a string, as it is for Python code execution. Otherwise `None`. |

### `AgentToolApproval.arguments`

```text
arguments() -> Any
```

Parse `arguments_json` and return the result as Python objects.


## `AgentToolApprovalDecision`

A richer answer for an approval callback than `True` or `False`. Use it to send the model a reason
for a denial, or to approve every later action in the session.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `decision` | `AgentToolApprovalDecisionKind` | required | Approve or deny. |
| `remember_for_session` | `bool` | `False` | On approve, skip approval for later actions in the same `session_id`. |
| `message` | `str \| None` | `None` | On deny, text returned to the model as the tool result. |

### `AgentToolApprovalDecision.approve`

```text
approve(
    remember_for_session: bool = False,
) -> 'AgentToolApprovalDecision'
```

Approve the action.

### `AgentToolApprovalDecision.deny`

```text
deny(message: str | None = None) -> 'AgentToolApprovalDecision'
```

Deny the action, optionally with a message for the model.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
