---
title: Agent approvals
description: "Request and decision types for agent action approval callbacks."
sidebar:
  order: 10
---
## `AgentToolMetadata`

Stable metadata for the agent action being approved.

| Field | Type |
| --- | --- |
| `source` | `AgentToolSource` |
| `kind` | `AgentToolKind` |
| `label` | `str` |


## `AgentToolApproval`

Approval request passed to `ChatCompletionRequest.agent_approval_callback`.

| Field | Type | Default |
| --- | --- | --- |
| `approval_id` | `str` | required |
| `session_id` | `str` | required |
| `round` | `int` | required |
| `tool` | `AgentToolMetadata` | required |
| `arguments_json` | `str` | required |
| `code` | `str \| None` | `None` |

### `AgentToolApproval.arguments`

```text
arguments() -> Any
```


## `AgentToolApprovalDecision`

Approval callback return value with HTTP/Rust parity.

| Field | Type | Default |
| --- | --- | --- |
| `decision` | `AgentToolApprovalDecisionKind` | required |
| `remember_for_session` | `bool` | `False` |
| `message` | `str \| None` | `None` |

### `AgentToolApprovalDecision.approve`

```text
approve(
    remember_for_session: bool = False,
) -> 'AgentToolApprovalDecision'
```

### `AgentToolApprovalDecision.deny`

```text
deny(message: str | None = None) -> 'AgentToolApprovalDecision'
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
