---
title: MCP
description: "MCP client configuration types."
sidebar:
  order: 11
---
### `McpServerSourcePy`

MCP server transport source configuration

| Member | Value |
| --- | --- |
| `McpServerSourcePy.Http` | `'Http'` |
| `McpServerSourcePy.Process` | `'Process'` |
| `McpServerSourcePy.WebSocket` | `'WebSocket'` |


### `McpServerConfigPy`

Configuration for an individual MCP server

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `name` | `str` | required |
| `source` | `McpServerSourcePy` | required |
| `enabled` | `bool` | `True` |
| `tool_prefix` | `Optional[str]` | `None` |
| `resources` | `Optional[list[str]]` | `None` |
| `bearer_token` | `Optional[str]` | `None` |


### `McpClientConfigPy`

Configuration for MCP client integration

| Field | Type | Default |
| --- | --- | --- |
| `servers` | `list[McpServerConfigPy]` | required |
| `auto_register_tools` | `bool` | `True` |
| `tool_timeout_secs` | `Optional[int]` | `None` |
| `max_concurrent_calls` | `Optional[int]` | `None` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
