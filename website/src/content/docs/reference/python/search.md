---
title: Search
description: "Types for web-search tool configuration."
sidebar:
  order: 7
---
### `WebSearchOptions`

| Field | Type |
| --- | --- |
| `search_context_size` | `Optional[SearchContextSize]` |
| `user_location` | `Optional[WebSearchUserLocation]` |


### `WebSearchUserLocation`

| Field | Type |
| --- | --- |
| `type` | `Literal['approximate']` |
| `approximate` | `ApproximateUserLocation` |


### `ApproximateUserLocation`

| Field | Type |
| --- | --- |
| `city` | `str` |
| `country` | `str` |
| `region` | `str` |
| `timezone` | `str` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
