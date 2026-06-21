---
title: Search
description: "Types for web-search tool configuration."
sidebar:
  order: 7
---
## `WebSearchOptions`

| Field | Type | Default |
| --- | --- | --- |
| `search_context_size` | `Optional[SearchContextSize]` | `None` |
| `user_location` | `Optional[WebSearchUserLocation]` | `None` |
| `search_description` | `Optional[str]` | `None` |
| `extract_description` | `Optional[str]` | `None` |


## `WebSearchUserLocation`

### `WebSearchUserLocation.approximate`

```text
approximate(
    approximate: ApproximateUserLocation,
) -> 'WebSearchUserLocation'
```


## `ApproximateUserLocation`

| Field | Type |
| --- | --- |
| `city` | `str` |
| `country` | `str` |
| `region` | `str` |
| `timezone` | `str` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
