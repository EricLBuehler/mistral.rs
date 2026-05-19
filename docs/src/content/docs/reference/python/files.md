---
title: Files
description: "First-class output files surfaced from agentic runs."
sidebar:
  order: 10
---
## `RequestedFile`

A required output file declared on a request. The runtime tells the
model about declared files; if produced by a tool, they surface in
`ChatCompletionResponse.files`. If missing, an error placeholder is
surfaced instead.

| Field | Type |
| --- | --- |
| `name` | `str` |
| `format` | `str \| None` |
| `description` | `str \| None` |

### `RequestedFile.__init__`

```text
__init__(
    name: str,
    format: str | None = None,
    description: str | None = None,
) -> None
```


## `FileSource`

Where a file was produced.

| Field | Type |
| --- | --- |
| `tool` | `str` |
| `round` | `int` |
| `turn` | `int` |


## `File`

First-class output from an agentic run.

Files exist independently of the transcript. The body is inline for
small files (`text` for text content, `data_base64` for binary). Large
files have a server-side url and `text`/`data_base64` will be `None` -
use `is_truncated()` to detect.

| Field | Type |
| --- | --- |
| `id` | `str` |
| `name` | `str` |
| `format` | `str \| None` |
| `mime_type` | `str \| None` |
| `bytes` | `int` |
| `source` | `FileSource` |
| `text` | `str \| None` |
| `data_base64` | `str \| None` |
| `preview` | `str \| None` |

### `File.is_text`

```text
is_text() -> bool
```

### `File.is_binary`

```text
is_binary() -> bool
```

### `File.is_image`

```text
is_image() -> bool
```

### `File.is_video`

```text
is_video() -> bool
```

### `File.is_error`

```text
is_error() -> bool
```

### `File.is_truncated`

```text
is_truncated() -> bool
```

### `File.save`

```text
save(path: str) -> None
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
