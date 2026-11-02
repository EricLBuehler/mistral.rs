---
title: Files
description: "Input files and first-class output files surfaced from agentic runs."
sidebar:
  order: 11
---
## `RequestedFile`

An output file the model must produce. Pass a list of them as `ChatCompletionRequest.files`.

The model is told to write each file with a tool, so the request also needs a tool that can, such
as code execution or the shell. Otherwise the request fails. Written files appear in
`ChatCompletionResponse.files`. A file the model never wrote appears there as an error placeholder,
for which `File.is_error()` returns True.

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | File name the model must write. |
| `format` | `str \| None` | File format, such as `csv`. Inferred from the name's extension if unset. |
| `description` | `str \| None` | What the file should contain, shown to the model. |

### `RequestedFile.__init__`

```text
__init__(
    name: str,
    format: str | None = None,
    description: str | None = None,
) -> None
```


## `InputFile`

A file you give the model. Pass a list of them as `ChatCompletionRequest.input_files`.

- The model sees each file's name, ID, MIME type, and size.
- For a text file, it also sees the first 4,096 characters, and up to 32,768 across all files. It can
  read the rest with the built-in `mistralrs_read_file` tool.
- When the Python or shell tool runs, every file, text or binary, is copied into its working directory.

See [file inputs](/guides/agents/file-inputs/).

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | ID assigned when the object is created. |
| `name` | `str` | File name shown to the model. |
| `mime_type` | `str \| None` | MIME type. Inferred from the name's extension if unset. |
| `bytes` | `int` | Size in bytes. |

### `InputFile.__init__`

```text
__init__(name: str, data: bytes, mime_type: str | None = None) -> None
```

### `InputFile.from_text`

```text
from_text(
    name: str,
    text: str,
    mime_type: str = 'text/plain',
) -> 'InputFile'
```

### `InputFile.from_path`

```text
from_path(
    path: str,
    mime_type: str | None = None,
    name: str | None = None,
) -> 'InputFile'
```


## `FileSource`

Where a file was produced.

| Field | Type | Description |
| --- | --- | --- |
| `tool` | `str` | Name of the tool that wrote it. |
| `round` | `int` | Tool-call round within the turn, counting from 0. |
| `turn` | `int` | Turn within the session, counting from 0. Always 0 without a session. |


## `File`

A file that the model wrote during an agent run, such as a CSV from the Python tool. Read them from
`ChatCompletionResponse.files`.

A file's body is in `text` for text files and in `data_base64` for binary ones. Bodies over 8 MiB
are left out of the response: `text` and `data_base64` are `None` and `is_truncated()` returns True.
Fetch the full file with `Runner.find_file(file.id)`.

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | File ID, for `Runner.find_file`. |
| `name` | `str` | File name. |
| `format` | `str \| None` | File format, such as `csv`, from the name's extension. |
| `mime_type` | `str \| None` | MIME type. |
| `bytes` | `int` | Size in bytes. |
| `source` | `FileSource` | Which tool wrote the file, and when. |
| `text` | `str \| None` | Body of a text file. `None` for binary files and truncated bodies. |
| `data_base64` | `str \| None` | Base64 body of a binary file. `None` for text files and truncated bodies. |
| `preview` | `str \| None` | Short preview of a text file, or the full text when there is no preview. |

### `File.is_text`

```text
is_text() -> bool
```

True for a text file.

### `File.is_binary`

```text
is_binary() -> bool
```

True for a binary file.

### `File.is_image`

```text
is_image() -> bool
```

True if the MIME type starts with `image/`.

### `File.is_video`

```text
is_video() -> bool
```

True if the MIME type starts with `video/`.

### `File.is_error`

```text
is_error() -> bool
```

True for the placeholder of a requested file the model never wrote.

### `File.is_truncated`

```text
is_truncated() -> bool
```

True if the body was left out of the response because it was too large.

### `File.save`

```text
save(path: str) -> None
```

Write the body to `path`. Raises `OSError` for a truncated body or an error placeholder.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
