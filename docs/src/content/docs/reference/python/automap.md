---
title: Auto-mapping
description: "Hints for automatic device mapping."
sidebar:
  order: 13
---
## `TextAutoMapParams`

Auto-mapping parameters for a text model.
These affect automatic device mapping but are not a hard limit.

| Field | Type | Default |
| --- | --- | --- |
| `max_seq_len` | `int` | `4 * 1024` |
| `max_batch_size` | `int` | `1` |


## `MultimodalAutoMapParams`

Auto-mapping parameters for a multimodal model.
These affect automatic device mapping but are not a hard limit.

| Field | Type | Default |
| --- | --- | --- |
| `max_seq_len` | `int` | `4 * 1024` |
| `max_batch_size` | `int` | `1` |
| `max_num_images` | `int` | `1` |
| `max_image_length` | `int` | `1024` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
