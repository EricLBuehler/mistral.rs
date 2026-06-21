---
title: AnyMoE
description: "AnyMoE expert and config types."
sidebar:
  order: 8
---
## `AnyMoeExpertType`

Expert type for an AnyMoE model. May be:
- `AnyMoeExpertType.FineTuned()`
- `AnyMoeExpertType.LoraAdapter(rank: int, alpha: float, target_modules: list[str])`

### `AnyMoeExpertType.FineTuned`

### `AnyMoeExpertType.LoraAdapter`

| Field | Type |
| --- | --- |
| `rank` | `int` |
| `alpha` | `float` |
| `target_modules` | `list[str]` |


## `AnyMoeConfig`

### `AnyMoeConfig.__init__`

```text
__init__(
    hidden_size: int,
    dataset_json: str,
    prefix: str,
    mlp: str,
    model_ids: list[str],
    expert_type: AnyMoeExpertType,
    layers: list[int] = [],
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 4,
    gate_model_id: str | None = None,
    training: bool = True,
    loss_csv_path: str | None = None,
) -> None
```

Create an AnyMoE config from the hidden size, dataset, and other metadata. The model IDs may be local paths.

To find the prefix/mlp values:

- Go to `https://huggingface.co/<MODEL ID>/tree/main?show_file_info=model.safetensors.index.json`
- Look for the mlp layers: for example `model.layers.27.mlp.down_proj.weight` means the prefix is `model.layers` and the mlp is `mlp`.

To find the hidden size:

- Look it up in `https://huggingface.co/<BASE MODEL ID>/blob/main/config.json`.

Note: `gate_model_id` specifies the gating model ID. If `training == True`, safetensors are written here; otherwise the pretrained safetensors are loaded and no training occurs.

Note: if `training == True`, `loss_csv_path` has no effect. Otherwise, a CSV loss file is saved at that path.

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
