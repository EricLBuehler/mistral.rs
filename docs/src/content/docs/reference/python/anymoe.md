---
title: AnyMoE
description: "AnyMoE expert and config types."
sidebar:
  order: 8
---
## `AnyMoeExpertType`

What each AnyMoE expert in `AnyMoeConfig.model_ids` is: a full fine-tune or a LoRA adapter.

### `AnyMoeExpertType.FineTuned`

Each expert is a full fine-tune of the base model. Its MLP weights become the expert.

### `AnyMoeExpertType.LoraAdapter`

Each expert is a LoRA adapter, applied to the base model's MLP.

| Field | Type | Description |
| --- | --- | --- |
| `rank` | `int` | LoRA rank of the adapters. |
| `alpha` | `float` | LoRA alpha of the adapters. |
| `target_modules` | `list[str]` | MLP modules the adapters change. |


## `AnyMoeConfig`

Builds a mixture of experts from a base model and fine-tunes of it, with a small gate that learns
which expert should handle each token. Pass it as `Runner(anymoe_config=...)`. See
[AnyMoE](/guides/customize/lora-adapters/#anymoe).

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

To find `prefix` and `mlp`, open the model's `model.safetensors.index.json` and find an MLP weight.
For example, `model.layers.27.mlp.down_proj.weight` means `prefix` is `model.layers` and `mlp` is `mlp`.
`hidden_size` is in the base model's `config.json`.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `hidden_size` | `int` | required | The base model's hidden size. |
| `dataset_json` | `str` | required | Training data: a JSON file with a `rows` list of `{"prompt": ..., "expert": N}` objects. Expert `0` is the base model, `1` the first entry in `model_ids`, and so on. |
| `prefix` | `str` | required | Weight name prefix before the layer number, such as `model.layers`. |
| `mlp` | `str` | required | Name of the MLP module within each layer, such as `mlp`. |
| `model_ids` | `list[str]` | required | One Hugging Face ID or local path per expert. |
| `expert_type` | `AnyMoeExpertType` | required | Whether the experts are full fine-tunes or LoRA adapters. |
| `layers` | `list[int]` | `[]` | Layers to turn into mixtures. Empty means every layer. |
| `lr` | `float` | `0.001` | Learning rate for training the gate. |
| `epochs` | `int` | `100` | Training epochs. |
| `batch_size` | `int` | `4` | Training batch size. |
| `gate_model_id` | `str \| None` | `None` | With `training=True`, a directory to save the trained gate to, as `gate.safetensors`. With `training=False`, a Hub ID or local directory to load a trained gate from. |
| `training` | `bool` | `True` | Train the gate at load time. Set it to False to load a saved gate from `gate_model_id`. |
| `loss_csv_path` | `str \| None` | `None` | With `training=True`, save each step's training loss to this `.csv` file. Ignored when not training. |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
