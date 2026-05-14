---
title: Which
description: "Variants that select which kind of model to load."
sidebar:
  order: 3
---
## `Which`

Which model to select. See the docs for the `Which` enum in API.md for more details.
Usage:
```python
>>> Which.Plain(...)
```

### `Which.Plain`

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `Architecture \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `organization` | `IsqOrganization \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `imatrix` | `str \| None` | `None` |
| `calibration_file` | `str \| None` | `None` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |
| `hf_cache_path` | `str \| None` | `None` |
| `matformer_config_path` | `str \| None` | `None` |
| `matformer_slice_name` | `str \| None` | `None` |

### `Which.Embedding`

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `EmbeddingArchitecture \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `hf_cache_path` | `str \| None` | `None` |

### `Which.XLora`

| Field | Type | Default |
| --- | --- | --- |
| `xlora_model_id` | `str` | required |
| `order` | `str` | required |
| `arch` | `Architecture \| None` | `None` |
| `model_id` | `str \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `tgt_non_granular_index` | `int \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |
| `hf_cache_path` | `str \| None` | `None` |

### `Which.Lora`

| Field | Type | Default |
| --- | --- | --- |
| `adapter_model_ids` | `list[str]` | required |
| `arch` | `Architecture \| None` | `None` |
| `model_id` | `str \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |
| `hf_cache_path` | `str \| None` | `None` |

### `Which.GGUF`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str \| list[str]` | required |
| `tok_model_id` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.XLoraGGUF`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str \| list[str]` | required |
| `xlora_model_id` | `str` | required |
| `order` | `str` | required |
| `tok_model_id` | `str \| None` | `None` |
| `tgt_non_granular_index` | `int \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.LoraGGUF`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str \| list[str]` | required |
| `adapters_model_id` | `str` | required |
| `order` | `str` | required |
| `tok_model_id` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.GGML`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str` | required |
| `tok_model_id` | `str` | required |
| `tokenizer_json` | `str \| None` | `None` |
| `gqa` | `int` | `1` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.XLoraGGML`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str` | required |
| `xlora_model_id` | `str` | required |
| `order` | `str` | required |
| `tok_model_id` | `str \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `tgt_non_granular_index` | `int \| None` | `None` |
| `gqa` | `int` | `1` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.LoraGGML`

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str` | required |
| `adapters_model_id` | `str` | required |
| `order` | `str` | required |
| `tok_model_id` | `str \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `gqa` | `int` | `1` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |

### `Which.MultimodalPlain`

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `MultimodalArchitecture \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `max_edge` | `int \| None` | `None` |
| `calibration_file` | `str \| None` | `None` |
| `imatrix` | `str \| None` | `None` |
| `auto_map_params` | `MultimodalAutoMapParams \| None` | `None` |
| `hf_cache_path` | `str \| None` | `None` |
| `matformer_config_path` | `str \| None` | `None` |
| `matformer_slice_name` | `str \| None` | `None` |
| `organization` | `IsqOrganization \| None` | `None` |

### `Which.DiffusionPlain`

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `DiffusionArchitecture` | required |
| `dtype` | `ModelDType` | `ModelDType.Auto` |

### `Which.Speech`

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `SpeechLoaderType` | required |
| `dac_model_id` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
