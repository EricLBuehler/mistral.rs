---
title: Which
description: "Variants that select which kind of model to load."
sidebar:
  order: 3
---
## `LoraAdapter`

| Field | Type | Default |
| --- | --- | --- |
| `alias` | `str` | required |
| `source` | `str` | required |
| `revision` | `str \| None` | `None` |


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
| `imatrix` | `str \| None` | `None` |
| `calibration_file` | `str \| None` | `None` |

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
| `model_id` | `str` | required |
| `adapters` | `list[LoraAdapter] \| None` | `None` |
| `arch` | `Architecture \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `write_uqff` | `str \| None` | `None` |
| `from_uqff` | `str \| list[str] \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |
| `hf_cache_path` | `str \| None` | `None` |
| `max_adapters` | `int` | `16` |
| `max_rank` | `int` | `256` |
| `max_bytes` | `int` | `8589934592` |

### `Which.GGUF`

Select a GGUF model.

Pass `adapters=[]` or set a non-default LoRA limit to enable an empty dynamic LoRA runtime.
With `mmproj_filename`, adapters apply to the language model.
Pass `in_situ_quant` to `Runner` to requantize compatible GGUF weights while loading.

| Field | Type | Default |
| --- | --- | --- |
| `quantized_model_id` | `str` | required |
| `quantized_filename` | `str \| list[str]` | required |
| `tok_model_id` | `str \| None` | `None` |
| `topology` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |
| `auto_map_params` | `TextAutoMapParams \| None` | `None` |
| `tokenizer_json` | `str \| None` | `None (keyword-only)` |
| `mmproj_filename` | `str \| list[str] \| None` | `None (keyword-only)` |
| `organization` | `IsqOrganization \| None` | `None (keyword-only)` |
| `write_uqff` | `str \| None` | `None (keyword-only)` |
| `imatrix` | `str \| None` | `None (keyword-only)` |
| `calibration_file` | `str \| None` | `None (keyword-only)` |
| `max_edge` | `int \| None` | `None (keyword-only)` |
| `multimodal_auto_map_params` | `MultimodalAutoMapParams \| None` | `None (keyword-only)` |
| `adapters` | `list[LoraAdapter] \| None` | `None (keyword-only)` |
| `max_adapters` | `int` | `16 (keyword-only)` |
| `max_rank` | `int` | `256 (keyword-only)` |
| `max_bytes` | `int` | `8589934592 (keyword-only)` |
| `hf_cache_path` | `str \| None` | `None (keyword-only)` |
| `matformer_config_path` | `str \| None` | `None (keyword-only)` |
| `matformer_slice_name` | `str \| None` | `None (keyword-only)` |

### `Which.XLoraGGUF`

Select X-LoRA for a Phi3 GGUF configuration.

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

Select legacy static LoRA for a Phi3 GGUF configuration.

For dynamic adapters on a supported GGUF, pass `adapters` to `Which.GGUF`.

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
