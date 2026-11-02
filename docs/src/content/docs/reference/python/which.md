---
title: Which
description: "Variants that select which kind of model to load."
sidebar:
  order: 3
---
## `LoraAdapter`

A LoRA adapter to load at startup, for `Which.Lora` or `Which.GGUF`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `alias` | `str` | required | Name that requests use to select the adapter. Case-sensitive, up to 256 bytes. |
| `source` | `str` | required | Hugging Face repo or local directory holding the adapter. |
| `revision` | `str \| None` | `None` | Branch, tag, or commit to download for a Hub `source`. Defaults to `main`. |


## `Which`

Chooses the kind of model to load and where its weights come from. Pass one variant to `Runner(which=...)`.

```python
runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B"))
```

### `Which.Plain`

A text model from safetensors weights.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | Hugging Face model ID or local directory. |
| `arch` | `Architecture \| None` | `None` | Architecture to load it as. Detected from `config.json` if unset. |
| `tokenizer_json` | `str \| None` | `None` | Path to a `tokenizer.json` to use instead of the model's own. |
| `topology` | `str \| None` | `None` | Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer. |
| `organization` | `IsqOrganization \| None` | `None` | Which layers in-situ quantization touches. See `IsqOrganization`. |
| `write_uqff` | `str \| None` | `None` | Save the quantized weights to this `.uqff` path while loading. Use with `Runner(in_situ_quant=...)`. |
| `from_uqff` | `str \| list[str] \| None` | `None` | Pre-quantized [UQFF](/guides/quantization/uqff/) file, or list of files, to load from the model's repo. |
| `dtype` | `ModelDType` | `ModelDType.Auto` | Data type to load the model in. |
| `imatrix` | `str \| None` | `None` | Importance matrix (`.imatrix` or `.cimatrix` file) that guides in-situ quantization. |
| `calibration_file` | `str \| None` | `None` | Text file to run through the model at load time to build an importance matrix. |
| `hf_cache_path` | `str \| None` | `None` | Hugging Face cache directory to use instead of the default. |
| `matformer_config_path` | `str \| None` | `None` | [MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n. |
| `matformer_slice_name` | `str \| None` | `None` | Slice to load from that table. Without it, the full model loads. |

### `Which.Embedding`

An embedding model from safetensors weights. Fields match `Which.Plain`.

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
| `hf_cache_path` | `str \| None` | `None` |

### `Which.Lora`

A text model that can switch LoRA adapters per request. See [LoRA adapters](/guides/customize/lora-adapters/).

Other fields match `Which.Plain`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required |  |
| `adapters` | `list[LoraAdapter] \| None` | `None` | Adapters to load at startup. Leave unset to start with none and load them later. |
| `arch` | `Architecture \| None` | `None` |  |
| `tokenizer_json` | `str \| None` | `None` |  |
| `topology` | `str \| None` | `None` |  |
| `write_uqff` | `str \| None` | `None` |  |
| `from_uqff` | `str \| list[str] \| None` | `None` |  |
| `dtype` | `ModelDType` | `ModelDType.Auto` |  |
| `hf_cache_path` | `str \| None` | `None` |  |
| `max_adapters` | `int` | `16` | Limit on loaded aliases, and separately on adapter generations in memory, counting old ones still finishing requests. |
| `max_rank` | `int` | `256` | Highest adapter rank accepted. |
| `max_bytes` | `int` | `8589934592` | Memory limit for all loaded adapters, in bytes (8 GiB by default). |

### `Which.GGUF`

A model from a GGUF file. See [run GGUF models](/guides/models/run-gguf/).

- Dynamic LoRA turns on when you pass `adapters`, even `[]`, or change `max_adapters`, `max_rank`, or `max_bytes`.
- With `mmproj_filename`, adapters apply to the language model only.
- To re-quantize compatible GGUF weights while loading, pass `in_situ_quant` to `Runner`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `quantized_model_id` | `str` | required | Hugging Face repo or local directory holding the GGUF file. |
| `quantized_filename` | `str \| list[str]` | required | GGUF file name, or a list of names for a split model. |
| `tok_model_id` | `str \| None` | `None` | The original model's repo or local directory, for config, tokenizer, and processor files the GGUF doesn't include. |
| `topology` | `str \| None` | `None` | Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer. |
| `dtype` | `ModelDType` | `ModelDType.Auto` | Data type to load the model in. |
| `tokenizer_json` | `str \| None` | `None (keyword-only)` | Path to a `tokenizer.json` to use instead of the model's own. |
| `mmproj_filename` | `str \| list[str] \| None` | `None (keyword-only)` | Projector GGUF file, or files, holding a multimodal model's vision or audio encoder. |
| `organization` | `IsqOrganization \| None` | `None (keyword-only)` | Which layers to quantize when `in_situ_quant` re-quantizes the weights. |
| `write_uqff` | `str \| None` | `None (keyword-only)` | Save the re-quantized weights to this `.uqff` path while loading. |
| `imatrix` | `str \| None` | `None (keyword-only)` | Importance matrix (`.imatrix` or `.cimatrix` file) that guides re-quantization. |
| `calibration_file` | `str \| None` | `None (keyword-only)` | Text file to run through the model at load time to build an importance matrix. |
| `max_edge` | `int \| None` | `None (keyword-only)` | Shrink images so their longest edge is at most this many pixels, keeping the aspect ratio. |
| `adapters` | `list[LoraAdapter] \| None` | `None (keyword-only)` | LoRA adapters to load at startup. Pass `[]` to start with none and load them later. |
| `max_adapters` | `int` | `16 (keyword-only)` | Limit on loaded aliases, and separately on adapter generations in memory. |
| `max_rank` | `int` | `256 (keyword-only)` | Highest adapter rank accepted. |
| `max_bytes` | `int` | `8589934592 (keyword-only)` | Memory limit for all loaded adapters, in bytes (8 GiB by default). |
| `hf_cache_path` | `str \| None` | `None (keyword-only)` | Hugging Face cache directory to use instead of the default. |
| `matformer_config_path` | `str \| None` | `None (keyword-only)` | [MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n. |
| `matformer_slice_name` | `str \| None` | `None (keyword-only)` | Slice to load from that table. Without it, the full model loads. |
| `encoder_cache_memory_bytes` | `int \| None` | `None (keyword-only)` | Memory limit, in bytes, for cached image, audio, and video encoder outputs. Unset keeps the model's default entry-count limit. |

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

### `Which.MultimodalPlain`

A model that takes images, audio, or video, from safetensors weights.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | `str` | required | Hugging Face model ID or local directory. |
| `arch` | `MultimodalArchitecture \| None` | `None` | Architecture to load it as. Detected from `config.json` if unset. |
| `tokenizer_json` | `str \| None` | `None` | Path to a `tokenizer.json` to use instead of the model's own. |
| `topology` | `str \| None` | `None` | Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer. |
| `write_uqff` | `str \| None` | `None` | Save the quantized weights to this `.uqff` path while loading. Use with `Runner(in_situ_quant=...)`. |
| `from_uqff` | `str \| list[str] \| None` | `None` | Pre-quantized [UQFF](/guides/quantization/uqff/) file, or list of files, to load from the model's repo. |
| `dtype` | `ModelDType` | `ModelDType.Auto` | Data type to load the model in. |
| `max_edge` | `int \| None` | `None` | Shrink images so their longest edge is at most this many pixels, keeping the aspect ratio. |
| `calibration_file` | `str \| None` | `None` | Text file to run through the model at load time to build an importance matrix. |
| `imatrix` | `str \| None` | `None` | Importance matrix (`.imatrix` or `.cimatrix` file) that guides in-situ quantization. |
| `hf_cache_path` | `str \| None` | `None` | Hugging Face cache directory to use instead of the default. |
| `matformer_config_path` | `str \| None` | `None` | [MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n. |
| `matformer_slice_name` | `str \| None` | `None` | Slice to load from that table. Without it, the full model loads. |
| `organization` | `IsqOrganization \| None` | `None` | Which layers in-situ quantization touches. See `IsqOrganization`. |
| `encoder_cache_memory_bytes` | `int \| None` | `None` | Memory limit, in bytes, for cached image, audio, and video encoder outputs. Unset keeps the model's default entry-count limit. |

### `Which.DiffusionPlain`

An image generation model.

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `DiffusionArchitecture` | required |
| `dtype` | `ModelDType` | `ModelDType.Auto` |

### `Which.Speech`

A dedicated speech generation model, such as Dia.

| Field | Type | Default |
| --- | --- | --- |
| `model_id` | `str` | required |
| `arch` | `SpeechLoaderType` | required |
| `dac_model_id` | `str \| None` | `None` |
| `dtype` | `ModelDType` | `ModelDType.Auto` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
