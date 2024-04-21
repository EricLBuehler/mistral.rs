# Adapter model support
An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. For both X-LoRA and LoRA, an ordering file (see [this section](#adapter-ordering-file) for preparing the ordering file) must be provided. The ordering file describes the ordering of layers and which adapters to use (and what order to use them in for X-LoRA).

When using an adapter model with a quantized base model, if the ordering file specifies unsupported layers you will receive an error.

**Supported X-LoRA or LoRA quantized layers**
- model.layers.{layer_idx}.self_attn.q_proj
- model.layers.{layer_idx}.self_attn.k_proj
- model.layers.{layer_idx}.self_attn.v_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.up_proj
- model.layers.{layer_idx}.mlp.down_proj
- model.layers.{layer_idx}.mlp.gate_proj

## Adapter ordering file
**Preparing the X-LoRA/LoRA Ordering File**
The X-LoRA/LoRA ordering file is necessary to prepare before inference with an X-LoRA model. However, it is easy with a provided [`script`](scripts/create_ordering.py)!

The X-LoRA/LoRA ordering JSON file contains 2 parts. The first is the order of the adapters and the second, the layer ordering. The layer ordering has been automatically generated and should not be manipulated as it controls the application of scalings. However the order of adapter should be an array of strings which are the adapter names corresponding to the order the adapters were specified during training. For example, if the adapters were specified as a dictionary:

```python
adapters = {
    "math": ...,
    "reasoning": ...,
    "biology": ...
}
```

The specified order would be `["math", "reasoning", "biology"]`.

For LoRA models, the order of the adapters does not matter. You can reorder them or remove some to control which adapters will be used. However, for an X-LoRA model, the order of the adapters in the ordering file is important.

There are 2 scripts to prepare the ordering file. The ordering file is specific to each architecture and set of target modules. Therefore, if either are changed, it is necessary to create a new ordering file using the first option. If only the adapter order or adapters changed, then it the second option should be used.

1) From scratch: No ordering file for the architecture and target modules

    A script [`create_ordering.py`](scripts/create_ordering.py) is provided which prompts the user for the model ID, target modules, and adapter names. The user is prompted for an output file location, relative to the working directory.

2) Create a new ordering file from an existing ordering file for an architecture and target modules

    A script [`modify_names.py`](scripts/modify_names.py) is provided which prompts the user for the adapter names and the old ordering file. The user is prompted for an output file location, relative to the working directory.

A provide a [ordering file](scripts/xlora-paper-ordering.json) which contains the ordering for the X-LoRA model associated with [the paper](https://arxiv.org/abs/2402.07148) and the Huggingface repository: https://huggingface.co/lamm-mit/x-lora.

**Quantized X-LoRA or LoRA models**

Mistral.rs supports running quantized models with X-LoRA or LoRA. The X-LoRA or LoRA adapter layers will not be quantized, only the base model. Please note that using a high quantization level (eg., 4-bit) can distort the signal and prevent the classifier from acting properly. Therefore, it is better to use slightly lower levels such as 8-bit.


## Avoiding the scaling pass with non-granular scalings

The X-LoRA implementation supports non-granular scalings. This caches the scalings after `k` completion tokens are generated and they will be used for the remaining passes avoiding the scaling pass. The number of tokens to generate before caching is defined by setting `tgt_non_granular_index`. Setting `tgt_non_granular_index` will restrict the maximum running sequences to 1.