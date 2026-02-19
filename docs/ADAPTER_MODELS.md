# Adapter model support
An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting an `XLora*` architecture, and LoRA support by selecting the `Lora*` architecture. For both X-LoRA and LoRA, an ordering file (see [this section](#adapter-ordering-file) for preparing the ordering file) must be provided. The ordering file describes the ordering of layers and which adapters to use (and what order to use them in for X-LoRA).

When using an adapter model with a quantized base model, if the ordering file specifies unsupported layers you will receive an error.

## Supported X-LoRA or LoRA quantized layers**

**Llama architecture:**

- model.layers.{layer_idx}.self_attn.q_proj
- model.layers.{layer_idx}.self_attn.k_proj
- model.layers.{layer_idx}.self_attn.v_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.up_proj
- model.layers.{layer_idx}.mlp.down_proj
- model.layers.{layer_idx}.mlp.gate_proj
- lm_head

**Phi 3 architecture:**
- model.layers.{layer_idx}.self_attn.qkv_proj
- model.layers.{layer_idx}.self_attn.o_proj
- model.layers.{layer_idx}.mlp.gate_up_proj
- model.layers.{layer_idx}.mlp.down_proj
- lm_head

## Adapter ordering file
**Preparing the X-LoRA/LoRA Ordering File**
The X-LoRA/LoRA ordering file is necessary to prepare before inference with an X-LoRA model. However, it is easy with a provided [`script`](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/create_ordering.py)!

### X-LoRA case
An ordering JSON file for X-LoRA contains 2 major parts. 

1) The adapter names `order`
    - The order matters!
    - Should be an array of strings which are the adapter names corresponding to the order the adapters were specified during training. For example, if the adapters were specified as a dictionary:
2) The layer ordering `layers`
    - Automatically generated and should not be manipulated as it controls the application of scalings. 

```python
adapters = {
    "math": ...,
    "reasoning": ...,
    "biology": ...
}
```

The specified order would be `["math", "reasoning", "biology"]`.

We provide an [ordering file](https://github.com/EricLBuehler/mistral.rs/blob/master/orderings/xlora-paper-ordering.json) which contains the ordering for the X-LoRA model associated with [the paper](https://arxiv.org/abs/2402.07148) and the Huggingface repository: https://huggingface.co/lamm-mit/x-lora.

### LoRA case
An ordering JSON file for LoRA contains 2 major parts:
1) The adapter names `order` (optional):
    - The order does not matter
    - Come controls which adapters will be initially activated
    - If this key is not specified, then no adapters will be activated initially
2) Preload adapter section `preload_adapters` (optional): [see this section](#adapter-model-dynamic-adapter-activation)
    - Order does not matter
    - Specifies the adapter name and the model ID to find them, which may be a local path.

### Preparing the ordering file (LoRA or X-LoRA cases)
There are 2 scripts to prepare the ordering file and which work for both X-LoRA and LoRA. The ordering file is specific to each architecture and set of target modules. Therefore, if either are changed, it is necessary to create a new ordering file using the first option. If only the adapter order or adapters changed, then the second option should be used.

1) From scratch: No ordering file for the architecture and target modules

    A script [`create_ordering.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/create_ordering.py) is provided which prompts the user for the model ID, target modules, and adapter names. The user is prompted for an output file location, relative to the working directory.

2) Create a new ordering file from an existing ordering file for an architecture and target modules

    A script [`set_names.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/set_names.py) is provided which prompts the user for the adapter names and the old ordering file. The user is prompted for an output file location, relative to the working directory.

### Quantized X-LoRA or LoRA models

Mistral.rs supports running quantized models with X-LoRA or LoRA. The X-LoRA or LoRA adapter layers will not be quantized, only the base model. P

In the X-LoRA case, please note that using a high quantization level (eg., 4-bit) can distort the signal and prevent the classifier from acting properly. Therefore, it is better to use slightly lower levels such as 8-bit.


## Avoiding the scaling pass with non-granular scalings

The X-LoRA implementation supports non-granular scalings. This caches the scalings after `k` completion tokens are generated and they will be used for the remaining passes avoiding the scaling pass. The number of tokens to generate before caching is defined by setting `tgt_non_granular_index`. Setting `tgt_non_granular_index` will restrict the maximum running sequences to 1.

Please see [this page](NON_GRANULAR.md) for more details and examples.

## Adapter model dynamic adapter activation

We support dynamic adapter activation for LoRA models, allowing you to activate a set of adapters at runtime. There is a Python, Rust and HTTP API:

- Rust: [example](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/lora/main.rs)
- Python: [example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lora_zephyr.py)
- HTTP: [example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/adapter_chat.py)

To use this feature, you should add a `preload_adapters` key to your ordering file:
```diff
{
    "order": ["..."],
    "layers": {"...": "123"},
    "base_model_id": "...",
+    "preload_adapters": [{"name": "...", "adapter_model_id": "..."}] # New field here
}
```

This allows mistral.rs to preload the adapter and enable runtime activation.