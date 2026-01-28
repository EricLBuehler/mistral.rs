# Structured model loading with .toml files

Mistral.rs supports loading models from a .toml file, and the fields are the same as for the CLI. Please find some example toml selectors [here](https://github.com/EricLBuehler/mistral.rs/tree/master/toml-selectors/).

There are a few cases which add functionality that cannot be found in the CLI.

## Speculative decoding

### What to specify
**Under `[speculative]`**
- Specify the `gamma` parameter

**Under `[speculative.draft_model]`**
- Choose a draft model, just like under `[model]` (only requirement is that they have the same tokenizer)

```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[speculative]
gamma = 32

[speculative.draft_model]
tok_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
quantized_model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
quantized_filename = "mistral-7b-instruct-v0.1.Q2_K.gguf"
```

```bash
mistralrs from-config -f toml-selectors/speculative-gguf.toml
```

## AnyMoE

### What to specify
**Under `[anymoe]`, required unless specified**
- Specify the dataset
- Find and specify the prefix/mlp values
    - Go to `https://huggingface.co/<MODEL ID>/tree/main?show_file_info=model.safetensors.index.json`
    - Look for the mlp layers: For example `model.layers.27.mlp.down_proj.weight` means that the prefix is `model.layers` and the mlp is `mlp`.
- Specify the expert or LoRA adapter model IDs
- (Optional) Specify layers to apply AnyMoE to.

**Under `[anymoe.config]`**
- Hidden size, typically found at `https://huggingface.co/<BASE MODEL ID>/blob/main/config.json`

**(For LoRA experts) Under `[anymoe.config.expert_type.lora_adapter]`**
- Rank
- Alpha
- Target modules

```bash
mistralrs from-config -f toml-selectors/anymoe.toml
```

### With fine-tuned experts
```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_json = "test.csv"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["HuggingFaceH4/zephyr-7b-beta"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096
expert_type = "fine_tuned"
```

### With LoRA adapter experts
```toml
[model]
model_id = "HuggingFaceH4/zephyr-7b-beta"
arch = "mistral"

[anymoe]
dataset_json = "test.csv"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["EricB/example_adapter"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096

[anymoe.config.expert_type.lora_adapter]
rank = 16
alpha = 16
target_modules = ["gate_proj"]
```
