# AnyMoE: Build an MoE model from anything, quickly

AnyMoE is technique to dynamically and efficiently create MoE models. By providing a set of experts and a small pretraining dataset, you can create an MoE.

## Dataset
Currently, AnyMoE expets a CSV dataset with 2 columns: `prompt` and `expert`. For example:
```csv
prompt,expert
Discuss the impact of Renaissance art on modern aesthetics,0
Explain the significance of the theory of relativity in modern physics,1
Analyze the themes of existentialism in 20th-century literature,0
Describe the process of photosynthesis and its importance to ecosystems,1
Evaluate the role of classical music in contemporary film scores,0
Outline the steps of the scientific method and their importance in experiments,1
Compare and contrast the philosophies of Socrates and Nietzsche,0
Discuss the ethical implications of artificial intelligence in society,1
Interpret the symbolism in Salvador Dalí's paintings,0
Describe the function and structure of DNA in genetic inheritance,1
```

## Experts
AnyMoE experts can be either fine-tuned models or LoRA adapter models. Only the mlp layers will be loaded from each. The experts must be homogenous: they must be all fine-tuned or all adapter.

### With fine-tuned experts
```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_csv = "test.csv"
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
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_csv = "test.csv"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["typeof/zephyr-7b-beta-lora"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096

[anymoe.config.expert_type.lora_adapter]
rank = 64
alpha = 16
```

## CLI usage

CLI usage is via the [TOML selector](TOML_SELECTOR.md#anymoe).
