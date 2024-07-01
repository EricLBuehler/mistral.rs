# AnyMoE: Build an MoE model from anything, quickly

AnyMoE is technique to dynamically and efficiently create MoE models. By providing a set of experts and a small pretraining dataset, you can create an MoE locally!

It has the following features:
- Apply AnyMoE to any supported model
    - `plain`
- Specify the layers to apply AnyMoE to for efficient training

## Dataset
Currently, AnyMoE expects a CSV dataset with 2 columns: `prompt` and `expert`. For example:
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
AnyMoE experts can be either fine-tuned models or LoRA adapter models. Only the mlp layers will be loaded from each. The experts must be homogeneous: they must be all fine-tuned or all adapter. Additionally, certain layers can be specified to apply AnyMoE.

> Note: When using LoRA adapter experts, it may not be necessary to set the layers where AnyMoE will be applied due to the lower memory usage.

### With fine-tuned experts
```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_csv = "examples/amoe.csv"
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
dataset_csv = "examples/amoe.csv"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["EricB/example_adapter"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096

[anymoe.config.expert_type.lora_adapter]
rank = 16
alpha = 16
target_modules = ["up_proj", "down_proj", "gate_proj"]
```

## Examples

## `mistralrs-server`

CLI usage is via the [TOML selector](TOML_SELECTOR.md#anymoe).

For example, to use the demo fine-tuned expert:
```
./mistralrs_server -i toml -f toml-selectors/anymoe.toml
```

To use the demo LoRA expert:
```
./mistralrs_server -i toml -f toml-selectors/anymoe_lora.toml
```

## Python API
```py
from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    AnyMoeConfig,
    AnyMoeExpertType,
)

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer_json=None,
        repeat_last_n=64,
        arch=Architecture.Mistral,
    ),
    anymoe_config=AnyMoeConfig(
        hidden_size=4096,
        dataset_csv="examples/amoe.csv",
        prefix="model.layers",
        mlp="mlp",
        lr=1e-3,
        epochs=100,
        batch_size=4,
        expert_type=AnyMoeExpertType.FineTuned(),
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Rust API
