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

### Example of TOML selector with fine-tuned experts
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

### Example of TOML selector with LoRA adapter experts
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

CLI usage is via the [TOML selector](TOML_SELECTOR.md#anymoe) where you can also find docs on the required fields.

For example, to use the demo fine-tuned expert:
```
./mistralrs_server -i toml -f toml-selectors/anymoe.toml
```

To use the demo LoRA expert:
```
./mistralrs_server -i toml -f toml-selectors/anymoe_lora.toml
```

## Python example
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
        expert_type=AnyMoeExpertType.FineTuned(),
        lr=1e-3,
        epochs=100,
        batch_size=4,
        model_ids=["HuggingFaceH4/zephyr-7b-beta"],
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
```rust
use either::Either;
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    AnyMoeConfig, AnyMoeExpertType, AnyMoeLoader, Constraint, Device, DeviceMapMetadata, Loader,
    MistralRs, MistralRsBuilder, ModelDType, NormalLoaderBuilder, NormalLoaderType, NormalRequest,
    NormalSpecificConfig, Request, RequestMessage, Response, Result, SamplingParams,
    SchedulerMethod, TokenSource,
};

/// Gets the best device, cpu, cuda if compiled with CUDA
pub(crate) fn best_device() -> Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        None,
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
    )
    .build(NormalLoaderType::Mistral);
    let loader: Box<dyn Loader> = Box::new(AnyMoeLoader {
        target: loader,
        config: AnyMoeConfig {
            hidden_size: 4096,
            lr: 1e-3,
            epochs: 100,
            batch_size: 4,
            expert_type: AnyMoeExpertType::LoraAdapter {
                rank: 64,
                alpha: 16.,
                target_modules: vec![
                    "up_proj".to_string(),
                    "down_proj".to_string(),
                    "gate_proj".to_string(),
                ],
            },
        },
        prefix: "model.layers".to_string(),
        mlp: "mlp".to_string(),
        path: "examples/amoe.csv".to_string(),
        model_ids: vec!["typeof/zephyr-7b-beta-lora".to_string()],
        layers: vec![],
    });
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &best_device()?,
        false,
        DeviceMapMetadata::dummy(),
        None,
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
}
```
