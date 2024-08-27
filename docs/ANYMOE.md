# Build a memory-efficient MoE model from anything, in seconds

AnyMoE is technique to dynamically and efficiently create MoE models. By providing a set of experts and a small pretraining dataset, you can create an MoE locally!

It has the following features:
- Apply AnyMoE to any supported model
    - `plain`
    - `vision-plain`
- Specify the layers to apply AnyMoE to for efficient training

https://github.com/EricLBuehler/mistral.rs/assets/65165915/33593903-d907-4c08-a0ac-d349d7bf33de

> Note: By default, this has the capability to create an csv loss image. When building from source (for Python or CLI), you may use `--no-default-features` command line to disable this. This may be necessary if networking is unavailable.

## Dataset
Currently, AnyMoE expects a JSON dataset with one top-level key `row`, which is an array of objects with keys `prompt` (string), `expert` (integer), and `image_urls` (optional array of strings). For example:
```json
{
    "rows": [
        {
            "prompt": "Discuss the impact of Renaissance art on modern aesthetics",
            "expert": 0
        },
        {
            "prompt": "Explain the significance of the theory of relativity in modern physics",
            "expert": 1
        },
    ]
}
  
```

For a vision model, `image_urls` may contain an array of image URLs/local paths or Base64 encoded images.

## Experts
AnyMoE experts can be either fine-tuned models or LoRA adapter models. Only the mlp layers will be loaded from each. The experts must be homogeneous: they must be all fine-tuned or all adapter. Additionally, certain layers can be specified to apply AnyMoE.

> Note: When using LoRA adapter experts, it may not be necessary to set the layers where AnyMoE will be applied due to the lower memory usage.

### Example of TOML selector with fine-tuned experts
```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_json = "examples/amoe.json"
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
dataset_json = "examples/amoe.json"
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
        arch=Architecture.Mistral,
    ),
    anymoe_config=AnyMoeConfig(
        hidden_size=4096,
        dataset_json="examples/amoe.json",
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
You can find this example [here](../mistralrs/examples/anymoe/main.rs).

```rust
use either::Either;
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    AnyMoeConfig, AnyMoeExpertType, AnyMoeLoader, Constraint, DefaultSchedulerMethod, Device,
    DeviceMapMetadata, Loader, MistralRs, MistralRsBuilder, ModelDType, NormalLoaderBuilder,
    NormalLoaderType, NormalRequest, NormalSpecificConfig, Request, RequestMessage, Response,
    Result, SamplingParams, SchedulerConfig, TokenSource,
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
            expert_type: AnyMoeExpertType::FineTuned,
            gate_model_id: None, // Set this to Some("path/to/model/id") for the pretrained gating model id
            training: true,
            loss_csv_path: None,
        },
        prefix: "model.layers".to_string(),
        mlp: "mlp".to_string(),
        path: "examples/amoe.json".to_string(),
        model_ids: vec!["HuggingFaceH4/zephyr-7b-beta".to_string()],
        layers: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
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
        None, // No PagedAttention.
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}
```
