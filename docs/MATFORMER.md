# Matformer (Matryoshka Transformer) Support

Mistral.rs supports Matformer models, which are elastic transformers that can operate at different computational budgets by selectively using subsets of their layers. This is particularly useful for deploying models on devices with varying computational resources.

## Overview

Matformer models, also known as Matryoshka Transformers, are designed with a nested structure where smaller, efficient models are embedded within larger ones. By skipping certain layers during inference, you can achieve different trade-offs between model quality and computational cost.

## Supported Models

Currently, mistral.rs supports Matformer configurations for:
- **Gemma3n** (Gemma 3n) - Google's efficient multimodal model with vision and audio capabilities

## Configuration

Matformer models require two components:
1. A CSV configuration file that defines different "slices" of the model
2. A slice name to select which configuration to use

### CSV Configuration Format

The Matformer configuration file is a CSV with the following columns:
- `name`: The name of the slice configuration
- `# Layers`: Total number of layers in this configuration
- `# Effective Params (B)`: Effective parameter count in billions
- `MMLU PT accuracy`: Model accuracy on MMLU benchmark (informational)
- `FFN Hidden Dims`: List of FFN hidden dimensions for each layer (e.g., `[8192, 8192, ...]`)
- `Layers Skipped`: Optional list of layer indices to skip (e.g., `[5, 10, 15]`)

Example CSV:
```csv
name,# Layers,# Effective Params (B),MMLU PT accuracy,FFN Hidden Dims,Layers Skipped
Config for official E2B Model,26,2.0,69.4,"[2_048 * 4, 2_048 * 4, ...]","[5, 10, 15, 20, 25, 30, 35, 40]"
Full Model,46,3.8,72.1,"[2_048 * 8, 2_048 * 8, ...]",
```

### Using Matformer Models

#### Command Line Interface

```bash
# Run with a specific Matformer slice
cargo run --release --features cuda -- \
  -i plain \
  -m google/gemma-3n-E4B-it \
  -a gemma3n \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for official E2B Model"
```

#### Rust API

```rust
use mistralrs::{TextModelBuilder, DeviceMapSetting};

let model = TextModelBuilder::new("google/gemma-3n-E4B-it")
    .with_arch(NormalLoaderType::Gemma3n)
    .with_matformer_config_path("matformer_configs/gemma3n.csv".into())
    .with_matformer_slice_name("Config for official E2B Model".to_string())
    .build()
    .await?;
```

#### Python API

```python
from mistralrs import Runner, Which, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="google/gemma-3n-E4B-it",
        arch=Architecture.Gemma3n,
        matformer_config_path="matformer_configs/gemma3n.csv",
        matformer_slice_name="Config for official E2B Model"
    )
)
```

## How It Works

### Layer Skipping

When a Matformer slice is selected:
1. The model loads all weights from the base model
2. Layers specified in `Layers Skipped` are removed from the computation graph
3. The remaining layers are renumbered sequentially
4. Activations flow through only the active layers

### FFN Dimension Adjustment

Matformer models can have different FFN (feed-forward network) dimensions per layer. The `FFN Hidden Dims` field specifies the intermediate size for each layer's MLP block, allowing for more fine-grained control over model capacity.

### Automatic Device Mapping

When using automatic device mapping with Matformer models:
- The device mapper initially considers all layers from the base model
- After matformer slicing is applied, skipped layers are not loaded onto devices
- This ensures efficient memory usage - you only pay for the layers you actually use

Example with device mapping:
```rust
use mistralrs::{TextModelBuilder, DeviceMapSetting, AutoDeviceMapParams};

let model = TextModelBuilder::new("google/gemma-3n-E4B-it")
    .with_arch(NormalLoaderType::Gemma3n)
    .with_matformer_config_path("matformer_configs/gemma3n.csv".into())
    .with_matformer_slice_name("Config for official E2B Model".to_string())
    .with_device_mapping(DeviceMapSetting::Auto(
        AutoDeviceMapParams::Text {
            max_seq_len: 4096,
            max_batch_size: 2,
        }
    ))
    .build()
    .await?;
```

## Creating Custom Matformer Configurations

To create your own Matformer configurations:

1. Start with the base model's architecture
2. Identify which layers can be skipped while maintaining acceptable quality
3. Optionally adjust FFN dimensions for remaining layers
4. Test different configurations to find optimal quality/performance trade-offs

### Guidelines for Layer Selection

- **Early layers** (0-10): Usually critical for basic feature extraction
- **Middle layers** (10-30): Can often be selectively skipped
- **Late layers** (30+): Important for final representations
- **Special layers**: Some models have KV-sharing or attention pattern layers that should not be skipped

### Example: Creating a Small Slice

For a model with 46 layers, you might create a "small" configuration:
```csv
name,# Layers,# Effective Params (B),FFN Hidden Dims,Layers Skipped
Small,20,1.2,"[2048 * 2, 2048 * 2, ...]","[5,6,7,8,15,16,17,18,25,26,27,28,35,36,37,38,40,41,42,43,44,45]"
```

## Performance Considerations

1. **Memory Usage**: Matformer slices use less memory proportional to the number of skipped layers
2. **Inference Speed**: Fewer layers means faster inference, roughly linear with layer count
3. **Quality Trade-off**: Smaller slices may have reduced accuracy but can be much faster
4. **Loading Time**: Initial model loading still loads all weights, but skipped layers are not transferred to GPU

## Troubleshooting

### Common Issues

1. **"Matformer slice not found"**: Ensure the slice name exactly matches one in your CSV file
2. **"Layers X and Y are reserved"**: Some models have special layers (e.g., KV-sharing layers) that cannot be skipped
3. **Device mapping failures**: If using very small slices, adjust `max_seq_len` and `max_batch_size` accordingly

### Debugging

Enable verbose logging to see which layers are being skipped:
```bash
RUST_LOG=mistralrs_core=info cargo run ...
```

This will show:
- Which Matformer configuration file is loaded
- Which slice is selected
- How many layers are skipped
- The final layer count after slicing

## Future Enhancements

Planned improvements for Matformer support:
- Dynamic slice switching during runtime
- Automatic slice selection based on available resources
- Training utilities for creating new Matformer models
- Support for more model architectures

## References

- [Matryoshka Transformer Paper](https://arxiv.org/abs/2410.23265)
- [Gemma 3 Nano Technical Report](https://github.com/google-deepmind/gemma)
- [Elastic Model Serving Best Practices](https://github.com/EricLBuehler/mistral.rs/discussions)