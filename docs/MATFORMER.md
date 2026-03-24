# Matformer (Matryoshka Transformer) Support

Matformer allows you to dynamically resize transformer models at runtime, trading compute/memory for quality. This enables deploying the same model across devices with different resource constraints - from edge devices to powerful GPUs.

## Quick Start

### Command Line

```bash
# Run Gemma 3n with the E2.49B configuration (2.49B params instead of 3.98B)
mistralrs run -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

### Python

```python
from mistralrs import Runner, Which, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=VisionArchitecture.Gemma3n,
        matformer_config_path="matformer_configs/gemma3n.csv",
        matformer_slice_name="Config for E2.49B (block-level)",
    ),
)
```

### Rust

```rust
use mistralrs::VisionModelBuilder;
use std::path::PathBuf;

let model = VisionModelBuilder::new("google/gemma-3n-E4B-it")
    .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
    .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
    .build()
    .await?;
```

## How It Works

Matformer models are pre-trained with a special architecture that allows certain layers to be skipped at inference time while maintaining reasonable quality. When you select a "slice":

1. **Layer Skipping**: Specified layers are completely removed from computation
2. **FFN Resizing**: Feed-forward network dimensions can be adjusted per layer
3. **Automatic Remapping**: Remaining layers are renumbered sequentially

For example, the Gemma 3n E2.49B (block-level) slice:
- Keeps all 35 layers (no layer skipping)
- Uses mixed FFN dimensions: 8192 for layers 0-19, 16384 for layers 20-24, 8192 for layers 25-34
- Cuts parameters from 3.98B to 2.49B (~37% reduction)
- Maintains ~87% of the full model's quality

## Configuration Files

Matformer configurations are CSV files with these columns:

```csv
name,# Layers,# Effective Params (B),MMLU PT accuracy,FFN Hidden Dims,Layers Skipped
Main model,35,3.98,62.30%,"[16384, 16384, ...]",
Config for E2.49B (block-level),35,2.49,54.50%,"[8192, 8192, ..., 16384, 16384, ..., 8192, 8192, ...]",
```

- **name**: Slice identifier used in `matformer_slice_name`
- **# Layers**: Number of active layers after skipping
- **# Effective Params (B)**: Approximate parameter count in billions
- **MMLU PT accuracy**: Benchmark score (informational)
- **FFN Hidden Dims**: List of FFN dimensions for each layer
- **Layers Skipped**: Which layers to remove (0-indexed)

## Supported Models

Currently supported:
- **Gemma 3n** (`google/gemma-3n-E4B-it`) - Multimodal model with vision and audio

See [`matformer_configs/`](https://github.com/EricLBuehler/mistral.rs/tree/master/matformer_configs/) for available configurations.

## Performance Guide

### Memory Usage

Memory scales approximately with parameter count:
- Full model (3.98B): ~8GB VRAM
- E2.49B slice: ~5GB VRAM
- E2B slice (1.91B): ~4GB VRAM  
- Smaller slices: Proportionally less

### Inference Speed

Speed improvement is roughly linear with layer count:
- 30 layers vs 35 layers = ~14% faster
- 20 layers vs 35 layers = ~43% faster

### Quality Trade-offs

Example accuracy on MMLU benchmark:
- Full model: 62.3%
- E2.98B: 59.5% (-4.5%)
- E2.49B: 54.5% (-12.5%)
- E2B: 50.9% (-18.3%)

Choose based on your requirements:
- **Maximum quality**: Use full model (omit matformer args)
- **Balanced**: E2.49B to E2.98B configurations (block-level configs recommended)
- **Resource-constrained**: E2B configuration (1.91B params)
- **Extreme efficiency**: E1.96B configuration

## Advanced Usage

### With Quantization

Combine Matformer with ISQ for maximum efficiency:

```python
runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=VisionArchitecture.Gemma3n,
        matformer_config_path="matformer_configs/gemma3n.csv",
        matformer_slice_name="Config for E2.49B (block-level)",
    ),
    in_situ_quant="Q4K"  # 4-bit quantization
)
```

### With Device Mapping

Matformer works seamlessly with automatic device mapping:

```rust
use mistralrs::{VisionModelBuilder, DeviceMapSetting, AutoDeviceMapParams};

let model = VisionModelBuilder::new("google/gemma-3n-E4B-it")
    .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
    .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
    .with_device_mapping(DeviceMapSetting::Auto(
        AutoDeviceMapParams::default_vision()
    ))
    .build()
    .await?;
```

Only active layers are loaded to GPU, saving memory.

## Creating Custom Configurations

To create your own Matformer configuration:

1. **Start with the full model** as baseline
2. **Identify skippable layers**:
   - Middle layers (10-30) are often good candidates
   - Avoid early layers (feature extraction) and late layers (final representations)
   - Never skip special layers (KV-sharing, attention patterns)
3. **Test quality degradation** at each configuration
4. **Create CSV file** with your configurations

Example minimal configuration:
```csv
name,# Layers,# Effective Params (B),FFN Hidden Dims,Layers Skipped
Tiny,15,0.8,"[4096, 4096, ...]","[5,6,7,10,11,12,15,16,17,20,21,22,25,26,27,30,31,32,33,34]"
```

## API Reference

### Command Line Arguments

- `--matformer-config-path PATH`: Path to CSV configuration file
- `--matformer-slice-name NAME`: Exact name of slice from CSV

### Python Parameters

```python
Which.VisionPlain(
    model_id: str,
    arch: VisionArchitecture,
    matformer_config_path: str = None,  # Path to CSV
    matformer_slice_name: str = None,   # Slice name
    # ... other parameters
)
```

### Rust Methods

```rust
// For VisionModelBuilder
.with_matformer_config_path(path: PathBuf)
.with_matformer_slice_name(name: String)

// For TextModelBuilder (when supported)
.with_matformer_config_path(path: PathBuf)  
.with_matformer_slice_name(name: String)
```

## Troubleshooting

### Common Issues

**"Matformer slice 'X' not found"**
- Check slice name matches exactly (case-sensitive)
- Verify CSV file path is correct

**"Layers X and Y are reserved and cannot be skipped"**
- Some models have special layers that must not be skipped
- Try different layer combinations

**Memory not reduced as expected**
- Ensure you're using the slice (check logs)
- Skipped layers still need to be loaded initially
- Consider combining with quantization

### Debugging

Enable logging to see Matformer details:
```bash
RUST_LOG=mistralrs_core=info mistralrs ...
```

This shows:
- Configuration file loaded
- Selected slice details  
- Layers being skipped
- Final layer count

## Future Plans

- Support for more model architectures
- Dynamic slice switching during runtime
- Automatic slice selection based on available resources
- Fine-tuning tools for creating new Matformer models

## References

- [Matryoshka Transformer Paper](https://arxiv.org/abs/2410.23265)
- [Example Configurations](https://github.com/EricLBuehler/mistral.rs/tree/master/matformer_configs/)