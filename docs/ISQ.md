# In situ quantization

In situ quantization works by quantizing models inplace, with the chief benefit being reduced memory footprint when running the model. This enables larger model to be run on devices which would not fit the full weights, and may increase model inference performance.

**Quick start**: Just use `--isq 4` (or 2, 3, 5, 6, 8) and mistral.rs will pick the best quantization for your hardware:
```
mistralrs run --isq 4 -m meta-llama/Llama-3.2-3B-Instruct
```

An API is exposed on the Python and Rust SDKs which provides the ability to dynamically re-ISQ models at runtime.

To set the ISQ type for individual layers, use a model [`topology`](TOPOLOGY.md).

> Note: 🔥 AFQ (affine) quantization is designed to be fast on **Metal** but is only supported on Metal.

## Automatic ISQ (just use a number!)
Instead of specifying a quantization type like `Q4K`, you can just pass an integer (2, 3, 4, 5, 6, or 8) and mistral.rs will automatically select the best quantization method for your platform.

On Metal, this uses fast AFQ quantization (for 2, 3, 4, 6, or 8 bits). On other platforms, it falls back to Q/K quantization.

```
mistralrs run --isq 4 -m meta-llama/Llama-3.2-3B-Instruct
```

## ISQ quantization types
- AFQ2 (*AFQ is only available on Metal*)
- AFQ3
- AFQ4
- AFQ6
- AFQ8
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q8_1 (*not available on CUDA*)
- Q2K
- Q3K
- Q4K
- Q5K
- Q6K
- Q8K  (*not available on CUDA*)
- HQQ4
- HQQ8
- FP8

```
mistralrs run --isq 4 -m meta-llama/Llama-3.2-3B-Instruct
```

ISQ supports two quantization strategies, selected automatically:

- **Immediate ISQ** (default): Weights are quantized per-layer during model construction. This is memory-efficient because only one layer's unquantized weights need to be in memory at a time. On discrete GPUs, weights are loaded to CPU, quantized, and moved to the device. On integrated/unified memory systems (e.g. Apple Silicon, NVIDIA Grace Blackwell), weights are loaded directly to the device.

- **Deferred ISQ**: The full model is loaded to CPU memory first, then all layers are quantized in a post-processing pass via `get_layers` and loaded to the correct device. This path is used when an imatrix file (`--imatrix`) or calibration file (`--calibration-file`) is provided, since these require either the full model or a forward pass before quantization can begin.

For Mixture of Expert models, a method called [MoQE](https://arxiv.org/abs/2310.02410) can be applied to only quantize MoE layers. This is configured via the ISQ "organization" parameter in all APIs. The following models support MoQE:
- [Phi 3.5 MoE](PHI3.5MOE.md)
- [DeepSeek V2](DEEPSEEKV2.md)
- [DeepSeek V3 / DeepSeek R1](DEEPSEEKV3.md)
- [GLM4-MoE](GLM4_MOE.md)
- [GLM4-MoE-Lite](GLM4_MOE_LITE.md)
- [Qwen 3 (MoE variants)](QWEN3.md)
- [Qwen3-VL-MoE (MoE variants)](QWEN3VL.md)

## Accuracy

Accuracy of ISQ can be measured by the performance degradation versus the unquantized model.
This is commonly measured with perplexity. Please see the [perplexity](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/perplexity/README.md) example.

To improve the accuracy of a model with ISQ, use an imatrix file. These can be found online (for example, on Hugging Face), and should be passed with the `--imatrix` flag for `plain` models. This will increase the accuracy of the quantization significantly and bring the ISQ quantization up to par with the GGUF counterpart.

Check out the [imatrix docs](IMATRIX.md).

## Python Example
```python
runner = Runner(
    which=Which.Plain(
        model_id="Qwen/Qwen3-0.6B",
    ),
    in_situ_quant="4",
)
```

## Rust Example
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/isq/main.rs).

```rust
let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
    .with_isq(IsqType::Q8_0)
    .with_logging()
    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
    .build()
    .await?;
```

## Server example
```
mistralrs serve --port 1234 --isq 4 -m mistralai/Mistral-7B-Instruct-v0.1
```

Or with a specific quantization type:
```
mistralrs serve --port 1234 --isq Q4K -m mistralai/Mistral-7B-Instruct-v0.1
```