# In situ quantization (ISQ)

In situ quantization (ISQ) quantizes model weights in place as they are loaded, so the full unquantized model never needs to fit in memory. Using with I/O and parallel pipelining, this means you can load and run a model that is larger than the total amount of RAM (CPU or GPU) on your system.

If the quantized weights are small enough to fit even though the original weights would not, you can still run the model! Like all quantization, ISQ may also increase inference performance due to reduced memory bandwidth pressure.

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
- F8Q8

```
mistralrs run --isq 4 -m meta-llama/Llama-3.2-3B-Instruct
```

For Mixture of Expert models, a method called [MoQE](https://arxiv.org/abs/2310.02410) can be applied to only quantize MoE layers. This is configured via the ISQ "organization" parameter in all APIs. The following models support MoQE:
- [Phi 3.5 MoE](PHI3.5MOE.md)
- [DeepSeek V2](DEEPSEEKV2.md)
- [DeepSeek V3 / DeepSeek R1](DEEPSEEKV3.md)
- [GLM4-MoE](GLM4_MOE.md)
- [GLM4-MoE-Lite](GLM4_MOE_LITE.md)
- [Qwen 3 (MoE variants)](QWEN3.md)
- [Qwen3-VL-MoE (MoE variants)](QWEN3VL.md)

## Quantization strategies

ISQ supports two quantization strategies, selected automatically based on your configuration:

### Immediate ISQ (default)

Immediate ISQ quantizes each weight as it is loaded during model construction rather than loading all weights first, then quantizing. This means only a small number of unquantized weight tensors need to be in CPU memory at any given time, enabling ISQ for models that would not otherwise fit in memory.

Quantization is parallelized across a thread pool on all devices. Multiple weights are quantized concurrently on CPU during loading, then moved to the target device. The number of threads depends on the ISQ type: GGML types (Q2K-Q8K) use all available CPU threads, while GPU-quantized types (HQQ, AFQ) use a single thread since the GPU work is serialized by a guard.

Set `MISTRALRS_ISQ_SINGLETHREAD=1` to force single-threaded quantization.

### Deferred ISQ

Deferred ISQ loads the full unquantized model into CPU memory first, then quantizes all weights in parallel in a post-processing pass. This path is used when an imatrix file (`--imatrix`) or calibration file (`--calibration-file`) is provided, since these require access to the full model or a forward pass before quantization can begin. Peak CPU memory usage is higher than immediate ISQ because the entire unquantized model must fit in memory during the quantization pass.

## Accuracy

Accuracy of ISQ can be measured by the performance degradation versus the unquantized model.
This is commonly measured with perplexity. Please see the [perplexity](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/perplexity/main.rs) example.

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