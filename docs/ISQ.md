# In situ quantization

In situ quantization works by quantizing models inplace, with the chief benefit being reduced memory footprint when running the model. This enables larger model to be run on devices which would not fit the full weights, and may increase model inference performance.

An API is exposed on the Python and Rust APIs which provide the ability to dynamically re-ISQ models at runtime.

To set the ISQ type for individual layers, use a model [`topology`](TOPOLOGY.md).

> Note: 🔥 AFQ (affine) quantization is designed to be fast on **Metal** but is only supported on Metal.

## Automatic ISQ
Automatic ISQ is an opt-in feature that selects the most accurate and fastest quantization method for the platform.

If the provided ISQ value is a valid integer (one of 2, 3, 4, 5, 6, or 8), the best quantization type for the platform will be chosen.
Note that the fallback is always a Q/K quantization. On Metal, for 2, 3, 4, 6, or 8 bits, fast AFQ is used.

```
cargo run --release --features ... -- -i --isq 4 plain -m meta-llama/Llama-3.2-3B-Instruct
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
cargo run --release --features ... -- -i --isq 4 plain -m meta-llama/Llama-3.2-3B-Instruct
```

When using ISQ, it will automatically load ISQ-able weights into CPU memory before applying ISQ. The ISQ application process moves the weights to device memory. This process is implemented to avoid memory spikes from loading the model in full precision.

For Mixture of Expert models, a method called [MoQE](https://arxiv.org/abs/2310.02410) can be applied to only quantize MoE layers. This is configured via the ISQ "organization" parameter in all APIs. The following models support MoQE:
- [Phi 3.5 MoE](PHI3.5MOE.md)
- [DeepSeek V2](DEEPSEEKV2.md)
- [DeepSeek V3 / DeepSeek R1](DEEPSEEKV3.md)

## Accuracy

Accuracy of ISQ can be measured by the performance degradation versus the unquantized model.
This is commonly measured with perplexity. Please see the [perplexity](../mistralrs/examples/perplexity/README.md) example.

To improve the accuracy of a model with ISQ, use an imatrix file. These can be found online (for example, on Hugging Face), and should be passed with the `--imatrix` flag for `plain` models. This will increase the accuracy of the quantization significantly and bring the ISQ quantization up to par with the GGUF counterpart.

Check out the [imatrix docs](IMATRIX.md).

## Python Example
```python
runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    in_situ_quant="4",
)
```

## Rust Example
You can find this example [here](../mistralrs/examples/isq/main.rs).

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
cargo run --release --features "cuda flash-attn" -- --port 1234 --log output.txt --isq Q2K plain -m mistralai/Mistral-7B-Instruct-v0.1
```