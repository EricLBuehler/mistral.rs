# In situ quantization

In situ quantization works by quantizing non GGUF or GGML models in-place. This allows you to take advantage of flash attention, and reduces memory footprint when running the model. Currently, all layers which would be `Linear` are able to be quantized.

An API is exposed on the Python and Rust APIs which provide the ability to dynamically re-ISQ models at runtime.

To set the ISQ type for individual layers, use a model [`topology`](TOPOLOGY.md).

## ISQ quantization types
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

When using ISQ, it will automatically load ISQ-able weights into CPU memory before applying ISQ. The ISQ application process moves the weights to device memory. This process is implemented to avoid memory spikes from loading the model in full precision.

**Fallback rules for GGUF quantization**
If a tensor cannot be quantized, the fallback process is as follows:
1) If using a `K` quant, fallback to a similar `Q` quant.
2) If that is not possible, use `F32` as the data type.

## Python Example
```python
runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    in_situ_quant="Q4K",
)
```

## Rust Example
You can find this example [here](../mistralrs/examples/isq/main.rs).

```rust
let pipeline = loader.load_model_from_hf(
    None,
    TokenSource::CacheToken,
    None,
    &Device::cuda_if_available(0)?,
    false,
    DeviceMapMetadata::dummy(),
    Some(GgmlDType::Q4K),
    None, // No PagedAttention yet
)?;
```

## Server example
```
cargo run --release --features "cuda flash-attn" -- --port 1234 --log output.txt --isq Q2K plain -m mistralai/Mistral-7B-Instruct-v0.1 -a mistral
```