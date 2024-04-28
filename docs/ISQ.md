# In situ quantization

In situ quantization works by quantizing non GGUF or GGML models in-place. This allows you to take advantage of flash attention, and reduces memory footprint when running the model. Currently, all layers which would be `Linear` are able to be quantized. An API is exposed on the Python and Rust APIs which provide the ability to dynamically re-ISQ models. 

Possible values for ISQ quantization:
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q8_1
- Q2K
- Q3K
- Q4K
- Q5K
- Q6K
- Q8K

When using ISQ, it will automatically load non ISQ-able weights into CPU memory before applying ISQ. The ISQ application process moves the weights to device memory. This process is implemented to avoid memory spikes from loading the model in full precision.

## Python Example
```python
runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    ),
    in_situ_quant="Q4K",
)
```

## Rust Example
```rust
let pipeline = loader.load_model(
    None,
    TokenSource::CacheToken,
    None,
    &Device::cuda_if_available(0)?,
    false,
    DeviceMapMetadata::dummy(),
    Some(GgmlDType::Q4K),
)?;
```

## Server example
```
cargo run --release --features "cuda flash-attn" -- --port 1234 --log output.txt --isq Q2K plain -m mistralai/Mistral-7B-Instruct-v0.1 -a mistral
```