# Quantization in mistral.rs

Mistral.rs supports the following quantization:
- GGUF/GGML
    - Q, K type
    - Supported in GGUF/GGML and GGUF/GGML adapter models
    - I quants coming!
    - CPU, CUDA, Metal (all supported devices)
- GPTQ
    - Supported in all plain and adapter models
    - CUDA only
- ISQ
    - Q, K type GGUF quants
    - Supported in all plain and adapter models
    - I quants coming!
    - GPTQ quants coming!
    - CPU, CUDA, Metal (all supported devices)

## Using a GGUF quantized model
- Use the `gguf` (cli) / `GGUF` (Python) model selector
- Provide the GGUF file

```
cargo run --features cuda -- -i gguf -f my-gguf-file.gguf
```

## Using ISQ
See the [docs](ISQ.md)

```
cargo run --features cuda -- -i --isq Q4K plain -m microsoft/Phi-3-mini-4k-instruct -a phi3
```

## Using a GPTQ quantized model
- Use the `plain` (cli) / `Plain` (Python) model selector
- Provide the model ID for the GPTQ model
- Mistral.rs will automatically detect and use GPTQ quantization.

```
cargo run --features cuda -- -i plain -m kaitchup/Phi-3-mini-4k-instruct-gptq-4bit -a phi3
```