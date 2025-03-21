# Quantization in mistral.rs

Mistral.rs supports the following quantization:
- GGUF/GGML
    - Q, K type
    - Supported in GGUF/GGML and GGUF/GGML adapter models
    - Supported in all plain and adapter models
    - Imatrix quantization is supported
    - I quants coming!
    - CPU, CUDA, Metal (all supported devices)
    - 2, 3, 4, 5, 6, 8 bit
- GPTQ
    - Supported in all plain and adapter models
    - CUDA only
    - 2, 3, 4, 8 bit
    - [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- HQQ
    - Supported in all plain and adapter models via ISQ
    - 4, 8 bit
    - CPU, CUDA, Metal (all supported devices)
- FP8
    - Supported in all plain and adapter models
    - CPU, CUDA, Metal (all supported devices)
- BNB
    - Supported in all plai models
    - bitsandbytes int8, fp4, nf4 support
- ISQ
    - Q, K type GGUF quants
    - Supported in all plain and adapter models
    - HQQ quants
    - FP8
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
- The [Marlin](https://github.com/IST-DASLab/marlin) kernel will automatically be used in 4-bit and 8-bit.

```
cargo run --features cuda -- -i plain -m kaitchup/Phi-3-mini-4k-instruct-gptq-4bit -a phi3
```