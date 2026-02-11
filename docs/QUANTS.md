# Quantization in mistral.rs

Mistral.rs supports the following quantization:
- ⭐ ISQ ([read more detail](ISQ.md))
    - Supported in all plain/vision and adapter models
    - Works on all supported devices
    - Automatic selection to use the fastest and most accurate method
    - Supports:
      - Q, K type GGUF quants
      - AFQ
      - HQQ
      - FP8
      - F8Q8
- GGUF/GGML
    - Q, K type
    - Supported in GGUF/GGML and GGUF/GGML adapter models
    - Supported in all plain/vision and adapter models
    - Imatrix quantization is supported
    - I quants coming!
    - CPU, CUDA, Metal (all supported devices)
    - 2, 3, 4, 5, 6, 8 bit
- GPTQ (convert with [this script](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/convert_to_gptq.py))
    - Supported in all plain/vision and adapter models
    - CUDA only
    - 2, 3, 4, 8 bit
    - [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- AWQ (convert with [this script](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/convert_awq_marlin.py))
    - Supported in all plain/vision and adapter models
    - CUDA only
    - 4 and 8 bit
    - [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- HQQ
    - Supported in all plain/vision and adapter models via ISQ
    - 4, 8 bit
    - CPU, CUDA, Metal (all supported devices)
- FP8
    - Supported in all plain/vision and adapter models
    - CPU, CUDA, Metal (all supported devices)
- BNB
    - Supported in all plain/vision and adapter models
    - bitsandbytes int8, fp4, nf4 support
- AFQ
    - 2, 3, 4, 6, 8 bit
    - 🔥 Designed to be fast on **Metal**!
    - Only supported on Metal.
- MLX prequantized
    - Supported in all plain/vision and adapter models

## Using a GGUF quantized model
- Use the `gguf` (cli) / `GGUF` (Python) model selector
- Provide the GGUF file

```
mistralrs run --format gguf -f my-gguf-file.gguf
```

## Using ISQ
See the [docs](ISQ.md)

```
mistralrs run --isq 4 -m microsoft/Phi-3-mini-4k-instruct
```

## Using a GPTQ quantized model
- Provide the model ID for the GPTQ model
- Mistral.rs will automatically detect and use GPTQ quantization for plain and vision models!
- The [Marlin](https://github.com/IST-DASLab/marlin) kernel will automatically be used for 4-bit and 8-bit.

```
mistralrs run -m kaitchup/Phi-3-mini-4k-instruct-gptq-4bit
```

You can create your own GPTQ model using [`scripts/convert_to_gptq.py`][../scripts/convert_to_gptq.py]:
```
pip install gptqmodel transformers datasets

python3 scripts/convert_to_gptq.py --src path/to/model --dst output/model/path --bits 4
```

## Using a MLX prequantized model (on Metal)
- Provide the model ID for the MLX prequantized model
- Mistral.rs will automatically detect and use quantization for plain and vision models!
- Specialized kernels will be used to accelerate inference!

```
mistralrs run -m mlx-community/Llama-3.8-1B-8bit
```