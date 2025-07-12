# Quantization in mistral.rs

Mistral.rs supports the following quantization:
- GGUF/GGML
    - Q, K type
    - Supported in GGUF/GGML and GGUF/GGML adapter models
    - Supported in all plain/vision and adapter models
    - Imatrix quantization is supported
    - I quants coming!
    - CPU, CUDA, Metal (all supported devices)
    - 2, 3, 4, 5, 6, 8 bit
- GPTQ (convert with [this script](../scripts/convert_to_gptq.py))
    - Supported in all plain/vision and adapter models
    - CUDA only
    - 2, 3, 4, 8 bit
    - [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- AWQ (convert with [this script](../scripts/convert_awq_marlin.py))
    - Supported in all plain/vision and adapter models
    - CUDA only
    - 4 and 8 bit
    - [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- HQQ
    - Supported in all plain/vision and adapter models via ISQ
    - 4, 8 bit
    - CPU, CUDA, Metal (all supported devices)
- FP8
    - **Blockwise FP8**:
        - Loaded via `config.json` with `quant_method: "fp8"` and `weight_block_size` specified.
        - Supports loading pre-quantized F8E4M3 weights and F32 scales from safetensors.
        - Supports UQFF save/load.
        - Execution currently involves dequantization before GEMM.
        - CPU, CUDA, Metal (all supported devices for dequant path).
    - **Hybrid Precision FP8 (e.g., Llama 3.1 8B style)**:
        - Loaded via `config.json` with `quant_method: "cutlass_fp8_e4m3w_e5m2a"`.
        - Expects pre-quantized E4M3 weights (`.weight`) and BF16/FP16 per-channel scales (`.weight_scale`) in safetensors.
        - Activations are dynamically quantized to E5M2 per-token.
        - GEMM execution leverages cuBLASLt for high performance (CUDA only).
        - Requires compatible GPU (Hopper, Ada, or newer for best FP8 performance).
        - `activation_dtype` (e.g., "bf16", "f16") can be specified in `config.json` for this method to define the layer's interface and accumulation precision.
- BNB
    - Supported in all plain/vision and adapter models
    - bitsandbytes int8, fp4, nf4 support
- AFQ
    - 2, 3, 4, 6, 8 bit
    - 🔥 Designed to be fast on **Metal**!
    - Only supported on Metal.
- ISQ
    - Supported in all plain/vision and adapter models
    - Works on all supported devices
    - Automatic selection to use the fastest and most accurate method
    - Supports:
      - Q, K type GGUF quants
      - AFQ
      - HQQ
      - FP8
- MLX prequantized
    - Supported in all plain/vision and adapter models

## Using a GGUF quantized model
- Use the `gguf` (cli) / `GGUF` (Python) model selector
- Provide the GGUF file

```
cargo run --features cuda -- -i gguf -f my-gguf-file.gguf
```

## Using ISQ
See the [docs](ISQ.md)

```
cargo run --features cuda -- -i --isq 4 plain -m microsoft/Phi-3-mini-4k-instruct
```

## Using a GPTQ quantized model
- Provide the model ID for the GPTQ model
- Mistral.rs will automatically detect and use GPTQ quantization for plain and vision models!
- The [Marlin](https://github.com/IST-DASLab/marlin) kernel will automatically be used for 4-bit and 8-bit.

```
cargo run --features cuda --release -- -i plain -m kaitchup/Phi-3-mini-4k-instruct-gptq-4bit
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
cargo run --features ... --release -- -i plain -m mlx-community/Llama-3.8-1B-8bit
```