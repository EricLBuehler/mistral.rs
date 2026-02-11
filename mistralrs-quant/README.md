# `mistralrs-quant`

An advanced and highly diverse set of quantization techniques. This crate supports both quantization and optimized inference.

It has grown beyon simply quantization and is used by `mistral.rs` to power:
- ISQ
- Imatrix collection
- General quantization features
- Specific CUDA and Metal features
- cuBLASlt integration

Currently supported:
- AFQ: `GgufMatMul` (2-8 bit quantization optimized for Metal and compatible with MLX)
- GGUF: `GgufMatMul` (2-8 bit quantization, with imatrix)
- Gptq/Awq: `GptqAwqLayer` (with CUDA marlin kernel)
- Hqq: `HqqLayer` (4, 8 bit quantization)
- FP8: `FP8Linear`
- F8Q8: `F8Q8Linear`
- Unquantized (used for ISQ): `UnquantLinear`
- Bnb: `BnbLinear` (int8, fp4, nf4)

Some kernels are copied or based on implementations in:
- https://github.com/vllm-project/vllm
- https://github.com/mobiusml/hqq
- https://github.com/bitsandbytes-foundation/bitsandbytes
