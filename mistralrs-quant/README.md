# `mistralrs-quant`

An advanced and highly diverse set of quantization techniques. This crate supports both quantization and optimized inference, making it truly unique in its breadth of useability.

It is used by `mistral.rs` to power ISQ, imatrix collection, and general quantization features.

Currently supported:
- GGUF: `GgufMatMul`(2-8 bit quantization, with imatrix)
- Gptq: `GptqLayer`(with CUDA marlin kernel)
- Hqq: `HqqLayer` (4, 8 bit quantization)
- FP8: `FP8Linear`(optimized on CUDA)
- Unquantized (used for ISQ): `UnquantLinear`
- Bnb: `BnbLinear` (int8, fp4, nf4)

Some kernels are copied or based on implementations in:
- https://github.com/vllm-project/vllm
- https://github.com/mobiusml/hqq
- https://github.com/bitsandbytes-foundation/bitsandbytes
