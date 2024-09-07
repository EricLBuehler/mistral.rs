# `mistralrs-quant`

Quantization techniques for mistral.rs. This implements a common trait for all quantization methods to implement for ease of extension and development.

Currently supported:
- GGUF: `GgufMatMul`
- Gptq: `GptqLayer`
- Hqq: `HqqLayer`
- Unquantized (used for ISQ): `UnquantLinear`

Some kernels are copied or based on implementations in:
- https://github.com/vllm-project/vllm
- https://github.com/mobiusml/hqq
