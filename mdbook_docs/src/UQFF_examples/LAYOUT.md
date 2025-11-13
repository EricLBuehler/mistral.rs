# UQFF internal structure

The following describes the exact memory layout of UQFF tensors of version 0.1.0.

## ToC
- [GGUF quantization](#gguf-quantization)
- [HQQ quantization](#hqq-quantization)
- [Uquantized layers](#unquantized-layers)
- [FP8 layers](#fp8-layers)
- [Standard tensors](#standard-tensors)


## GGUF quantization

| ID | Element type | Endianness |
| -------- | -------- | -------- |
| UQFF version | u32 | little endian  |
| ISQ type (0) | u8 | little endian  |
| Tensor data length in bytes | u32 | little endian  |
| Whether bias data is included (boolean) | u8 | little endian  |
| Quantized dtype | u32 | little endian  |
| Num shape dims | u32 | little endian  |
| **Array** quantized weight shape dims | u32 | little endian  |
| **Array** quantized weight data | u8 | little endian  |
| **[Optional]** **Array** Bias tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |

## Unquantized layers
| ID | Element type | Endianness |
| -------- | -------- | -------- |
| UQFF version | u32 | little endian  |
| ISQ type (1) | u8 | little endian  |
| Whether bias data is included (boolean) | u8 | little endian  |
| **Array** Weight tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |
| **[Optional]** **Array** Bias tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |

## FP8 layers
| ID | Element type | Endianness |
| -------- | -------- | -------- |
| UQFF version | u32 | little endian  |
| ISQ type (1) | u8 | little endian  |
| Whether bias data is included (boolean) | u8 | little endian  |
| **Array** Weight tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |
| Dequant W scalar | f32 | little endian
| Dequant X scalar | f32 | little endian
| Quant scalar | f32 | little endian
| Quantization type | u32 | little endian
| **[Optional]** **Array** Bias tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |

## HQQ quantization
| ID | Element type | Endianness |
| -------- | -------- | -------- |
| UQFF version | u32 | little endian  |
| ISQ type (2) | u8 | little endian  |
| Whether bias data is included (boolean) | u8 | little endian  |
| **Array** Q weight, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors) |
| **Array** Q scale, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors) |
| **Array** Q zeroes, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors) |
| Dequant weight num shape dims | u32 | little endian  |
| **Array** dequant weight shape dims | u32 | little endian  |
| CFG bits | u8 | little endian  |
| CFG group size | u32 | little endian  |
| CFG axis | u8 | little endian  |
| CFG optimization steps (0 means `Option::None` for now) | u32 | little endian  |
| CFG round zeroes (boolean) | u8 | little endian  |
| CFG channel wise (boolean) | u8 | little endian  |

## FP8 layers
| ID | Element type | Endianness |
| -------- | -------- | -------- |
| UQFF version | u32 | little endian  |
| ISQ type (3) | u8 | little endian  |
| Whether bias data is included (boolean) | u8 | little endian  |
| **Array** Weight tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  | 
| Dequant scale W | f32 | little endian  |
| Dequant scale X | f32 | little endian  |
| Quant scale | f32 | little endian  |
| Layer dtype | u32 | little endian  |
| **[Optional]** **Array** Bias tensor data, see [docs](#standard-tensors) | See [docs](#standard-tensors) | See [docs](#standard-tensors)  |

## Standard tensors
| ID | Element type | Endianness |
| -------- | -------- | -------- |
| Tensor data length in bytes | u32 | little endian  |
| Tensor dtype | u32 | little endian  |
| Num shape dims | u32 | little endian  |
| **Array** shape dims | u32 | little endian  |
| **Array** flattened (contiguous) tensor data | u8 | little endian  |

