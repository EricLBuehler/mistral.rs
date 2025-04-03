#include <cassert>
#include <cstdint>
#include <cuda.h>

#define QUANTIZE_SCALAR(TYPENAME, RUST_NAME)                                   \
  extern "C" void quantize_scalar_fp8_##RUST_NAME(                             \
      const TYPENAME *d_in, uint8_t *d_out, float *s_out,                      \
      const uint32_t elem_count, cudaStream_t stream) {                        \
    assert(false);                                                             \
  }

QUANTIZE_SCALAR(float, f32)
QUANTIZE_SCALAR(uint16_t, bf16)
QUANTIZE_SCALAR(uint16_t, f16)