#include "codecs.cuh"

using namespace routed_gemv;

namespace {
// AFQ scales and biases share the activation dtype.
template <int Bits>
cudaError_t launch_afq(const Launch &a, int dtype, int mode, int lanes,
                       cudaStream_t stream) {
  switch (dtype) {
  case kF16:
    return launch_mode<Afq<Bits, __half>, __half, __half>(a, mode, lanes,
                                                         stream);
  case kBF16:
    return launch_mode<Afq<Bits, __nv_bfloat16>, __nv_bfloat16,
                       __nv_bfloat16>(a, mode, lanes, stream);
  default:
    return launch_mode<Afq<Bits, float>, float, float>(a, mode, lanes, stream);
  }
}
} // namespace

extern "C" int routed_gemv_afq(const Launch *a, int bits, int dtype, int mode,
                               int lanes, cudaStream_t stream) {
  switch (bits) {
  case 2:
    return launch_afq<2>(*a, dtype, mode, lanes, stream);
  case 3:
    return launch_afq<3>(*a, dtype, mode, lanes, stream);
  case 4:
    return launch_afq<4>(*a, dtype, mode, lanes, stream);
  case 6:
    return launch_afq<6>(*a, dtype, mode, lanes, stream);
  case 8:
    return launch_afq<8>(*a, dtype, mode, lanes, stream);
  default:
    return cudaErrorInvalidValue;
  }
}
