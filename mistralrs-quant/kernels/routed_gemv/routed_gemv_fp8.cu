#include "codecs.cuh"

using namespace routed_gemv;

extern "C" int routed_gemv_mxfp8(const Launch *a, int dtype, int mode,
                                 int lanes, cudaStream_t stream) {
  return launch_float<Mxfp8>(*a, dtype, mode, lanes, stream);
}

extern "C" int routed_gemv_fp8(const Launch *a, int dtype, int mode, int lanes,
                               cudaStream_t stream) {
  return launch_float<Fp8>(*a, dtype, mode, lanes, stream);
}
