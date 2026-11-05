#include "codecs.cuh"

using namespace routed_gemv;

extern "C" int routed_gemv_nvfp4(const Launch *a, int unit, int dtype,
                                 int mode, int lanes, cudaStream_t stream) {
  return unit == 16
             ? launch_float<Nvfp4<16>>(*a, dtype, mode, lanes, stream)
             : launch_float<Nvfp4<32>>(*a, dtype, mode, lanes, stream);
}

extern "C" int routed_gemv_mxfp4(const Launch *a, int dtype, int mode,
                                 int lanes, cudaStream_t stream) {
  return launch_float<Mxfp4>(*a, dtype, mode, lanes, stream);
}
