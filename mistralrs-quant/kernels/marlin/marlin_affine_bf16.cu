#include "marlin_kernel.cuh"

namespace {

template <marlin::ScalarTypeID weight_type, int bits>
int launch_affine(const void *inputs, const void *weight, void *scales,
                  void *offsets, void *output, int m, int k, int n,
                  int group_size, void *workspace, int64_t stream) {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (group_size != 16 && group_size != 32) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  try {
    marlin_matmul<nv_bfloat16, weight_type, true, bits, true>(
        inputs, weight, scales, offsets, output, m, k, n, workspace, group_size,
        stream);
    return static_cast<int>(cudaGetLastError());
  } catch (...) {
    return -1;
  }
}

} // namespace

extern "C" int marlin_affine_u4_bf16(const void *inputs, const void *weight,
                                     void *scales, void *offsets, void *output,
                                     int m, int k, int n, int group_size,
                                     void *workspace, int64_t stream) {
  return launch_affine<marlin::ScalarTypeID::kU4, 4>(
      inputs, weight, scales, offsets, output, m, k, n, group_size, workspace,
      stream);
}

extern "C" int marlin_affine_u8_bf16(const void *inputs, const void *weight,
                                     void *scales, void *offsets, void *output,
                                     int m, int k, int n, int group_size,
                                     void *workspace, int64_t stream) {
  return launch_affine<marlin::ScalarTypeID::kU8, 8>(
      inputs, weight, scales, offsets, output, m, k, n, group_size, workspace,
      stream);
}
