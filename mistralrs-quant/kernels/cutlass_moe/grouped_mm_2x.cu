// CUTLASS 2.x grouped GEMM (Sm80 tensor-op, bf16 in / f32 accumulate / bf16 out) for the MoE
// fallback path. One GEMM group per expert; problem sizes and pointers live in device memory
// (GroupScheduleMode::kDeviceOnly) so the forward never syncs to host.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

namespace {

using Element = cutlass::bfloat16_t;
using ElementAccum = float;

// A: expert-sorted activations [M_e, K] row-major. B: expert weights stored [N, K] row-major,
// consumed as a [K, N] column-major view with ld = K. D: [M_e, N] row-major.
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::RowMajor, ElementAccum,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, ElementAccum,
                                                 ElementAccum>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

int threadblock_count_for_device() {
  static int cached = -1;
  if (cached < 0) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           device_id);
    int max_active = GemmGrouped::maximum_active_blocks();
    cached = (max_active > 0 ? max_active : 1) * sm_count;
  }
  return cached;
}

}  // namespace

// Returns 0 on success, a nonzero cutlass/cuda status otherwise.
extern "C" int launch_cutlass_moe_grouped_gemm_2x_bf16(
    const void** a_ptrs, const void** b_ptrs, void** d_ptrs,
    const int32_t* problem_sizes, int problem_count, int64_t* lda,
    int64_t* ldb, int64_t* ldd, void* workspace, size_t workspace_size,
    cudaStream_t stream) {
  // GemmCoord is three ints {m, n, k}; problem_sizes rows are laid out identically.
  auto* problem_sizes_dev = reinterpret_cast<cutlass::gemm::GemmCoord*>(
      const_cast<int32_t*>(problem_sizes));

  typename GemmGrouped::Arguments args(
      problem_sizes_dev, problem_count, threadblock_count_for_device(),
      {ElementAccum(1), ElementAccum(0)},
      reinterpret_cast<Element**>(const_cast<void**>(a_ptrs)),
      reinterpret_cast<Element**>(const_cast<void**>(b_ptrs)),
      reinterpret_cast<Element**>(d_ptrs),
      reinterpret_cast<Element**>(d_ptrs), lda, ldb, ldd, ldd,
      /*host_problem_sizes=*/nullptr);

  GemmGrouped gemm;
  cutlass::Status status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.run(stream);
  return static_cast<int>(status);
}

extern "C" size_t cutlass_moe_grouped_gemm_2x_workspace_size(int problem_count) {
  return GemmGrouped::get_workspace_size(typename GemmGrouped::Arguments(
      nullptr, problem_count, 1, {ElementAccum(1), ElementAccum(0)}, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
}
