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
template <typename ThreadblockShape, typename WarpShape>
using GroupedKernelFor = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::RowMajor, ElementAccum,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
    WarpShape, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, ElementAccum,
                                                 ElementAccum>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

// Tile configs by average rows per expert group; large-M wants wide tiles, small-M wants
// less predicated waste per group.
using GemmLarge = cutlass::gemm::device::GemmGrouped<GroupedKernelFor<
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>>;
using GemmMedium = cutlass::gemm::device::GemmGrouped<GroupedKernelFor<
    cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>>>;
using GemmSmall = cutlass::gemm::device::GemmGrouped<GroupedKernelFor<
    cutlass::gemm::GemmShape<32, 128, 32>, cutlass::gemm::GemmShape<32, 32, 32>>>;

template <typename Gemm>
int threadblock_count_for_device() {
  static int cached = -1;
  if (cached < 0) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           device_id);
    int max_active = Gemm::maximum_active_blocks();
    cached = (max_active > 0 ? max_active : 1) * sm_count;
  }
  return cached;
}

template <typename Gemm>
int run_grouped(const void** a_ptrs, const void** b_ptrs, void** d_ptrs,
                const int32_t* problem_sizes, int problem_count, int64_t* lda,
                int64_t* ldb, int64_t* ldd, void* workspace,
                cudaStream_t stream) {
  auto* problem_sizes_dev = reinterpret_cast<cutlass::gemm::GemmCoord*>(
      const_cast<int32_t*>(problem_sizes));

  typename Gemm::Arguments args(
      problem_sizes_dev, problem_count, threadblock_count_for_device<Gemm>(),
      {ElementAccum(1), ElementAccum(0)},
      reinterpret_cast<Element**>(const_cast<void**>(a_ptrs)),
      reinterpret_cast<Element**>(const_cast<void**>(b_ptrs)),
      reinterpret_cast<Element**>(d_ptrs),
      reinterpret_cast<Element**>(d_ptrs), lda, ldb, ldd, ldd,
      /*host_problem_sizes=*/nullptr);

  Gemm gemm;
  cutlass::Status status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.run(stream);
  return static_cast<int>(status);
}

}  // namespace

// Returns 0 on success, a nonzero cutlass/cuda status otherwise. GemmCoord is three ints
// {m, n, k}; problem_sizes rows are laid out identically. tile_cfg: 0 = large (avg M >= 96),
// 1 = medium, 2 = small (avg M < 48).
extern "C" int launch_cutlass_moe_grouped_gemm_2x_bf16(
    const void** a_ptrs, const void** b_ptrs, void** d_ptrs,
    const int32_t* problem_sizes, int problem_count, int64_t* lda,
    int64_t* ldb, int64_t* ldd, void* workspace, size_t workspace_size,
    int tile_cfg, cudaStream_t stream) {
  (void)workspace_size;
  switch (tile_cfg) {
    case 0:
      return run_grouped<GemmLarge>(a_ptrs, b_ptrs, d_ptrs, problem_sizes,
                                    problem_count, lda, ldb, ldd, workspace,
                                    stream);
    case 1:
      return run_grouped<GemmMedium>(a_ptrs, b_ptrs, d_ptrs, problem_sizes,
                                     problem_count, lda, ldb, ldd, workspace,
                                     stream);
    default:
      return run_grouped<GemmSmall>(a_ptrs, b_ptrs, d_ptrs, problem_sizes,
                                    problem_count, lda, ldb, ldd, workspace,
                                    stream);
  }
}

extern "C" size_t cutlass_moe_grouped_gemm_2x_workspace_size(int problem_count) {
  size_t ws = GemmLarge::get_workspace_size(typename GemmLarge::Arguments(
      nullptr, problem_count, 1, {ElementAccum(1), ElementAccum(0)}, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  size_t ws_m = GemmMedium::get_workspace_size(typename GemmMedium::Arguments(
      nullptr, problem_count, 1, {ElementAccum(1), ElementAccum(0)}, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  size_t ws_s = GemmSmall::get_workspace_size(typename GemmSmall::Arguments(
      nullptr, problem_count, 1, {ElementAccum(1), ElementAccum(0)}, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  if (ws_m > ws) ws = ws_m;
  if (ws_s > ws) ws = ws_s;
  return ws;
}
