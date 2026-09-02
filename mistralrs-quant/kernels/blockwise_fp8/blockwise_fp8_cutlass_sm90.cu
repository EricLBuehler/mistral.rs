#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "blockwise_fp8_cutlass_sm90.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

// Adapted from vLLM's SM90 blockwise scaled GEMM dispatch (Apache-2.0).
namespace mistralrs::fp8 {

using namespace cute;

constexpr int kActivationGroupSize = 128;
constexpr int kThreadsPerGroup = 16;
constexpr int kMaxGroupsPerBlock = 16;
constexpr int kSmallMThreshold = 32;
constexpr int kSmallMOutputTileN = 128;
// CUTLASS's blockwise builder only accepts auto-carveout policies; this yields eight stages for the 128x16 tile.
constexpr int kSmallMCooperativeCarveoutBytes = 64 * 1024;
constexpr float kFp8Max = 448.0f;
constexpr float kScaleEpsilon = 1.0e-10f;

template <class Output, int ScaleM, int ScaleN, int ScaleK,
          class MmaTileShape, class ClusterShape, class EpilogueScheduler,
          class MainloopScheduler, bool SwapAB = false,
          int MainloopCarveoutBytes = 0,
          class TileScheduler = void>
struct BlockwiseGemm {
  static constexpr bool kSwapAB = SwapAB;

  using ElementAB = cutlass::float_e4m3_t;
  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutAT = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutBT = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = Output;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutDT = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = LayoutD;
  using LayoutCT = LayoutDT;
  static constexpr int AlignmentC = AlignmentD;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = conditional_t<
      SwapAB,
      cutlass::detail::Sm90BlockwiseScaleConfig<
          ScaleM, ScaleN, ScaleK, cute::GMMA::Major::K,
          cute::GMMA::Major::MN>,
      cutlass::detail::Sm90BlockwiseScaleConfig<
          ScaleM, ScaleN, ScaleK, cute::GMMA::Major::MN,
          cute::GMMA::Major::K>>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC,
          conditional_t<SwapAB, LayoutCT, LayoutC>, AlignmentC, ElementD,
          conditional_t<SwapAB, LayoutDT, LayoutD>, AlignmentD,
          EpilogueScheduler, EpilogueOperation>::CollectiveOp;

  using MainloopStageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) +
      MainloopCarveoutBytes>;

  using CollectiveMainloop = conditional_t<
      SwapAB,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB, tuple<LayoutBT, LayoutSFA>,
          AlignmentB, ElementA, tuple<LayoutAT, LayoutSFB>, AlignmentA,
          ElementAccumulator, MmaTileShape, ClusterShape,
          MainloopStageCount, MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, tuple<LayoutA, LayoutSFA>,
          AlignmentA, ElementB, tuple<LayoutB, LayoutSFB>, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          MainloopStageCount, MainloopScheduler>::CollectiveOp>;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileScheduler>;
};

template <typename Output>
using LargeMGemm = BlockwiseGemm<
    Output, 1, 128, 128, Shape<_128, _128, _128>, Shape<_1, _2, _1>,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>;

template <typename Output>
using SmallMGemm = BlockwiseGemm<
    Output, 128, 1, 128, Shape<_128, _16, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum, true>;

template <typename Output>
using SmallMCooperativeGemm = BlockwiseGemm<
    Output, 128, 1, 128, Shape<_128, _16, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum,
    true, kSmallMCooperativeCarveoutBytes>;

template <typename Output>
using SmallMStreamKGemm = BlockwiseGemm<
    Output, 128, 1, 128, Shape<_128, _16, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum,
    true, kSmallMCooperativeCarveoutBytes,
    cutlass::gemm::StreamKScheduler>;

bool use_small_m_gemm(int m) {
  return m < kSmallMThreshold || m % 4 != 0;
}

bool use_stream_k_gemm(int n, int sm_count) {
  const int output_tiles = (n + kSmallMOutputTileN - 1) / kSmallMOutputTileN;
  return output_tiles * 2 < sm_count;
}

bool use_cooperative_gemm(int n, int sm_count) {
  const int output_tiles = (n + kSmallMOutputTileN - 1) / kSmallMOutputTileN;
  return output_tiles >= sm_count * 2;
}

template <typename Gemm>
typename Gemm::Kernel::Arguments make_arguments(
    const void *activation, const void *weight, const float *activation_scales,
    const float *weight_scales, void *output, int m, int n, int k,
    int sm_count) {
  using Kernel = typename Gemm::Kernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, Gemm::kSwapAB ? cute::make_shape(n, m, 1)
                              : cute::make_shape(m, n, 1));

  auto problem_shape = Gemm::kSwapAB ? cute::make_shape(n, m, k, 1)
                                     : cute::make_shape(m, n, k, 1);
  auto layout_sfa = Gemm::kSwapAB
                        ? Gemm::ScaleConfig::tile_atom_to_shape_SFA(
                              cute::make_shape(n, m, k, 1))
                        : Gemm::ScaleConfig::tile_atom_to_shape_SFA(
                              cute::make_shape(m, n, k, 1));
  auto layout_sfb = Gemm::kSwapAB
                        ? Gemm::ScaleConfig::tile_atom_to_shape_SFB(
                              cute::make_shape(n, m, k, 1))
                        : Gemm::ScaleConfig::tile_atom_to_shape_SFB(
                              cute::make_shape(m, n, k, 1));

  auto activation_ptr = static_cast<const ElementAB *>(activation);
  auto weight_ptr = static_cast<const ElementAB *>(weight);
  typename Kernel::MainloopArguments mainloop{};
  mainloop.layout_SFA = layout_sfa;
  mainloop.layout_SFB = layout_sfb;
  if constexpr (Gemm::kSwapAB) {
    mainloop.ptr_A = weight_ptr;
    mainloop.dA = stride_b;
    mainloop.ptr_B = activation_ptr;
    mainloop.dB = stride_a;
    mainloop.ptr_SFA = weight_scales;
    mainloop.ptr_SFB = activation_scales;
  } else {
    mainloop.ptr_A = activation_ptr;
    mainloop.dA = stride_a;
    mainloop.ptr_B = weight_ptr;
    mainloop.dB = stride_b;
    mainloop.ptr_SFA = activation_scales;
    mainloop.ptr_SFB = weight_scales;
  }

  auto output_ptr = static_cast<ElementD *>(output);
  typename Kernel::EpilogueArguments epilogue{{}, output_ptr, stride_c,
                                               output_ptr, stride_c};
  cutlass::KernelHardwareInfo hardware{0, sm_count};
  return {cutlass::gemm::GemmUniversalMode::kGemm, problem_shape, mainloop,
          epilogue, hardware, {}};
}

template <typename Gemm>
int prepare_kernel() {
  using Kernel = typename Gemm::Kernel;
  if constexpr (Kernel::SharedStorageSize >= 48 << 10) {
    cudaError_t error = cudaFuncSetAttribute(
        cutlass::device_kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize);
    if (error != cudaSuccess) {
      return -static_cast<int>(error);
    }
  }
  return 0;
}

template <typename Gemm>
int workspace_size(int m, int n, int k, int sm_count, size_t *bytes) {
  if (bytes == nullptr || m <= 0 || n <= 0 || k <= 0 || sm_count <= 0) {
    return -static_cast<int>(cudaErrorInvalidValue);
  }
  auto args = make_arguments<Gemm>(nullptr, nullptr, nullptr, nullptr, nullptr,
                                   m, n, k, sm_count);
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<typename Gemm::Kernel>;
  *bytes = Adapter::get_workspace_size(args);
  return static_cast<int>(Adapter::can_implement(args));
}

template <typename Gemm>
int launch_gemm(const void *activation, const void *weight,
                const float *activation_scales, const float *weight_scales,
                void *output, int m, int n, int k, void *workspace,
                size_t workspace_bytes, int sm_count, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || activation_scales == nullptr ||
      weight_scales == nullptr || output == nullptr || m <= 0 || n <= 0 ||
      k <= 0 || sm_count <= 0) {
    return -static_cast<int>(cudaErrorInvalidValue);
  }
  auto args = make_arguments<Gemm>(activation, weight, activation_scales,
                                   weight_scales, output, m, n, k, sm_count);
  using Kernel = typename Gemm::Kernel;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto status = Adapter::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  const size_t required_workspace = Adapter::get_workspace_size(args);
  if (required_workspace > workspace_bytes ||
      (required_workspace != 0 && workspace == nullptr)) {
    return static_cast<int>(cutlass::Status::kErrorWorkspaceNull);
  }
  status = Kernel::initialize_workspace(args, workspace, stream, nullptr);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  auto params = Kernel::to_underlying_arguments(args, workspace);
  return static_cast<int>(Adapter::run(params, stream, nullptr, true));
}

template <typename T>
__global__ void quantize_activation_kernel(const T *__restrict__ input,
                                           __nv_fp8_e4m3 *__restrict__ output,
                                           float *__restrict__ scales,
                                           int total_groups,
                                           int groups_in_block,
                                           int groups_per_row,
                                           int rows) {
  const int group_in_block = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;
  const int group = blockIdx.x * groups_in_block + group_in_block;
  const int offset = group * kActivationGroupSize + lane * 8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  if (group >= total_groups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  union InputPack {
    uint4 packed;
    T values[8];
  } values;
  values.packed = reinterpret_cast<const uint4 *>(input + offset)[0];

  float local_max = kScaleEpsilon;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(values.values[i])));
  }
  const unsigned int mask = (threadIdx.x & 16) == 0 ? 0x0000ffff : 0xffff0000;
#pragma unroll
  for (int delta = 8; delta >= 1; delta /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, delta));
  }
  const float scale = local_max / kFp8Max;
  if (lane == 0) {
    const int row = group / groups_per_row;
    const int column_group = group % groups_per_row;
    scales[column_group * rows + row] = scale;
  }

  union OutputPack {
    uint64_t packed;
    __nv_fp8x2_storage_t pairs[4];
  } quantized;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 pair = make_float2(
        fminf(fmaxf(static_cast<float>(values.values[2 * i]) / scale,
                    -kFp8Max),
              kFp8Max),
        fminf(fmaxf(static_cast<float>(values.values[2 * i + 1]) / scale,
                    -kFp8Max),
              kFp8Max));
    quantized.pairs[i] =
        __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
  }
  reinterpret_cast<uint64_t *>(output + offset)[0] = quantized.packed;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
int launch_quantize(const T *input, __nv_fp8_e4m3 *output, float *scales,
                    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 ||
      cols <= 0 || cols % kActivationGroupSize != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint64_t) != 0 ||
      reinterpret_cast<uintptr_t>(scales) % alignof(float) != 0) {
    return -static_cast<int>(cudaErrorInvalidValue);
  }
  const int groups_per_row = cols / kActivationGroupSize;
  if (rows > std::numeric_limits<int>::max() / groups_per_row) {
    return -static_cast<int>(cudaErrorInvalidValue);
  }
  const int total_groups = rows * groups_per_row;
  const int group_count =
      total_groups < kMaxGroupsPerBlock ? total_groups : kMaxGroupsPerBlock;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1 + (total_groups - 1) / group_count);
  config.blockDim = dim3(group_count * kThreadsPerGroup);
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaError_t error = cudaLaunchKernelEx(
      &config, quantize_activation_kernel<T>, input, output, scales,
      total_groups, group_count, groups_per_row, rows);
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

template <typename Output>
int dispatch_workspace_size(int m, int n, int k, int sm_count, size_t *bytes) {
  if (use_small_m_gemm(m)) {
    if (use_stream_k_gemm(n, sm_count)) {
      return workspace_size<SmallMStreamKGemm<Output>>(m, n, k, sm_count,
                                                       bytes);
    }
    if (use_cooperative_gemm(n, sm_count)) {
      return workspace_size<SmallMCooperativeGemm<Output>>(m, n, k, sm_count,
                                                          bytes);
    }
    return workspace_size<SmallMGemm<Output>>(m, n, k, sm_count, bytes);
  }
  return workspace_size<LargeMGemm<Output>>(m, n, k, sm_count, bytes);
}

template <typename Output>
int dispatch_gemm(const void *activation, const void *weight,
                  const float *activation_scales, const float *weight_scales,
                  void *output, int m, int n, int k, void *workspace,
                  size_t workspace_bytes, int sm_count, cudaStream_t stream) {
  if (use_small_m_gemm(m)) {
    if (use_stream_k_gemm(n, sm_count)) {
      return launch_gemm<SmallMStreamKGemm<Output>>(
          activation, weight, activation_scales, weight_scales, output, m, n,
          k, workspace, workspace_bytes, sm_count, stream);
    }
    if (use_cooperative_gemm(n, sm_count)) {
      return launch_gemm<SmallMCooperativeGemm<Output>>(
          activation, weight, activation_scales, weight_scales, output, m, n,
          k, workspace, workspace_bytes, sm_count, stream);
    }
    return launch_gemm<SmallMGemm<Output>>(
        activation, weight, activation_scales, weight_scales, output, m, n, k,
        workspace, workspace_bytes, sm_count, stream);
  }
  return launch_gemm<LargeMGemm<Output>>(
      activation, weight, activation_scales, weight_scales, output, m, n, k,
      workspace, workspace_bytes, sm_count, stream);
}

} // namespace mistralrs::fp8

extern "C" const char *mistralrs_cutlass_fp8_error_string(int status) {
  if (status < 0) {
    return cudaGetErrorString(
        static_cast<cudaError_t>(-static_cast<int64_t>(status)));
  }
  return cutlassGetStatusString(static_cast<cutlass::Status>(status));
}

extern "C" int mistralrs_cutlass_fp8_blockwise_prepare() {
  int status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::LargeMGemm<cutlass::half_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMGemm<cutlass::half_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMCooperativeGemm<cutlass::half_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMStreamKGemm<cutlass::half_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::LargeMGemm<cutlass::bfloat16_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMGemm<cutlass::bfloat16_t>>();
  if (status != 0) {
    return status;
  }
  status = mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMCooperativeGemm<cutlass::bfloat16_t>>();
  if (status != 0) {
    return status;
  }
  return mistralrs::fp8::prepare_kernel<
      mistralrs::fp8::SmallMStreamKGemm<cutlass::bfloat16_t>>();
}

extern "C" int mistralrs_cutlass_fp8_blockwise_workspace_size(
    int m, int n, int k, int output_dtype, int sm_count, size_t *bytes) {
  if (output_dtype == MISTRALRS_CUTLASS_FP8_OUTPUT_F16) {
    return mistralrs::fp8::dispatch_workspace_size<cutlass::half_t>(
        m, n, k, sm_count, bytes);
  }
  if (output_dtype == MISTRALRS_CUTLASS_FP8_OUTPUT_BF16) {
    return mistralrs::fp8::dispatch_workspace_size<cutlass::bfloat16_t>(
        m, n, k, sm_count, bytes);
  }
  return -static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int mistralrs_cutlass_fp8_blockwise_gemm(
    const void *activation, const void *weight,
    const float *activation_scales, const float *weight_scales, void *output,
    int m, int n, int k, int output_dtype, void *workspace,
    size_t workspace_bytes, int sm_count, void *stream) {
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (output_dtype == MISTRALRS_CUTLASS_FP8_OUTPUT_F16) {
    return mistralrs::fp8::dispatch_gemm<cutlass::half_t>(
        activation, weight, activation_scales, weight_scales, output, m, n, k,
        workspace, workspace_bytes, sm_count, cuda_stream);
  }
  if (output_dtype == MISTRALRS_CUTLASS_FP8_OUTPUT_BF16) {
    return mistralrs::fp8::dispatch_gemm<cutlass::bfloat16_t>(
        activation, weight, activation_scales, weight_scales, output, m, n, k,
        workspace, workspace_bytes, sm_count, cuda_stream);
  }
  return -static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int mistralrs_fp8_quantize_activation_f16(
    const void *input, void *output, float *scales, int rows, int cols,
    void *stream) {
  return mistralrs::fp8::launch_quantize(
      static_cast<const __half *>(input), static_cast<__nv_fp8_e4m3 *>(output),
      scales, rows, cols, static_cast<cudaStream_t>(stream));
}

extern "C" int mistralrs_fp8_quantize_activation_bf16(
    const void *input, void *output, float *scales, int rows, int cols,
    void *stream) {
  return mistralrs::fp8::launch_quantize(
      static_cast<const __nv_bfloat16 *>(input),
      static_cast<__nv_fp8_e4m3 *>(output), scales, rows, cols,
      static_cast<cudaStream_t>(stream));
}
