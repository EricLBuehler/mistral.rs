#include "nvfp4_cutlass.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

static_assert(CUDART_VERSION >= 13030);

namespace mistralrs_nvfp4 {
using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

constexpr int kBlockSize = 16;
constexpr int kScaleRows = 128;
constexpr int kScaleCols = 4;
constexpr int kPointerAlignment = 16;
constexpr int kWeightAlignment = 32;
constexpr int kReductionAlignment = 64;
constexpr int kSwizzleThreads = 256;
constexpr int kSchedulerSwizzle = 8;
constexpr int kDecodeSchedulerSwizzle = 1;
constexpr size_t kMaxSwizzleBlocks = 65535;
constexpr size_t kDefaultSharedBytes = 48 * 1024;

template <class T> struct MultiplyRn;
template <> struct MultiplyRn<float> {
  CUTLASS_HOST_DEVICE float operator()(float a, float b) const {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
  }
};
template <int N, bool RegisterSized>
struct MultiplyRn<cutlass::Array<float, N, RegisterSized>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float, N, RegisterSized>
  operator()(const cutlass::Array<float, N, RegisterSized> &a,
             const cutlass::Array<float, N, RegisterSized> &b) const {
    cutlass::Array<float, N, RegisterSized> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
      out[i] = MultiplyRn<float>{}(a[i], b[i]);
    return out;
  }
};

template <class OutputType, bool WeightFirst = false> struct KernelConfig {
  static constexpr bool kWeightFirst = WeightFirst;
  static constexpr int kSwizzle =
      WeightFirst ? kDecodeSchedulerSwizzle : kSchedulerSwizzle;
  using Output = OutputType;
  using TileShape = std::conditional_t<WeightFirst, Shape<_128, _32, _256>,
                                       Shape<_128, _128, _128>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Packed = cutlass::float_e2m1_t;
  using BlockScale = cutlass::float_ue4m3_t;
  using InputPair = cutlass::nv_float4_t<Packed>;
  using ScaleLayout = cutlass::detail::Sm1xxBlockScaledConfig<kBlockSize>;
  using WeightGlobal = std::conditional_t<
      WeightFirst,
      fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, _0>,
                               1, false>,
      fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<_0, _1, _0>,
                               1, false>>;
  using ActivationGlobal = fusion::Sm90ScalarBroadcast<float>;
  using WeightScaled = fusion::Sm90EVT<
      fusion::Sm90Compute<MultiplyRn, float, float,
                          cutlass::FloatRoundStyle::round_to_nearest>,
      fusion::Sm90AccFetch, WeightGlobal>;
  using OutputScaled = fusion::Sm90EVT<
      fusion::Sm90Compute<MultiplyRn, Output, float,
                          cutlass::FloatRoundStyle::round_to_nearest>,
      WeightScaled, ActivationGlobal>;
  using OutputLayout =
      std::conditional_t<WeightFirst, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>;
  using EpilogueSchedule =
      std::conditional_t<WeightFirst, cutlass::epilogue::TmaWarpSpecialized,
                         cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, OutputLayout, 8, Output, OutputLayout, 8,
      EpilogueSchedule, OutputScaled>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      InputPair, cutlass::layout::RowMajor, kWeightAlignment, InputPair,
      cutlass::layout::ColumnMajor, kWeightAlignment, float, TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Scheduler =
      std::conditional_t<WeightFirst, cutlass::gemm::StaticPersistentScheduler,
                         void>;
  using Implementation =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop,
                                           Epilogue, Scheduler>;
};

struct Bf16Kernel : KernelConfig<cutlass::bfloat16_t>::Implementation {
  using Config = KernelConfig<cutlass::bfloat16_t>;
  using Base = Config::Implementation;
  struct Params : Base::Params {
    Params(const Base::Params &params) : Base::Params(params) {}
  };
  static Params to_underlying_arguments(const Arguments &args,
                                        void *workspace) {
    return Base::to_underlying_arguments(args, workspace);
  };
};
static_assert(sizeof(Bf16Kernel::Params) == sizeof(Bf16Kernel::Base::Params));
static_assert(alignof(Bf16Kernel::Params) == alignof(Bf16Kernel::Base::Params));

struct F16Kernel : KernelConfig<cutlass::half_t>::Implementation {
  using Config = KernelConfig<cutlass::half_t>;
  using Base = Config::Implementation;
  struct Params : Base::Params {
    Params(const Base::Params &params) : Base::Params(params) {}
  };
  static Params to_underlying_arguments(const Arguments &args,
                                        void *workspace) {
    return Base::to_underlying_arguments(args, workspace);
  };
};
static_assert(sizeof(F16Kernel::Params) == sizeof(F16Kernel::Base::Params));
static_assert(alignof(F16Kernel::Params) == alignof(F16Kernel::Base::Params));

struct DecodeBf16Kernel
    : KernelConfig<cutlass::bfloat16_t, true>::Implementation {
  using Config = KernelConfig<cutlass::bfloat16_t, true>;
  using Base = Config::Implementation;
  struct Params : Base::Params {
    Params(const Base::Params &params) : Base::Params(params) {}
  };
  static Params to_underlying_arguments(const Arguments &args,
                                        void *workspace) {
    return Base::to_underlying_arguments(args, workspace);
  };
};
static_assert(sizeof(DecodeBf16Kernel::Params) ==
              sizeof(DecodeBf16Kernel::Base::Params));
static_assert(alignof(DecodeBf16Kernel::Params) ==
              alignof(DecodeBf16Kernel::Base::Params));
static_assert(!DecodeBf16Kernel::TileScheduler::IsDynamicPersistent);

struct DecodeF16Kernel : KernelConfig<cutlass::half_t, true>::Implementation {
  using Config = KernelConfig<cutlass::half_t, true>;
  using Base = Config::Implementation;
  struct Params : Base::Params {
    Params(const Base::Params &params) : Base::Params(params) {}
  };
  static Params to_underlying_arguments(const Arguments &args,
                                        void *workspace) {
    return Base::to_underlying_arguments(args, workspace);
  };
};
static_assert(sizeof(DecodeF16Kernel::Params) ==
              sizeof(DecodeF16Kernel::Base::Params));
static_assert(alignof(DecodeF16Kernel::Params) ==
              alignof(DecodeF16Kernel::Base::Params));
static_assert(!DecodeF16Kernel::TileScheduler::IsDynamicPersistent);

bool valid_dtype(int32_t dtype) {
  return dtype == MISTRALRS_NVFP4_BF16 || dtype == MISTRALRS_NVFP4_F16;
}

bool valid_kernel(int32_t kernel) {
  return kernel == MISTRALRS_NVFP4_PREFILL ||
         kernel == MISTRALRS_NVFP4_DECODE_DP32;
}

bool valid_context(const mistralrs_nvfp4_context *context) {
  return context && context->device >= 0 && context->sm_count > 0 &&
         valid_dtype(context->dtype) && valid_kernel(context->kernel);
}

bool valid_shape(const mistralrs_nvfp4_shape *shape) {
  return shape && shape->m > 0 && shape->n > 0 && shape->k > 0 &&
         shape->n % kWeightAlignment == 0 &&
         shape->k % kReductionAlignment == 0;
}

bool aligned(const void *pointer, uintptr_t alignment = kPointerAlignment) {
  return pointer && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

template <class Kernel>
typename Kernel::Arguments
make_arguments(const mistralrs_nvfp4_launch &launch) {
  using Config = typename Kernel::Config;
  using Output = typename Config::Output;
  using Packed = typename Config::Packed;
  using BlockScale = typename Config::BlockScale;
  using ScaleLayout = typename Config::ScaleLayout;
  using WeightScaled = typename Config::WeightScaled;
  using OutputScaled = typename Config::OutputScaled;
  const auto &shape = launch.shape;
  const int m = Config::kWeightFirst ? shape.n : shape.m;
  const int n = Config::kWeightFirst ? shape.m : shape.n;
  auto problem = make_shape(m, n, shape.k, 1);
  typename Kernel::MainloopArguments mainloop{};
  mainloop.ptr_A = static_cast<const Packed *>(
      Config::kWeightFirst ? launch.w_packed : launch.a_packed);
  mainloop.dA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},
                                                 make_shape(m, shape.k, 1));
  mainloop.ptr_B = static_cast<const Packed *>(
      Config::kWeightFirst ? launch.a_packed : launch.w_packed);
  mainloop.dB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},
                                                 make_shape(n, shape.k, 1));
  mainloop.ptr_SFA = static_cast<const BlockScale *>(
      Config::kWeightFirst ? launch.w_scale_swizzled : launch.a_scale_swizzled);
  mainloop.ptr_SFB = static_cast<const BlockScale *>(
      Config::kWeightFirst ? launch.a_scale_swizzled : launch.w_scale_swizzled);
  mainloop.layout_SFA = ScaleLayout::tile_atom_to_shape_SFA(problem);
  mainloop.layout_SFB = ScaleLayout::tile_atom_to_shape_SFB(problem);
  typename WeightScaled::Arguments weight_scaled{
      {}, {launch.weight_global, 0.0f, {}}, {}};
  typename OutputScaled::Arguments output_scaled{
      weight_scaled, {{0.0f}, {launch.activation_global}, {}}, {}};
  auto stride = cutlass::make_cute_packed_stride(typename Kernel::StrideD{},
                                                 make_shape(m, n, 1));
  typename Kernel::EpilogueArguments epilogue{
      output_scaled, nullptr, stride, static_cast<Output *>(launch.output),
      stride};
  typename Kernel::TileScheduler::Arguments scheduler{};
  scheduler.max_swizzle_size = Config::kSwizzle;
  return {cutlass::gemm::GemmUniversalMode::kGemm,
          problem,
          mainloop,
          epilogue,
          cutlass::KernelHardwareInfo{launch.context.device,
                                      launch.context.sm_count},
          scheduler};
}

int scale_sizes(int32_t rows, int32_t k, size_t *cols, size_t *padded_cols,
                size_t *bytes) {
  if (rows <= 0 || k <= 0 || k % kBlockSize != 0 || !bytes)
    return -static_cast<int>(cudaErrorInvalidValue);
  *cols = static_cast<size_t>(k) / kBlockSize;
  *padded_cols = (*cols + kScaleCols - 1) / kScaleCols * kScaleCols;
  size_t padded_rows =
      (static_cast<size_t>(rows) + kScaleRows - 1) / kScaleRows * kScaleRows;
  if (*padded_cols > std::numeric_limits<size_t>::max() / padded_rows)
    return -static_cast<int>(cudaErrorInvalidValue);
  *bytes = padded_rows * *padded_cols;
  return 0;
}

__host__ __device__ size_t scale_offset(size_t row, size_t col,
                                        size_t padded_cols) {
  return (((row / kScaleRows) * (padded_cols / kScaleCols) + col / kScaleCols) *
              32 +
          row % 32) *
             16 +
         (row % kScaleRows) / 32 * 4 + col % 4;
}

__global__ void swizzle_scales(const uint8_t *source, uint8_t *dest,
                               size_t rows, size_t cols, size_t padded_cols,
                               size_t bytes) {
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < bytes; index += step) {
    size_t part = index / 4;
    size_t row_quarter = part % 4;
    part /= 4;
    size_t row_lane = part % 32;
    part /= 32;
    size_t col = part % (padded_cols / 4) * 4 + index % 4;
    size_t row = part / (padded_cols / 4) * 128 + row_quarter * 32 + row_lane;
    dest[index] = row < rows && col < cols ? source[row * cols + col] : 0;
  }
}
template <class Kernel>
int prepare_kernel(const mistralrs_nvfp4_context &context,
                   const cudaDeviceProp &props,
                   mistralrs_nvfp4_resources *resources) {
  if constexpr (Kernel::SharedStorageSize >= kDefaultSharedBytes) {
    auto error = cudaFuncSetAttribute(
        cutlass::device_kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize);
    if (error != cudaSuccess)
      return -static_cast<int>(error);
  }
  cudaFuncAttributes attributes{};
  auto error =
      cudaFuncGetAttributes(&attributes, cutlass::device_kernel<Kernel>);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  *resources = {context,
                props.major,
                props.minor,
                Kernel::MaxThreadsPerBlock,
                attributes.numRegs,
                Kernel::SharedStorageSize,
                attributes.localSizeBytes};
  return 0;
}

template <class Kernel>
int workspace_size(const mistralrs_nvfp4_context &context,
                   const mistralrs_nvfp4_shape &shape, size_t *bytes) {
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  mistralrs_nvfp4_launch launch{};
  launch.shape = shape;
  launch.context = context;
  auto arguments = make_arguments<Kernel>(launch);
  auto status = Adapter::can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  *bytes = Adapter::get_workspace_size(arguments);
  return 0;
}

template <class Kernel> int gemm(const mistralrs_nvfp4_launch &launch) {
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto arguments = make_arguments<Kernel>(launch);
  auto status = Adapter::can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  size_t bytes = Adapter::get_workspace_size(arguments);
  if (bytes > launch.workspace_bytes || (bytes && !launch.workspace))
    return static_cast<int>(cutlass::Status::kErrorWorkspaceNull);
  if (bytes && !aligned(launch.workspace))
    return -static_cast<int>(cudaErrorInvalidValue);
  auto stream = static_cast<cudaStream_t>(launch.stream);
  status = Kernel::initialize_workspace(arguments, launch.workspace, stream,
                                        nullptr);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  auto params = Kernel::to_underlying_arguments(arguments, launch.workspace);
  return static_cast<int>(Adapter::run(params, stream, nullptr, false));
}
} // namespace mistralrs_nvfp4
using namespace mistralrs_nvfp4;

extern "C" const char *mistralrs_nvfp4_error_string(int status) {
  return status < 0 ? cudaGetErrorString(static_cast<cudaError_t>(-status))
                    : cutlass::cutlassGetStatusString(
                          static_cast<cutlass::Status>(status));
}

extern "C" int mistralrs_nvfp4_prepare(int32_t device, int32_t dtype,
                                       int32_t kernel,
                                       mistralrs_nvfp4_resources *resources) {
  if (!resources || device < 0 || !valid_dtype(dtype) || !valid_kernel(kernel))
    return -static_cast<int>(cudaErrorInvalidValue);
  int current = 0;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  if (current != device)
    return -static_cast<int>(cudaErrorInvalidDevice);
  cudaDeviceProp props{};
  error = cudaGetDeviceProperties(&props, device);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  if (props.major != 12 || props.minor != 1)
    return -static_cast<int>(cudaErrorNotSupported);
  mistralrs_nvfp4_context context{device, props.multiProcessorCount, dtype,
                                  kernel};
  if (kernel == MISTRALRS_NVFP4_DECODE_DP32)
    return dtype == MISTRALRS_NVFP4_BF16
               ? prepare_kernel<DecodeBf16Kernel>(context, props, resources)
               : prepare_kernel<DecodeF16Kernel>(context, props, resources);
  return dtype == MISTRALRS_NVFP4_BF16
             ? prepare_kernel<Bf16Kernel>(context, props, resources)
             : prepare_kernel<F16Kernel>(context, props, resources);
}

extern "C" int
mistralrs_nvfp4_workspace_size(const mistralrs_nvfp4_context *context,
                               const mistralrs_nvfp4_shape *shape,
                               size_t *bytes) {
  if (!valid_context(context) || !valid_shape(shape) || !bytes)
    return -static_cast<int>(cudaErrorInvalidValue);
  if (context->kernel == MISTRALRS_NVFP4_DECODE_DP32)
    return context->dtype == MISTRALRS_NVFP4_BF16
               ? workspace_size<DecodeBf16Kernel>(*context, *shape, bytes)
               : workspace_size<DecodeF16Kernel>(*context, *shape, bytes);
  return context->dtype == MISTRALRS_NVFP4_BF16
             ? workspace_size<Bf16Kernel>(*context, *shape, bytes)
             : workspace_size<F16Kernel>(*context, *shape, bytes);
}

extern "C" int mistralrs_nvfp4_gemm(const mistralrs_nvfp4_launch *launch) {
  if (!launch || !valid_context(&launch->context) ||
      !valid_shape(&launch->shape) || !aligned(launch->a_packed) ||
      !aligned(launch->w_packed) || !aligned(launch->a_scale_swizzled) ||
      !aligned(launch->w_scale_swizzled) || !aligned(launch->output) ||
      !aligned(launch->weight_global, sizeof(float)) ||
      !aligned(launch->activation_global, sizeof(float)))
    return -static_cast<int>(cudaErrorInvalidValue);
  if (launch->context.kernel == MISTRALRS_NVFP4_DECODE_DP32)
    return launch->context.dtype == MISTRALRS_NVFP4_BF16
               ? gemm<DecodeBf16Kernel>(*launch)
               : gemm<DecodeF16Kernel>(*launch);
  return launch->context.dtype == MISTRALRS_NVFP4_BF16
             ? gemm<Bf16Kernel>(*launch)
             : gemm<F16Kernel>(*launch);
}

extern "C" int mistralrs_nvfp4_scale_bytes(int32_t rows, int32_t k,
                                           size_t *bytes) {
  size_t cols = 0, padded_cols = 0;
  return scale_sizes(rows, k, &cols, &padded_cols, bytes);
}

extern "C" int mistralrs_nvfp4_swizzle_host(const void *source, void *dest,
                                            int32_t rows, int32_t k,
                                            size_t dest_bytes) {
  size_t cols = 0, padded_cols = 0, bytes = 0;
  int status = scale_sizes(rows, k, &cols, &padded_cols, &bytes);
  if (status)
    return status;
  if (!source || !dest || source == dest || dest_bytes < bytes)
    return -static_cast<int>(cudaErrorInvalidValue);
  std::memset(dest, 0, bytes);
  auto input = static_cast<const uint8_t *>(source);
  auto output = static_cast<uint8_t *>(dest);
  for (size_t row = 0; row < static_cast<size_t>(rows); ++row)
    for (size_t col = 0; col < cols; ++col)
      output[scale_offset(row, col, padded_cols)] = input[row * cols + col];
  return 0;
}

extern "C" int mistralrs_nvfp4_swizzle_cuda(const void *source, void *dest,
                                            int32_t rows, int32_t k,
                                            size_t dest_bytes, void *stream) {
  size_t cols = 0, padded_cols = 0, bytes = 0;
  int status = scale_sizes(rows, k, &cols, &padded_cols, &bytes);
  if (status)
    return status;
  if (!source || !dest || source == dest || dest_bytes < bytes)
    return -static_cast<int>(cudaErrorInvalidValue);
  size_t blocks = (bytes + kSwizzleThreads - 1) / kSwizzleThreads;
  if (blocks > kMaxSwizzleBlocks)
    blocks = kMaxSwizzleBlocks;
  swizzle_scales<<<static_cast<unsigned>(blocks), kSwizzleThreads, 0,
                   static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t *>(source), static_cast<uint8_t *>(dest), rows,
      cols, padded_cols, bytes);
  return -static_cast<int>(cudaGetLastError());
}
