#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct WeightedReduce {
  const void *input;
  const float *weights;
  void *output;
  size_t tokens;
  size_t hidden;
  size_t topk;
  int input_type;
  int output_type;
};

namespace {
constexpr int kThreads = 256;

template <typename Input, typename Output>
__global__ void weighted_reduce(WeightedReduce args) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= args.tokens * args.hidden)
    return;
  const size_t token = index / args.hidden;
  const size_t column = index % args.hidden;
  const auto *input = static_cast<const Input *>(args.input);
  auto *output = static_cast<Output *>(args.output);
  float sum = 0.0f;
  for (size_t route = 0; route < args.topk; ++route) {
    const size_t row = token * args.topk + route;
    const float value = static_cast<float>(input[row * args.hidden + column]);
    // Keep the F32 product rounding from the unfused multiply and reduction.
    sum = __fadd_rn(sum, __fmul_rn(value, args.weights[row]));
  }
  output[index] = static_cast<Output>(sum);
}

template <typename Input>
void dispatch_output(const WeightedReduce &args, cudaStream_t stream) {
  const dim3 grid((args.tokens * args.hidden + kThreads - 1) / kThreads);
  switch (args.output_type) {
  case 0:
    weighted_reduce<Input, __half><<<grid, kThreads, 0, stream>>>(args);
    break;
  case 1:
    weighted_reduce<Input, __nv_bfloat16><<<grid, kThreads, 0, stream>>>(args);
    break;
  case 2:
    weighted_reduce<Input, float><<<grid, kThreads, 0, stream>>>(args);
    break;
  }
}
} // namespace

extern "C" int mistralrs_moe_weighted_reduce(const WeightedReduce *args,
                                            cudaStream_t stream) {
  switch (args->input_type) {
  case 0:
    dispatch_output<__half>(*args, stream);
    break;
  case 1:
    dispatch_output<__nv_bfloat16>(*args, stream);
    break;
  case 2:
    dispatch_output<float>(*args, stream);
    break;
  }
  return cudaGetLastError();
}
