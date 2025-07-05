#include "marlin_kernel.cuh"

extern "C" void marlin_gptq_4bit_bf16(const void *A, const void *B,
                                      void *scales, void *zeros, void *C,
                                      int prob_m, int prob_k, int prob_n,
                                      void *workspace, int groupsize,
                                      int64_t stream) {
  marlin_matmul<nv_bfloat16, ScalarTypeID::kU4B8, false, 4>(
      A, B, scales, zeros, C, prob_m, prob_k, prob_n, workspace, groupsize,
      stream);
}
