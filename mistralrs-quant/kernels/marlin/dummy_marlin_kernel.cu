#include <cassert>

extern "C" void marlin_4bit_f16(const void *A, const void *B, void *s, void *C,
                                int prob_m, int prob_k, int prob_n,
                                void *workspace, int groupsize) {
  assert(false);
}

extern "C" void marlin_4bit_bf16(const void *A, const void *B, void *s, void *C,
                                 int prob_m, int prob_k, int prob_n,
                                 void *workspace, int groupsize) {
  assert(false);
}

extern "C" void gptq_marlin_repack(void *weight, void *perm, void *out,
                                   int size_k, int size_n, int num_bits) {
  assert(false);
}
