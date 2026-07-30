#include "utils.metal"
#include <metal_stdlib>
using namespace metal;

[[kernel]] void softcap_float(const device float *input [[buffer(0)]],
                              device float *output [[buffer(1)]],
                              constant uint &n_elements [[buffer(2)]],
                              constant float &cap [[buffer(3)]],
                              uint tid [[thread_position_in_grid]]) {
  if (tid < n_elements) {
    output[tid] = precise::tanh(input[tid] / cap) * cap;
  }
}
