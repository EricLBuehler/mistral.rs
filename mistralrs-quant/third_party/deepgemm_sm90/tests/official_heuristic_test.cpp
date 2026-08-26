// SPDX-License-Identifier: MIT

#include <deep_gemm/official_sm90.cuh>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>

int main() {
  using mistralrs::deepgemm_official::getBestGemmConfig;

  const auto block_m_candidate = getBestGemmConfig(4096, 5120, 6144, 132);
  assert(block_m_candidate.valid());
  assert(block_m_candidate.block_m == 256);

  const std::array<std::array<uint32_t, 2>, 5> shapes = {{{16384, 5120},
                                                          {14336, 5120},
                                                          {5120, 6144},
                                                          {34816, 5120},
                                                          {5120, 17408}}};
  for (const auto& shape : shapes) {
    const auto config = getBestGemmConfig(512, shape[0], shape[1], 132);
    assert(config.valid());
    assert(config.block_k == 128);
    assert(config.block_m <= 256);
    assert(config.block_n <= 192);
    assert(config.num_stages >= 3);
    assert(config.cluster_size() <= 2);
    std::printf("N=%u K=%u BM=%u BN=%u stages=%u cluster=%ux%u smem=%u\n", shape[0],
                shape[1], config.block_m, config.block_n, config.num_stages,
                config.cluster_m, config.cluster_n, config.smem_bytes);
  }
  assert(!getBestGemmConfig(512, 5120, 6144, 131).valid());
  return 0;
}
