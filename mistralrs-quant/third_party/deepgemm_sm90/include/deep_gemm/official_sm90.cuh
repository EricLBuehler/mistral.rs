/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 * Modified by the mistral.rs project in 2026.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace mistralrs::deepgemm_official {

constexpr uint32_t kBlockK = 128;
constexpr uint32_t kSmemCapacity = 232448;
constexpr uint32_t kMaxStages = 16;

struct GemmConfig {
  uint32_t block_m = 0;
  uint32_t block_n = 0;
  uint32_t block_k = 0;
  uint32_t cluster_m = 0;
  uint32_t cluster_n = 0;
  uint32_t swizzle_a = 0;
  uint32_t swizzle_b = 0;
  uint32_t swizzle_d = 0;
  uint32_t num_stages = 0;
  uint32_t smem_bytes = 0;
  uint32_t num_threads = 0;
  int64_t estimated_cycles = std::numeric_limits<int64_t>::max();

  [[nodiscard]] bool valid() const { return block_m != 0; }
  [[nodiscard]] uint32_t cluster_size() const { return cluster_m * cluster_n; }
  [[nodiscard]] bool multicast_on_a() const { return cluster_n > 1; }
};

template <typename T>
constexpr T ceilDiv(T value, T divisor) {
  return (value + divisor - 1) / divisor;
}

template <typename T>
constexpr T alignUp(T value, T alignment) {
  return ceilDiv(value, alignment) * alignment;
}

inline uint32_t swizzleMode(uint32_t block_size, uint32_t element_size) {
  for (uint32_t mode : {128U, 64U, 32U, 16U}) {
    if ((block_size * element_size) % mode == 0) {
      return mode;
    }
  }
  return 0;
}

inline std::vector<GemmConfig> getGemmConfigs(uint32_t m, uint32_t n, uint32_t k,
                                              uint32_t num_sms) {
  if (m == 0 || n == 0 || k == 0 || num_sms == 0 || num_sms % 2 != 0) {
    return {};
  }

  std::vector<uint32_t> block_m_candidates = {64, 128};
  if (m <= 16) {
    block_m_candidates.push_back(16);
  }
  if (m <= 32) {
    block_m_candidates.push_back(32);
  }
  block_m_candidates.push_back(256);

  std::vector<uint32_t> block_n_candidates;
  const uint32_t block_n_step = std::lcm(16U, 1U);
  for (uint32_t block_n = block_n_step; block_n <= 192; block_n += block_n_step) {
    block_n_candidates.push_back(block_n);
  }

  std::vector<GemmConfig> configs;
  for (uint32_t cluster_m = 1; cluster_m <= 2; ++cluster_m) {
    for (uint32_t cluster_n = 1; cluster_n <= 2; ++cluster_n) {
      const uint32_t cluster_size = cluster_m * cluster_n;
      if (cluster_size > 2 || num_sms % cluster_size != 0) {
        continue;
      }

      for (uint32_t block_m : block_m_candidates) {
        for (uint32_t block_n : block_n_candidates) {
          if (block_n > kBlockK && block_n % (block_n - kBlockK) != 0 &&
              kBlockK % (block_n - kBlockK) != 0) {
            continue;
          }
          if (block_m > 128 && block_n > 128) {
            continue;
          }

          const uint32_t swizzle_a = swizzleMode(kBlockK, 1);
          const uint32_t swizzle_b = swizzleMode(kBlockK, 1);
          const uint32_t swizzle_d = swizzleMode(block_n, 2);
          if (swizzle_a % 64 != 0 || swizzle_b % 64 != 0 || swizzle_d == 0) {
            continue;
          }

          const uint32_t smem_d = alignUp(block_m * block_n * 2U, 1024U);
          constexpr uint32_t smem_barriers = kMaxStages * 8U * 2U;
          const uint32_t smem_a_per_stage = block_m * kBlockK;
          const uint32_t smem_b_per_stage = block_n * kBlockK;
          const uint32_t smem_sfa_per_stage = alignUp(block_m * 4U, 128U);
          const uint32_t uniform_sfb = kBlockK % block_n == 0 ? 1U : 2U;
          const uint32_t smem_extra_sfb =
              alignUp(ceilDiv(k, kBlockK) * 4U * uniform_sfb, 8U);
          const uint32_t smem_extra = smem_d + smem_barriers + smem_extra_sfb;
          const uint32_t smem_per_stage =
              smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
          if (smem_extra >= kSmemCapacity) {
            continue;
          }
          const uint32_t num_stages = std::min(
              (kSmemCapacity - smem_extra) / smem_per_stage, kMaxStages);
          if (num_stages < 3 || (block_m * block_n < 128U * 192U && num_stages < 4)) {
            continue;
          }

          const int64_t num_blocks =
              static_cast<int64_t>(ceilDiv(m, block_m)) * ceilDiv(n, block_n);
          const int64_t num_waves = ceilDiv(num_blocks, static_cast<int64_t>(num_sms));
          const int l2_bandwidth_per_cycle = static_cast<int>(
              std::min(64.0 * num_sms, 8e6 / 1.3e3));
          const int l1_bandwidth_per_cycle = 128 * static_cast<int>(num_sms);
          const int64_t num_bytes_l2_ab = static_cast<int64_t>(k) *
              (block_m / cluster_n + block_n / cluster_m);
          const int64_t num_bytes_l1_ab =
              static_cast<int64_t>(k) * (block_m + block_n);
          const int64_t num_bytes_l1_tc = static_cast<int64_t>(k) *
                                                  (std::max(64U, block_m) + block_n) +
                                              static_cast<int64_t>(block_m) * block_n * 2;
          const int64_t num_bytes_l1_l2_d =
              static_cast<int64_t>(block_m) * block_n * 2;
          const int64_t num_l2_cycles =
              (num_bytes_l2_ab + num_bytes_l1_l2_d) * num_blocks /
              l2_bandwidth_per_cycle;
          const int64_t num_l1_cycles =
              (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_d) * num_blocks /
              l1_bandwidth_per_cycle;
          const float wave_efficiency = static_cast<float>(num_blocks) /
                                        static_cast<float>(num_waves * num_sms);
          int64_t estimated_cycles = static_cast<int64_t>(
              std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency);
          if (cluster_size > 1 && num_waves <= 1) {
            estimated_cycles = std::numeric_limits<int64_t>::max();
          }

          configs.push_back({
              block_m,
              block_n,
              kBlockK,
              cluster_m,
              cluster_n,
              swizzle_a,
              swizzle_b,
              swizzle_d,
              num_stages,
              smem_extra + num_stages * smem_per_stage,
              128U + (block_m <= 64 ? 128U : 256U),
              estimated_cycles,
          });
        }
      }
    }
  }
  return configs;
}

inline GemmConfig getBestGemmConfig(uint32_t m, uint32_t n, uint32_t k,
                                    uint32_t num_sms) {
  const auto configs = getGemmConfigs(m, n, k, num_sms);
  if (configs.empty()) {
    return {};
  }
  return *std::min_element(configs.begin(), configs.end(),
                           [](const GemmConfig& lhs, const GemmConfig& rhs) {
                             return lhs.estimated_cycles < rhs.estimated_cycles;
                           });
}

inline GemmConfig getGemmConfig(uint32_t m, uint32_t n, uint32_t k,
                                uint32_t num_sms, uint32_t block_m,
                                uint32_t block_n, uint32_t cluster_m,
                                uint32_t cluster_n) {
  const auto configs = getGemmConfigs(m, n, k, num_sms);
  const auto config = std::find_if(
      configs.begin(), configs.end(), [&](const GemmConfig& candidate) {
        return candidate.block_m == block_m && candidate.block_n == block_n &&
               candidate.cluster_m == cluster_m &&
               candidate.cluster_n == cluster_n;
      });
  return config == configs.end() ? GemmConfig{} : *config;
}

}  // namespace mistralrs::deepgemm_official
