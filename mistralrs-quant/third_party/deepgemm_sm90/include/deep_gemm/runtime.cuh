/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modified by the mistral.rs project in 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "jit_utils.cuh"
#include "scheduler.cuh"

namespace deep_gemm::jit {

inline constexpr char kKernelName[] = "kernel.cubin";

class Runtime {
 public:
  Runtime(std::string path, std::vector<char> cubin, deep_gemm::GemmType gemm_type)
      : path_(std::move(path)), cubin_(std::move(cubin)), gemm_type_(gemm_type) {
    if (cubin_.empty() && !isPathValid(path_)) {
      throw std::runtime_error("DeepGEMM runtime has no usable cubin");
    }
  }

  // The process-global cache keeps libraries alive for every captured graph and CUDA context.
  ~Runtime() = default;

  static bool isPathValid(const std::string& path) {
    std::error_code error;
    auto cubin_path = std::filesystem::path(path) / kKernelName;
    return std::filesystem::is_regular_file(cubin_path, error) &&
           std::filesystem::file_size(cubin_path, error) > 0;
  }

  CUkernel getKernel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (kernel_ != nullptr) {
      return kernel_;
    }
    if (cubin_.empty()) {
      auto cubin_path = std::filesystem::path(path_) / kKernelName;
      std::ifstream input(cubin_path, std::ios::binary);
      if (!input) {
        throw std::runtime_error("Failed to open DeepGEMM cubin: " + cubin_path.string());
      }
      cubin_ = std::vector<char>(std::istreambuf_iterator<char>(input), {});
      if (cubin_.empty()) {
        throw std::runtime_error("DeepGEMM cubin is empty: " + cubin_path.string());
      }
    }

    CUlibrary loaded_library = nullptr;
    CUkernel loaded_kernel = nullptr;
    try {
      CHECK_CUDA(cuLibraryLoadData(&loaded_library, cubin_.data(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
      unsigned int count = 0;
      CHECK_CUDA(cuLibraryGetKernelCount(&count, loaded_library));
      std::vector<CUkernel> kernels(count);
      CHECK_CUDA(cuLibraryEnumerateKernels(kernels.data(), count, loaded_library));
      for (auto kernel : kernels) {
        const char* name = nullptr;
        CHECK_CUDA(cuKernelGetName(&name, kernel));
        if (name != nullptr &&
            (std::string(name).find("sm90_fp8_gemm_1d2d_impl") != std::string::npos ||
             std::string(name).find("fp8_gemm_kernel") != std::string::npos)) {
          loaded_kernel = kernel;
          break;
        }
      }
      if (loaded_kernel == nullptr) {
        throw std::runtime_error("DeepGEMM cubin does not contain an FP8 GEMM kernel");
      }
    } catch (...) {
      if (loaded_library != nullptr) {
        static_cast<void>(cuLibraryUnload(loaded_library));
      }
      throw;
    }
    library_ = loaded_library;
    kernel_ = loaded_kernel;
    return kernel_;
  }

 private:
  std::string path_;
  std::vector<char> cubin_;
  deep_gemm::GemmType gemm_type_;
  CUlibrary library_ = nullptr;
  CUkernel kernel_ = nullptr;
  std::mutex mutex_;
};

class RuntimeCache {
 public:
  static RuntimeCache& getInstance() {
    static RuntimeCache instance;
    return instance;
  }

  Runtime* get(const std::string& path, deep_gemm::GemmType gemm_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = path;
    auto cached = cache_.find(key);
    if (cached != cache_.end()) {
      return cached->second.get();
    }
    if (!Runtime::isPathValid(path)) {
      return nullptr;
    }
    try {
      auto runtime = std::make_unique<Runtime>(path, std::vector<char>(), gemm_type);
      static_cast<void>(runtime->getKernel());
      auto* result = runtime.get();
      cache_.emplace(std::move(key), std::move(runtime));
      return result;
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove(std::filesystem::path(path) / kKernelName, ignored);
      return nullptr;
    }
  }

  Runtime* insert(const std::string& path, std::unique_ptr<Runtime> runtime) {
    auto key = path;
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = cache_.find(key);
    if (cached != cache_.end()) {
      return cached->second.get();
    }
    static_cast<void>(runtime->getKernel());
    auto entry = cache_.try_emplace(std::move(key), std::move(runtime));
    return entry.first->second.get();
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Runtime>> cache_;
};

inline RuntimeCache& getGlobalRuntimeCache() { return RuntimeCache::getInstance(); }

}  // namespace deep_gemm::jit
