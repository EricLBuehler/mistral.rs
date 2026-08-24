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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "jit_utils.cuh"
#include "runtime.cuh"
#include "scheduler.cuh"

#ifndef _WIN32
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace deep_gemm::jit {

#ifndef MISTRALRS_DEEPGEMM_SOURCE_HASH
#error "MISTRALRS_DEEPGEMM_SOURCE_HASH must identify the bundled JIT sources"
#endif

#define MISTRALRS_DEEPGEMM_STRINGIFY_IMPL(value) #value
#define MISTRALRS_DEEPGEMM_STRINGIFY(value) MISTRALRS_DEEPGEMM_STRINGIFY_IMPL(value)

inline std::string getCacheVersion() {
  return "source-" MISTRALRS_DEEPGEMM_STRINGIFY(MISTRALRS_DEEPGEMM_SOURCE_HASH);
}

#undef MISTRALRS_DEEPGEMM_STRINGIFY
#undef MISTRALRS_DEEPGEMM_STRINGIFY_IMPL

inline std::filesystem::path getDefaultUserDir() {
  if (const char* override_dir = std::getenv("MISTRALRS_DEEPGEMM_CACHE_DIR");
      override_dir != nullptr && override_dir[0] != '\0') {
    return override_dir;
  }
  if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
      xdg_cache != nullptr && xdg_cache[0] != '\0') {
    return std::filesystem::path(xdg_cache) / "mistralrs" / "deepgemm-sm90";
  }
  if (const char* user_home = std::getenv("HOME"); user_home != nullptr && user_home[0] != '\0') {
    return std::filesystem::path(user_home) / ".cache" / "mistralrs" / "deepgemm-sm90";
  }
#ifdef _WIN32
  return std::filesystem::temp_directory_path() / "mistralrs-deepgemm-sm90";
#else
  return std::filesystem::temp_directory_path() /
         ("mistralrs-deepgemm-sm90-" + std::to_string(static_cast<uint64_t>(geteuid())));
#endif
}

inline std::filesystem::path getTmpDir() {
  return getDefaultUserDir() / getCacheVersion() / "tmp";
}

inline std::filesystem::path getCacheDir() {
  return getDefaultUserDir() / getCacheVersion() / "kernels";
}

inline void ensurePrivateCacheRoot() {
  auto root = getDefaultUserDir();
  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    throw std::system_error(error, "Failed to create DeepGEMM cache root");
  }
#ifndef _WIN32
  struct stat metadata {};
  if (lstat(root.c_str(), &metadata) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "Failed to inspect DeepGEMM cache root");
  }
  if (!S_ISDIR(metadata.st_mode) || metadata.st_uid != geteuid()) {
    throw std::runtime_error("DeepGEMM cache root must be an owner-controlled directory");
  }
  if ((metadata.st_mode & 0777) != 0700 && chmod(root.c_str(), 0700) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "Failed to secure DeepGEMM cache root");
  }
#endif
}

inline std::string getNvccCompiler() {
  if (const char* nvcc = std::getenv("MISTRALRS_DEEPGEMM_NVCC");
      nvcc != nullptr && nvcc[0] != '\0') {
    return nvcc;
  }
  if (const char* cuda_home = std::getenv("CUDA_HOME");
      cuda_home != nullptr && cuda_home[0] != '\0') {
#ifdef _WIN32
    return (std::filesystem::path(cuda_home) / "bin" / "nvcc.exe").string();
#else
    return (std::filesystem::path(cuda_home) / "bin" / "nvcc").string();
#endif
  }
#ifdef _WIN32
  return "nvcc.exe";
#else
  return "nvcc";
#endif
}

inline bool executableExists(const std::string& executable) {
#ifdef _WIN32
  std::error_code error;
  return std::filesystem::is_regular_file(executable, error);
#else
  if (executable.find('/') != std::string::npos) {
    return access(executable.c_str(), X_OK) == 0;
  }
  const char* path_env = std::getenv("PATH");
  if (path_env == nullptr) {
    return false;
  }
  std::string path(path_env);
  size_t begin = 0;
  while (begin <= path.size()) {
    size_t end = path.find(':', begin);
    auto directory = path.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    auto candidate = std::filesystem::path(directory.empty() ? "." : directory) / executable;
    if (access(candidate.c_str(), X_OK) == 0) {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
#endif
}

inline std::vector<std::filesystem::path>& getJitIncludeDirs() {
  static std::vector<std::filesystem::path> include_dirs;
  return include_dirs;
}

inline void setJitIncludeDirs(const std::vector<std::filesystem::path>& dirs) {
  getJitIncludeDirs() = dirs;
}

inline std::string generateUniqueId() {
  static std::atomic<uint64_t> sequence{0};
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
  uint64_t process = 0;
#else
  uint64_t process = static_cast<uint64_t>(getpid());
#endif
  return std::to_string(process) + "_" + std::to_string(now) + "_" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

struct CommandResult {
  int exit_code;
  std::string output;
};

inline CommandResult runCompiler(const std::vector<std::string>& command) {
  if (command.empty()) {
    throw std::runtime_error("DeepGEMM compiler command is empty");
  }
#ifdef _WIN32
  throw std::runtime_error("DeepGEMM runtime compilation is unsupported on Windows");
#else
  int output_pipe[2];
  if (pipe(output_pipe) != 0) {
    throw std::system_error(errno, std::generic_category(), "Failed to create compiler pipe");
  }

  posix_spawn_file_actions_t actions;
  int action_status = posix_spawn_file_actions_init(&actions);
  if (action_status != 0) {
    close(output_pipe[0]);
    close(output_pipe[1]);
    throw std::system_error(action_status, std::generic_category(),
                            "Failed to initialize compiler process");
  }
  posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
  posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, output_pipe[1]);

  std::vector<char*> arguments;
  arguments.reserve(command.size() + 1);
  for (const auto& argument : command) {
    arguments.push_back(const_cast<char*>(argument.c_str()));
  }
  arguments.push_back(nullptr);

  pid_t pid = 0;
  int spawn_status =
      posix_spawnp(&pid, command.front().c_str(), &actions, nullptr, arguments.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(output_pipe[1]);
  if (spawn_status != 0) {
    close(output_pipe[0]);
    throw std::system_error(spawn_status, std::generic_category(),
                            "Failed to start DeepGEMM nvcc");
  }

  std::string output;
  char buffer[4096];
  for (;;) {
    ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
    if (count > 0) {
      output.append(buffer, static_cast<size_t>(count));
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      close(output_pipe[0]);
      static_cast<void>(waitpid(pid, nullptr, 0));
      throw std::system_error(errno, std::generic_category(),
                              "Failed to read DeepGEMM nvcc output");
    }
  }
  close(output_pipe[0]);

  int wait_status = 0;
  while (waitpid(pid, &wait_status, 0) < 0) {
    if (errno != EINTR) {
      throw std::system_error(errno, std::generic_category(),
                              "Failed to wait for DeepGEMM nvcc");
    }
  }
  int exit_code = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : 128;
  return {exit_code, std::move(output)};
#endif
}

inline std::string generateKernel(uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
                                  uint32_t block_n, uint32_t block_k, uint32_t num_groups,
                                  uint32_t num_stages, uint32_t num_tma_multicast,
                                  deep_gemm::GemmType gemm_type, bool swap_ab = false) {
  constexpr uint32_t kNumTMAThreads = 128;
  constexpr uint32_t kNumMathThreadsPerGroup = 128;

  std::string input_type;
  if (!swap_ab) {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedContiguous:
        input_type = "GroupedContiguousSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedMasked:
        input_type = "GroupedMaskedSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInput";
        break;
      case deep_gemm::GemmType::StridedBatched:
        input_type = "StridedBatchedSchedulerInput";
        break;
      default:
        throw std::runtime_error("Unsupported DeepGEMM type");
    }
  } else {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInputSwapAB";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInputSwapAB";
        break;
      default:
        throw std::runtime_error("Unsupported swapped DeepGEMM type");
    }
  }

  std::string kernel_name = swap_ab ? "fp8_gemm_kernel_swapAB" : "fp8_gemm_kernel";
  std::string scheduler_name = swap_ab ? "SchedulerSelectorSwapAB" : "SchedulerSelector";
  return R"(
#include <string>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <deep_gemm/fp8_gemm_impl.cuh>

using namespace deep_gemm;

using SchedulerType =
typename )" + scheduler_name +
         R"(<GemmType::)" + gemm_type_to_string(gemm_type) + R"(, )" +
         std::to_string(shape_n) + R"(, )" + std::to_string(shape_k) + R"(, )" +
         std::to_string(block_m) + R"(, )" + std::to_string(block_n) + R"(, )" +
         std::to_string(block_k) + R"(, )" + std::to_string(num_groups) + R"(, )" +
         std::to_string(num_tma_multicast) + R"(>::type;

__global__ void dummy_kernel() {
  void *ptr = (void *)&)" +
         kernel_name + R"(<)" + std::to_string(shape_n) + R"(, )" +
         std::to_string(shape_k) + R"(, )" + std::to_string(block_m) + R"(, )" +
         std::to_string(block_n) + R"(, )" + std::to_string(block_k) + R"(, )" +
         std::to_string(num_groups) + R"(, )" + std::to_string(num_stages) + R"(, )" +
         std::to_string(kNumTMAThreads) + R"(, )" +
         std::to_string(kNumMathThreadsPerGroup) + R"(, )" +
         std::to_string(num_tma_multicast) + R"(, SchedulerType, )" + input_type + R"(>;
}
)";
}

class Compiler {
 public:
  static Compiler& getInstance() {
    static Compiler instance;
    return instance;
  }

  [[nodiscard]] bool isValid() const {
    if (getJitIncludeDirs().empty() || !executableExists(getNvccCompiler())) {
      return false;
    }
    std::error_code error;
    for (const auto& directory : getJitIncludeDirs()) {
      if (!std::filesystem::is_directory(directory, error)) {
        return false;
      }
    }
    return true;
  }

  static void setIncludeDirs(const std::vector<std::filesystem::path>& dirs) {
    setJitIncludeDirs(dirs);
  }

  Runtime* build(uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
                 uint32_t block_k, uint32_t num_groups, uint32_t num_stages,
                 uint32_t num_tma_multicast, deep_gemm::GemmType gemm_type,
                 bool swap_ab = false) {
    CUcontext context = nullptr;
    CHECK_CUDA(cuCtxGetCurrent(&context));
    if (context == nullptr) {
      throw std::runtime_error("DeepGEMM requires a current CUDA context");
    }
    CUdevice device = 0;
    CHECK_CUDA(cuCtxGetDevice(&device));
    int major = 0;
    int minor = 0;
    CHECK_CUDA(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    if (major != 9 || minor != 0) {
      throw std::runtime_error("DeepGEMM requires an SM90 CUDA device");
    }
    ensurePrivateCacheRoot();

    std::string name = std::string(swap_ab ? "gemm_swap_ab_" : "gemm_") +
                       std::to_string(shape_n) + "_" + std::to_string(shape_k) + "_" +
                       std::to_string(block_m) + "_" + std::to_string(block_n) + "_" +
                       std::to_string(block_k) + "_" + std::to_string(num_groups) + "_" +
                       std::to_string(num_stages) + "_" + std::to_string(num_tma_multicast) + "_" +
                       gemm_type_to_string(gemm_type);
    auto path = getCacheDir() / name;
    auto& runtime_cache = getGlobalRuntimeCache();
    if (auto* cached = runtime_cache.get(path.string(), gemm_type); cached != nullptr) {
      return cached;
    }
    if (!isValid()) {
      throw std::runtime_error(
          "DeepGEMM kernel is not cached and a runtime nvcc or include bundle is unavailable");
    }

    std::error_code error;
    std::filesystem::create_directories(getTmpDir(), error);
    if (error) {
      throw std::system_error(error, "Failed to create DeepGEMM temporary directory");
    }
    std::filesystem::create_directories(path, error);
    if (error) {
      throw std::system_error(error, "Failed to create DeepGEMM cache directory");
    }

    auto tmp_path = getTmpDir() / (name + "_" + generateUniqueId());
    std::filesystem::create_directories(tmp_path, error);
    if (error) {
      throw std::system_error(error, "Failed to create DeepGEMM compilation directory");
    }
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    } cleanup{tmp_path};

    auto source_path = tmp_path / "kernel.cu";
    auto temporary_cubin = tmp_path / kKernelName;
    auto cached_cubin = path / kKernelName;
    {
      std::ofstream source(source_path, std::ios::binary);
      source << generateKernel(shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages,
                               num_tma_multicast, gemm_type, swap_ab);
      if (!source) {
        throw std::runtime_error("Failed to write DeepGEMM kernel source");
      }
    }

    std::vector<std::string> command = {
        getNvccCompiler(),
        source_path.string(),
        "-o",
        temporary_cubin.string(),
        "-std=c++17",
        "--gpu-architecture=sm_90a",
        "--ptxas-options=-allow-expensive-optimizations=true",
        "--ptxas-options=--register-usage-level=10",
        "--diag-suppress=161,174,177,940",
        "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1",
        "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1",
        "-O3",
        "-cubin",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi",
    };
    for (const auto& directory : getJitIncludeDirs()) {
      command.push_back("-I" + directory.string());
    }
    auto compile = runCompiler(command);
    if (compile.exit_code != 0) {
      throw std::runtime_error("DeepGEMM nvcc failed with exit code " +
                               std::to_string(compile.exit_code) + ":\n" + compile.output);
    }
    if (!Runtime::isPathValid(tmp_path.string())) {
      throw std::runtime_error("DeepGEMM nvcc succeeded without producing a cubin");
    }

    std::filesystem::rename(temporary_cubin, cached_cubin, error);
    if (error && !Runtime::isPathValid(path.string())) {
      throw std::system_error(error, "Failed to publish DeepGEMM cubin");
    }
    auto runtime = std::make_unique<Runtime>(path.string(), std::vector<char>(), gemm_type);
    try {
      return runtime_cache.insert(path.string(), std::move(runtime));
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove(cached_cubin, ignored);
      throw;
    }
  }

 private:
  Compiler() = default;
  Compiler(const Compiler&) = delete;
  Compiler& operator=(const Compiler&) = delete;
};

inline Compiler& getGlobalCompiler() { return Compiler::getInstance(); }

}  // namespace deep_gemm::jit
