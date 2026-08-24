/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda_runtime_api.h>

struct FlashAttentionCudaError {
    cudaError_t status;
};

#define CHECK_CUDA(call) \
    do { \
        const cudaError_t status_ = (call); \
        if (status_ != cudaSuccess) { \
            throw FlashAttentionCudaError{status_}; \
        } \
    } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
