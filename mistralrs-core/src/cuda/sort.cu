#include<stdint.h>
#include <limits>
#include "cuda_fp16.h"
#include "cuda_bf16.h"
template<typename T>
inline __device__ void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T, bool ascending>
__global__ void bitonic_sort_kernel(T* arr, uint32_t * dst, int j, int k) {
    unsigned int i, ij;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ij = i ^ j;

    if (ij > i) {
        if constexpr (ascending) {
            if ((i & k) == 0) {
                if (arr[i] > arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            } else {
                if (arr[i] < arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
        }

        if constexpr (!ascending) {
            if ((i & k) != 0) {
                if (arr[i] > arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            } else {
                if (arr[i] < arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
        }
    }
    __syncthreads();
}

int next_power_of_2(int x){
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#define ASORT_OP(T, RUST_NAME, ASC) \
extern "C" void RUST_NAME(  \
    void * x1, void * dst1, const int nrows, const int ncols, bool inplace, int64_t stream \
) { \
    T* x = reinterpret_cast<T*>(x1);\
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst1);\
    const cudaStream_t custream = (cudaStream_t)stream;\
    int ncols_pad = next_power_of_2(ncols);\
    T* x_row_padded;\
    uint32_t * dst_row_padded;\
    cudaMallocAsync((void**)&x_row_padded, ncols_pad * sizeof(T), custream);\
    cudaMallocAsync((void**)&dst_row_padded, ncols_pad * sizeof(uint32_t), custream);\
    uint32_t * indices_padded = (uint32_t*)malloc(ncols_pad * sizeof(uint32_t));\
    for (int i=0; i<ncols_pad; i++) {\
        indices_padded[i] = i;\
    }\
    T * values_padded = (T*)malloc((ncols_pad - ncols) * sizeof(T));\
    for (int i=0; i<ncols_pad - ncols; i++) {\
        values_padded[i] = ASC ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();\
    }\
    int max_threads_per_block = 1024;\
    int threads_per_block = max_threads_per_block > ncols_pad ? ncols_pad : max_threads_per_block;\
    int blocks_per_row = (ncols_pad + threads_per_block - 1) / threads_per_block;\
    for (int row=0; row< nrows; row++) {\
        T* x_row = x + row * ncols;\
        uint32_t* dst_row = dst + row * ncols;\
        cudaMemcpyAsync(x_row_padded, x_row, ncols * sizeof(T), cudaMemcpyDeviceToDevice, custream);\
        if (ncols_pad - ncols > 0)\
            cudaMemcpyAsync(x_row_padded + ncols, values_padded, (ncols_pad - ncols) * sizeof(T), cudaMemcpyHostToDevice, custream);\
        cudaMemcpyAsync(dst_row_padded, indices_padded, ncols_pad * sizeof(uint32_t), cudaMemcpyHostToDevice, custream);\
        for (int k = 2; k <= ncols_pad; k <<= 1) {\
            for (int j = k >> 1; j > 0; j = j >> 1) {\
                bitonic_sort_kernel<T, ASC><<<blocks_per_row, threads_per_block, 0, custream>>>(x_row_padded, dst_row_padded, j, k);\
            }\
        }\
        if (inplace)\
            cudaMemcpyAsync(x_row, x_row_padded, ncols * sizeof(T), cudaMemcpyDeviceToDevice, custream);\
        cudaMemcpyAsync(dst_row, dst_row_padded, ncols * sizeof(uint32_t), cudaMemcpyDeviceToDevice, custream);\
    }\
    cudaFreeAsync(x_row_padded, custream);\
    cudaFreeAsync(dst_row_padded, custream);\
    cudaStreamSynchronize(custream);\
    free(indices_padded);\
    free(values_padded);\
}\

ASORT_OP(__nv_bfloat16, asort_asc_bf16, true)
ASORT_OP(__nv_bfloat16, asort_desc_bf16, false)

ASORT_OP(__half, asort_asc_f16, true)
ASORT_OP(__half, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(int64_t, asort_asc_i64, true)

ASORT_OP(float, asort_desc_f32, false)
ASORT_OP(double, asort_desc_f64, false)
ASORT_OP(uint8_t, asort_desc_u8, false)
ASORT_OP(uint32_t, asort_desc_u32, false)
ASORT_OP(int64_t, asort_desc_i64, false)
