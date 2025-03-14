// https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
#endif
#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
#endif

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}
#define BLOCK_SIZE 256  //~256
#define SHARED_SIZE 512 //~512

/*******************************************************************************************************************************************/
/************* 8-bit *************/
/*******************************************************************************************************************************************/
// Simple
template <typename T>
__global__ void dequantize_8bit_u8_kernel(unsigned char *Wq_packed, T *scale,
                                          T *zero, T *W_r, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = h * w;
  if (i >= n)
    return;

  int j = i % w;
  W_r[i] = ((T)(Wq_packed[i]) - zero[j]) * scale[j];
}

extern "C" void dequantize_8bit_u8_kernel_f32(unsigned char *Wq_packed,
                                              float *scale, float *zero,
                                              float *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_8bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}

#if __CUDA_ARCH__ >= 530
extern "C" void dequantize_8bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              __half *scale, __half *zero,
                                              __half *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_8bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_8bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              uint16_t *scale, uint16_t *zero,
                                              uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" void dequantize_8bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               __nv_bfloat16 *scale,
                                               __nv_bfloat16 *zero,
                                               __nv_bfloat16 *W_r, int h,
                                               int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_8bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_8bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               uint16_t *scale, uint16_t *zero,
                                               uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

/*******************************************************************************************************************************************/
/************* 4-bit *************/
/*******************************************************************************************************************************************/

// Simple
/*__global__ void unpack_4bit_u8_kernel(unsigned char* Wq_packed, unsigned char*
Wq_unpacked, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=n)
return;

        Wq_unpacked[i]     = (Wq_packed[i] & 0xF0) >> 4;  //First chunk
        Wq_unpacked[i + n] = (Wq_packed[i] & 0x0F);       //Second chunk
}*/

// Simple
template <typename T>
__global__ void dequantize_4bit_u8_kernel(unsigned char *Wq_packed, T *scale,
                                          T *zero, T *W_r, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = h * w;
  if (i >= n)
    return;

  int j = i % w;
  // W_r[i]     = (T)((Wq_packed[i] & 0xF0) >> 4);//((T)((Wq_packed[i] & 0xF0)
  // >> 4) - zero[j])*scale[j];  //First chunk W_r[i + n] = (T)((Wq_packed[i] &
  // 0x0F)) + (T)10000;//((T)((Wq_packed[i] & 0x0F))      - zero[j])*scale[j];
  // //Second chunk
  W_r[i] = ((T)((Wq_packed[i] & 0xF0) >> 4) - zero[j]) * scale[j]; // First
                                                                   // chunk
  W_r[i + n] = ((T)((Wq_packed[i] & 0x0F)) - zero[j]) * scale[j];  // Second
                                                                   // chunk
}

extern "C" void dequantize_4bit_u8_kernel_f32(unsigned char *Wq_packed,
                                              float *scale, float *zero,
                                              float *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_4bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}

#if __CUDA_ARCH__ >= 530
extern "C" void dequantize_4bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              __half *scale, __half *zero,
                                              __half *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_4bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_4bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              uint16_t *scale, uint16_t *zero,
                                              uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" void dequantize_4bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               __nv_bfloat16 *scale,
                                               __nv_bfloat16 *zero,
                                               __nv_bfloat16 *W_r, int h,
                                               int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_4bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_4bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               uint16_t *scale, uint16_t *zero,
                                               uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

/*******************************************************************************************************************************************/
/************* 2-bit *************/
/*******************************************************************************************************************************************/

// Simple
/*__global__ void unpack_2bit_u8_kernel(unsigned char* Wq_packed, unsigned char*
Wq_unpacked, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=n)
return;

        Wq_unpacked[i]       = (Wq_packed[i] & 0xC0) >> 6;  //1st chunk
        Wq_unpacked[i + n]   = (Wq_packed[i] & 0x30) >> 4;  //2nd chunk
        Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x0C) >> 2;  //3rd chunk
        Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x03);       //4th chunk
}*/

// Simple
template <typename T>
__global__ void dequantize_2bit_u8_kernel(unsigned char *Wq_packed, T *scale,
                                          T *zero, T *W_r, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = h * w;
  if (i >= n)
    return;

  int j = i % w;
  W_r[i] = ((T)((Wq_packed[i] & 0xC0) >> 6) - zero[j]) * scale[j]; // 1st chunk
  W_r[i + n] =
      ((T)((Wq_packed[i] & 0x30) >> 4) - zero[j]) * scale[j]; // 2nd chunk
  W_r[i + n * 2] =
      ((T)((Wq_packed[i] & 0x0C) >> 2) - zero[j]) * scale[j]; // 3rd chunk
  W_r[i + n * 3] =
      ((T)((Wq_packed[i] & 0x03)) - zero[j]) * scale[j]; // 4th chunk
}

extern "C" void dequantize_2bit_u8_kernel_f32(unsigned char *Wq_packed,
                                              float *scale, float *zero,
                                              float *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_2bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}

#if __CUDA_ARCH__ >= 530
extern "C" void dequantize_2bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              __half *scale, __half *zero,
                                              __half *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_2bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_2bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              uint16_t *scale, uint16_t *zero,
                                              uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" void dequantize_2bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               __nv_bfloat16 *scale,
                                               __nv_bfloat16 *zero,
                                               __nv_bfloat16 *W_r, int h,
                                               int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_2bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_2bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               uint16_t *scale, uint16_t *zero,
                                               uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

// //Shared
// template <typename scalar_t>
// __global__ void dequantize_2bit_u8_kernel(unsigned char* Wq_packed, scalar_t*
// scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 	int i =
// blockIdx.x*blockDim.x + threadIdx.x; 	int n = h*w; 	int s =
// threadIdx.x;

// 	if(i>=n) return;

// 	__shared__ unsigned char shared[BLOCK_SIZE];
// 	__shared__ scalar_t shared_meta[BLOCK_SIZE][2];

// 	int j             = i % w;
// 	shared[s]         = Wq_packed[i];
// 	shared_meta[s][0] = zero[j];
// 	shared_meta[s][1] = scale[j];
// 	__syncthreads();

// 	W_r[i]       = (scalar_t((shared[s] & 0xC0) >> 6) -
// shared_meta[s][0])*shared_meta[s][1];  //1st chunk 	W_r[i + n]   =
// (scalar_t((shared[s] & 0x30) >> 4) - shared_meta[s][0])*shared_meta[s][1];
// //2nd chunk 	W_r[i + n*2] = (scalar_t((shared[s] & 0x0C) >> 2) -
// shared_meta[s][0])*shared_meta[s][1];  //3rd chunk 	W_r[i + n*3] =
// (scalar_t((shared[s] & 0x03))      - shared_meta[s][0])*shared_meta[s][1];
// //4th chunk
// }

/*******************************************************************************************************************************************/
/************* 1-bit *************/
/*******************************************************************************************************************************************/

// Simple
/*__global__ void unpack_1bit_u8_kernel(unsigned char* Wq_packed, unsigned char*
Wq_unpacked, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=n)
return;

        Wq_unpacked[i]       = (Wq_packed[i] & 0x80) >> 7;  //1st chunk
        Wq_unpacked[i + n]   = (Wq_packed[i] & 0x40) >> 6;  //2nd chunk
        Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x20) >> 5;  //3rd chunk
        Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x10) >> 4;  //4th chunk
        Wq_unpacked[i + n*4] = (Wq_packed[i] & 0x08) >> 3;  //5th chunk
        Wq_unpacked[i + n*5] = (Wq_packed[i] & 0x04) >> 2;  //6th chunk
        Wq_unpacked[i + n*6] = (Wq_packed[i] & 0x02) >> 1;  //7th chunk
        Wq_unpacked[i + n*7] = (Wq_packed[i] & 0x01);       //8th chunk
}*/

// Simple
template <typename T>
__global__ void dequantize_1bit_u8_kernel(unsigned char *Wq_packed, T *scale,
                                          T *zero, T *W_r, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = h * w;
  if (i >= n)
    return;

  int j = i % w;
  W_r[i] = ((T)((Wq_packed[i] & 0x80) >> 7) - zero[j]) * scale[j]; // 1st chunk
  W_r[i + n] =
      ((T)((Wq_packed[i] & 0x40) >> 6) - zero[j]) * scale[j]; // 2nd chunk
  W_r[i + n * 2] =
      ((T)((Wq_packed[i] & 0x20) >> 5) - zero[j]) * scale[j]; // 3rd chunk
  W_r[i + n * 3] =
      ((T)((Wq_packed[i] & 0x10) >> 4) - zero[j]) * scale[j]; // 4th chunk
  W_r[i + n * 4] =
      ((T)((Wq_packed[i] & 0x08) >> 3) - zero[j]) * scale[j]; // 5th chunk
  W_r[i + n * 5] =
      ((T)((Wq_packed[i] & 0x04) >> 2) - zero[j]) * scale[j]; // 6th chunk
  W_r[i + n * 6] =
      ((T)((Wq_packed[i] & 0x02) >> 1) - zero[j]) * scale[j]; // 7th chunk
  W_r[i + n * 7] =
      ((T)((Wq_packed[i] & 0x01)) - zero[j]) * scale[j]; // 8th chunk
}

extern "C" void dequantize_1bit_u8_kernel_f32(unsigned char *Wq_packed,
                                              float *scale, float *zero,
                                              float *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_1bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}

#if __CUDA_ARCH__ >= 530
extern "C" void dequantize_1bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              __half *scale, __half *zero,
                                              __half *W_r, int h, int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_1bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_1bit_u8_kernel_f16(unsigned char *Wq_packed,
                                              uint16_t *scale, uint16_t *zero,
                                              uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" void dequantize_1bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               __nv_bfloat16 *scale,
                                               __nv_bfloat16 *zero,
                                               __nv_bfloat16 *W_r, int h,
                                               int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_1bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_1bit_u8_kernel_bf16(unsigned char *Wq_packed,
                                               uint16_t *scale, uint16_t *zero,
                                               uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

// //Shared
// template <typename scalar_t>
// __global__ void dequantize_1bit_u8_kernel(unsigned char* Wq_packed, scalar_t*
// scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 	int i =
// blockIdx.x*blockDim.x + threadIdx.x; 	int s = threadIdx.x; 	int n =
// h*w; 	if(i>=n) return;

// 	__shared__ unsigned char shared[BLOCK_SIZE];
// 	__shared__ scalar_t shared_meta[BLOCK_SIZE][2];

// 	int j             = i % w;
// 	shared[s]         = Wq_packed[i];
// 	shared_meta[s][0] = zero[j];
// 	shared_meta[s][1] = scale[j];
// 	__syncthreads();

// 	W_r[i]       = (scalar_t((shared[s] & 0x80) >> 7) -
// shared_meta[s][0])*shared_meta[s][1]; //1st chunk 	W_r[i + n]   =
// (scalar_t((shared[s] & 0x40) >> 6) - shared_meta[s][0])*shared_meta[s][1];
// //2nd chunk 	W_r[i + n*2] = (scalar_t((shared[s] & 0x20) >> 5) -
// shared_meta[s][0])*shared_meta[s][1]; //3rd chunk 	W_r[i + n*3] =
// (scalar_t((shared[s] & 0x10) >> 4) - shared_meta[s][0])*shared_meta[s][1];
// //4th chunk 	W_r[i + n*4] = (scalar_t((shared[s] & 0x08) >> 3) -
// shared_meta[s][0])*shared_meta[s][1]; //5th chunk 	W_r[i + n*5] =
// (scalar_t((shared[s] & 0x04) >> 2) - shared_meta[s][0])*shared_meta[s][1];
// //6th chunk 	W_r[i + n*6] = (scalar_t((shared[s] & 0x02) >> 1) -
// shared_meta[s][0])*shared_meta[s][1]; //7th chunk 	W_r[i + n*7] =
// (scalar_t((shared[s] & 0x01))      - shared_meta[s][0])*shared_meta[s][1];
// //8th chunk
// }

/*******************************************************************************************************************************************/
/************* 3-bit *************/
/*******************************************************************************************************************************************/

// Simple
/*__global__ void unpack_3bit_32_kernel(int32_t* Wq_packed, unsigned char*
Wq_unpacked, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=n)
return;

        Wq_unpacked[i]       = (Wq_packed[i] & 0x38000000) >> 27;  //1st chunk
        Wq_unpacked[i + n]   = (Wq_packed[i] & 0x07000000) >> 24;  //2nd chunk
        Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x00E00000) >> 21;  //3rd chunk
        Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x001C0000) >> 18;  //4th chunk
        Wq_unpacked[i + n*4] = (Wq_packed[i] & 0x00038000) >> 15;  //5th chunk
        Wq_unpacked[i + n*5] = (Wq_packed[i] & 0x00007000) >> 12;  //6th chunk
        Wq_unpacked[i + n*6] = (Wq_packed[i] & 0x00000E00) >> 9;   //7th chunk
        Wq_unpacked[i + n*7] = (Wq_packed[i] & 0x000001C0) >> 6;   //8th chunk
        Wq_unpacked[i + n*8] = (Wq_packed[i] & 0x00000038) >> 3;   //9th chunk
        Wq_unpacked[i + n*9] = (Wq_packed[i] & 0x00000007);        //10th chunk
}*/

// Simple
template <typename T>
__global__ void dequantize_3bit_32_kernel(int32_t *Wq_packed, T *scale, T *zero,
                                          T *W_r, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = h * w;
  if (i >= n)
    return;

  int j = i % w;
  W_r[i] = ((T)((Wq_packed[i] & 0x38000000) >> 27) - zero[j]) *
           scale[j]; // 1st chunk
  W_r[i + n] = ((T)((Wq_packed[i] & 0x07000000) >> 24) - zero[j]) *
               scale[j]; // 2nd chunk
  W_r[i + n * 2] = ((T)((Wq_packed[i] & 0x00E00000) >> 21) - zero[j]) *
                   scale[j]; // 3rd chunk
  W_r[i + n * 3] = ((T)((Wq_packed[i] & 0x001C0000) >> 18) - zero[j]) *
                   scale[j]; // 4th chunk
  W_r[i + n * 4] = ((T)((Wq_packed[i] & 0x00038000) >> 15) - zero[j]) *
                   scale[j]; // 5th chunk
  W_r[i + n * 5] = ((T)((Wq_packed[i] & 0x00007000) >> 12) - zero[j]) *
                   scale[j]; // 6th chunk
  W_r[i + n * 6] =
      ((T)((Wq_packed[i] & 0x00000E00) >> 9) - zero[j]) * scale[j]; // 7th chunk
  W_r[i + n * 7] =
      ((T)((Wq_packed[i] & 0x000001C0) >> 6) - zero[j]) * scale[j]; // 8th chunk
  W_r[i + n * 8] =
      ((T)((Wq_packed[i] & 0x00000038) >> 3) - zero[j]) * scale[j]; // 9th chunk
  W_r[i + n * 9] =
      ((T)((Wq_packed[i] & 0x00000007)) - zero[j]) * scale[j]; // 10th chunk
}

extern "C" void dequantize_3bit_32_kernel_f32(int32_t *Wq_packed, float *scale,
                                              float *zero, float *W_r, int h,
                                              int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_3bit_32_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}

#if __CUDA_ARCH__ >= 530
extern "C" void dequantize_3bit_32_kernel_f16(int32_t *Wq_packed, __half *scale,
                                              __half *zero, __half *W_r, int h,
                                              int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_3bit_32_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_3bit_32_kernel_f16(unsigned char *Wq_packed,
                                              uint16_t *scale, uint16_t *zero,
                                              uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" void dequantize_3bit_32_kernel_bf16(int32_t *Wq_packed,
                                               __nv_bfloat16 *scale,
                                               __nv_bfloat16 *zero,
                                               __nv_bfloat16 *W_r, int h,
                                               int w) {
  int blocks = cdiv(h * w, BLOCK_SIZE);
  dequantize_3bit_32_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed, scale, zero, W_r,
                                                    h, w);
}
#else
extern "C" void dequantize_3bit_32_kernel_bf16(unsigned char *Wq_packed,
                                               uint16_t *scale, uint16_t *zero,
                                               uint16_t *W_r, int h, int w) {
  assert(false);
}
#endif