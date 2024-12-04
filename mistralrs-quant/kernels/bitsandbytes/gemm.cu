#include <cub/warp/warp_reduce.cuh>
#include <cub/cub.cuh>

#define num_values_4bit 32
template <typename T, int THREADS, int BITS> __global__ void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{
  // per threadblock:
  // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
  // 4 warps -> 4 loads per iter
  // 1x32 * 32x4 -> 1x4 outputs per thread block
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[THREADS/32];

  const int warp_idx = threadIdx.x / 32;
  const int warp_lane = threadIdx.x % 32;
  const int row_B = (THREADS/32)*blockIdx.x + warp_idx;
  const int num_values_8bit = num_values_4bit/2;
  float local_C = 0.0f;

  unsigned char local_B_4bit[num_values_8bit];
  T local_B[num_values_4bit/4];
  T local_A[num_values_4bit/4];
  __shared__ T quant_map[16];
	T local_absmax = T(0.0f);

  for(int i = threadIdx.x; i < 16; i++)
    quant_map[i] = T(datatype[i]);
  __syncthreads();

  // A: [1, K]
  // B: [N, K]
  for(int inner_idx = warp_lane*num_values_4bit; inner_idx < K; inner_idx += 32*num_values_4bit)
  {
    int inner_idx_halved = inner_idx/2;
    int offset_B = ldb*row_B;
    int absidx = ((2*offset_B)+inner_idx)/blocksize;
	  local_absmax = __ldg(&(absmax[absidx]));

    if(row_B < M)
    {
      if((inner_idx_halved + num_values_8bit) < (K/2))
      {
        // this is the most important for performance considerations
        reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] = reinterpret_cast<int4*>(B)[(offset_B+(inner_idx_halved))/(num_values_8bit)];
      }
      else
      {
        #pragma unroll
        for(int j = 0; j < (num_values_8bit); j++)
          if((inner_idx_halved) + j < (K/2))
            local_B_4bit[j] = B[offset_B+inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    }
    else
    {
      #pragma unroll
      for(int j = 0; j < (num_values_8bit); j++)
          local_B_4bit[j] = 0b01110111;
    }

    for(int i = 0; i < 4; i++)
    {
      #pragma unroll
      for(int k = 0; k < num_values_8bit/4; k++)
      {
        #if __CUDA_ARCH__ >= 800
          local_B[k*2] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*local_absmax;
          local_B[k*2 + 1] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*local_absmax;
        #else
          // bf16 multipliation not supported
          local_B[k*2] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*(float)local_absmax);
          local_B[k*2 + 1] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*(float)local_absmax);
        #endif
      }

      if(inner_idx+(num_values_4bit/4) + (i*num_values_4bit/4) < K)
      {
        // this is also relatively important for performance
        if(BITS==16)
        {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/4) + i];
        }
        else
        {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 0];
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] = reinterpret_cast<int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 1];
        }

      }
      else
        #pragma unroll
        for(int k = 0; k < num_values_4bit/4; k++)
          if(inner_idx + (i*num_values_4bit/4) + k < K)
            local_A[k] = A[inner_idx + k + (i*num_values_4bit/4)];
          else
            local_A[k] = T(0.0f);


      // accumulate in float; small performance hit for Ampere, but lower error for outputs
      #pragma unroll
      for(int k = 0; k < num_values_4bit/4; k++)
      {
        #if __CUDA_ARCH__ >= 800
          local_C += (float)(local_A[k]*local_B[k]);
        #else
          // bf16 multipliation not supported
          local_C += ((float)local_A[k]*(float)local_B[k]);
        #endif
      }
    }
  }

  local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

  if(row_B < M && warp_lane == 0)
    out[row_B] = T(local_C);

}

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize, cudaStream_t stream)
{

  int num_blocks = (m+3)/4;
  kgemm_4bit_inference_naive<T, 128, BITS><<< num_blocks, 128, 0, stream>>>(m,  n,  k, A,  B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}

extern "C" void gemm_4bit_inference_naive_f16(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *code, half * out,  int lda, int ldb, int ldc, int blocksize, cudaStream_t stream)
{ gemm_4bit_inference_naive<half, 16>(m, n, k, A, B, absmax,  code, out, lda, ldb, ldc, blocksize, stream); }

extern "C" void gemm_4bit_inference_naive_bf16(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *code, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize, cudaStream_t stream)
{ gemm_4bit_inference_naive<__nv_bfloat16, 16>(m, n, k, A, B, absmax,  code, out, lda, ldb, ldc, blocksize, stream); }

extern "C" void gemm_4bit_inference_naive_f32(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *code, float * out,  int lda, int ldb, int ldc, int blocksize, cudaStream_t stream)
{ gemm_4bit_inference_naive<float, 32>(m, n, k, A, B, absmax,  code, out, lda, ldb, ldc, blocksize, stream); }

