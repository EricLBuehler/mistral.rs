#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel for packing 1-bit values
// Takes 8 1-bit values and packs them into a single 8-bit integer
__global__ void pack_1bit_kernel(const uint8_t *__restrict__ input,
                                 uint8_t *__restrict__ output,
                                 size_t num_input_elements,
                                 size_t input_width) {

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = num_input_elements / 8;

  if (tid < step * input_width) {
    const size_t row = tid / input_width;
    const size_t col = tid % input_width;

    if (row < step) {
      // Gather 8 1-bit values
      uint8_t packed = 0;
      for (int i = 0; i < 8; i++) {
        size_t idx = (row + i * step) * input_width + col;
        if (row + i * step < num_input_elements) {
          uint8_t bit = input[idx] & 0x1;
          packed |= (bit << (7 - i));
        }
      }

      output[row * input_width + col] = packed;
    }
  }
}

// Kernel for packing 2-bit values
// Takes 4 2-bit values and packs them into a single 8-bit integer
__global__ void pack_2bit_kernel(const uint8_t *__restrict__ input,
                                 uint8_t *__restrict__ output,
                                 size_t num_input_elements,
                                 size_t input_width) {

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = num_input_elements / 4;

  if (tid < step * input_width) {
    const size_t row = tid / input_width;
    const size_t col = tid % input_width;

    if (row < step) {
      // Gather 4 2-bit values
      uint8_t packed = 0;
      for (int i = 0; i < 4; i++) {
        size_t idx = (row + i * step) * input_width + col;
        if (row + i * step < num_input_elements) {
          uint8_t val = input[idx] & 0x3;
          packed |= (val << (6 - i * 2));
        }
      }

      output[row * input_width + col] = packed;
    }
  }
}

// Kernel for packing 3-bit values
// Takes 10 3-bit values and packs them into a single 32-bit integer
__global__ void pack_3bit_kernel(const uint32_t *__restrict__ input,
                                 int32_t *__restrict__ output,
                                 size_t num_input_elements,
                                 size_t input_width) {

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = num_input_elements / 10;

  if (tid < step * input_width) {
    const size_t row = tid / input_width;
    const size_t col = tid % input_width;

    if (row < step) {
      // Gather 10 3-bit values
      int32_t packed = 0;
      for (int i = 0; i < 10; i++) {
        size_t idx = (row + i * step) * input_width + col;
        if (row + i * step < num_input_elements) {
          int32_t val = input[idx] & 0x7;
          packed |= (val << (27 - i * 3));
        }
      }

      output[row * input_width + col] = packed;
    }
  }
}

// Kernel for packing 4-bit values
// Takes 2 4-bit values and packs them into a single 8-bit integer
__global__ void pack_4bit_kernel(const uint8_t *__restrict__ input,
                                 uint8_t *__restrict__ output,
                                 size_t num_input_elements,
                                 size_t input_width) {

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = num_input_elements / 2;

  if (tid < step * input_width) {
    const size_t row = tid / input_width;
    const size_t col = tid % input_width;

    if (row < step) {
      // Gather 2 4-bit values
      uint8_t packed = 0;
      for (int i = 0; i < 2; i++) {
        size_t idx = (row + i * step) * input_width + col;
        if (row + i * step < num_input_elements) {
          uint8_t val = input[idx] & 0xF;
          packed |= (val << (4 - i * 4));
        }
      }

      output[row * input_width + col] = packed;
    }
  }
}

// 8-bit doesn't need packing, just copy
__global__ void pack_8bit_kernel(const uint8_t *__restrict__ input,
                                 uint8_t *__restrict__ output,
                                 size_t num_elements) {

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_elements) {
    output[tid] = input[tid];
  }
}

// Launch functions
extern "C" void launch_pack_1bit_kernel(const uint8_t *d_input,
                                        uint8_t *d_output,
                                        size_t num_input_elements,
                                        size_t input_width,
                                        cudaStream_t stream) {

  const size_t step = num_input_elements / 8;
  const size_t total_threads = step * input_width;
  const int block_size = 256;
  const int num_blocks = (total_threads + block_size - 1) / block_size;

  pack_1bit_kernel<<<num_blocks, block_size, 0, stream>>>(
      d_input, d_output, num_input_elements, input_width);
}

extern "C" void launch_pack_2bit_kernel(const uint8_t *d_input,
                                        uint8_t *d_output,
                                        size_t num_input_elements,
                                        size_t input_width,
                                        cudaStream_t stream) {

  const size_t step = num_input_elements / 4;
  const size_t total_threads = step * input_width;
  const int block_size = 256;
  const int num_blocks = (total_threads + block_size - 1) / block_size;

  pack_2bit_kernel<<<num_blocks, block_size, 0, stream>>>(
      d_input, d_output, num_input_elements, input_width);
}

extern "C" void launch_pack_3bit_kernel(const uint32_t *d_input,
                                        int32_t *d_output,
                                        size_t num_input_elements,
                                        size_t input_width,
                                        cudaStream_t stream) {

  const size_t step = num_input_elements / 10;
  const size_t total_threads = step * input_width;
  const int block_size = 256;
  const int num_blocks = (total_threads + block_size - 1) / block_size;

  pack_3bit_kernel<<<num_blocks, block_size, 0, stream>>>(
      d_input, d_output, num_input_elements, input_width);
}

extern "C" void launch_pack_4bit_kernel(const uint8_t *d_input,
                                        uint8_t *d_output,
                                        size_t num_input_elements,
                                        size_t input_width,
                                        cudaStream_t stream) {

  const size_t step = num_input_elements / 2;
  const size_t total_threads = step * input_width;
  const int block_size = 256;
  const int num_blocks = (total_threads + block_size - 1) / block_size;

  pack_4bit_kernel<<<num_blocks, block_size, 0, stream>>>(
      d_input, d_output, num_input_elements, input_width);
}

extern "C" void launch_pack_8bit_kernel(const uint8_t *d_input,
                                        uint8_t *d_output, size_t num_elements,
                                        cudaStream_t stream) {

  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  pack_8bit_kernel<<<num_blocks, block_size, 0, stream>>>(d_input, d_output,
                                                          num_elements);
}