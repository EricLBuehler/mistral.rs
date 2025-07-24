#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

// Block sizes for tiled matrix multiplication
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define WARP_SIZE 32

// Structure to hold sorted token-expert pairs
struct TokenExpertPair {
    int token_idx;
    int expert_idx;
    float routing_weight;
    int original_idx;  // Position in original routing weights
};

// Helper function for SiLU activation
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Helper function for GELU activation
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Simple optimized fused MoE kernel with shared memory
template<typename T>
__global__ void fused_moe_kernel(
    const T* input,              // [num_tokens, hidden_dim]
    const T* gate_weights,       // [num_experts, hidden_dim, intermediate_dim]
    const T* up_weights,         // [num_experts, hidden_dim, intermediate_dim]
    const T* down_weights,       // [num_experts, intermediate_dim, hidden_dim]
    const float* routing_weights,// [num_tokens, num_selected_experts]
    const uint32_t* expert_indices,// [num_tokens, num_selected_experts]
    T* output,                   // [num_tokens, hidden_dim]
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int activation_type          // 0: SiLU, 1: GELU, 2: ReLU
) {
    extern __shared__ char shared_mem[];
    T* shared_input = (T*)shared_mem;
    T* shared_intermediate = shared_input + hidden_dim;
    
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    if (token_idx >= num_tokens) {
        return;
    }
    
    // Load input to shared memory
    for (int i = tid; i < hidden_dim; i += block_size) {
        shared_input[i] = input[token_idx * hidden_dim + i];
    }
    __syncthreads();
    
    // Initialize output to zero
    for (int i = tid; i < hidden_dim; i += block_size) {
        output[token_idx * hidden_dim + i] = T(0.0f);
    }
    __syncthreads();
    
    // Process each selected expert
    for (int k = 0; k < num_selected_experts; k++) {
        int expert_id = expert_indices[token_idx * num_selected_experts + k];
        float routing_weight = routing_weights[token_idx * num_selected_experts + k];
        
        const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
        const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;
        const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;
        
        // Compute gate and up projections
        for (int i = tid; i < intermediate_dim; i += block_size) {
            float gate_val = 0.0f;
            float up_val = 0.0f;
            
            for (int j = 0; j < hidden_dim; j++) {
                float input_val = float(shared_input[j]);
                gate_val += input_val * float(gate_w[j * intermediate_dim + i]);
                up_val += input_val * float(up_w[j * intermediate_dim + i]);
            }
            
            // Apply activation to gate
            if (activation_type == 0) { // SiLU
                gate_val = gate_val / (1.0f + expf(-gate_val));
            } else if (activation_type == 1) { // GELU
                gate_val = 0.5f * gate_val * (1.0f + tanhf(0.7978845608f * (gate_val + 0.044715f * gate_val * gate_val * gate_val)));
            } else if (activation_type == 2) { // ReLU
                gate_val = fmaxf(0.0f, gate_val);
            }
            
            // Multiply gate and up
            shared_intermediate[i] = T(gate_val * up_val);
        }
        __syncthreads();
        
        // Compute down projection and accumulate to output
        for (int i = tid; i < hidden_dim; i += block_size) {
            float down_val = 0.0f;
            
            for (int j = 0; j < intermediate_dim; j++) {
                down_val += float(shared_intermediate[j]) * float(down_w[j * hidden_dim + i]);
            }
            
            output[token_idx * hidden_dim + i] += T(down_val * routing_weight);
        }
        __syncthreads();
    }
}

// Optimized fused MoE kernel with tiling and better memory access patterns
template<typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void fused_moe_kernel_optimized(
    const T* __restrict__ input,              // [num_tokens, hidden_dim]
    const T* __restrict__ gate_weights,       // [num_experts, hidden_dim, intermediate_dim]
    const T* __restrict__ up_weights,         // [num_experts, hidden_dim, intermediate_dim]
    const T* __restrict__ down_weights,       // [num_experts, intermediate_dim, hidden_dim]
    const TokenExpertPair* __restrict__ sorted_pairs,  // Sorted token-expert pairs
    const int* __restrict__ expert_offsets,   // Start offset for each expert in sorted_pairs
    T* __restrict__ output,                   // [num_tokens, hidden_dim]
    T* __restrict__ intermediate_cache,       // [num_tokens, intermediate_dim] workspace
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int total_pairs,
    int activation_type                       // 0: SiLU, 1: GELU, 2: ReLU
) {
    // Thread block processes one tile of the output
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Shared memory for tiles
    extern __shared__ char shared_mem[];
    T* tile_input = (T*)shared_mem;
    T* tile_weight = tile_input + TILE_M * TILE_K;
    float* tile_accum = (float*)(tile_weight + TILE_K * TILE_N);
    
    // Grid-stride loop over expert groups
    for (int expert_id = blockIdx.z; expert_id < num_experts; expert_id += gridDim.z) {
        int start_idx = (expert_id == 0) ? 0 : expert_offsets[expert_id - 1];
        int end_idx = expert_offsets[expert_id];
        
        if (start_idx >= end_idx) continue;
        
        // Get expert weight pointers
        const T* gate_w = gate_weights + expert_id * hidden_dim * intermediate_dim;
        const T* up_w = up_weights + expert_id * hidden_dim * intermediate_dim;
        const T* down_w = down_weights + expert_id * intermediate_dim * hidden_dim;
        
        // Process tokens assigned to this expert in blocks
        for (int token_block = start_idx + blockIdx.x * TILE_M; 
             token_block < end_idx; 
             token_block += gridDim.x * TILE_M) {
            
            // Phase 1: Compute gate and up projections
            for (int out_tile = blockIdx.y * TILE_N; 
                 out_tile < intermediate_dim; 
                 out_tile += gridDim.y * TILE_N) {
                
                // Initialize accumulator
                float accum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float accum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                
                // Loop over K dimension in tiles
                for (int k_tile = 0; k_tile < hidden_dim; k_tile += TILE_K) {
                    // Cooperatively load input tile
                    __syncthreads();
                    for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
                        int local_m = idx / TILE_K;
                        int local_k = idx % TILE_K;
                        int token_idx = token_block + local_m;
                        
                        if (token_idx < end_idx && k_tile + local_k < hidden_dim) {
                            int actual_token = sorted_pairs[token_idx].token_idx;
                            tile_input[local_m * TILE_K + local_k] = 
                                input[actual_token * hidden_dim + k_tile + local_k];
                        } else {
                            tile_input[local_m * TILE_K + local_k] = T(0);
                        }
                    }
                    
                    // Load weight tiles for gate and up
                    for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x * blockDim.y) {
                        int local_k = idx / TILE_N;
                        int local_n = idx % TILE_N;
                        
                        if (k_tile + local_k < hidden_dim && out_tile + local_n < intermediate_dim) {
                            // Gate weights
                            tile_weight[local_k * TILE_N + local_n] = 
                                gate_w[(k_tile + local_k) * intermediate_dim + out_tile + local_n];
                        } else {
                            tile_weight[local_k * TILE_N + local_n] = T(0);
                        }
                    }
                    __syncthreads();
                    
                    // Compute partial dot products for gate
                    int local_m = threadIdx.y;
                    int local_n = threadIdx.x;
                    
                    if (local_m < TILE_M && local_n < TILE_N) {
                        for (int k = 0; k < TILE_K; k++) {
                            accum_gate[0] += float(tile_input[local_m * TILE_K + k]) * 
                                           float(tile_weight[k * TILE_N + local_n]);
                        }
                    }
                    
                    // Load up weights
                    __syncthreads();
                    for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x * blockDim.y) {
                        int local_k = idx / TILE_N;
                        int local_n = idx % TILE_N;
                        
                        if (k_tile + local_k < hidden_dim && out_tile + local_n < intermediate_dim) {
                            tile_weight[local_k * TILE_N + local_n] = 
                                up_w[(k_tile + local_k) * intermediate_dim + out_tile + local_n];
                        }
                    }
                    __syncthreads();
                    
                    // Compute partial dot products for up
                    if (local_m < TILE_M && local_n < TILE_N) {
                        for (int k = 0; k < TILE_K; k++) {
                            accum_up[0] += float(tile_input[local_m * TILE_K + k]) * 
                                         float(tile_weight[k * TILE_N + local_n]);
                        }
                    }
                }
                
                // Apply activation and store intermediate results
                __syncthreads();
                int local_m = threadIdx.y;
                int local_n = threadIdx.x;
                
                if (local_m < TILE_M && local_n < TILE_N) {
                    int token_idx = token_block + local_m;
                    int out_idx = out_tile + local_n;
                    
                    if (token_idx < end_idx && out_idx < intermediate_dim) {
                        float gate_val = accum_gate[0];
                        float up_val = accum_up[0];
                        
                        // Apply activation
                        if (activation_type == 0) {
                            gate_val = silu(gate_val);
                        } else if (activation_type == 1) {
                            gate_val = gelu(gate_val);
                        } else {
                            gate_val = fmaxf(0.0f, gate_val);
                        }
                        
                        // Store to intermediate cache
                        int actual_token = sorted_pairs[token_idx].token_idx;
                        intermediate_cache[actual_token * intermediate_dim + out_idx] = 
                            T(gate_val * up_val);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Phase 2: Down projection
        for (int token_block = start_idx + blockIdx.x * TILE_M; 
             token_block < end_idx; 
             token_block += gridDim.x * TILE_M) {
            
            for (int out_tile = blockIdx.y * TILE_N; 
                 out_tile < hidden_dim; 
                 out_tile += gridDim.y * TILE_N) {
                
                float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                
                // Loop over K dimension (intermediate_dim)
                for (int k_tile = 0; k_tile < intermediate_dim; k_tile += TILE_K) {
                    // Load intermediate results
                    __syncthreads();
                    for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
                        int local_m = idx / TILE_K;
                        int local_k = idx % TILE_K;
                        int token_idx = token_block + local_m;
                        
                        if (token_idx < end_idx && k_tile + local_k < intermediate_dim) {
                            int actual_token = sorted_pairs[token_idx].token_idx;
                            tile_input[local_m * TILE_K + local_k] = 
                                intermediate_cache[actual_token * intermediate_dim + k_tile + local_k];
                        } else {
                            tile_input[local_m * TILE_K + local_k] = T(0);
                        }
                    }
                    
                    // Load down weights
                    for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x * blockDim.y) {
                        int local_k = idx / TILE_N;
                        int local_n = idx % TILE_N;
                        
                        if (k_tile + local_k < intermediate_dim && out_tile + local_n < hidden_dim) {
                            tile_weight[local_k * TILE_N + local_n] = 
                                down_w[(k_tile + local_k) * hidden_dim + out_tile + local_n];
                        }
                    }
                    __syncthreads();
                    
                    // Compute partial products
                    int local_m = threadIdx.y;
                    int local_n = threadIdx.x;
                    
                    if (local_m < TILE_M && local_n < TILE_N) {
                        for (int k = 0; k < TILE_K; k++) {
                            accum[0] += float(tile_input[local_m * TILE_K + k]) * 
                                       float(tile_weight[k * TILE_N + local_n]);
                        }
                    }
                }
                
                // Accumulate to output with routing weights
                int local_m = threadIdx.y;
                int local_n = threadIdx.x;
                
                if (local_m < TILE_M && local_n < TILE_N) {
                    int token_idx = token_block + local_m;
                    int out_idx = out_tile + local_n;
                    
                    if (token_idx < end_idx && out_idx < hidden_dim) {
                        int actual_token = sorted_pairs[token_idx].token_idx;
                        float routing_weight = sorted_pairs[token_idx].routing_weight;
                        
                        atomicAdd(&output[actual_token * hidden_dim + out_idx], 
                                  T(accum[0] * routing_weight));
                    }
                }
            }
        }
    }
}

// Kernel to sort tokens by expert for better data locality
__global__ void prepare_sorted_pairs(
    const uint32_t* expert_indices,    // [num_tokens, num_selected_experts]
    const float* routing_weights,       // [num_tokens, num_selected_experts]
    TokenExpertPair* sorted_pairs,     // Output: sorted pairs
    int* expert_counts,                 // Output: count per expert
    int num_tokens,
    int num_selected_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_tokens * num_selected_experts) {
        int token_idx = tid / num_selected_experts;
        int k = tid % num_selected_experts;
        
        TokenExpertPair pair;
        pair.token_idx = token_idx;
        pair.expert_idx = expert_indices[tid];
        pair.routing_weight = routing_weights[tid];
        pair.original_idx = tid;
        
        sorted_pairs[tid] = pair;
        
        // Count tokens per expert
        atomicAdd(&expert_counts[pair.expert_idx], 1);
    }
}

// C interface for optimized fused MoE
extern "C" {

void fused_moe_forward_optimized_f32(
    const float* input,
    const float* gate_weights,
    const float* up_weights,
    const float* down_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    float* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int num_experts,
    int activation_type,
    cudaStream_t stream
) {
    // For now, use a simpler optimized kernel without sorting
    // This still provides tiling and shared memory benefits
    
    const int threads = 256;
    const int shared_mem_size = (hidden_dim + intermediate_dim) * sizeof(float);
    
    // Clear output
    CUDA_CHECK(cudaMemset(output, 0, num_tokens * hidden_dim * sizeof(float)));
    
    // Launch the kernel with one block per token
    fused_moe_kernel<float><<<num_tokens, threads, shared_mem_size, stream>>>(
        input, gate_weights, up_weights, down_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim,
        num_selected_experts, activation_type
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Chunked execution for large batches
void fused_moe_forward_chunked_f32(
    const float* input,
    const float* gate_weights,
    const float* up_weights,
    const float* down_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    float* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int num_experts,
    int activation_type,
    int chunk_size,
    cudaStream_t stream
) {
    // Process in chunks to avoid memory issues
    for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += chunk_size) {
        int chunk_tokens = std::min(chunk_size, num_tokens - chunk_start);
        
        fused_moe_forward_optimized_f32(
            input + chunk_start * hidden_dim,
            gate_weights,
            up_weights,
            down_weights,
            routing_weights + chunk_start * num_selected_experts,
            expert_indices + chunk_start * num_selected_experts,
            output + chunk_start * hidden_dim,
            chunk_tokens,
            hidden_dim,
            intermediate_dim,
            num_selected_experts,
            num_experts,
            activation_type,
            stream
        );
    }
}

} // extern "C"