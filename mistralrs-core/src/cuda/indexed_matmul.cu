#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

// CUDA kernel for indexed matrix multiplication (MoE)
// This performs: C[indices[i]] += A[i] * B[expert_id] * weights[i]
// where B contains all expert weights concatenated

template<typename T>
__global__ void indexed_matmul_kernel(
    const T* input,           // Input tensor [num_tokens, hidden_dim]
    const T* expert_weights,  // All expert weights [num_experts, in_dim, out_dim]
    const float* routing_weights, // Routing weights [num_tokens, num_selected_experts]
    const uint32_t* expert_indices, // Expert indices [num_tokens, num_selected_experts]
    T* output,                // Output tensor [num_tokens, out_dim]
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int expert_stride        // in_dim * out_dim
) {
    const int token_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (token_idx >= num_tokens || out_idx >= out_dim) {
        return;
    }
    
    float sum = 0.0f;
    
    // Process each selected expert for this token
    for (int k = 0; k < num_selected_experts; k++) {
        int expert_id = expert_indices[token_idx * num_selected_experts + k];
        float weight = routing_weights[token_idx * num_selected_experts + k];
        
        // Compute dot product for this output element
        float dot_product = 0.0f;
        const T* expert_matrix = expert_weights + expert_id * expert_stride + out_idx;
        
        for (int h = 0; h < hidden_dim; h++) {
            dot_product += float(input[token_idx * hidden_dim + h]) * 
                          float(expert_matrix[h * out_dim]);
        }
        
        sum += dot_product * weight;
    }
    
    output[token_idx * out_dim + out_idx] = T(sum);
}

// Optimized version using shared memory for better memory access patterns
template<typename T>
__global__ void indexed_matmul_kernel_optimized(
    const T* input,
    const T* expert_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    T* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int expert_stride
) {
    extern __shared__ char shared_mem[];
    T* shared_input = (T*)shared_mem;
    
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
    
    // Each thread computes multiple output elements
    for (int out_idx = tid; out_idx < out_dim; out_idx += block_size) {
        float sum = 0.0f;
        
        for (int k = 0; k < num_selected_experts; k++) {
            int expert_id = expert_indices[token_idx * num_selected_experts + k];
            float weight = routing_weights[token_idx * num_selected_experts + k];
            
            float dot_product = 0.0f;
            const T* expert_matrix = expert_weights + expert_id * expert_stride + out_idx;
            
            for (int h = 0; h < hidden_dim; h++) {
                dot_product += float(shared_input[h]) * float(expert_matrix[h * out_dim]);
            }
            
            sum += dot_product * weight;
        }
        
        output[token_idx * out_dim + out_idx] = T(sum);
    }
}

// Fused MoE forward pass: combines gate, up, and down projections
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

// C interface
extern "C" {

void indexed_matmul_f32(
    const float* input,
    const float* expert_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    float* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int num_experts,
    cudaStream_t stream
) {
    const int threads = 256;
    dim3 blocks(num_tokens, (out_dim + threads - 1) / threads);
    
    indexed_matmul_kernel<float><<<blocks, threads, 0, stream>>>(
        input, expert_weights, routing_weights, expert_indices, output,
        num_tokens, hidden_dim, out_dim, num_selected_experts,
        hidden_dim * out_dim
    );
}

void indexed_matmul_f16(
    const half* input,
    const half* expert_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    half* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int num_experts,
    cudaStream_t stream
) {
    const int threads = 256;
    dim3 blocks(num_tokens, (out_dim + threads - 1) / threads);
    
    indexed_matmul_kernel<half><<<blocks, threads, 0, stream>>>(
        input, expert_weights, routing_weights, expert_indices, output,
        num_tokens, hidden_dim, out_dim, num_selected_experts,
        hidden_dim * out_dim
    );
}

void indexed_matmul_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* expert_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int out_dim,
    int num_selected_experts,
    int num_experts,
    cudaStream_t stream
) {
    const int threads = 256;
    dim3 blocks(num_tokens, (out_dim + threads - 1) / threads);
    
    indexed_matmul_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        input, expert_weights, routing_weights, expert_indices, output,
        num_tokens, hidden_dim, out_dim, num_selected_experts,
        hidden_dim * out_dim
    );
}

void fused_moe_forward_f32(
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
    const int threads = 256;
    const int shared_mem_size = (hidden_dim + intermediate_dim) * sizeof(float);
    
    fused_moe_kernel<float><<<num_tokens, threads, shared_mem_size, stream>>>(
        input, gate_weights, up_weights, down_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim,
        num_selected_experts, activation_type
    );

    CUDA_CHECK(cudaGetLastError());
}

void fused_moe_forward_f16(
    const half* input,
    const half* gate_weights,
    const half* up_weights,
    const half* down_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    half* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int num_experts,
    int activation_type,
    cudaStream_t stream
) {
    const int threads = 256;
    const int shared_mem_size = (hidden_dim + intermediate_dim) * sizeof(half);
    
    fused_moe_kernel<half><<<num_tokens, threads, shared_mem_size, stream>>>(
        input, gate_weights, up_weights, down_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim,
        num_selected_experts, activation_type
    );

    CUDA_CHECK(cudaGetLastError());
}


void fused_moe_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weights,
    const __nv_bfloat16* up_weights,
    const __nv_bfloat16* down_weights,
    const float* routing_weights,
    const uint32_t* expert_indices,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_selected_experts,
    int num_experts,
    int activation_type,
    cudaStream_t stream
) {
    const int threads = 256;
    const int shared_mem_size = (hidden_dim + intermediate_dim) * sizeof(__nv_bfloat16);
    
    fused_moe_kernel<__nv_bfloat16><<<num_tokens, threads, shared_mem_size, stream>>>(
        input, gate_weights, up_weights, down_weights,
        routing_weights, expert_indices, output,
        num_tokens, hidden_dim, intermediate_dim,
        num_selected_experts, activation_type
    );

    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"