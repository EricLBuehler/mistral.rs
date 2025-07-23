use candle_core::{CudaDevice, DType, Device, Result, Storage, Tensor};

use super::ffi;
use crate::ops::mul_and_act;

/// Performs indexed matrix multiplication for MoE layers
/// output[token_idx] = sum(input[token_idx] @ experts[expert_idx] * routing_weight)
pub struct IndexedMatmul {
    num_experts: usize,
    num_selected_experts: usize,
}

impl IndexedMatmul {
    pub fn new(num_experts: usize, num_selected_experts: usize) -> Self {
        Self {
            num_experts,
            num_selected_experts,
        }
    }

    /// Performs indexed matrix multiplication
    /// Args:
    /// - input: [num_tokens, hidden_dim]
    /// - expert_weights: [num_experts, hidden_dim, out_dim]
    /// - routing_weights: [num_tokens, num_selected_experts]
    /// - expert_indices: [num_tokens, num_selected_experts]
    /// Returns:
    /// - output: [num_tokens, out_dim]
    pub fn forward(
        &self,
        input: &Tensor,
        expert_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Tensor> {
        let device = input.device();
        
        // Validate inputs
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (ne, hd, out_dim) = expert_weights.dims3()?;
        let (nt, nse) = routing_weights.dims2()?;
        let (nt2, nse2) = expert_indices.dims2()?;
        
        if ne != self.num_experts {
            candle_core::bail!("expert_weights has {} experts, expected {}", ne, self.num_experts);
        }
        if hd != hidden_dim {
            candle_core::bail!("expert_weights hidden_dim {} doesn't match input {}", hd, hidden_dim);
        }
        if nt != num_tokens || nt2 != num_tokens {
            candle_core::bail!("routing_weights/expert_indices num_tokens mismatch");
        }
        if nse != self.num_selected_experts || nse2 != self.num_selected_experts {
            candle_core::bail!("num_selected_experts mismatch");
        }
        
        // Create output tensor
        let output = Tensor::zeros((num_tokens, out_dim), input.dtype(), device)?;
        
        // Call CUDA kernel
        match device {
            Device::Cuda(cuda_device) => {
                self.cuda_fwd(
                    input,
                    expert_weights,
                    routing_weights,
                    expert_indices,
                    &output,
                    cuda_device,
                )?;
            }
            _ => {
                candle_core::bail!("IndexedMatmul only supports CUDA device");
            }
        }
        
        Ok(output)
    }

    fn cuda_fwd(
        &self,
        input: &Tensor,
        expert_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        output: &Tensor,
        device: &CudaDevice,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use std::ffi::c_void;
        
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (_, _, out_dim) = expert_weights.dims3()?;
        
        // Get storage and layouts
        let (input_storage, input_layout) = input.storage_and_layout();
        let (expert_weights_storage, expert_weights_layout) = expert_weights.storage_and_layout();
        let (routing_weights_storage, routing_weights_layout) = routing_weights.storage_and_layout();
        let (expert_indices_storage, expert_indices_layout) = expert_indices.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();
        
        // Extract CUDA storage
        let input_cuda = match &*input_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("input must be a cuda tensor"),
        };
        let expert_weights_cuda = match &*expert_weights_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("expert_weights must be a cuda tensor"),
        };
        let routing_weights_cuda = match &*routing_weights_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("routing_weights must be a cuda tensor"),
        };
        let expert_indices_cuda = match &*expert_indices_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("expert_indices must be a cuda tensor"),
        };
        let output_cuda = match &*output_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("output must be a cuda tensor"),
        };
        
        let stream = device.cuda_stream().cu_stream() as i64;
        
        match (input.dtype(), expert_indices.dtype()) {
            (DType::F32, DType::I32) => {
                let input_slice = input_cuda.as_cuda_slice::<f32>()?.slice(input_layout.start_offset()..);
                let expert_weights_slice = expert_weights_cuda.as_cuda_slice::<f32>()?.slice(expert_weights_layout.start_offset()..);
                let routing_weights_slice = routing_weights_cuda.as_cuda_slice::<f32>()?.slice(routing_weights_layout.start_offset()..);
                let expert_indices_slice = expert_indices_cuda.as_cuda_slice::<i32>()?.slice(expert_indices_layout.start_offset()..);
                let output_slice = output_cuda.as_cuda_slice::<f32>()?.slice(output_layout.start_offset()..);
                
                unsafe {
                    let (input_ptr, _input_guard) = input_slice.device_ptr(input_slice.stream());
                    let (expert_weights_ptr, _ew_guard) = expert_weights_slice.device_ptr(expert_weights_slice.stream());
                    let (routing_weights_ptr, _rw_guard) = routing_weights_slice.device_ptr(routing_weights_slice.stream());
                    let (expert_indices_ptr, _ei_guard) = expert_indices_slice.device_ptr(expert_indices_slice.stream());
                    let (output_ptr, _output_guard) = output_slice.device_ptr(output_slice.stream());
                    
                    ffi::indexed_matmul_f32(
                        input_ptr as *const c_void,
                        expert_weights_ptr as *const c_void,
                        routing_weights_ptr as *const c_void,
                        expert_indices_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        num_tokens as i32,
                        hidden_dim as i32,
                        out_dim as i32,
                        self.num_selected_experts as i32,
                        self.num_experts as i32,
                        stream,
                    );
                }
            }
            (DType::F16, DType::I32) => {
                let input_slice = input_cuda.as_cuda_slice::<half::f16>()?.slice(input_layout.start_offset()..);
                let expert_weights_slice = expert_weights_cuda.as_cuda_slice::<half::f16>()?.slice(expert_weights_layout.start_offset()..);
                let routing_weights_slice = routing_weights_cuda.as_cuda_slice::<f32>()?.slice(routing_weights_layout.start_offset()..);
                let expert_indices_slice = expert_indices_cuda.as_cuda_slice::<i32>()?.slice(expert_indices_layout.start_offset()..);
                let output_slice = output_cuda.as_cuda_slice::<half::f16>()?.slice(output_layout.start_offset()..);
                
                unsafe {
                    let (input_ptr, _input_guard) = input_slice.device_ptr(input_slice.stream());
                    let (expert_weights_ptr, _ew_guard) = expert_weights_slice.device_ptr(expert_weights_slice.stream());
                    let (routing_weights_ptr, _rw_guard) = routing_weights_slice.device_ptr(routing_weights_slice.stream());
                    let (expert_indices_ptr, _ei_guard) = expert_indices_slice.device_ptr(expert_indices_slice.stream());
                    let (output_ptr, _output_guard) = output_slice.device_ptr(output_slice.stream());
                    
                    ffi::indexed_matmul_f16(
                        input_ptr as *const c_void,
                        expert_weights_ptr as *const c_void,
                        routing_weights_ptr as *const c_void,
                        expert_indices_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        num_tokens as i32,
                        hidden_dim as i32,
                        out_dim as i32,
                        self.num_selected_experts as i32,
                        self.num_experts as i32,
                        stream,
                    );
                }
            }
            (DType::BF16, DType::I32) => {
                let input_slice = input_cuda.as_cuda_slice::<half::bf16>()?.slice(input_layout.start_offset()..);
                let expert_weights_slice = expert_weights_cuda.as_cuda_slice::<half::bf16>()?.slice(expert_weights_layout.start_offset()..);
                let routing_weights_slice = routing_weights_cuda.as_cuda_slice::<f32>()?.slice(routing_weights_layout.start_offset()..);
                let expert_indices_slice = expert_indices_cuda.as_cuda_slice::<i32>()?.slice(expert_indices_layout.start_offset()..);
                let output_slice = output_cuda.as_cuda_slice::<half::bf16>()?.slice(output_layout.start_offset()..);
                
                unsafe {
                    let (input_ptr, _input_guard) = input_slice.device_ptr(input_slice.stream());
                    let (expert_weights_ptr, _ew_guard) = expert_weights_slice.device_ptr(expert_weights_slice.stream());
                    let (routing_weights_ptr, _rw_guard) = routing_weights_slice.device_ptr(routing_weights_slice.stream());
                    let (expert_indices_ptr, _ei_guard) = expert_indices_slice.device_ptr(expert_indices_slice.stream());
                    let (output_ptr, _output_guard) = output_slice.device_ptr(output_slice.stream());
                    
                    ffi::indexed_matmul_bf16(
                        input_ptr as *const c_void,
                        expert_weights_ptr as *const c_void,
                        routing_weights_ptr as *const c_void,
                        expert_indices_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        num_tokens as i32,
                        hidden_dim as i32,
                        out_dim as i32,
                        self.num_selected_experts as i32,
                        self.num_experts as i32,
                        stream,
                    );
                }
            }
            _ => candle_core::bail!("Unsupported dtype combination for IndexedMatmul"),
        }
        
        Ok(())
    }
}

/// Performs fused MoE forward pass with gate, up, and down projections
pub struct FusedMoeForward {
    num_experts: usize,
    num_selected_experts: usize,
    activation: Activation,
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Silu,
    Gelu,
    Relu,
}

impl Activation {
    fn to_int(&self) -> i32 {
        match self {
            Activation::Silu => 0,
            Activation::Gelu => 1,
            Activation::Relu => 2,
        }
    }
}

impl FusedMoeForward {
    pub fn new(num_experts: usize, num_selected_experts: usize, activation: Activation) -> Self {
        Self {
            num_experts,
            num_selected_experts,
            activation,
        }
    }

    /// Performs fused MoE forward pass
    /// Args:
    /// - input: [num_tokens, hidden_dim]
    /// - gate_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - up_weights: [num_experts, hidden_dim, intermediate_dim]
    /// - down_weights: [num_experts, intermediate_dim, hidden_dim]
    /// - routing_weights: [num_tokens, num_selected_experts]
    /// - expert_indices: [num_tokens, num_selected_experts]
    /// Returns:
    /// - output: [num_tokens, hidden_dim]
    pub fn forward(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Tensor> {
        let device = input.device();
        
        // Validate inputs
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (ne_g, hd_g, intermediate_dim) = gate_weights.dims3()?;
        let (ne_u, hd_u, id_u) = up_weights.dims3()?;
        let (ne_d, id_d, hd_d) = down_weights.dims3()?;
        let (nt, nse) = routing_weights.dims2()?;
        let (nt2, nse2) = expert_indices.dims2()?;
        
        // Validate dimensions
        if ne_g != self.num_experts || ne_u != self.num_experts || ne_d != self.num_experts {
            candle_core::bail!("Number of experts mismatch");
        }
        if hd_g != hidden_dim || hd_u != hidden_dim || hd_d != hidden_dim {
            candle_core::bail!("Hidden dimension mismatch");
        }
        if id_u != intermediate_dim || id_d != intermediate_dim {
            candle_core::bail!("Intermediate dimension mismatch");
        }
        if nt != num_tokens || nt2 != num_tokens {
            candle_core::bail!("Number of tokens mismatch");
        }
        if nse != self.num_selected_experts || nse2 != self.num_selected_experts {
            candle_core::bail!("Number of selected experts mismatch");
        }
        
        // Create output tensor
        let output = Tensor::zeros((num_tokens, hidden_dim), input.dtype(), device)?;
        
        // Call CUDA kernel
        match device {
            Device::Cuda(cuda_device) => {
                self.cuda_fwd(
                    input,
                    gate_weights,
                    up_weights,
                    down_weights,
                    routing_weights,
                    expert_indices,
                    &output,
                    cuda_device,
                )?;
            }
            _ => {
                candle_core::bail!("FusedMoeForward only supports CUDA device");
            }
        }
        
        Ok(output)
    }

    fn cuda_fwd(
        &self,
        input: &Tensor,
        gate_weights: &Tensor,
        up_weights: &Tensor,
        down_weights: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &Tensor,
        output: &Tensor,
        device: &CudaDevice,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use std::ffi::c_void;
        
        let (num_tokens, hidden_dim) = input.dims2()?;
        let (_, _, intermediate_dim) = gate_weights.dims3()?;
        
        // Get storage and layouts
        let (input_storage, input_layout) = input.storage_and_layout();
        let (gate_storage, gate_layout) = gate_weights.storage_and_layout();
        let (up_storage, up_layout) = up_weights.storage_and_layout();
        let (down_storage, down_layout) = down_weights.storage_and_layout();
        let (routing_storage, routing_layout) = routing_weights.storage_and_layout();
        let (indices_storage, indices_layout) = expert_indices.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();
        
        // Extract CUDA storage
        let input_cuda = match &*input_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("input must be a cuda tensor"),
        };
        let gate_cuda = match &*gate_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("gate_weights must be a cuda tensor"),
        };
        let up_cuda = match &*up_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("up_weights must be a cuda tensor"),
        };
        let down_cuda = match &*down_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("down_weights must be a cuda tensor"),
        };
        let routing_cuda = match &*routing_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("routing_weights must be a cuda tensor"),
        };
        let indices_cuda = match &*indices_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("expert_indices must be a cuda tensor"),
        };
        let output_cuda = match &*output_storage {
            Storage::Cuda(cuda_storage) => cuda_storage,
            _ => candle_core::bail!("output must be a cuda tensor"),
        };
        
        let stream = device.cuda_stream().cu_stream() as i64;
        
        match (input.dtype(), expert_indices.dtype()) {
            (DType::F32, DType::I32) => {
                let input_slice = input_cuda.as_cuda_slice::<f32>()?.slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda.as_cuda_slice::<f32>()?.slice(gate_layout.start_offset()..);
                let up_slice = up_cuda.as_cuda_slice::<f32>()?.slice(up_layout.start_offset()..);
                let down_slice = down_cuda.as_cuda_slice::<f32>()?.slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda.as_cuda_slice::<f32>()?.slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda.as_cuda_slice::<i32>()?.slice(indices_layout.start_offset()..);
                let output_slice = output_cuda.as_cuda_slice::<f32>()?.slice(output_layout.start_offset()..);
                
                unsafe {
                    let (input_ptr, _input_guard) = input_slice.device_ptr(input_slice.stream());
                    let (gate_ptr, _gate_guard) = gate_slice.device_ptr(gate_slice.stream());
                    let (up_ptr, _up_guard) = up_slice.device_ptr(up_slice.stream());
                    let (down_ptr, _down_guard) = down_slice.device_ptr(down_slice.stream());
                    let (routing_ptr, _routing_guard) = routing_slice.device_ptr(routing_slice.stream());
                    let (indices_ptr, _indices_guard) = indices_slice.device_ptr(indices_slice.stream());
                    let (output_ptr, _output_guard) = output_slice.device_ptr(output_slice.stream());
                    
                    ffi::fused_moe_forward_f32(
                        input_ptr as *const c_void,
                        gate_ptr as *const c_void,
                        up_ptr as *const c_void,
                        down_ptr as *const c_void,
                        routing_ptr as *const c_void,
                        indices_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.num_experts as i32,
                        self.activation.to_int(),
                        stream,
                    );
                }
            }
            (DType::F16, DType::I32) => {
                let input_slice = input_cuda.as_cuda_slice::<half::f16>()?.slice(input_layout.start_offset()..);
                let gate_slice = gate_cuda.as_cuda_slice::<half::f16>()?.slice(gate_layout.start_offset()..);
                let up_slice = up_cuda.as_cuda_slice::<half::f16>()?.slice(up_layout.start_offset()..);
                let down_slice = down_cuda.as_cuda_slice::<half::f16>()?.slice(down_layout.start_offset()..);
                let routing_slice = routing_cuda.as_cuda_slice::<f32>()?.slice(routing_layout.start_offset()..);
                let indices_slice = indices_cuda.as_cuda_slice::<i32>()?.slice(indices_layout.start_offset()..);
                let output_slice = output_cuda.as_cuda_slice::<half::f16>()?.slice(output_layout.start_offset()..);
                
                unsafe {
                    let (input_ptr, _input_guard) = input_slice.device_ptr(input_slice.stream());
                    let (gate_ptr, _gate_guard) = gate_slice.device_ptr(gate_slice.stream());
                    let (up_ptr, _up_guard) = up_slice.device_ptr(up_slice.stream());
                    let (down_ptr, _down_guard) = down_slice.device_ptr(down_slice.stream());
                    let (routing_ptr, _routing_guard) = routing_slice.device_ptr(routing_slice.stream());
                    let (indices_ptr, _indices_guard) = indices_slice.device_ptr(indices_slice.stream());
                    let (output_ptr, _output_guard) = output_slice.device_ptr(output_slice.stream());
                    
                    ffi::fused_moe_forward_f16(
                        input_ptr as *const c_void,
                        gate_ptr as *const c_void,
                        up_ptr as *const c_void,
                        down_ptr as *const c_void,
                        routing_ptr as *const c_void,
                        indices_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        num_tokens as i32,
                        hidden_dim as i32,
                        intermediate_dim as i32,
                        self.num_selected_experts as i32,
                        self.num_experts as i32,
                        self.activation.to_int(),
                        stream,
                    );
                }
            }
            _ => candle_core::bail!("Unsupported dtype combination for FusedMoeForward"),
        }
        
        Ok(())
    }
}