use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc, Mutex},
};

use candle_core::{
    cuda::{
        cudarc::{
            cublas::{result::hgemm, sys::cublasOperation_t},
            driver::{CudaSlice, DevicePtr},
        },
        CudaStorageSlice, WrapErr,
    },
    from_storage_no_op, CudaStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use half::f16;

use crate::{
    utils::{get_cuda_device, get_cuda_slice},
    IsqType, QuantMethod, QuantMethodConfig,
};

use super::ffi::{
    exl2_reconstruct_q_matrix, 
    exl2_create_q_matrix, 
    exl2_destroy_q_matrix
};

const MAX_Q_GEMM_ROWS: i32 = 32;
const BLOCK_M_SIZE_MAX: i32 = 8;


#[derive(Debug)]
pub struct Exl2Layer {
    q_weight: Tensor,
    q_scale: Tensor,
    q_scale_max: Tensor,
    q_groups: Tensor,
    q_perm: Tensor,
    q_invperm: Tensor,
    q_group_map: Tensor,
    bias: Option<Tensor>,
    bits: i32,
    exllama_state: i32,
    q_matrix: *mut std::ffi::c_void,
}

impl Exl2Layer {
    fn exl2_gemm(&self, a: Tensor) -> Result<Tensor> {
        let dev = get_cuda_device(&a)?;
        let a_ptr = get_cuda_slice::<f16>(&a)?;

        if self.exllama_state == 0 {

            self.q_scale_max = (self.q_scale_max / 256.0)?;
            self.q_invperm = self.q_invperm.to_dtype(DType::I16)?;
            self.q_perm = self.q_invperm.arg_sort_last_dim(false)?.to_dtype(DType::I16)?; 
            self.q_group_map = make_group_map(&self.q_groups, self.q_weight.dim(0)?)?;

            // QMatrix entries
            let dev_ord = dev.ordinal() as i32;
            let b_width = self.q_weight.dims()[1] as i32;
            let b_height = self.q_perm.dims()[0] as i32;
            let b_groups = self.q_scale.dims()[0] as i32;
            let b_q_weight = get_cuda_slice::<i32>(&self.q_weight)? as *const u32;
            let b_q_perm = get_cuda_slice::<i16>(&self.q_perm)? as *const u16;
            let b_q_invperm = get_cuda_slice::<i16>(&self.q_invperm)? as *const u16;
            let b_q_scale = get_cuda_slice::<f32>(&self.q_scale)? as *const u32;
            let b_q_scale_max = get_cuda_slice::<f16>(&self.q_scale_max)?;
            let b_q_groups = get_cuda_slice::<i32>(&self.q_groups)? as *const u16;
            let b_q_group_map = get_cuda_slice::<i16>(&self.q_group_map)? as *const u16;

            self.q_matrix = unsafe {
                exl2_create_q_matrix(
                    dev_ord,
                    b_height,
                    b_width,
                    b_groups,
                    b_q_weight,
                    b_q_perm,
                    b_q_invperm,
                    b_q_scale,
                    b_q_scale_max,
                    b_q_groups,
                    b_q_group_map,
                )
            };
            self.exllama_state = 1;
        }



        let qm_width = self.q_weight.dim(1)?;                  
        let c_shape = Shape::from_dims(&[a.dims()[0], qm_width]);

        let (m, n, k) = (
            c_shape.dims()[0] as i32,
            c_shape.dims()[1] as i32,
            a.dims()[1] as i32,
        );

        let c = unsafe { dev.alloc::<f16>(c_shape.elem_count()).w()? };
        let c_ptr = *c.device_ptr() as *mut f16;

        // Create temp_dq as a Tensor, using a zero-sized tensor when not needed 
        // (TODO: review if this is the best solution here)
        let temp_dq = if c_shape.dims()[0] > MAX_Q_GEMM_ROWS as usize {
            Tensor::zeros(&[a.dims()[1], qm_width], DType::F16, a.device())?
        } else {
            Tensor::zeros(&[0, 0], DType::F16, a.device())?
        };
        
        
        let temp_dq_ptr = get_cuda_slice::<f16>(&temp_dq)?;

        if m > MAX_Q_GEMM_ROWS {
            // Reconstruct FP16 matrix, then cuBLAS
            unsafe {
                exl2_reconstruct_q_matrix(self.q_matrix);
            }
            
            let alpha = f16::from_f32(1.0);
            let beta = f16::from_f32(0.0);
            let cublas_handle = match a.device() {
                Device::Cuda(dev) => dev.cublas_handle(),
                _ => unreachable!(), // invariant enforced earlier
            };

            unsafe {
                hgemm(
                    *cublas_handle.handle(),
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    temp_dq_ptr as *const _,
                    n,
                    a_ptr as *const _,
                    k,
                    &beta,
                    c_ptr,
                    n,
                )
                .w()?
            };

        } else {
            todo!()
        }
        todo!()
    }
}



fn make_group_map(q_groups: &Tensor, num_qrows: usize) -> Result<Tensor> {
    let gr = q_groups.to_vec1::<i16>()?;
    let mut group_map = Vec::new();
    let num_groups = gr.len() / 2;

    for i in 0..num_groups {
        let bits = gr[i * 2] as usize;
        let qrows = if i < num_groups - 1 {
            gr[i * 2 + 3] as usize - gr[i * 2 + 1] as usize
        } else {
            num_qrows - gr[i * 2 + 1] as usize
        };
        let rows = qrows * 32 / bits;
        for j in 0..rows {
            group_map.push(i as i16);
            group_map.push((rows - j) as i16);
        }
    }

    Tensor::from_vec(group_map.clone(), (group_map.len(),), q_groups.device())
}


impl QuantMethod for Exl2Layer {
    fn new(method: QuantMethodConfig) -> Result<Self> {
        match method {
            QuantMethodConfig::Exl2 {
                q_weight,
                q_scale,
                q_scale_max,
                q_groups,
                q_perm,
                q_invperm,
                q_group_map,
                bias,
                bits,
            } => {

                Ok(Self {
                    q_weight,
                    q_scale,
                    q_scale_max,
                    q_groups,
                    q_perm,
                    q_invperm,
                    q_group_map,
                    bias,
                    bits,
                    exllama_state: 0,
                    q_matrix: std::ptr::null_mut(),
                })
            }
            QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. } => {
                unreachable!()
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out_shape = Shape::from_dims(
            &[
                &x.dims()[..x.dims().len() - 1],
                &[self.q_weight.dim(candle_core::D::Minus1)?],
            ]
            .concat(),
        );
        let reshaped_x = x.reshape(((), x.dim(candle_core::D::Minus1)?))?;
        let mut output = self.exl2_gemm(reshaped_x)?;
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?;
        }
        output.reshape(out_shape)
    }
    
    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F16)
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("EXL2 quantization does not support adding weight delta.")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        todo!()
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        None
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("EXL2 quantization does not support ISQ.")
    }

    fn get_max_isq_cpu_threads(&self, _dtype: IsqType) -> Option<NonZeroUsize> {
        None
    }
}


impl Drop for Exl2Layer {
    fn drop(&mut self) {
        if !self.q_matrix.is_null() {
            unsafe {
                exl2_destroy_q_matrix(self.q_matrix);
            }
        }
    }
}