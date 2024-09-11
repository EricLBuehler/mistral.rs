use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_core::cudarc::{
    cublas::{result::hgemm, sys::cublasOperation_t},
    driver::{CudaSlice, DevicePtr},
};
use std::sync::Arc;

const MAX_Q_GEMM_ROWS: i32 = 32;
const BLOCK_M_SIZE_MAX: i32 = 8;



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
            _ => candle_core::bail!("Expected Exl2 config"),
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

        if self.exllama_state == 0 {
            let dev = get_cuda_device(&x)?;
            self.prepare_weights(dev.id())?;
        }

        let mut output = self.exl2_gemm(reshaped_x)?;
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?;
        }
        output.reshape(out_shape)
    }

    // Implement other required methods...
}

impl Exl2Layer {
    fn prepare_weights(&mut self, device_id: i32) -> Result<()> {
        self.q_scale_max = &self.q_scale_max / 256.0;
        self.q_invperm = self.q_invperm.to_dtype(DType::U16)?;

        let q_perm = self.q_invperm.argsort()?.to_dtype(DType::U16)?; 
        let q_group_map = make_group_map(&q_groups, q_weight.dim(0)?)?;

        self.q_matrix = unsafe {
            exl2_create_q_matrix(
                device_id,

                self.q_perm.dims(0)? as i32,
                self.q_weight.dim(1)? as i32,
                self.q_scale.dim(0)? as i32,

                self.q_weight.as_ptr()?,
                self.q_perm.as_ptr()?,
                self.q_invperm.as_ptr()?,
                self.q_scale.as_ptr()?,
                self.q_scale_max.as_ptr()?,
                self.q_groups.as_ptr()?,
                self.q_group_map.as_ptr()?,
            )
        };
        self.exllama_state = 1;
        Ok(())
    }

    fn exl2_gemm(&self, a: Tensor) -> Result<Tensor> {

        let dev = get_cuda_device(&a)?;
        let qm_width = self.q_weight.dims()[1]?;                  
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
            Tensor::zeros(&[a.dims()[1], qm_width], DType::F16, &dev)?
        } else {
            Tensor::zeros(&[0, 0], DType::F16, &dev)?
        };
        
        let a_ptr = get_cuda_slice::<f16>(a)?;
        let temp_dq_ptr = temp_dq.device_ptr() as *const f16;

        if m > MAX_Q_GEMM_ROWS {
            // Reconstruct FP16 matrix, then cuBLAS
            unsafe {
                super::ffi::exl2_reconstruct_q_matrix(self.q_matrix);
            }
            
            let alpha = f16::from_f32(1.0);
            let beta = if clear { f16::from_f32(0.0) } else { f16::from_f32(1.0) };

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
            // Quantized matmul
        }

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

fn make_group_map(q_groups: &Tensor, num_qrows: usize) -> Result<Tensor> {
    let gr = q_groups.to_vec1::<u16>()?;
    let mut group_map = Vec::new();
    let num_groups = gr.len() / 2;

    let mut row = 0;
    for i in 0..num_groups {
        let bits = gr[i * 2] as usize;
        let rows = if i < num_groups - 1 {
            let qrows = gr[i * 2 + 3] as usize - gr[i * 2 + 1] as usize;
            qrows * 32 / bits
        } else {
            num_qrows - gr[i * 2 + 1] as usize
        };
        
        for _ in 0..rows {
            group_map.push(i as u16);
            group_map.push(rows as u16);
        }
        row += rows;
    }

    Tensor::from_vec(group_map, (group_map.len(),), q_groups.device())
}

