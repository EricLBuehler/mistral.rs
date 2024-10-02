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
use lazy_static::lazy_static;

use crate::{
    utils::{get_cuda_device, get_cuda_slice},
    IsqType, QuantMethod, QuantMethodConfig, QuantizedSerde,
};

use super::ffi::{
    gemm_half_q_half_alt, gemm_half_q_half_cuda_part, reconstruct_exllama, reconstruct_gptq,
};

const MAX_Q_GEMM_ROWS_8BIT: i32 = 24;
const MAX_Q_GEMM_ROWS: i32 = 50;
const MAX_ALT_GEMM_ROWS: i32 = 8;
const BLOCK_M_SIZE_MAX: i32 = 8;

lazy_static! {
    static ref TMP_DQS: Mutex<HashMap<usize, CudaSlice<f16>>> = Mutex::new(HashMap::new());
}

#[derive(Debug)]
pub struct GptqLayer {
    q_weight: Tensor,    // u32
    gptq_qzeros: Tensor, // u32
    gptq_scales: Tensor, // f16
    bias: Tensor,        // f16
    g_idx: Tensor,       // i32
    bits: i32,
    use_exllama: bool,
}

impl GptqLayer {
    // https://github.com/vllm-project/vllm/blob/966fe72141e8365721840b7ababfb78601c23ead/csrc/quantization/gptq/q_gemm.cu#L1490
    // https://github.com/vllm-project/vllm/blob/966fe72141e8365721840b7ababfb78601c23ead/csrc/quantization/gptq/q_gemm.cu#L1823
    fn gptq_gemm(&self, a: Tensor, groups: i32, use_exllama: bool) -> Result<Tensor> {
        if !a.is_contiguous() {
            candle_core::bail!(
                "Expected `a` to be contiguous, got strides {:?}",
                a.layout().stride()
            )
        }
        let a_ptr = get_cuda_slice::<f16>(&a)?;
        let b_q_weight = get_cuda_slice::<i32>(&self.q_weight)? as *const u32;
        let b_gptq_qzeros = get_cuda_slice::<i32>(&self.gptq_qzeros)? as *const u32;
        let b_gptq_scales = get_cuda_slice::<f16>(&self.gptq_scales)?;
        let b_g_idx = get_cuda_slice::<i32>(&self.g_idx)?;

        let dev = get_cuda_device(&a)?;

        let c_shape = Shape::from_dims(&[a.dims()[0], self.q_weight.dims()[1]]);

        let (m, n, k) = (
            c_shape.dims()[0] as i32,
            c_shape.dims()[1] as i32,
            a.dims()[1] as i32,
        );

        let c = unsafe { dev.alloc::<f16>(c_shape.elem_count()).w()? };

        let c_ptr = *c.device_ptr() as *mut f16;

        let len = (self.q_weight.dims()[0] * 32 / self.bits as usize) * self.q_weight.dims()[1];
        let temp_dq_ptr = *TMP_DQS.try_lock().unwrap().get(&len).unwrap().device_ptr() as *mut f16;

        let use_reconstruct = if use_exllama {
            (self.bits == 8 && m > MAX_Q_GEMM_ROWS_8BIT) || (self.bits != 8 && m > MAX_Q_GEMM_ROWS)
        } else {
            // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
            // we disabled them for now.
            self.bits < 4 || m > MAX_ALT_GEMM_ROWS
        };

        if use_reconstruct {
            // Reconstruct FP16 matrix, then cuBLAS

            let cublas_handle = match a.device() {
                Device::Cuda(dev) => dev.cublas_handle(),
                _ => unreachable!(), // invariant enforced earlier
            };

            let reconstruct_kernel = if use_exllama {
                reconstruct_exllama
            } else {
                reconstruct_gptq
            };
            unsafe {
                reconstruct_kernel(
                    b_q_weight,
                    b_gptq_qzeros,
                    b_gptq_scales,
                    b_g_idx,
                    temp_dq_ptr,
                    k,
                    n,
                    groups,
                    self.bits,
                )
            };

            let alpha = f16::from_f32_const(1.0);
            let beta = f16::from_f32_const(0.0);

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
        } else if use_exllama {
            let max_chunks = m / BLOCK_M_SIZE_MAX;
            let last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
            let last_chunk_size = m - last_chunk;

            if max_chunks > 0 {
                unsafe {
                    gemm_half_q_half_cuda_part(
                        a_ptr as *const _,
                        b_q_weight,
                        b_gptq_qzeros,
                        b_gptq_scales,
                        b_g_idx,
                        c_ptr,
                        last_chunk,
                        n,
                        k,
                        BLOCK_M_SIZE_MAX,
                        groups,
                        self.bits,
                    )
                }
            }
            if last_chunk_size > 0 {
                unsafe {
                    gemm_half_q_half_cuda_part(
                        a_ptr.add((last_chunk * k) as usize),
                        b_q_weight,
                        b_gptq_qzeros,
                        b_gptq_scales,
                        b_g_idx,
                        c_ptr.add((last_chunk * n) as usize),
                        last_chunk_size,
                        n,
                        k,
                        last_chunk_size,
                        groups,
                        self.bits,
                    )
                }
            }
        } else {
            unsafe {
                gemm_half_q_half_alt(
                    a_ptr as *const _,
                    b_q_weight,
                    b_gptq_qzeros,
                    b_gptq_scales,
                    b_g_idx,
                    c_ptr,
                    m,
                    n,
                    k,
                    self.bits,
                )
            }
        }

        let storage = CudaStorage {
            slice: CudaStorageSlice::F16(c),
            device: dev.clone(),
        };
        let storage = Storage::Cuda(storage);

        Ok(from_storage_no_op(storage, c_shape, false))
    }
}

impl QuantMethod for GptqLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gptq {
                bits,
                use_exllama,
                q_weight,
                gptq_qzeros,
                gptq_scales,
                g_idx,
                bias,
            } => {
                let dev = get_cuda_device(&g_idx)?;
                let len = (q_weight.dims()[0] * 32 / bits as usize) * q_weight.dims()[1];
                // SAFETY: used in the kernel as a tmp space, just preallocating it here.
                if let std::collections::hash_map::Entry::Vacant(e) =
                    TMP_DQS.lock().unwrap().entry(len)
                {
                    e.insert(unsafe { dev.alloc::<f16>(len).w()? });
                }
                Ok(Self {
                    q_weight,
                    gptq_qzeros,
                    gptq_scales,
                    g_idx,
                    bits,
                    use_exllama,
                    bias,
                })
            }
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy => {
                unreachable!()
            }
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // https://github.com/vllm-project/vllm/blob/ba991d5c84adbc0685075af88333c688ddb06011/vllm/model_executor/layers/quantization/gptq.py#L200
        let out_shape = Shape::from_dims(
            &[
                &a.dims()[..a.dims().len() - 1],
                &[self.q_weight.dim(D::Minus1)?],
            ]
            .concat(),
        );
        let reshaped_a = a.reshape(((), a.dim(D::Minus1)?))?;
        if !reshaped_a.device().is_cuda() {
            candle_core::bail!("Expected CUDA input to GptqLayer");
        }
        let out = self.gptq_gemm(
            reshaped_a,
            self.gptq_qzeros.dim(0)? as i32,
            self.use_exllama,
        )?;
        out.reshape(out_shape)?.broadcast_add(&self.bias)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F16)
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("GPTQ quantization does not support adding weight delta.")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.gptq_scales.dtype(), self.gptq_scales.device().clone())
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
        candle_core::bail!("GPTQ quantization does not support ISQ.")
    }

    fn get_max_isq_cpu_threads(&self, _dtype: IsqType) -> Option<NonZeroUsize> {
        None
    }
}

impl QuantizedSerde for GptqLayer {
    fn name(&self) -> &'static str {
        "gptq"
    }
}
