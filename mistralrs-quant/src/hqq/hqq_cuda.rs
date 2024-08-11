use std::sync::Arc;

use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaStorageSlice, WrapErr},
    from_storage_no_op,
    quantized::QMatMul,
    CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use crate::{
    util_cuda::{get_cuda_device, get_cuda_slice},
    QuantMethod, QuantMethodConfig,
};

use super::ffi::{eight_bit, four_bit, one_bit, three_bit, two_bit};

pub struct HqqMatMul {
    w_q: Tensor,
    zeros: Tensor,
    scales: Tensor,
    bits: i32,
    bias: Option<Tensor>,
}

impl HqqMatMul {
    fn dequantize(&self) -> Result<Tensor> {
        match (self.w_q.dtype(), self.scales.dtype(), self.zeros.dtype()) {
            (DType::F16, DType::F16, DType::F16)
            | (DType::BF16, DType::BF16, DType::BF16)
            | (DType::F32, DType::F32, DType::F32) => (),
            (a, b, c) => {
                candle_core::bail!("Expected all dtypes to be the same, got ({a:?}, {b:?}, {c:?}).")
            }
        }
        let (h, w) = self.w_q.dims2()?;
        let dev = get_cuda_device(&self.w_q);

        match (self.bits, self.w_q.dtype()) {
            (8, DType::F32) => {
                let w_slice = get_cuda_slice::<u8>(&self.w_q);
                let scale_slice = get_cuda_slice::<f32>(&self.scales);
                let zero_slice = get_cuda_slice::<f32>(&self.zeros);

                let num_packed_elems = 1;
                let out_shape = Shape::from_dims(&[num_packed_elems * h, w]);

                let out = unsafe { dev.alloc::<f32>(out_shape.elem_count()).w()? };
                let out_ptr = *out.device_ptr() as *mut f32;
                unsafe {
                    eight_bit::dequantize_8bit_u8_kernel_f32(
                        w_slice,
                        scale_slice,
                        zero_slice,
                        out_ptr,
                        h as i32,
                        w as i32,
                    );
                }

                let storage = CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                };
                let storage = Storage::Cuda(storage);

                Ok(from_storage_no_op(storage, out_shape, false))
            }
            (bits, dtype) => candle_core::bail!("Unsupported bit width {bits} and dtype {dtype:?}"),
        }
    }
}

impl QuantMethod for HqqMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight: _, b: _ }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Gptq {
                bits: _,
                use_exllama: _,
                q_weight: _,
                gptq_qzeros: _,
                gptq_scales: _,
                g_idx: _,
                bias: _,
            } => {
                unreachable!()
            }
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let res = a.matmul(&self.dequantize()?.t()?)?;
        if let Some(ref bias) = self.bias {
            res + bias
        } else {
            Ok(res)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        todo!()
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        todo!()
    }

    fn get_qmatmul(&mut self) -> Option<&mut QMatMul> {
        todo!()
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        todo!()
    }

    fn convert_to_isq(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }
}
