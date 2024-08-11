use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaStorageSlice, WrapErr},
    from_storage_no_op,
    quantized::QMatMul,
    CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};
use half::{bf16, f16};
use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    util_cuda::{get_cuda_device, get_cuda_slice},
    QuantMethod, QuantMethodConfig,
};

#[cfg(feature = "cuda")]
use super::ffi::{eight_bit, four_bit, one_bit, three_bit, two_bit};

macro_rules! dequant_for_dtype {
    ($this:expr, w=$wq_t:ty, sz=$scale_t:ty, $dtype:ident, pack=$pack:expr, $dev:expr, $bit_thing:ident, $postfix:tt) => {{
        paste::paste! {
            let w_slice = get_cuda_slice::<$wq_t>(&$this.w_q);
            let scale_slice = get_cuda_slice::<$scale_t>(&$this.scales);
            let zero_slice = get_cuda_slice::<$scale_t>(&$this.zeros);

            let (h, w) = $this.w_q.dims2()?;
            let num_packed_elems = $pack;
            let out_shape = Shape::from_dims(&[num_packed_elems * h, w]);

            let out = unsafe { $dev.alloc::<$scale_t>(out_shape.elem_count()).w()? };
            let out_ptr = *out.device_ptr() as *mut $scale_t;
            unsafe {
                $bit_thing::[< dequantize_ $postfix >](
                    w_slice,
                    scale_slice,
                    zero_slice,
                    out_ptr,
                    h as i32,
                    w as i32,
                );
            }

            let storage = CudaStorage {
                slice: CudaStorageSlice::$dtype(out),
                device: $dev.clone(),
            };
            let storage = Storage::Cuda(storage);

            from_storage_no_op(storage, out_shape, false).reshape($this.w_shape.clone())
        }
    }};
}

#[derive(Clone, Copy)]
pub struct HqqConfig {
    pub bits: i32,
    pub group_size: Option<NonZeroUsize>,
    pub axis: usize,
}

pub struct HqqMatMul {
    w_q: Tensor,
    zeros: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    w_shape: Shape,
    cfg: HqqConfig,
}

impl HqqMatMul {
    /// Dequantize `self` into a tensor of shape `scales` or `zeros`.
    #[cfg(not(feature = "cuda"))]
    fn dequantize(&self) -> Result<Tensor> {
        use crate::hqq::hqq_cpu::{
            Dequant1Bit, Dequant2Bit, Dequant3Bit, Dequant4Bit, Dequant8Bit,
        };

        match (self.w_q.dtype(), self.scales.dtype(), self.zeros.dtype()) {
            (DType::F16, DType::F16, DType::F16)
            | (DType::BF16, DType::BF16, DType::BF16)
            | (DType::F32, DType::F32, DType::F32) => (),
            (a, b, c) => {
                candle_core::bail!("Expected all dtypes to be the same, got ({a:?}, {b:?}, {c:?}).")
            }
        }
        if !(self.w_q.is_contiguous() && self.scales.is_contiguous() && self.zeros.is_contiguous())
        {
            candle_core::bail!("All tensors must be contiguous!");
        }
        if self.cfg.axis != 0 {
            candle_core::bail!(
                "CPU HQQ dequantization requires axis == 0, got {}.",
                self.cfg.axis
            );
        }
        let (h, w) = self.w_q.dims2()?;

        match self.cfg.bits {
            8 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant8Bit { h, w }),
            4 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant4Bit { h, w }),
            3 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant3Bit { h, w }),
            2 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant2Bit { h, w }),
            1 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant1Bit { h, w }),
            b => candle_core::bail!("Unreachable bits {b}"),
        }
    }

    /// Dequantize `self` into a tensor of shape `scales` or `zeros`.
    #[cfg(feature = "cuda")]
    fn dequantize(&self) -> Result<Tensor> {
        match (self.w_q.dtype(), self.scales.dtype(), self.zeros.dtype()) {
            (DType::F16, DType::F16, DType::F16)
            | (DType::BF16, DType::BF16, DType::BF16)
            | (DType::F32, DType::F32, DType::F32) => (),
            (a, b, c) => {
                candle_core::bail!("Expected all dtypes to be the same, got ({a:?}, {b:?}, {c:?}).")
            }
        }
        if !(self.w_q.is_contiguous() && self.scales.is_contiguous() && self.zeros.is_contiguous())
        {
            candle_core::bail!("All tensors must be contiguous!");
        }
        if self.cfg.axis != 0 {
            candle_core::bail!(
                "CUDA HQQ dequantization requires axis == 0, got {}.",
                self.cfg.axis
            );
        }
        let dev = get_cuda_device(&self.w_q);

        match (self.cfg.bits, self.w_q.dtype()) {
            // 8 bits
            (8, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_f32
                )
            }
            (8, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_f16
                )
            }
            (8, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_bf16
                )
            }

            // 4 bits
            (4, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_f32
                )
            }
            (4, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_f16
                )
            }
            (4, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_bf16
                )
            }

            // 3 bits
            // https://github.com/mobiusml/hqq/blob/306e30d9400629523c8e0af70101d8d7073cb3d5/hqq/kernels/hqq_aten_cuda.cpp#L42-L45
            (3, DType::F32) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = f32,
                    F32,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_f32
                )?;
                if let Some(grp_size) = self.cfg.group_size {
                    res.narrow(self.cfg.axis, 0, grp_size.into())
                } else {
                    Ok(res)
                }
            }
            (3, DType::F16) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = f16,
                    F16,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_f16
                )?;
                if let Some(grp_size) = self.cfg.group_size {
                    res.narrow(self.cfg.axis, 0, grp_size.into())
                } else {
                    Ok(res)
                }
            }
            (3, DType::BF16) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = bf16,
                    BF16,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_bf16
                )?;
                if let Some(grp_size) = self.cfg.group_size {
                    res.narrow(self.cfg.axis, 0, grp_size.into())
                } else {
                    Ok(res)
                }
            }

            // 2 bits
            (2, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_f32
                )
            }
            (2, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_f16
                )
            }
            (2, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_bf16
                )
            }

            // 1 bit
            (1, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_f32
                )
            }
            (1, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_f16
                )
            }
            (1, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_bf16
                )
            }
            (bits, dtype) => candle_core::bail!("Unsupported bit width {bits} and dtype {dtype:?}"),
        }
    }

    fn dequantize_matmul(&self, a: &Tensor) -> Result<Tensor> {
        let res = a.matmul(&self.dequantize()?.t()?.contiguous()?)?;
        if let Some(ref bias) = self.bias {
            res + bias
        } else {
            Ok(res)
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
        /*
        if self.cfg.force_dequantize {
            self.dequantize_matmul(a)
        } else {
            todo!()
        } */
        self.dequantize_matmul(a)
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
