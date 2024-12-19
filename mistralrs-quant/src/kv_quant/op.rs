use candle_core::{
    quantized::k_quants::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0},
    CustomOp1, DType, InplaceOp2, Result, Tensor,
};

const Q8_0_TYPE_SIZE: usize = std::mem::size_of::<BlockQ8_0>();
const Q4_0_TYPE_SIZE: usize = std::mem::size_of::<BlockQ4_0>();

pub struct KvQuantizeOp8Bit;

impl KvQuantizeOp8Bit {
    /// Compute the element count for qs for a tensor divisible by blocksize.
    pub fn compute_qs_elem_count(xs: &Tensor) -> Result<usize> {
        let x_el_count = xs.shape().elem_count();
        if x_el_count % QK8_0 != 0 {
            candle_core::bail!("xs elem count must be a multiple of the block size {QK8_0}");
        }
        Ok(x_el_count / QK8_0 * Q8_0_TYPE_SIZE)
    }

    pub fn narrow_qs(qs: &Tensor, start_block: usize, len_blocks: usize) -> Result<Tensor> {
        qs.narrow(0, start_block * Q8_0_TYPE_SIZE, len_blocks * Q8_0_TYPE_SIZE)
    }

    pub fn narrow_xs(qs: &Tensor, start_block: usize, len_blocks: usize) -> Result<Tensor> {
        qs.narrow(0, start_block * QK8_0, len_blocks * QK8_0)
    }
}

impl InplaceOp2 for KvQuantizeOp8Bit {
    fn name(&self) -> &'static str {
        "kv-quant-8-bit"
    }

    fn cpu_fwd(
        &self,
        _xs: &mut candle_core::CpuStorage,
        _xl: &candle_core::Layout,
        _qs: &candle_core::CpuStorage,
        _ql: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &mut candle_core::MetalStorage,
        xl: &candle_core::Layout,
        qs: &candle_core::MetalStorage,
        ql: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::{backend::BackendStorage, DType};

        use crate::metal_kernels;

        if qs.dtype() != DType::U8 {
            candle_core::bail!("qs dtype must be u8");
        }

        if !(xl.is_contiguous() && ql.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }

        let command_buffer = qs.device().command_buffer()?;
        command_buffer.set_label(self.name());
        let device = qs.device();

        let kv_n_blocks = ql.shape().elem_count().div_ceil(Q8_0_TYPE_SIZE);
        let x_el_count = xl.shape().elem_count();
        if x_el_count % QK8_0 != 0 {
            candle_core::bail!("xs elem count must be a multiple of the block size {QK8_0}");
        }

        // Ensure there is enough space
        if kv_n_blocks != x_el_count / QK8_0 {
            candle_core::bail!(
                "number of blocks in xs {} does not match the number of blocks in qs {kv_n_blocks}",
                x_el_count / QK8_0
            );
        }

        metal_kernels::call_quantize_q8_0_kv(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            xs.dtype(),
            xs.buffer(),
            xl.start_offset() * xs.dtype().size_in_bytes(),
            qs.buffer(),
            ql.start_offset() * qs.dtype().size_in_bytes(),
            kv_n_blocks,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

/// Quantize xs into qs.
///
/// - The element count of qs determines the number of blocks
/// - The element count of xs must be a multiple of the blocksize
/// - xs is quantized blockwise and into qs, so xs must contain the same number of blocks as qs
pub fn quantize_inplace_8bit(qs: &mut Tensor, xs: &Tensor) -> Result<()> {
    xs.inplace_op2(qs, &KvQuantizeOp8Bit)
}

pub struct KvDequantizeOp8Bit {
    out_ty: DType,
}

impl CustomOp1 for KvDequantizeOp8Bit {
    fn name(&self) -> &'static str {
        "kv-dequant-8-bit"
    }

    fn cpu_fwd(
        &self,
        _qs: &candle_core::CpuStorage,
        _ql: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        qs: &candle_core::MetalStorage,
        ql: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, Shape};

        use crate::metal_kernels;

        if !ql.is_contiguous() {
            candle_core::bail!("All inputs must be contiguous");
        }

        let command_buffer = qs.device().command_buffer()?;
        command_buffer.set_label(self.name());
        let device = qs.device();

        let kv_n_blocks = ql.shape().elem_count().div_ceil(Q8_0_TYPE_SIZE);

        let xs_elem_count = kv_n_blocks * QK8_0;
        let xs = device.new_buffer(xs_elem_count, self.out_ty, self.name())?;

        metal_kernels::call_dequantize_q8_0_kv(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            self.out_ty,
            &xs,
            0,
            qs.buffer(),
            ql.start_offset() * qs.dtype().size_in_bytes(),
            kv_n_blocks,
        )
        .map_err(candle_core::Error::wrap)?;

        let out_shape = Shape::from_dims(&[xs_elem_count]);
        let newstorage =
            candle_core::MetalStorage::new(xs, device.clone(), out_shape.elem_count(), self.out_ty);
        Ok((newstorage, out_shape.clone()))
    }
}

/// Dequantize qs. This returns a flattened tensor which has (num blocks)*(block size)
///
/// - The element count of qs determines the number of blocks
pub fn dequantize_8bit(qs: &Tensor, out_ty: DType) -> Result<Tensor> {
    qs.apply_op1_no_bwd(&KvDequantizeOp8Bit { out_ty })
}

pub struct KvQuantizeOp4Bit;

impl KvQuantizeOp4Bit {
    /// Compute the element count for qs for a tensor divisible by blocksize.
    pub fn compute_qs_elem_count(xs: &Tensor) -> Result<usize> {
        let x_el_count = xs.shape().elem_count();
        if x_el_count % QK4_0 != 0 {
            candle_core::bail!("xs elem count must be a multiple of the block size {QK4_0}");
        }
        Ok(x_el_count / QK4_0 * Q4_0_TYPE_SIZE)
    }

    pub fn narrow_qs(qs: &Tensor, start_block: usize, len_blocks: usize) -> Result<Tensor> {
        qs.narrow(0, start_block * Q4_0_TYPE_SIZE, len_blocks * Q4_0_TYPE_SIZE)
    }

    pub fn narrow_xs(qs: &Tensor, start_block: usize, len_blocks: usize) -> Result<Tensor> {
        qs.narrow(0, start_block * QK4_0, len_blocks * QK4_0)
    }
}

impl InplaceOp2 for KvQuantizeOp4Bit {
    fn name(&self) -> &'static str {
        "kv-quant-4-bit"
    }

    fn cpu_fwd(
        &self,
        _xs: &mut candle_core::CpuStorage,
        _xl: &candle_core::Layout,
        _qs: &candle_core::CpuStorage,
        _ql: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &mut candle_core::MetalStorage,
        xl: &candle_core::Layout,
        qs: &candle_core::MetalStorage,
        ql: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::{backend::BackendStorage, DType};

        use crate::metal_kernels;

        if qs.dtype() != DType::U8 {
            candle_core::bail!("qs dtype must be u8");
        }

        if !(xl.is_contiguous() && ql.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }

        let command_buffer = qs.device().command_buffer()?;
        command_buffer.set_label(self.name());
        let device = qs.device();

        let kv_n_blocks = ql.shape().elem_count().div_ceil(Q4_0_TYPE_SIZE);
        let x_el_count = xl.shape().elem_count();
        if x_el_count % QK4_0 != 0 {
            candle_core::bail!("xs elem count must be a multiple of the block size {QK4_0}");
        }

        // Ensure there is enough space
        if kv_n_blocks != x_el_count / QK4_0 {
            candle_core::bail!(
                "number of blocks in xs {} does not match the number of blocks in qs {kv_n_blocks}",
                x_el_count / QK4_0
            );
        }

        metal_kernels::call_quantize_q4_0_kv(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            xs.dtype(),
            xs.buffer(),
            xl.start_offset() * xs.dtype().size_in_bytes(),
            qs.buffer(),
            ql.start_offset() * qs.dtype().size_in_bytes(),
            kv_n_blocks,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

/// Quantize xs into qs.
///
/// - The element count of qs determines the number of blocks
/// - The element count of xs must be a multiple of the blocksize
/// - xs is quantized blockwise and into qs, so xs must contain the same number of blocks as qs
pub fn quantize_inplace_4bit(qs: &mut Tensor, xs: &Tensor) -> Result<()> {
    xs.inplace_op2(qs, &KvQuantizeOp4Bit)
}

pub struct KvDequantizeOp4Bit {
    out_ty: DType,
}

impl CustomOp1 for KvDequantizeOp4Bit {
    fn name(&self) -> &'static str {
        "kv-dequant-4-bit"
    }

    fn cpu_fwd(
        &self,
        _qs: &candle_core::CpuStorage,
        _ql: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        qs: &candle_core::MetalStorage,
        ql: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, Shape};

        use crate::metal_kernels;

        if !ql.is_contiguous() {
            candle_core::bail!("All inputs must be contiguous");
        }

        let command_buffer = qs.device().command_buffer()?;
        command_buffer.set_label(self.name());
        let device = qs.device();

        let kv_n_blocks = ql.shape().elem_count().div_ceil(Q4_0_TYPE_SIZE);

        let xs_elem_count = kv_n_blocks * QK4_0;
        let xs = device.new_buffer(xs_elem_count, self.out_ty, self.name())?;

        metal_kernels::call_dequantize_q4_0_kv(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            self.out_ty,
            &xs,
            0,
            qs.buffer(),
            ql.start_offset() * qs.dtype().size_in_bytes(),
            kv_n_blocks,
        )
        .map_err(candle_core::Error::wrap)?;

        let out_shape = Shape::from_dims(&[xs_elem_count]);
        let newstorage =
            candle_core::MetalStorage::new(xs, device.clone(), out_shape.elem_count(), self.out_ty);
        Ok((newstorage, out_shape.clone()))
    }
}

/// Dequantize qs. This returns a flattened tensor which has (num blocks)*(block size)
///
/// - The element count of qs determines the number of blocks
pub fn dequantize_4bit(qs: &Tensor, out_ty: DType) -> Result<Tensor> {
    qs.apply_op1_no_bwd(&KvDequantizeOp4Bit { out_ty })
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use candle_core::{DType, Device, Tensor};

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sum = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        sum / a.len() as f32
    }

    #[test]
    fn test_kvquant_roundtrip_8bit() -> candle_core::Result<()> {
        use crate::kv_quant::op::KvQuantizeOp8Bit;

        let dev = Device::new_metal(0)?;

        let xs = Tensor::new(
            (0..32 * 128)
                .map(|i| 0.1 + 2.0 * (i as f32).cos())
                .collect::<Vec<_>>(),
            &dev,
        )?;
        let mut qs = Tensor::zeros(
            KvQuantizeOp8Bit::compute_qs_elem_count(&xs)?,
            DType::U8,
            &dev,
        )?;

        super::quantize_inplace_8bit(&mut qs, &xs)?;

        let dequant = super::dequantize_8bit(&qs, DType::F32)?.reshape(xs.shape())?;

        let rmse = rmse(&xs.to_vec1::<f32>()?, &dequant.to_vec1::<f32>()?);
        assert!(rmse < 0.002, "{rmse}");

        Ok(())
    }

    #[test]
    fn test_kvquant_roundtrip_8bit_narrow() -> candle_core::Result<()> {
        use crate::kv_quant::op::KvQuantizeOp8Bit;

        let dev = Device::new_metal(0)?;

        let xs = Tensor::new(
            (0..32 * 128)
                .map(|i| 0.1 + 2.0 * (i as f32).cos())
                .collect::<Vec<_>>(),
            &dev,
        )?;
        let full_qs = Tensor::zeros(
            KvQuantizeOp8Bit::compute_qs_elem_count(&xs)?,
            DType::U8,
            &dev,
        )?;

        let xs = KvQuantizeOp8Bit::narrow_xs(&xs, 64, 64)?;
        let mut qs = KvQuantizeOp8Bit::narrow_qs(&full_qs, 64, 64)?;

        super::quantize_inplace_8bit(&mut qs, &xs)?;

        let dequant = super::dequantize_8bit(&qs, DType::F32)?.reshape(xs.shape())?;

        let rmse = rmse(&xs.to_vec1::<f32>()?, &dequant.to_vec1::<f32>()?);
        assert!(rmse < 0.002, "{rmse}");

        Ok(())
    }

    #[test]
    fn test_kvquant_roundtrip_4bit() -> candle_core::Result<()> {
        use crate::kv_quant::op::KvQuantizeOp4Bit;

        let dev = Device::new_metal(0)?;

        let xs = Tensor::new(
            (0..32 * 128)
                .map(|i| 0.1 + 2.0 * (i as f32).cos())
                .collect::<Vec<_>>(),
            &dev,
        )?;
        let mut qs = Tensor::zeros(
            KvQuantizeOp4Bit::compute_qs_elem_count(&xs)?,
            DType::U8,
            &dev,
        )?;

        super::quantize_inplace_4bit(&mut qs, &xs)?;

        let dequant = super::dequantize_4bit(&qs, DType::F32)?.reshape(xs.shape())?;

        let rmse = rmse(&xs.to_vec1::<f32>()?, &dequant.to_vec1::<f32>()?);
        assert!(rmse < 0.002, "{rmse}");

        Ok(())
    }

    #[test]
    fn test_kvquant_roundtrip_4bit_narrow() -> candle_core::Result<()> {
        use crate::kv_quant::op::KvQuantizeOp4Bit;

        let dev = Device::new_metal(0)?;

        let xs = Tensor::new(
            (0..32 * 128)
                .map(|i| 0.1 + 2.0 * (i as f32).cos())
                .collect::<Vec<_>>(),
            &dev,
        )?;
        let full_qs = Tensor::zeros(
            KvQuantizeOp4Bit::compute_qs_elem_count(&xs)?,
            DType::U8,
            &dev,
        )?;

        let xs = KvQuantizeOp4Bit::narrow_xs(&xs, 64, 64)?;
        let mut qs = KvQuantizeOp4Bit::narrow_qs(&full_qs, 64, 64)?;

        super::quantize_inplace_4bit(&mut qs, &xs)?;

        let dequant = super::dequantize_4bit(&qs, DType::F32)?.reshape(xs.shape())?;

        let rmse = rmse(&xs.to_vec1::<f32>()?, &dequant.to_vec1::<f32>()?);
        assert!(rmse < 0.002, "{rmse}");

        Ok(())
    }
}
