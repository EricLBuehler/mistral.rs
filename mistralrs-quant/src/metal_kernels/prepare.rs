use std::collections::BTreeSet;

use candle_core::{quantized::QMatMul, Device, Result, Tensor};
use candle_metal_kernels::source::Source;

use super::{
    afq_qmm_dispatch, afq_type_name, moe_weighted_reduce_kernel_name, pick_afq_gather_rhs_tile,
    AfqQmmShape, Kernels,
};
use crate::{afq_sorted_gather_supported, AfqBits, AfqGroupSize};

const AFQ_PREPARE_ROWS: &[usize] = &[1, 2, 4, 8, 16, 32, 256];
const AFQ_GATHER_PREPARE_ROWS: &[usize] = &[1, 16, 128, 129];
const GLU_ACTIVATIONS: std::ops::Range<u32> = 0..4;

pub(crate) fn prepare_afq(
    weight: &Tensor,
    scales: &Tensor,
    bits: AfqBits,
    group_size: AfqGroupSize,
) -> Result<()> {
    let Device::Metal(device) = weight.device() else {
        return Ok(());
    };
    let sorted_gather = afq_sorted_gather_supported(group_size, bits);
    let bits = bits as usize;
    let group_size = group_size as usize;
    let dtype = scales.dtype();
    let ty = afq_type_name(dtype).map_err(candle_core::Error::wrap)?;
    let input = scales.dim(candle_core::D::Minus1)? * group_size;
    let output = weight.dim(candle_core::D::Minus2)?;
    let stacked = weight.rank() > 2;
    let aligned = output.is_multiple_of(32);
    let mut names = BTreeSet::new();
    for &rows in AFQ_PREPARE_ROWS {
        let modes: &[(bool, bool)] = if stacked {
            &[(true, false), (false, true)]
        } else {
            &[(false, false)]
        };
        for &(batched, gather) in modes {
            let dispatch = afq_qmm_dispatch(
                device.device(),
                AfqQmmShape {
                    input,
                    output,
                    rows,
                    batches: 1,
                    batched,
                    gather,
                    transpose: true,
                },
                dtype,
                bits,
                group_size,
            )
            .map_err(candle_core::Error::wrap)?;
            names.insert(dispatch.name);
        }
    }
    if stacked && sorted_gather {
        for &rows in AFQ_GATHER_PREPARE_ROWS {
            let (bm, bn, bk, wm, wn) = pick_afq_gather_rhs_tile(rows);
            let am = if rows.is_multiple_of(bm) { "t" } else { "n" };
            let an = if output.is_multiple_of(bn) { "t" } else { "n" };
            let ak = if input.is_multiple_of(bk) { "t" } else { "n" };
            names.insert(format!(
                "affine_gather_qmm_rhs_{ty}_gs_{group_size}_b_{bits}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}_t_true_alM_{am}_alN_{an}_alK_{ak}"
            ));
            for act in GLU_ACTIVATIONS {
                names.insert(format!(
                    "affine_gather_qmm_rhs_gate_up_{ty}_gs_{group_size}_b_{bits}_act_{act}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}_alM_{am}_alN_{an}_alK_{ak}"
                ));
            }
        }
        names.insert(moe_weighted_reduce_kernel_name(dtype).map_err(candle_core::Error::wrap)?);
    } else if !stacked {
        if input / group_size >= 2 {
            names.insert(format!(
                "qmm_t_splitk_{ty}_gs_{group_size}_b_{bits}_alN_{aligned}"
            ));
        }
        for act in GLU_ACTIVATIONS {
            names.insert(format!(
                "qmm_t_gate_up_{ty}_gs_{group_size}_b_{bits}_act_{act}_alN_{aligned}"
            ));
        }
        if aligned {
            names.insert(format!("qmm_t_qkv_{ty}_gs_{group_size}_b_{bits}"));
        }
        names.insert(format!("affine_embedding_{ty}_gs_{group_size}_b_{bits}"));
    }
    for name in names {
        Kernels::global()
            .load_pipeline(device.device(), name)
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}

pub(crate) fn prepare_dense(weight: &Tensor) -> Result<()> {
    let Device::Metal(device) = weight.device() else {
        return Ok(());
    };
    prepare_dense_device(device)
}

fn prepare_dense_device(device: &candle_core::MetalDevice) -> Result<()> {
    // Candle keeps its shape-specific GEMM selector private; preload its source libraries here.
    for source in [Source::Gemm, Source::Gemv] {
        device
            .kernels()
            .load_library(device.device(), source)
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}

pub(crate) fn prepare_gguf(weight: &QMatMul) -> Result<()> {
    let q = match weight {
        QMatMul::QTensor(weight) => weight,
        QMatMul::Tensor(weight) | QMatMul::TensorF16(weight) => return prepare_dense(weight),
    };
    let device = q.device();
    let Device::Metal(device) = &device else {
        return Ok(());
    };
    use candle_core::quantized::GgmlDType;
    let name = match q.dtype() {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_K",
        GgmlDType::Q3K => "q3_K",
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q5K => "q5_K",
        GgmlDType::Q6K => "q6_K",
        GgmlDType::Q8K => "q8_K",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::F32 => "f32",
    };
    let mut names = Vec::new();
    if !matches!(
        q.dtype(),
        GgmlDType::BF16 | GgmlDType::Q8_1 | GgmlDType::Q8K
    ) {
        names.push(format!("kernel_mul_mv_{name}_f32"));
    }
    if !matches!(q.dtype(), GgmlDType::Q8_1 | GgmlDType::Q8K) {
        names.push(format!("kernel_mul_mm_{name}_f32"));
        names.push(format!("kernel_get_rows_{name}"));
    }
    for name in names {
        device
            .kernels()
            .load_pipeline(device.device(), Source::Quantized, name)
            .map_err(candle_core::Error::wrap)?;
    }
    if q.shape().rank() > 2
        || matches!(
            q.dtype(),
            GgmlDType::BF16 | GgmlDType::Q8_1 | GgmlDType::Q8K
        )
    {
        prepare_dense_device(device)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AfqBits, AfqGroupSize, AfqLayer, QuantMethod};
    use candle_core::DType;

    #[test]
    fn afq_preparation_loads_dense_and_expert_variants() -> Result<()> {
        let device = Device::new_metal(0)?;
        for (dtype, bits, group, shape) in [
            (DType::BF16, AfqBits::Four, AfqGroupSize::Low, vec![64, 512]),
            (DType::BF16, AfqBits::Six, AfqGroupSize::Med, vec![17, 256]),
            (
                DType::F16,
                AfqBits::Four,
                AfqGroupSize::High,
                vec![2, 32, 128],
            ),
            (
                DType::BF16,
                AfqBits::Three,
                AfqGroupSize::Low,
                vec![2, 32, 128],
            ),
            (
                DType::BF16,
                AfqBits::Four,
                AfqGroupSize::Med,
                vec![2, 32, 128],
            ),
            (
                DType::F16,
                AfqBits::Eight,
                AfqGroupSize::Med,
                vec![2, 32, 128],
            ),
            (
                DType::F32,
                AfqBits::Four,
                AfqGroupSize::Med,
                vec![2, 32, 128],
            ),
        ] {
            let mut packed_shape = shape.clone();
            *packed_shape.last_mut().unwrap() = shape[shape.len() - 1] * bits as usize / 32;
            let mut scale_shape = shape;
            *scale_shape.last_mut().unwrap() /= group as usize;
            let layer = AfqLayer::from_parts(
                Tensor::zeros(packed_shape, DType::U32, &device)?,
                Tensor::ones(scale_shape.clone(), dtype, &device)?,
                Tensor::zeros(scale_shape, dtype, &device)?,
                None,
                bits,
                group,
            );
            layer.prepare_metal_kernels(dtype)?;
            layer.prepare_metal_kernels(dtype)?;
        }
        Ok(())
    }

    #[test]
    fn gguf_preparation_loads_only_the_resident_format() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QTensor};
        let device = Device::new_metal(0)?;
        let weight = Tensor::zeros((64, 256), DType::F32, &device)?;
        let quantized = QTensor::quantize(&weight, GgmlDType::Q4K)?;
        let layer = crate::GgufMatMul::new(crate::QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(quantized),
            b: None,
        })?;
        layer.prepare_metal_kernels(DType::BF16)?;
        Ok(())
    }

    #[test]
    fn gguf_expert_preparation_preserves_formats_without_native_gemv() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QStorage, QTensor};
        const SHAPE: (usize, usize, usize) = (2, 32, 256);
        const ELEMENTS: usize = SHAPE.0 * SHAPE.1 * SHAPE.2;
        let device = Device::new_metal(0)?;
        for dtype in [GgmlDType::BF16, GgmlDType::Q8_1, GgmlDType::Q8K] {
            let storage = QStorage::from_data(
                vec![0u8; ELEMENTS / dtype.block_size() * dtype.type_size()].into(),
                &device,
                dtype,
            )?;
            let layer = crate::GgufMatMul::from_qtensor(QTensor::new(storage, SHAPE)?, None);
            layer.prepare_metal_kernels(DType::BF16)?;
            assert_eq!(layer.get_qtensor().unwrap().dtype(), dtype);
        }
        Ok(())
    }
}
