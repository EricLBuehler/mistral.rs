use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor, WithDType};

struct CpuRmsNorm {
    eps: f32,
}

trait NormElement: WithDType {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl NormElement for half::f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl NormElement for half::bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

#[allow(clippy::cast_precision_loss)]
fn normalize<T: NormElement>(
    input: &[T],
    layout: &Layout,
    weight: &[T],
    weight_layout: &Layout,
    eps: f32,
) -> Result<(CpuStorage, Shape)> {
    let dims = layout.dims();
    let cols = *dims
        .last()
        .ok_or_else(|| candle_core::Error::msg("RMSNorm requires at least one input dimension"))?;
    if cols == 0 || weight_layout.dims() != [cols] {
        candle_core::bail!("RMSNorm weight must match the nonempty final input dimension");
    }
    let rows = layout.shape().elem_count() / cols;
    let strides = layout.stride();
    let column_stride = strides[strides.len() - 1];
    let contiguous = layout.is_contiguous();
    let weight = &weight[weight_layout.start_offset()..];
    let weight_stride = weight_layout.stride()[0];
    let mut output = vec![T::from_f32(0.0); rows * cols];
    let output_ptr = output.as_mut_ptr() as usize;
    candle_core::utils::barrier_pool().execute_chunked(rows, |range| {
        for row in range {
            let mut offset = layout.start_offset();
            if contiguous {
                offset += row * cols;
            } else {
                let mut index = row;
                for (&dim, &stride) in dims[..dims.len() - 1]
                    .iter()
                    .zip(&strides[..strides.len() - 1])
                    .rev()
                {
                    offset += (index % dim) * stride;
                    index /= dim;
                }
            }
            let mut sum = 0f32;
            for col in 0..cols {
                let value = input[offset + col * column_stride].to_f32();
                sum += value * value;
            }
            let inv_rms = (sum / cols as f32 + eps).sqrt().recip();
            for col in 0..cols {
                let value = input[offset + col * column_stride].to_f32();
                // HF rounds the normalized value before multiplying by the learned weight.
                let normalized = T::from_f32(value * inv_rms);
                let value = T::from_f32(normalized.to_f32() * weight[col * weight_stride].to_f32());
                // Rows are disjoint and the pool joins before the output is returned.
                unsafe { *((output_ptr as *mut T).add(row * cols + col)) = value };
            }
        }
    });
    Ok((T::to_cpu_storage_owned(output), layout.shape().clone()))
}

impl CustomOp2 for CpuRmsNorm {
    fn name(&self) -> &'static str {
        "cpu-rms-norm-f32-accum"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        layout: &Layout,
        weight: &CpuStorage,
        weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        match (input, weight) {
            (CpuStorage::F16(input), CpuStorage::F16(weight)) => {
                normalize(input, layout, weight, weight_layout, self.eps)
            }
            (CpuStorage::BF16(input), CpuStorage::BF16(weight)) => {
                normalize(input, layout, weight, weight_layout, self.eps)
            }
            _ => candle_core::bail!("CPU half RMSNorm requires matching F16 or BF16 tensors"),
        }
    }
}

pub fn cpu_rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    input.apply_op2_no_bwd(weight, &CpuRmsNorm { eps })
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    const INPUT: [f32; 8] = [
        -0.041015625,
        0.306640625,
        0.65234375,
        1.0,
        1.34375,
        1.6953125,
        2.046875,
        2.390625,
    ];
    const WEIGHT: [f32; 8] = [
        0.5, 0.6328125, 0.76171875, 0.89453125, 1.0234375, 1.15625, 1.2890625, 1.4140625,
    ];
    const F16_EXPECTED: [f32; 8] = [
        -0.0144500732421875,
        0.13671875,
        0.35009765625,
        0.63037109375,
        0.96923828125,
        1.380859375,
        1.859375,
        2.3828125,
    ];
    const BF16_EXPECTED: [f32; 8] = [
        -0.01446533203125,
        0.13671875,
        0.349609375,
        0.62890625,
        0.96875,
        1.3828125,
        1.859375,
        2.390625,
    ];
    const EPS: f32 = 1e-5;

    #[test]
    fn cpu_half_rms_norm_rounds_after_normalization() -> Result<()> {
        for (dtype, expected) in [(DType::F16, F16_EXPECTED), (DType::BF16, BF16_EXPECTED)] {
            let input = Tensor::new(&INPUT, &Device::Cpu)?.to_dtype(dtype)?;
            let weight = Tensor::new(&WEIGHT, &Device::Cpu)?.to_dtype(dtype)?;
            let actual = cpu_rms_norm(&input, &weight, EPS)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn cpu_half_rms_norm_handles_strided_rows_and_weights() -> Result<()> {
        for (dtype, expected) in [(DType::F16, F16_EXPECTED), (DType::BF16, BF16_EXPECTED)] {
            let values = (0..4 * 8 * 35)
                .map(|index| INPUT[index / 35 % 8])
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, (4, 8, 35), &Device::Cpu)?
                .to_dtype(dtype)?
                .narrow(0, 1, 2)?
                .narrow(2, 1, 33)?
                .transpose(1, 2)?;
            let weights = WEIGHT
                .into_iter()
                .flat_map(|weight| [0.0, weight])
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(weights, (8, 2), &Device::Cpu)?
                .to_dtype(dtype)?
                .narrow(1, 1, 1)?
                .squeeze(1)?;
            let actual = cpu_rms_norm(&input, &weight, EPS)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected.repeat(66));
        }
        Ok(())
    }
}
