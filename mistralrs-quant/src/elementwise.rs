use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};
use half::bf16;

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Mul,
}

impl Binary {
    fn apply(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Add => lhs + rhs,
            Self::Mul => lhs * rhs,
        }
    }
}

fn row_offset(layout: &Layout, mut row: usize, cols: usize) -> usize {
    if layout.is_contiguous() {
        return layout.start_offset() + row * cols;
    }
    let mut offset = layout.start_offset();
    for (&dim, &stride) in layout.dims()[..layout.dims().len().saturating_sub(1)]
        .iter()
        .zip(layout.stride())
        .rev()
    {
        offset += row % dim * stride;
        row /= dim;
    }
    offset
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{bf16, Binary};

    const LANES: usize = 8;
    const ROUNDING_BIAS: u32 = 0x7fff;
    const MAGNITUDE_MASK: u32 = 0x7fff_ffff;
    const INFINITY: u32 = 0x7f80_0000;
    const QUIET_NAN: u32 = 0x0040_0000;

    pub(super) struct RowStrides {
        pub(super) lhs: usize,
        pub(super) rhs: usize,
    }

    unsafe fn round(value: float32x4_t) -> uint16x4_t {
        let bits = vreinterpretq_u32_f32(value);
        let bias = vaddq_u32(
            vdupq_n_u32(ROUNDING_BIAS),
            vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1)),
        );
        let nan = vcgtq_u32(
            vandq_u32(bits, vdupq_n_u32(MAGNITUDE_MASK)),
            vdupq_n_u32(INFINITY),
        );
        vshrn_n_u32::<16>(vbslq_u32(
            nan,
            vorrq_u32(bits, vdupq_n_u32(QUIET_NAN)),
            vaddq_u32(bits, bias),
        ))
    }

    pub(super) unsafe fn row(
        lhs: &[bf16],
        rhs: &[bf16],
        output: &mut [bf16],
        strides: RowStrides,
        op: Binary,
    ) -> usize {
        let RowStrides {
            lhs: lhs_stride,
            rhs: rhs_stride,
        } = strides;
        if lhs_stride > 1 || rhs_stride > 1 {
            return 0;
        }
        let end = output.len() / LANES * LANES;
        for index in (0..end).step_by(LANES) {
            let load = |input: &[bf16], stride| {
                if stride == 0 {
                    vdupq_n_u16(input[0].to_bits())
                } else {
                    vld1q_u16(input.as_ptr().add(index).cast())
                }
            };
            let lhs = load(lhs, lhs_stride);
            let rhs = load(rhs, rhs_stride);
            let lhs_low = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(lhs)));
            let lhs_high = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(lhs));
            let rhs_low = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(rhs)));
            let rhs_high = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(rhs));
            let (low, high) = match op {
                Binary::Add => (vaddq_f32(lhs_low, rhs_low), vaddq_f32(lhs_high, rhs_high)),
                Binary::Mul => (vmulq_f32(lhs_low, rhs_low), vmulq_f32(lhs_high, rhs_high)),
            };
            vst1q_u16(
                output.as_mut_ptr().add(index).cast(),
                vcombine_u16(round(low), round(high)),
            );
        }
        end
    }
}

impl CustomOp2 for Binary {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "cpu-bf16-add-rne",
            Self::Mul => "cpu-bf16-mul-rne",
        }
    }

    fn cpu_fwd(
        &self,
        lhs: &CpuStorage,
        lhs_layout: &Layout,
        rhs: &CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs)) = (lhs, rhs) else {
            candle_core::bail!("CPU BF16 binary operations require matching BF16 inputs");
        };
        let count = lhs_layout.shape().elem_count();
        let cols = lhs_layout.dims().last().copied().unwrap_or(1).max(1);
        let lhs_stride = lhs_layout.stride().last().copied().unwrap_or(0);
        let rhs_stride = rhs_layout.stride().last().copied().unwrap_or(0);
        let mut output = vec![bf16::ZERO; count];
        let output_ptr = output.as_mut_ptr() as usize;
        candle_core::utils::barrier_pool().execute_chunked(count / cols, |range| {
            for row in range {
                let lhs = &lhs[row_offset(lhs_layout, row, cols)..];
                let rhs = &rhs[row_offset(rhs_layout, row, cols)..];
                // Each worker owns complete disjoint rows until the pool joins.
                let output = unsafe {
                    std::slice::from_raw_parts_mut((output_ptr as *mut bf16).add(row * cols), cols)
                };
                #[cfg(target_arch = "aarch64")]
                let start = unsafe {
                    neon::row(
                        lhs,
                        rhs,
                        output,
                        neon::RowStrides {
                            lhs: lhs_stride,
                            rhs: rhs_stride,
                        },
                        *self,
                    )
                };
                #[cfg(not(target_arch = "aarch64"))]
                let start = 0;
                for col in start..cols {
                    output[col] = bf16::from_f32(self.apply(
                        lhs[col * lhs_stride].to_f32(),
                        rhs[col * rhs_stride].to_f32(),
                    ));
                }
            }
        });
        Ok((CpuStorage::BF16(output), lhs_layout.shape().clone()))
    }
}

fn binary(lhs: &Tensor, rhs: &Tensor, op: Binary) -> Result<Tensor> {
    if lhs.device().is_cpu() && lhs.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
        let shape = lhs
            .shape()
            .broadcast_shape_binary_op(rhs.shape(), op.name())?;
        lhs.broadcast_as(&shape)?
            .apply_op2_no_bwd(&rhs.broadcast_as(&shape)?, &op)
    } else {
        match op {
            Binary::Add => lhs.broadcast_add(rhs),
            Binary::Mul => lhs.broadcast_mul(rhs),
        }
    }
}

pub fn broadcast_add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    binary(lhs, rhs, Binary::Add)
}

pub fn broadcast_mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    binary(lhs, rhs, Binary::Mul)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn bf16_binary_rounds_vector_lanes_and_tails_to_nearest_even() -> Result<()> {
        for op in [Binary::Add, Binary::Mul] {
            let lhs = [1.4453125f32, -1.265625, 0.20300293].repeat(9);
            let rhs = [-0.00000730157f32, 0.00010442734, 0.20300293].repeat(9);
            let expected = lhs
                .iter()
                .zip(&rhs)
                .map(|(&a, &b)| {
                    bf16::from_f32(op.apply(bf16::from_f32(a).to_f32(), bf16::from_f32(b).to_f32()))
                        .to_f32()
                })
                .collect::<Vec<_>>();
            let lhs = Tensor::new(lhs, &Device::Cpu)?.to_dtype(DType::BF16)?;
            let rhs = Tensor::new(rhs, &Device::Cpu)?.to_dtype(DType::BF16)?;
            assert_eq!(
                binary(&lhs, &rhs, op)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                expected
            );
            assert_eq!(
                binary(&lhs.reshape((3, 9))?.t()?, &rhs.reshape((3, 9))?.t()?, op)?
                    .t()?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn bf16_binary_handles_bias_and_head_gate_broadcasts() -> Result<()> {
        let input = Tensor::full(1.4453125f32, (2, 3, 9), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let bias = Tensor::full(-0.00000730157f32, 9, &Device::Cpu)?.to_dtype(DType::BF16)?;
        assert_eq!(
            broadcast_add(&input, &bias)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1.4453125; 54]
        );
        let gate = Tensor::full(0.3190918f32, (2, 3, 1), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let expected = bf16::from_f32(1.4453125 * bf16::from_f32(0.3190918).to_f32()).to_f32();
        assert_eq!(
            broadcast_mul(&input, &gate)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![expected; 54]
        );
        Ok(())
    }
}
