use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

use super::config::RopeScaling;

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

#[derive(Debug, Clone)]
pub struct HunyuanVLRotaryEmbedding {
    inv_freq: Tensor,
    sections: Vec<usize>,
}

impl HunyuanVLRotaryEmbedding {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn new(base: f64, head_dim: usize, scaling: &RopeScaling, device: &Device) -> Result<Self> {
        let base = if matches!(scaling.rope_type.as_str(), "xdrope" | "dynamic") {
            if let Some(alpha) = scaling
                .alpha
                .filter(|alpha| alpha.is_finite() && *alpha > 0.0)
            {
                if head_dim <= 2 {
                    candle_core::bail!("HunYuan-VL xdrope requires head_dim greater than 2");
                }
                base * alpha.powf(head_dim as f64 / (head_dim as f64 - 2.0))
            } else {
                base
            }
        } else {
            base
        };
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (base as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;
        let sections = if scaling.xdrope_section.is_empty() {
            vec![head_dim / 2]
        } else {
            scaling.xdrope_section.clone()
        };
        if sections.iter().sum::<usize>() != head_dim / 2 {
            candle_core::bail!(
                "HunYuan-VL xdrope sections sum {} does not match head_dim/2 {}",
                sections.iter().sum::<usize>(),
                head_dim / 2
            );
        }
        Ok(Self { inv_freq, sections })
    }

    /// Compute cos/sin from 4D position ids of shape (4, batch, seq_len).
    /// Official xD-RoPE builds a full head-dim embedding for each coordinate
    /// plane, then cycles those planes across the repeated section chunks.
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        if position_ids.dim(0)? != self.sections.len() {
            candle_core::bail!(
                "HunYuan-VL position_ids first dim {} must match xdrope sections {}",
                position_ids.dim(0)?,
                self.sections.len()
            );
        }

        let x_dim = self.sections.len();
        let inv_freq = self.inv_freq.reshape((1, 1, ()))?;
        let mut cos_planes = Vec::with_capacity(x_dim);
        let mut sin_planes = Vec::with_capacity(x_dim);
        for dim_idx in 0..x_dim {
            let position = position_ids
                .i(dim_idx)?
                .to_dtype(inv_freq.dtype())?
                .unsqueeze(D::Minus1)?;
            let freqs = position.broadcast_mul(&inv_freq)?;
            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
            cos_planes.push(emb.cos()?);
            sin_planes.push(emb.sin()?);
        }

        let mut cos_parts = Vec::with_capacity(x_dim * 2);
        let mut sin_parts = Vec::with_capacity(x_dim * 2);
        let mut offset = 0usize;
        for (chunk_idx, section) in self.sections.iter().chain(self.sections.iter()).enumerate() {
            let plane_idx = chunk_idx % x_dim;
            cos_parts.push(cos_planes[plane_idx].narrow(D::Minus1, offset, *section)?);
            sin_parts.push(sin_planes[plane_idx].narrow(D::Minus1, offset, *section)?);
            offset += section;
        }
        Ok((
            Tensor::cat(&cos_parts, D::Minus1)?
                .to_dtype(dtype)?
                .contiguous()?,
            Tensor::cat(&sin_parts, D::Minus1)?
                .to_dtype(dtype)?
                .contiguous()?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn forward_qk_norm(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        q_eps: f64,
        k_eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let origin_q_dtype = q.dtype();
        let origin_k_dtype = k.dtype();
        let cos = cos.unsqueeze(1)?.to_dtype(DType::F32)?;
        let sin = sin.unsqueeze(1)?.to_dtype(DType::F32)?;
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let q_embed = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?
            .to_dtype(origin_q_dtype)?;
        let k_embed = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?
            .to_dtype(origin_k_dtype)?;
        let q_embed = candle_nn::ops::rms_norm(&q_embed.contiguous()?, q_weight, q_eps as f32)?;
        let k_embed = candle_nn::ops::rms_norm(&k_embed.contiguous()?, k_weight, k_eps as f32)?;
        Ok((q_embed, k_embed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scaling(sections: Vec<usize>) -> RopeScaling {
        RopeScaling {
            alpha: None,
            factor: None,
            mscale: None,
            mscale_all_dim: None,
            rope_type: "xdrope".to_string(),
            xdrope_section: sections,
        }
    }

    #[test]
    fn xdrope_cycles_full_head_dim_chunks_across_position_planes() -> Result<()> {
        let device = Device::Cpu;
        let rope = HunyuanVLRotaryEmbedding::new(10_000.0, 4, &test_scaling(vec![1, 1]), &device)?;
        let position_ids = Tensor::from_vec(vec![1i64, 2, 3, 10, 20, 30], (2, 1, 3), &device)?;
        let (cos, sin) = rope.compute_cos_sin(&position_ids, DType::F32)?;
        let cos = cos.to_vec3::<f32>()?;
        let sin = sin.to_vec3::<f32>()?;

        for (token_idx, (plane0, plane1)) in [(1f32, 10f32), (2f32, 20f32), (3f32, 30f32)]
            .into_iter()
            .enumerate()
        {
            let expected_angles = [plane0, plane1 * 0.01, plane0, plane1 * 0.01];
            for (dim_idx, angle) in expected_angles.into_iter().enumerate() {
                assert!((cos[0][token_idx][dim_idx] - angle.cos()).abs() < 1e-6);
                assert!((sin[0][token_idx][dim_idx] - angle.sin()).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn forward_qk_norm_restores_original_dtype_after_float_rope() -> Result<()> {
        let device = Device::Cpu;
        let rope = HunyuanVLRotaryEmbedding::new(10_000.0, 4, &test_scaling(vec![1, 1]), &device)?;
        let position_ids = Tensor::from_vec(vec![1i64, 2], (2, 1, 1), &device)?;
        let cos_sin = rope.compute_cos_sin(&position_ids, DType::BF16)?;
        let q = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(vec![4f32, 3., 2., 1.], (1, 1, 1, 4), &device)?
            .to_dtype(DType::BF16)?;
        let norm_weight = Tensor::from_vec(vec![1f32; 4], (4,), &device)?.to_dtype(DType::BF16)?;

        let (q_out, k_out) =
            rope.forward_qk_norm(&cos_sin, &q, &k, &norm_weight, &norm_weight, 1e-5, 1e-5)?;
        assert_eq!(q_out.dtype(), DType::BF16);
        assert_eq!(k_out.dtype(), DType::BF16);
        Ok(())
    }
}
