use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(max_period: f32, head_dim: usize, device: &candle_core::Device) -> Result<Self> {
        let d = head_dim / 2;
        let ds = Tensor::arange(0u32, d as u32, device)?.to_dtype(DType::F32)?;
        let inv_freq = ds
            .affine((-max_period.ln() * 2.0 / head_dim as f32) as f64, 0.0)?
            .exp()?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (b, t, h, d_full) = q.dims4()?;
        let (_bk, _tk, hk, _dk) = k.dims4()?;
        let d = d_full / 2;
        let dev = q.device();

        // ts = (arange(T) + offset).view(-1, 1, 1)
        let ts = if t == 1 {
            Tensor::new(&[offset as f32], dev)?
        } else {
            Tensor::arange(0u32, t as u32, dev)?
                .to_dtype(DType::F32)?
                .affine(1.0, offset as f64)?
        }
        .reshape((t, 1, 1))?;

        // freqs * ts -> shape (t, 1, d)
        let freqs_ts = self.inv_freq.reshape((1, 1, d))?.broadcast_mul(&ts)?;
        let cos = freqs_ts.cos()?;
        let sin = freqs_ts.sin()?;

        // Reshape q and k to (b, t, h, d, 2)
        let q = q.reshape((b, t, h, d, 2))?;
        let k = k.reshape((b, t, hk, d, 2))?;

        let qr = q.narrow(4, 0, 1)?.squeeze(4)?;
        let qi = q.narrow(4, 1, 1)?.squeeze(4)?;
        let kr = k.narrow(4, 0, 1)?.squeeze(4)?;
        let ki = k.narrow(4, 1, 1)?.squeeze(4)?;

        // qor = qr * cos - qi * sin
        // qoi = qr * sin + qi * cos
        let qor = (qr.broadcast_mul(&cos)? - qi.broadcast_mul(&sin)?)?;
        let qoi = (qr.broadcast_mul(&sin)? + qi.broadcast_mul(&cos)?)?;

        let kor = (kr.broadcast_mul(&cos)? - ki.broadcast_mul(&sin)?)?;
        let koi = (kr.broadcast_mul(&sin)? + ki.broadcast_mul(&cos)?)?;

        let qo = Tensor::stack(&[qor, qoi], 4)?.reshape((b, t, h, d_full))?;
        let ko = Tensor::stack(&[kor, koi], 4)?.reshape((b, t, hk, d_full))?;

        Ok((qo, ko))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_rope_shape() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 10, 4, 32), DType::F32, &device)?;
        let k = Tensor::zeros((1, 10, 4, 32), DType::F32, &device)?;
        let rope = RotaryEmbedding::new(10000.0, 32, &device)?;
        let (qo, ko) = rope.forward(&q, &k, 0)?;
        assert_eq!(qo.dims(), &[1, 10, 4, 32]);
        assert_eq!(ko.dims(), &[1, 10, 4, 32]);
        Ok(())
    }
}
