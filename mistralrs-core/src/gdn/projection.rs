use candle_core::{Result, Tensor, D};

use super::backend::l2_norm;
use super::config::GdnDims;

const QK_NORM_EPS: f64 = 1e-6;

pub struct GdnProjection {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

pub struct GdnRecurrentInput {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

impl GdnProjection {
    pub fn new(
        mixed_qkvz: Tensor,
        mixed_ba: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let group_size_qkvz = 2 * dims.head_k_dim + 2 * dims.v_per_group * dims.head_v_dim;
        let mixed_qkvz =
            mixed_qkvz.reshape((batch_size, seq_len, dims.num_k_heads, group_size_qkvz))?;
        let mixed_ba =
            mixed_ba.reshape((batch_size, seq_len, dims.num_k_heads, 2 * dims.v_per_group))?;

        let mut offset = 0;
        let q = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let k = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let v = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;
        offset += dims.v_per_group * dims.head_v_dim;
        let z = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;

        let b = mixed_ba.narrow(D::Minus1, 0, dims.v_per_group)?;
        let a = mixed_ba.narrow(D::Minus1, dims.v_per_group, dims.v_per_group)?;

        Ok(Self {
            q,
            k,
            v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            z: z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: a.reshape((batch_size, seq_len, dims.num_v_heads))?,
        })
    }

    pub fn conv_input(&self, dims: &GdnDims, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let q = self.q.reshape((batch_size, seq_len, dims.key_dim))?;
        let k = self.k.reshape((batch_size, seq_len, dims.key_dim))?;
        let v = self.v.reshape((batch_size, seq_len, dims.value_dim))?;
        Tensor::cat(&[&q, &k, &v], D::Minus1)
    }

    pub fn with_convolved_qkv(
        &self,
        mixed_qkv: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GdnRecurrentInput> {
        let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;

        Ok(GdnRecurrentInput {
            q: q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            k: k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            z: self.z.clone(),
            b: self.b.clone(),
            a: self.a.clone(),
        })
    }
}

impl GdnRecurrentInput {
    pub fn normalized_qk(
        &self,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (q, k) = if dims.v_per_group > 1 {
            let q = self
                .q
                .unsqueeze(3)?
                .repeat((1, 1, 1, dims.v_per_group, 1))?
                .reshape((batch_size, seq_len, dims.num_v_heads, dims.head_k_dim))?;
            let k = self
                .k
                .unsqueeze(3)?
                .repeat((1, 1, 1, dims.v_per_group, 1))?
                .reshape((batch_size, seq_len, dims.num_v_heads, dims.head_k_dim))?;
            (q, k)
        } else {
            (self.q.clone(), self.k.clone())
        };

        Ok((l2_norm(&q, QK_NORM_EPS)?, l2_norm(&k, QK_NORM_EPS)?))
    }
}
