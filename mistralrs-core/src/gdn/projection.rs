use super::config::GdnDims;
use candle_core::{Result, Tensor, D};

pub struct GdnProjection {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

impl GdnProjection {
    pub fn from_packed(
        mixed: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let mixed_qkvz = mixed.narrow(D::Minus1, 0, dims.qkvz_out_dim())?;
        let mixed_ba = mixed.narrow(D::Minus1, dims.qkvz_out_dim(), dims.ba_out_dim())?;
        Self::new(mixed_qkvz, mixed_ba, dims, batch_size, seq_len)
    }

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
}
