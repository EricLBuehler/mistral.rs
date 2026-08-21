use super::config::GdnDims;
use candle_core::{Result, Tensor, D};
use mistralrs_quant::QuantMethod;
use std::sync::Arc;

pub enum GdnInputProjection {
    Grouped {
        in_proj_qkvz: Arc<dyn QuantMethod>,
        in_proj_ba: Arc<dyn QuantMethod>,
    },
    Split {
        in_proj_qkv: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_b: Arc<dyn QuantMethod>,
        in_proj_a: Arc<dyn QuantMethod>,
        merged_qkv_z: Option<crate::ops::MergedDenseProjection>,
        merged_b_a: Option<crate::ops::MergedDenseProjection>,
    },
    SplitQkvzGroupedBa {
        in_proj_qkv: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_ba: Arc<dyn QuantMethod>,
        merged_qkv_z: Option<crate::ops::MergedDenseProjection>,
    },
}

impl GdnInputProjection {
    pub fn forward(
        &self,
        x: &Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GdnProjection> {
        match self {
            Self::Grouped {
                in_proj_qkvz,
                in_proj_ba,
            } => GdnProjection::from_grouped(
                in_proj_qkvz.forward(x)?,
                in_proj_ba.forward(x)?,
                dims,
                batch_size,
                seq_len,
            ),
            Self::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
                merged_qkv_z,
                merged_b_a,
            } => {
                let (mixed_qkv, mixed_z) = if let Some(merged_qkv_z) = merged_qkv_z {
                    let [mixed_qkv, mixed_z]: [Tensor; 2] =
                        merged_qkv_z.forward(x)?.try_into().map_err(|_| {
                            candle_core::Error::msg(
                                "packed GDN QKV/Z returned the wrong output count",
                            )
                        })?;
                    (mixed_qkv.contiguous()?, mixed_z)
                } else {
                    shared_qkv_z(x, in_proj_qkv, in_proj_z)?
                };
                let (mixed_b, mixed_a) = if let Some(merged_b_a) = merged_b_a {
                    let [mixed_b, mixed_a]: [Tensor; 2] =
                        merged_b_a.forward(x)?.try_into().map_err(|_| {
                            candle_core::Error::msg(
                                "packed GDN B/A returned the wrong output count",
                            )
                        })?;
                    (mixed_b, mixed_a)
                } else {
                    (in_proj_b.forward(x)?, in_proj_a.forward(x)?)
                };
                GdnProjection::from_split(
                    mixed_qkv, mixed_z, mixed_b, mixed_a, dims, batch_size, seq_len,
                )
            }
            Self::SplitQkvzGroupedBa {
                in_proj_qkv,
                in_proj_z,
                in_proj_ba,
                merged_qkv_z,
            } => {
                let (mixed_qkv, mixed_z) = if let Some(merged_qkv_z) = merged_qkv_z {
                    let [mixed_qkv, mixed_z]: [Tensor; 2] =
                        merged_qkv_z.forward(x)?.try_into().map_err(|_| {
                            candle_core::Error::msg(
                                "packed GDN QKV/Z returned the wrong output count",
                            )
                        })?;
                    (mixed_qkv.contiguous()?, mixed_z)
                } else {
                    shared_qkv_z(x, in_proj_qkv, in_proj_z)?
                };
                GdnProjection::from_split_grouped_ba(
                    mixed_qkv,
                    mixed_z,
                    in_proj_ba.forward(x)?,
                    dims,
                    batch_size,
                    seq_len,
                )
            }
        }
    }
}

fn shared_qkv_z(
    x: &Tensor,
    qkv: &Arc<dyn QuantMethod>,
    z: &Arc<dyn QuantMethod>,
) -> Result<(Tensor, Tensor)> {
    if let Some(outputs) = mistralrs_quant::try_forward_with_shared_quantized_activation(
        x,
        &[qkv.as_ref(), z.as_ref()],
    )? {
        let [qkv, z]: [Tensor; 2] = outputs.try_into().map_err(|_| {
            candle_core::Error::msg("shared GDN projection returned the wrong output count")
        })?;
        Ok((qkv, z))
    } else {
        Ok((qkv.forward(x)?, z.forward(x)?))
    }
}

pub struct GdnProjection {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
    // Set when the projection already produced `[q | k | v]` contiguously, so the conv needs no copy
    conv_src: Option<Tensor>,
}

impl GdnProjection {
    pub fn from_grouped(
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
            conv_src: None,
        })
    }

    pub fn from_split(
        mixed_qkv: Tensor,
        mixed_z: Tensor,
        mixed_b: Tensor,
        mixed_a: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
        let conv_src = mixed_qkv
            .is_contiguous()
            .then(|| mixed_qkv.reshape((batch_size, seq_len, dims.conv_dim)))
            .transpose()?;

        Ok(Self {
            q: q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            k: k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            z: mixed_z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: mixed_b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: mixed_a.reshape((batch_size, seq_len, dims.num_v_heads))?,
            conv_src,
        })
    }

    pub fn from_split_grouped_ba(
        mixed_qkv: Tensor,
        mixed_z: Tensor,
        mixed_ba: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let mixed_ba =
            mixed_ba.reshape((batch_size, seq_len, dims.num_k_heads, 2 * dims.v_per_group))?;
        let b = mixed_ba.narrow(D::Minus1, 0, dims.v_per_group)?.reshape((
            batch_size,
            seq_len,
            dims.num_v_heads,
        ))?;
        let a = mixed_ba
            .narrow(D::Minus1, dims.v_per_group, dims.v_per_group)?
            .reshape((batch_size, seq_len, dims.num_v_heads))?;
        Self::from_split(mixed_qkv, mixed_z, b, a, dims, batch_size, seq_len)
    }

    pub fn conv_input(&self, dims: &GdnDims, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(src) = &self.conv_src {
            return Ok(src.clone());
        }
        let q = self.q.reshape((batch_size, seq_len, dims.key_dim))?;
        let k = self.k.reshape((batch_size, seq_len, dims.key_dim))?;
        let v = self.v.reshape((batch_size, seq_len, dims.value_dim))?;
        Tensor::cat(&[&q, &k, &v], D::Minus1)
    }
}
