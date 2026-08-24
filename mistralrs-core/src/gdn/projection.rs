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
                    (mixed_qkv, mixed_z)
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
                    (mixed_qkv, mixed_z)
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
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
    conv_input: GdnConvInput,
}

enum GdnConvInput {
    Direct(Tensor),
    Segmented { q: Tensor, k: Tensor, v: Tensor },
}

impl GdnProjection {
    fn split_gate(
        gate: Tensor,
        batch_size: usize,
        seq_len: usize,
        num_v_heads: usize,
    ) -> Result<Tensor> {
        if gate.dims() == [batch_size, seq_len, num_v_heads] {
            Ok(gate)
        } else {
            gate.reshape((batch_size, seq_len, num_v_heads))
        }
    }

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
            z: z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: a.reshape((batch_size, seq_len, dims.num_v_heads))?,
            conv_input: GdnConvInput::Segmented {
                q,
                k,
                v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            },
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
        Ok(Self {
            z: mixed_z,
            b: Self::split_gate(mixed_b, batch_size, seq_len, dims.num_v_heads)?,
            a: Self::split_gate(mixed_a, batch_size, seq_len, dims.num_v_heads)?,
            conv_input: GdnConvInput::Direct(mixed_qkv),
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
        match &self.conv_input {
            GdnConvInput::Direct(src) => Ok(src.clone()),
            GdnConvInput::Segmented { q, k, v } => {
                let q = q.reshape((batch_size, seq_len, dims.key_dim))?;
                let k = k.reshape((batch_size, seq_len, dims.key_dim))?;
                let v = v.reshape((batch_size, seq_len, dims.value_dim))?;
                Tensor::cat(&[&q, &k, &v], D::Minus1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::gdn::config::GdnVHeadLayout;

    fn dims() -> GdnDims {
        GdnDims {
            hidden_size: 32,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 3,
            head_v_dim: 5,
            conv_kernel_size: 4,
            key_dim: 6,
            value_dim: 20,
            conv_dim: 32,
            v_per_group: 2,
            v_head_layout: GdnVHeadLayout::Grouped,
        }
    }

    #[test]
    fn split_projection_preserves_strided_qkv_z_and_gate_views() -> Result<()> {
        let dev = Device::Cpu;
        let dims = dims();
        let (batch_size, seq_len) = (3, 2);
        let qkv_physical = dims.conv_dim + 7;
        let z_physical = dims.value_dim + 5;
        let qkv = Tensor::from_vec(
            vec![0.0f32; batch_size * seq_len * qkv_physical],
            (batch_size, seq_len, qkv_physical),
            &dev,
        )?
        .narrow(2, 3, dims.conv_dim)?;
        let z = Tensor::from_vec(
            vec![0.0f32; batch_size * seq_len * z_physical],
            (batch_size, seq_len, z_physical),
            &dev,
        )?
        .narrow(2, 2, dims.value_dim)?;
        let packed_gates = Tensor::from_vec(
            (0..batch_size * seq_len * dims.num_v_heads * 2)
                .map(|idx| idx as f32)
                .collect::<Vec<_>>(),
            (batch_size, seq_len, dims.num_v_heads * 2),
            &dev,
        )?;
        let b = packed_gates.narrow(D::Minus1, 0, dims.num_v_heads)?;
        let a = packed_gates.narrow(D::Minus1, dims.num_v_heads, dims.num_v_heads)?;
        assert!(!b.is_contiguous());
        assert!(!a.is_contiguous());
        let b_stride = b.stride().to_vec();
        let a_stride = a.stride().to_vec();
        let b_offset = b.layout().start_offset();
        let a_offset = a.layout().start_offset();

        let projection =
            GdnProjection::from_split(qkv.clone(), z, b, a, &dims, batch_size, seq_len)?;
        let conv_input = projection.conv_input(&dims, batch_size, seq_len)?;
        assert_eq!(conv_input.shape(), qkv.shape());
        assert_eq!(conv_input.stride(), qkv.stride());
        assert_eq!(
            conv_input.layout().start_offset(),
            qkv.layout().start_offset()
        );
        assert!(!conv_input.is_contiguous());
        assert!(!projection.z.is_contiguous());
        assert!(projection.z.layout().start_offset() > 0);
        assert_eq!(projection.b.stride(), b_stride);
        assert_eq!(projection.a.stride(), a_stride);
        assert_eq!(projection.b.layout().start_offset(), b_offset);
        assert_eq!(projection.a.layout().start_offset(), a_offset);
        Ok(())
    }
}
