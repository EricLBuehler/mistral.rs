use super::config::GdnDims;
use super::packed::PackedGdnLayout;
use candle_core::{Result, Tensor, D};
use mistralrs_quant::{
    ActivationQuantizationScheme, ActivationScaleLayout, QuantMethod, QuantizedActivation,
};
use std::{ops::Range, sync::Arc};

const RECURRENT_IDENTITY_GATE: f32 = f32::NEG_INFINITY;

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
    pub(crate) fn is_dynamic_lora_active(&self) -> bool {
        match self {
            Self::Grouped {
                in_proj_qkvz,
                in_proj_ba,
            } => in_proj_qkvz.is_dynamic_lora_active() || in_proj_ba.is_dynamic_lora_active(),
            Self::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
                ..
            } => {
                in_proj_qkv.is_dynamic_lora_active()
                    || in_proj_z.is_dynamic_lora_active()
                    || in_proj_b.is_dynamic_lora_active()
                    || in_proj_a.is_dynamic_lora_active()
            }
            Self::SplitQkvzGroupedBa {
                in_proj_qkv,
                in_proj_z,
                in_proj_ba,
                ..
            } => {
                in_proj_qkv.is_dynamic_lora_active()
                    || in_proj_z.is_dynamic_lora_active()
                    || in_proj_ba.is_dynamic_lora_active()
            }
        }
    }

    pub(crate) fn activation_quantization_scheme_for(
        &self,
        x: &Tensor,
    ) -> Option<ActivationQuantizationScheme> {
        if self.is_dynamic_lora_active() {
            return None;
        }
        match self {
            Self::Split {
                merged_qkv_z: Some(merged_qkv_z),
                ..
            }
            | Self::SplitQkvzGroupedBa {
                merged_qkv_z: Some(merged_qkv_z),
                ..
            } => merged_qkv_z.activation_quantization_scheme_for(x),
            _ => None,
        }
    }

    pub(crate) fn preferred_activation_scale_layout_for(
        &self,
        x: &Tensor,
    ) -> Option<ActivationScaleLayout> {
        if self.is_dynamic_lora_active() {
            return None;
        }
        match self {
            Self::Split {
                merged_qkv_z: Some(merged_qkv_z),
                ..
            }
            | Self::SplitQkvzGroupedBa {
                merged_qkv_z: Some(merged_qkv_z),
                ..
            } => merged_qkv_z.preferred_activation_scale_layout_for(x),
            _ => None,
        }
    }

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

    pub(crate) fn try_forward_quantized(
        &self,
        x: &Tensor,
        activation: &QuantizedActivation,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Option<GdnProjection>> {
        if self.activation_quantization_scheme_for(x) != Some(activation.scheme())
            || self.preferred_activation_scale_layout_for(x) != Some(activation.scale_layout())
            || activation.source_shape() != x.dims()
            || activation.source_dtype() != x.dtype()
            || !activation.quantized().device().same_device(x.device())
        {
            return Ok(None);
        }
        match self {
            Self::Grouped { .. } => Ok(None),
            Self::Split {
                in_proj_b,
                in_proj_a,
                merged_qkv_z: Some(merged_qkv_z),
                merged_b_a,
                ..
            } => {
                let [mixed_qkv, mixed_z]: [Tensor; 2] = merged_qkv_z
                    .forward_quantized(activation)?
                    .try_into()
                    .map_err(|_| {
                        candle_core::Error::msg("packed GDN QKV/Z returned the wrong output count")
                    })?;
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
                .map(Some)
            }
            Self::Split { .. } => Ok(None),
            Self::SplitQkvzGroupedBa {
                in_proj_ba,
                merged_qkv_z: Some(merged_qkv_z),
                ..
            } => {
                let [mixed_qkv, mixed_z]: [Tensor; 2] = merged_qkv_z
                    .forward_quantized(activation)?
                    .try_into()
                    .map_err(|_| {
                        candle_core::Error::msg("packed GDN QKV/Z returned the wrong output count")
                    })?;
                GdnProjection::from_split_grouped_ba(
                    mixed_qkv,
                    mixed_z,
                    in_proj_ba.forward(x)?,
                    dims,
                    batch_size,
                    seq_len,
                )
                .map(Some)
            }
            Self::SplitQkvzGroupedBa { .. } => Ok(None),
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

pub(crate) struct GdnCoreProjection {
    pub b: Tensor,
    pub a: Tensor,
    conv_input: GdnConvInput,
}

#[derive(Clone)]
enum GdnConvInput {
    Direct(Tensor),
    Segmented { q: Tensor, k: Tensor, v: Tensor },
}

#[derive(Clone, Copy)]
enum GdnTokenPadding {
    Zero,
    RecurrentIdentity,
}

impl GdnConvInput {
    fn materialize_flat(&self) -> Result<Tensor> {
        match self {
            Self::Direct(src) => Ok(src.clone()),
            Self::Segmented { q, k, v } => Tensor::cat(
                &[
                    &q.flatten_from(2)?,
                    &k.flatten_from(2)?,
                    &v.flatten_from(2)?,
                ],
                D::Minus1,
            ),
        }
    }

    fn materialize(&self, dims: &GdnDims, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        match self {
            Self::Direct(src) => Ok(src.clone()),
            Self::Segmented { q, k, v } => {
                let q = q.reshape((batch_size, seq_len, dims.key_dim))?;
                let k = k.reshape((batch_size, seq_len, dims.key_dim))?;
                let v = v.reshape((batch_size, seq_len, dims.value_dim))?;
                Tensor::cat(&[&q, &k, &v], D::Minus1)
            }
        }
    }
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
        self.conv_input.materialize(dims, batch_size, seq_len)
    }

    pub(crate) fn core_projection(&self) -> GdnCoreProjection {
        GdnCoreProjection {
            b: self.b.clone(),
            a: self.a.clone(),
            conv_input: self.conv_input.clone(),
        }
    }

    pub(crate) fn gather_core_token_ranges(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<GdnCoreProjection> {
        Ok(GdnCoreProjection {
            b: gather_token_ranges(&self.b, ranges)?,
            a: gather_token_ranges(&self.a, ranges)?,
            conv_input: match &self.conv_input {
                GdnConvInput::Direct(src) => {
                    GdnConvInput::Direct(gather_token_ranges(src, ranges)?)
                }
                GdnConvInput::Segmented { q, k, v } => GdnConvInput::Segmented {
                    q: gather_token_ranges(q, ranges)?,
                    k: gather_token_ranges(k, ranges)?,
                    v: gather_token_ranges(v, ranges)?,
                },
            },
        })
    }

    pub(crate) fn pad_core_packed(
        &self,
        layout: &PackedGdnLayout,
        padded_len: usize,
    ) -> Result<GdnCoreProjection> {
        let conv_input = self.conv_input.materialize_flat()?;
        Ok(GdnCoreProjection {
            b: pad_packed_tokens(
                &self.b,
                layout,
                padded_len,
                GdnTokenPadding::RecurrentIdentity,
            )?,
            a: pad_packed_tokens(
                &self.a,
                layout,
                padded_len,
                GdnTokenPadding::RecurrentIdentity,
            )?,
            conv_input: GdnConvInput::Direct(pad_packed_tokens(
                &conv_input,
                layout,
                padded_len,
                GdnTokenPadding::Zero,
            )?),
        })
    }
}

impl GdnCoreProjection {
    pub(crate) fn conv_input(
        &self,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.conv_input.materialize(dims, batch_size, seq_len)
    }
}

fn gather_token_ranges(source: &Tensor, ranges: &[Range<usize>]) -> Result<Tensor> {
    let Some(first) = ranges.first() else {
        candle_core::bail!("packed GDN projection requires at least one token range");
    };
    if first.is_empty() || ranges.iter().any(|range| range.len() != first.len()) {
        candle_core::bail!("packed GDN projection ranges must have one shared nonzero length");
    }
    if ranges.len() == 1 {
        return source.narrow(1, first.start, first.len());
    }
    let rows = ranges
        .iter()
        .map(|range| source.narrow(1, range.start, range.len()))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&rows, 0)
}

fn pad_token_ranges(
    source: &Tensor,
    ranges: &[Range<usize>],
    padded_len: usize,
    padding: GdnTokenPadding,
) -> Result<Tensor> {
    let source_dims = source.dims();
    if source_dims.len() < 2 || source_dims[0] != 1 || ranges.is_empty() {
        candle_core::bail!("packed GDN projection has invalid token dimensions");
    }
    if ranges
        .iter()
        .any(|range| range.is_empty() || range.end > source_dims[1] || range.len() > padded_len)
    {
        candle_core::bail!("packed GDN projection has an invalid padded token range");
    }

    let max_padding = ranges
        .iter()
        .map(|range| padded_len - range.len())
        .max()
        .unwrap_or(0);
    let padding = if max_padding == 0 {
        None
    } else {
        let mut padding_shape = source_dims.to_vec();
        padding_shape[0] = 1;
        padding_shape[1] = max_padding;
        Some(match padding {
            GdnTokenPadding::Zero => Tensor::zeros(padding_shape, source.dtype(), source.device())?,
            GdnTokenPadding::RecurrentIdentity => {
                Tensor::full(RECURRENT_IDENTITY_GATE, padding_shape, source.device())?
                    .to_dtype(source.dtype())?
            }
        })
    };

    let mut pieces = Vec::with_capacity(ranges.len().saturating_mul(2));
    for range in ranges {
        pieces.push(source.narrow(1, range.start, range.len())?);
        let padding_len = padded_len - range.len();
        if padding_len > 0 {
            pieces.push(
                padding
                    .as_ref()
                    .expect("nonzero padding has a backing tensor")
                    .narrow(1, 0, padding_len)?,
            );
        }
    }
    let packed = Tensor::cat(&pieces, 1)?;
    let mut padded_shape = source_dims.to_vec();
    padded_shape[0] = ranges.len();
    padded_shape[1] = padded_len;
    packed.reshape(padded_shape)
}

fn pad_packed_tokens(
    source: &Tensor,
    layout: &PackedGdnLayout,
    padded_len: usize,
    padding: GdnTokenPadding,
) -> Result<Tensor> {
    let source_dims = source.dims();
    if source_dims.len() < 2
        || source_dims[0] != 1
        || source_dims[1] != layout.token_count()
        || layout.batch_size() == 0
        || padded_len < layout.max_seq_len()
    {
        candle_core::bail!("packed GDN projection has invalid token dimensions");
    }
    if let Some(cu_seqlens) = layout.cu_seqlens(source.device())? {
        let padding_value = match padding {
            GdnTokenPadding::Zero => 0.0,
            GdnTokenPadding::RecurrentIdentity => RECURRENT_IDENTITY_GATE,
        };
        if let Some(output) =
            crate::cuda::gdn::try_gdn_packed_to_padded_cuda(crate::cuda::gdn::GdnPackedToPadded {
                source,
                cu_seqlens,
                batch_size: layout.batch_size(),
                token_count: layout.token_count(),
                padded_len,
                padding_value,
            })?
        {
            return Ok(output);
        }
    }
    let ranges = layout.token_ranges().collect::<Vec<_>>();
    pad_token_ranges(source, &ranges, padded_len, padding)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

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

    #[test]
    fn grouped_core_projection_gathers_only_requested_token_rows() -> Result<()> {
        let dev = Device::Cpu;
        let dims = dims();
        let seq_len = 6;
        let qkvz = Tensor::from_vec(
            (0..seq_len * dims.qkvz_out_dim())
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            (1, seq_len, dims.qkvz_out_dim()),
            &dev,
        )?;
        let ba = Tensor::from_vec(
            (0..seq_len * dims.ba_out_dim())
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            (1, seq_len, dims.ba_out_dim()),
            &dev,
        )?;
        let projection = GdnProjection::from_grouped(qkvz, ba, &dims, 1, seq_len)?;
        let full_conv = projection.conv_input(&dims, 1, seq_len)?;
        let gathered = projection.gather_core_token_ranges(&[0..2, 4..6])?;
        let gathered_conv = gathered.conv_input(&dims, 2, 2)?;
        let expected_conv =
            Tensor::cat(&[full_conv.narrow(1, 0, 2)?, full_conv.narrow(1, 4, 2)?], 0)?;
        assert_eq!(
            gathered_conv.flatten_all()?.to_vec1::<f32>()?,
            expected_conv.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(gathered.b.dims(), &[2, 2, dims.num_v_heads]);
        assert_eq!(gathered.a.dims(), &[2, 2, dims.num_v_heads]);

        let layout = PackedGdnLayout::new(vec![2, 4], HashMap::new())?;
        let padded = projection.pad_core_packed(&layout, 4)?;
        let padded_conv = padded.conv_input(&dims, 2, 4)?.to_vec3::<f32>()?;
        let full_conv = full_conv.to_vec3::<f32>()?;
        assert_eq!(padded_conv[0][..2], full_conv[0][..2]);
        assert_eq!(padded_conv[1], full_conv[0][2..]);
        assert!(padded_conv[0][2..]
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn padded_core_projection_uses_identity_gate_tails() -> Result<()> {
        let dev = Device::Cpu;
        let dims = dims();
        let seq_len = 6;
        let qkv = Tensor::from_vec(
            (0..seq_len * dims.conv_dim)
                .map(|value| value as f32 + 1.0)
                .collect::<Vec<_>>(),
            (1, seq_len, dims.conv_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let z = Tensor::zeros((1, seq_len, dims.value_dim), DType::BF16, &dev)?;
        let b = Tensor::from_vec(
            (0..seq_len * dims.num_v_heads)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            (1, seq_len, dims.num_v_heads),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let a = b.clone();
        let projection = GdnProjection::from_split(qkv, z, b, a, &dims, 1, seq_len)?;
        let layout = PackedGdnLayout::new(vec![1, 2, 3], HashMap::new())?;
        let padded = projection.pad_core_packed(&layout, 3)?;

        let padded_b = padded.b.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let padded_a = padded.a.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        for gates in [&padded_b, &padded_a] {
            assert!(gates[0][1].iter().all(|value| *value == f32::NEG_INFINITY));
            assert!(gates[0][2].iter().all(|value| *value == f32::NEG_INFINITY));
            assert!(gates[1][2].iter().all(|value| *value == f32::NEG_INFINITY));
            assert!(gates[2].iter().flatten().all(|value| value.is_finite()));
        }

        let conv_input = padded
            .conv_input(&dims, 3, 3)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        assert!(conv_input[0][1].iter().all(|value| *value == 0.0));
        assert!(conv_input[0][2].iter().all(|value| *value == 0.0));
        assert!(conv_input[1][2].iter().all(|value| *value == 0.0));
        assert!(conv_input[2]
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value != 0.0));
        Ok(())
    }
}
