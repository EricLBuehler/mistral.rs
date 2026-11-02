use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Shape, Tensor, D};
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, Shard, ShardedVarBuilder, UqffReader, UqffTensor,
};

#[cfg(feature = "metal")]
mod group;
pub mod ops;

#[cfg(feature = "cuda")]
pub(crate) mod ffi;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfqBits {
    Two = 2,
    Three = 3,
    Four = 4,
    Six = 6,
    Eight = 8,
}

impl TryFrom<usize> for AfqBits {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            6 => Ok(Self::Six),
            8 => Ok(Self::Eight),
            x => candle_core::bail!("Invalid AFQ bits {x}."),
        }
    }
}

impl TryFrom<u8> for AfqBits {
    type Error = candle_core::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AfqGroupSize {
    Low = 32,
    #[default]
    Med = 64,
    High = 128,
}

impl TryFrom<usize> for AfqGroupSize {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            32 => Ok(Self::Low),
            64 => Ok(Self::Med),
            128 => Ok(Self::High),
            x => candle_core::bail!("Invalid AFQ group size {x}."),
        }
    }
}

impl TryFrom<u8> for AfqGroupSize {
    type Error = candle_core::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

#[derive(Debug)]
pub struct AfqLayer {
    w_q: Tensor,
    scales: Tensor,
    biases: Tensor,
    bias: Option<Tensor>,
    bits: AfqBits,
    group_size: AfqGroupSize,
    stats: crate::ImatrixLayerStats,
}

impl AfqLayer {
    fn prepare(self) -> Result<Self> {
        #[cfg(feature = "metal")]
        self.prepare_metal_kernels(self.scales.dtype())?;
        Ok(self)
    }

    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &[
            "weight",
            "weight.format",
            "weight.bits",
            "weight.group_size",
            "weight.scales",
            "weight.biases",
        ];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.bits", Dtype::U8)
            && layer.scalar("weight.group_size", Dtype::U8)
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Afq,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        tensors: &[UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        let bits = crate::uqff::u8_scalar_with_suffix(tensors, prefix, "weight.bits")?;
        Ok(afq_bits_label(bits))
    }
}

fn afq_bits_label(bits: u8) -> String {
    match bits {
        2 => "afq2",
        3 => "afq3",
        4 => "afq4",
        6 => "afq6",
        8 => "afq8",
        _ => "afq",
    }
    .to_string()
}

/// Cheap handle to an AfqLayer's storage tensors, used by fused QKV/gate-up paths.
#[derive(Clone)]
pub struct AfqInner {
    pub w_q: Tensor,
    pub scales: Tensor,
    pub biases: Tensor,
    pub bias: Option<Tensor>,
    pub bits: AfqBits,
    pub group_size: AfqGroupSize,
}

impl QuantMethod for AfqLayer {
    #[cfg(feature = "metal")]
    fn prepare_metal_kernels(&self, _activation_dtype: DType) -> Result<()> {
        crate::metal_kernels::prepare_afq(&self.w_q, &self.scales, self.bits, self.group_size)
    }

    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
            QuantMethodConfig::Afq {
                weight,
                bias,
                bits,
                group_size,
            } => {
                let (w_q, scales, biases) = ops::afq_quantize_op(&weight, group_size, bits)?;

                Self {
                    w_q,
                    scales,
                    biases,
                    bias,
                    bits,
                    group_size,
                    stats: crate::ImatrixLayerStats::empty(),
                }
                .prepare()
            }
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::afq_dequantize_op(
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )
    }

    fn begin_track_stats(&self) -> Result<()> {
        let in_dim = self.scales.dim(candle_core::D::Minus1)? * (self.group_size as usize);
        // Stacked [E, out, in] expert weights collect per expert via the routed path.
        if self.w_q.dims().len() == 3 {
            self.stats
                .enable_routed(self.w_q.dim(0)?, in_dim, self.w_q.device())
        } else {
            self.stats.enable(in_dim, self.w_q.device())
        }
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.stats.process_routed(x, ids)
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.stats.snapshot()
    }
    fn end_track_stats(&self) -> Result<Tensor> {
        if self.stats.is_enabled() {
            let imatrix = self.stats.compute_imatrix();
            self.stats.clear()?;
            imatrix
        } else {
            candle_core::bail!("`{}` is not tracking stats.", self.name())
        }
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        self.stats.process(x)?;
        let output = ops::afq_mm_op(
            x,
            &self.w_q,
            &self.scales,
            &self.biases,
            None,
            None,
            self.group_size,
            self.bits,
            true,
        )?;
        match &self.bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }

    fn routed_weights(&self) -> Option<crate::RoutedWeights> {
        let (experts, n, _) = self.w_q.dims3().ok()?;
        let group = self.group_size as usize;
        let k = self.scales.dim(2).ok()? * group;
        Some(crate::RoutedWeights::new(
            crate::RoutedCodec::Afq {
                bits: self.bits as usize,
                group,
            },
            self.w_q.clone(),
            self.scales.clone(),
            Some(self.biases.clone()),
            self.bias.clone(),
            [experts, n, k],
        ))
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(routed) = self.routed_weights() {
            if let Some(output) = crate::try_routed_gather(x, indices, &routed)? {
                return Ok(output);
            }
        }
        let output = if x.device().is_metal() {
            let canonical = x.rank() == 3 && indices.rank() == 2;
            let input = if canonical {
                x.unsqueeze(D::Minus2)?
            } else {
                x.clone()
            };
            let output = ops::afq_mm_op(
                &input,
                &self.w_q,
                &self.scales,
                &self.biases,
                None,
                Some(indices),
                self.group_size,
                self.bits,
                true,
            )?;
            if canonical {
                output.squeeze(D::Minus2)?
            } else {
                output
            }
        } else {
            self.gather_by_expert(x, indices)?
        };
        let Some(bias) = &self.bias else {
            return Ok(output);
        };
        if bias.rank() == 2 {
            let mut shape = indices.dims().to_vec();
            shape.push(bias.dim(1)?);
            let bias = bias
                .index_select(&indices.flatten_all()?, 0)?
                .reshape(Shape::from(shape))?;
            output.broadcast_add(&bias)
        } else {
            output.broadcast_add(bias)
        }
    }

    fn embedding_table(&self) -> Option<crate::EmbeddingTable<'_>> {
        use crate::embedding::{RowCodec, ScaleGrid, ScaleType};
        let (rows, _) = self.w_q.dims2().ok()?;
        let columns = self.scales.dim(1).ok()? * self.group_size as usize;
        let grid = ScaleGrid::per_row(
            ScaleType::from_dtype(self.scales.dtype()).ok()?,
            self.group_size as usize,
        );
        let codec = RowCodec::Afq {
            bits: self.bits as usize,
            grid,
        };
        Some(
            crate::EmbeddingTable::new(codec, rows, columns, &self.w_q)
                .with_scales(&self.scales)
                .with_aux(&self.biases),
        )
    }

    #[cfg(feature = "metal")]
    fn embedding_forward(&self, ids: &Tensor, out: crate::EmbeddingOutput) -> Result<Tensor> {
        if !self.w_q.device().is_metal() {
            return crate::embedding::forward(self, ids, out);
        }
        let rows = ops::afq_embedding_op(
            ids,
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )?;
        crate::embedding::finish_rows(rows, out)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    #[cfg(feature = "metal")]
    fn group_member(
        &self,
        xs: &Tensor,
        kind: crate::GroupKind<'_>,
    ) -> Result<Option<crate::GroupMember>> {
        self.afq_group_member(xs, kind)
    }

    fn afq_inner(&self) -> Option<crate::AfqInner> {
        Some(crate::AfqInner {
            w_q: self.w_q.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            bias: self.bias.clone(),
            bits: self.bits,
            group_size: self.group_size,
        })
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize_w()?;
        Ok(Arc::new(Self::new(QuantMethodConfig::Afq {
            weight: (dequant + delta)?,
            bias: self.bias.clone(),
            bits: self.bits,
            group_size: self.group_size,
        })?))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let mut shape = self.scales.dims().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = last.saturating_mul(self.group_size as usize);
        }
        if shape.len() == 3 && request.ty.is_some_and(|ty| !ty.supports_stacked_gather()) {
            candle_core::bail!(
                "Cannot requantize stacked AFQ expert weights to {}: that target does not support stacked expert gather.",
                request.ty.expect("rank-3 rejection requires an ISQ target")
            );
        }
        Ok(crate::plan_weight_isq(
            self.scales.dtype(),
            self.scales.device().clone(),
            shape,
            request,
            true,
        ))
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() || (dtype == self.uqff_type() && imatrix_weight.is_none()) {
            if self.w_q.device().same_device(&device) {
                return Ok(self);
            }
            return Ok(Arc::new(
                Self {
                    w_q: self.w_q.to_device(&device)?,
                    scales: self.scales.to_device(&device)?,
                    biases: self.biases.to_device(&device)?,
                    bias: self
                        .bias
                        .as_ref()
                        .map(|bias| bias.to_device(&device))
                        .transpose()?,
                    bits: self.bits,
                    group_size: self.group_size,
                    stats: self.stats.clone(),
                }
                .prepare()?,
            ));
        }

        match dtype {
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                let w = self.dequantize_w()?.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            _ => Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(self.dequantize_w()?, self.bias.clone()),
            ))?)
            .apply_isq(dtype, device, n_quantized, imatrix_weight, guard),
        }
    }
}

impl AfqLayer {
    // Route rows are grouped per expert on the host so each expert runs the dense qmm path once.
    fn gather_by_expert(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let in_dim = x.dim(candle_core::D::Minus1)?;
        let out_dim = self.w_q.dim(candle_core::D::Minus2)?;
        let routes = indices.elem_count();
        let top_k = indices.dim(candle_core::D::Minus1)?;
        let x_rank = x.rank();
        let x_rows = x.elem_count() / in_dim;
        let per_route = match x_rows {
            r if r == routes => true,
            r if r * top_k == routes => false,
            _ => candle_core::bail!(
                "AFQ gather_forward: input shape {:?} does not match indices shape {:?}",
                x.dims(),
                indices.dims()
            ),
        };
        let x = x.reshape((x_rows, in_dim))?;
        let ids = indices
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;

        let mut routes_by_expert = vec![Vec::new(); self.w_q.dim(0)?];
        for (route, &expert) in ids.iter().enumerate() {
            routes_by_expert[expert as usize].push(route as u32);
        }
        let mut outputs = Vec::new();
        let mut order = vec![0u32; routes];
        let mut next = 0u32;
        for (expert, expert_routes) in routes_by_expert.iter().enumerate() {
            if expert_routes.is_empty() {
                continue;
            }
            let rows = expert_routes
                .iter()
                .map(|&r| if per_route { r } else { r / top_k as u32 })
                .collect::<Vec<_>>();
            let rows = Tensor::from_vec(rows, expert_routes.len(), x.device())?;
            outputs.push(ops::afq_mm_op(
                &x.index_select(&rows, 0)?,
                &self.w_q.get(expert)?,
                &self.scales.get(expert)?,
                &self.biases.get(expert)?,
                None,
                None,
                self.group_size,
                self.bits,
                true,
            )?);
            for &route in expert_routes {
                order[route as usize] = next;
                next += 1;
            }
        }
        let order = Tensor::from_vec(order, routes, x.device())?;
        let mut out_shape = indices.dims().to_vec();
        out_shape.resize(x_rank - 1, 1);
        out_shape.push(out_dim);
        Tensor::cat(&outputs, 0)?
            .index_select(&order, 0)?
            .reshape(out_shape)
    }

    pub fn from_parts(
        w_q: Tensor,
        scales: Tensor,
        biases: Tensor,
        bias: Option<Tensor>,
        bits: AfqBits,
        group_size: AfqGroupSize,
    ) -> Self {
        Self {
            w_q,
            scales,
            biases,
            bias,
            bits,
            group_size,
            stats: crate::ImatrixLayerStats::empty(),
        }
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        let bits = AfqBits::try_from(reader.load_u8_scalar(&format!("{key}.weight.bits"))?)?;
        let group_size =
            AfqGroupSize::try_from(reader.load_u8_scalar(&format!("{key}.weight.group_size"))?)?;
        let group = group_size as usize;

        let mut dims = reader.tensor_dims(&format!("{key}.weight.scales"))?;
        *dims.last_mut().expect("AFQ scales are non-empty") *= group;
        let range = crate::uqff::shard_range(shard, &dims)?;

        let (w_q_range, group_range) = match range {
            None => (None, None),
            Some((dim, start, len)) if dim == dims.len() - 1 => {
                if !start.is_multiple_of(group) || !len.is_multiple_of(group) {
                    candle_core::bail!(
                        "Sharding the AFQ packed dim requires group alignment: start {start}, len {len}, group {group}."
                    );
                }
                (
                    Some((
                        dim,
                        start * bits as usize / u32::BITS as usize,
                        len * bits as usize / u32::BITS as usize,
                    )),
                    Some((dim, start / group, len / group)),
                )
            }
            some => (some, some),
        };

        let w_q = reader.load_tensor_sharded(&format!("{key}.weight"), device, w_q_range)?;
        let scales =
            reader.load_tensor_sharded(&format!("{key}.weight.scales"), device, group_range)?;
        let biases =
            reader.load_tensor_sharded(&format!("{key}.weight.biases"), device, group_range)?;
        let bias = reader.load_bias(key, device, range, dims.len())?;
        Self::from_parts(w_q, scales, biases, bias, bits, group_size).prepare()
    }

    pub fn afq_linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::Afq { bits, group_size } = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let w_q = vb.get_with_hints_dtype(
            (out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales =
            vb.get_with_hints((out_dim, in_dim / group_size), "scales", Default::default())?;
        let biases =
            vb.get_with_hints((out_dim, in_dim / group_size), "biases", Default::default())?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(
            Self {
                w_q,
                scales,
                bias,
                biases,
                bits: AfqBits::try_from(*bits)?,
                group_size: AfqGroupSize::try_from(*group_size)?,
                stats: crate::ImatrixLayerStats::empty(),
            }
            .prepare()?,
        ))
    }

    pub fn afq_packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::Afq { bits, group_size } = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let w_q = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "scales",
            Default::default(),
        )?;
        let biases = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "biases",
            Default::default(),
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(
            Self {
                w_q,
                scales,
                bias,
                biases,
                bits: AfqBits::try_from(*bits)?,
                group_size: AfqGroupSize::try_from(*group_size)?,
                stats: crate::ImatrixLayerStats::empty(),
            }
            .prepare()?,
        ))
    }
}

impl QuantizedSerde for AfqLayer {
    fn name(&self) -> &'static str {
        "afq-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn uqff_type(&self) -> Option<IsqType> {
        Some(match self.bits {
            AfqBits::Two => IsqType::AFQ2,
            AfqBits::Three => IsqType::AFQ3,
            AfqBits::Four => IsqType::AFQ4,
            AfqBits::Six => IsqType::AFQ6,
            AfqBits::Eight => IsqType::AFQ8,
        })
    }
    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        let actual_ty = self
            .uqff_type()
            .expect("AFQ layers always have an ISQ type");
        if ty != actual_ty {
            candle_core::bail!("Cannot serialize AFQ layer as {ty}; actual type is {actual_ty}.");
        }

        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Afq as u8,
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), self.bits as u8),
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.group_size"),
                self.group_size as u8,
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight"), &self.w_q)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.scales"), &self.scales)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.biases"), &self.biases)?,
        ];
        if let Some(bias) = &self.bias {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(reader: &UqffReader, prefix: &str) -> Result<IsqType> {
        match AfqBits::try_from(reader.load_u8_scalar(&format!("{prefix}.weight.bits"))? as usize)?
        {
            AfqBits::Two => Ok(IsqType::AFQ2),
            AfqBits::Three => Ok(IsqType::AFQ3),
            AfqBits::Four => Ok(IsqType::AFQ4),
            AfqBits::Six => Ok(IsqType::AFQ6),
            AfqBits::Eight => Ok(IsqType::AFQ8),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_isq_preserves_packed_afq_for_capture_and_exact_target() -> Result<()> {
        let source = Arc::new(AfqLayer::from_parts(
            Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?,
            Tensor::zeros((1, 1), DType::BF16, &Device::Cpu)?,
            Tensor::zeros((1, 1), DType::BF16, &Device::Cpu)?,
            Some(Tensor::zeros(1, DType::BF16, &Device::Cpu)?),
            AfqBits::Four,
            AfqGroupSize::Med,
        )) as Arc<dyn QuantMethod>;
        let n_quantized = AtomicUsize::new(0);

        let captured = source.clone().apply_isq(
            None,
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert!(Arc::ptr_eq(&source, &captured));
        assert_eq!(captured.uqff_type(), Some(IsqType::AFQ4));
        assert!(captured.has_bias());

        let exact = source.clone().apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert!(Arc::ptr_eq(&source, &exact));
        Ok(())
    }

    #[test]
    fn three_and_six_bit_uqff_loads_preserve_logical_shape_and_shards() -> Result<()> {
        const OUTPUT_SIZE: usize = 2;
        const INPUT_SIZE: usize = 128;
        const GROUP_SIZE: usize = AfqGroupSize::Low as usize;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("afq-3-6.uqff");
        let mut tensors = crate::uqff_version_tensors();
        for (prefix, bits, ty) in [
            ("afq3", AfqBits::Three, IsqType::AFQ3),
            ("afq6", AfqBits::Six, IsqType::AFQ6),
        ] {
            let packed_columns = INPUT_SIZE * bits as usize / u32::BITS as usize;
            let layer = AfqLayer::from_parts(
                Tensor::zeros((OUTPUT_SIZE, packed_columns), DType::U32, &Device::Cpu)?,
                Tensor::zeros(
                    (OUTPUT_SIZE, INPUT_SIZE / GROUP_SIZE),
                    DType::BF16,
                    &Device::Cpu,
                )?,
                Tensor::zeros(
                    (OUTPUT_SIZE, INPUT_SIZE / GROUP_SIZE),
                    DType::BF16,
                    &Device::Cpu,
                )?,
                None,
                bits,
                AfqGroupSize::Low,
            );
            tensors.extend(layer.serialize_uqff(prefix, ty)?);
        }
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )?;

        let reader = UqffReader::open(&[path])?;
        for (prefix, bits) in [("afq3", 3), ("afq6", 6)] {
            let full = AfqLayer::from_uqff(&reader, prefix, &Device::Cpu, Shard::default())?;
            assert_eq!(full.w_q.dims(), [OUTPUT_SIZE, INPUT_SIZE * bits / 32]);
            assert_eq!(full.scales.dims(), [OUTPUT_SIZE, INPUT_SIZE / GROUP_SIZE]);
            assert_eq!(full.dequantize_w()?.dims(), [OUTPUT_SIZE, INPUT_SIZE]);

            let shard = AfqLayer::from_uqff(
                &reader,
                prefix,
                &Device::Cpu,
                Shard::Simple {
                    dim: 1,
                    rank: 1,
                    world_size: 2,
                },
            )?;
            assert_eq!(shard.w_q.dims(), [OUTPUT_SIZE, INPUT_SIZE * bits / 64]);
            assert_eq!(
                shard.scales.dims(),
                [OUTPUT_SIZE, INPUT_SIZE / GROUP_SIZE / 2]
            );
            assert_eq!(shard.dequantize_w()?.dims(), [OUTPUT_SIZE, INPUT_SIZE / 2]);
        }
        Ok(())
    }

    const GATHER_EXPERTS: usize = 4;
    const GATHER_OUT: usize = 64;
    const GATHER_IN: usize = 128;
    const GATHER_TOKENS: usize = 3;
    const GATHER_TOPK: usize = 2;
    const GATHER_IDS: [u32; GATHER_TOKENS * GATHER_TOPK] = [3, 1, 0, 2, 2, 3];

    fn stacked_gather_layer(bits: AfqBits) -> Result<AfqLayer> {
        let values = (0..GATHER_EXPERTS * GATHER_OUT * GATHER_IN)
            .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i / (GATHER_OUT * GATHER_IN)) as f32))
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(
            values,
            (GATHER_EXPERTS, GATHER_OUT, GATHER_IN),
            &Device::Cpu,
        )?;
        let (w_q, scales, biases) = ops::afq_quantize_op(&weight, AfqGroupSize::Med, bits)?;
        Ok(AfqLayer::from_parts(
            w_q,
            scales,
            biases,
            None,
            bits,
            AfqGroupSize::Med,
        ))
    }

    fn reference_gather(layer: &AfqLayer, x: &Tensor, per_route: bool) -> Result<Tensor> {
        let w = layer
            .dequantize_w()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?;
        let x = x.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let x = x.reshape((x.elem_count() / GATHER_IN, GATHER_IN))?;
        let mut rows = Vec::new();
        for (route, &expert) in GATHER_IDS.iter().enumerate() {
            let x_row = if per_route {
                route
            } else {
                route / GATHER_TOPK
            };
            rows.push(
                x.narrow(0, x_row, 1)?
                    .matmul(&w.get(expert as usize)?.t()?)?,
            );
        }
        Tensor::cat(&rows, 0)?.reshape((GATHER_TOKENS, GATHER_TOPK, GATHER_OUT))
    }

    fn gather_input(rows: usize) -> Result<Tensor> {
        let values = (0..rows * GATHER_IN)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, GATHER_IN), &Device::Cpu)
    }

    fn assert_close(got: &Tensor, expected: &Tensor, tol: f32) -> Result<()> {
        assert_eq!(got.dims(), expected.dims());
        let diff = (got.to_device(&Device::Cpu)?.to_dtype(DType::F32)? - expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < tol, "max abs diff {diff}");
        Ok(())
    }

    fn check_stacked_gather(device: &Device, dtype: DType, tol: f32) -> Result<()> {
        for bits in [AfqBits::Four, AfqBits::Eight] {
            let cpu = stacked_gather_layer(bits)?;
            let layer = AfqLayer::from_parts(
                cpu.w_q.to_device(device)?,
                cpu.scales.to_dtype(dtype)?.to_device(device)?,
                cpu.biases.to_dtype(dtype)?.to_device(device)?,
                None,
                bits,
                AfqGroupSize::Med,
            );
            let ids = Tensor::from_vec(GATHER_IDS.to_vec(), (GATHER_TOKENS, GATHER_TOPK), device)?;

            let shared = gather_input(GATHER_TOKENS)?;
            let got = layer.gather_forward(
                &shared
                    .reshape((GATHER_TOKENS, 1, GATHER_IN))?
                    .to_dtype(dtype)?
                    .to_device(device)?,
                &ids,
            )?;
            assert_close(&got, &reference_gather(&layer, &shared, false)?, tol)?;

            let routed = gather_input(GATHER_TOKENS * GATHER_TOPK)?;
            let got = layer.gather_forward(
                &routed
                    .reshape((GATHER_TOKENS, GATHER_TOPK, GATHER_IN))?
                    .to_dtype(dtype)?
                    .to_device(device)?,
                &ids,
            )?;
            assert_close(&got, &reference_gather(&layer, &routed, true)?, tol)?;
        }
        Ok(())
    }

    #[test]
    fn stacked_gather_honors_expert_ids_cpu() -> Result<()> {
        check_stacked_gather(&Device::Cpu, DType::F32, 1e-3)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn stacked_gather_honors_expert_ids_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        check_stacked_gather(&device, DType::F16, 0.25)?;
        check_stacked_gather(&device, DType::BF16, 0.5)
    }
}
