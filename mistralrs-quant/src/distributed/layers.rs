use std::sync::Arc;

use candle_core::{Context, Result, Tensor};
use candle_nn::Linear;

use crate::{
    blockwise_fp8::blockwise_fp8_linear_b, distributed, gptq::gptq_linear,
    lora::merge_lora_weights, AfqLayer, BnbLinear, DistributedKind, DummyLayer, FP8Linear,
    GgufMatMul, HqqLayer, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, QuantizedSerdeType, Shard, ShardedVarBuilder, UnquantLinear,
};

use super::{Comm, SumAllReduce};

fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}

/// This layer has a weight that is parallelized along the input dimension,
/// returning the "full" output dimension.
#[derive(Debug)]
pub struct RowParallelLayer {
    weight: Arc<dyn QuantMethod>,
    bias: Option<Tensor>,
    all_reduce: distributed::SumAllReduce,
}

impl RowParallelLayer {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(1, rank, world_size);

        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::Gptq { .. }
                    | QuantizedConfig::Bitsandbytes { .. }
                    | QuantizedConfig::Afq { .. }
            ) && comm.world_size() != 1
            {
                candle_core::bail!(
                    "GPTQ and BNB and AFQ quantization types to not support tensor parallelism, but got a world size of {}",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::Gptq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { .. } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    blockwise_fp8_linear_b(in_dim, out_dim, quant_conf, false, shard, vb.clone())?
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            // Handle the case where the layer is dummy (no tensors)
            if !vb.contains_tensor("weight") {
                let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            weight,
            bias,
            all_reduce: distributed::SumAllReduce::new(comm),
        }))
    }
}

impl QuantMethod for RowParallelLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("RowParallelLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let mut xs = self.weight.forward(a)?;
        xs = self.all_reduce.sum_all_reduce(&xs.contiguous()?)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.weight.add_delta_w(delta)?;
        Ok(Arc::new(Self {
            weight,
            bias: self.bias.clone(),
            all_reduce: self.all_reduce.clone(),
        }))
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.weight.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.weight.dtype_and_device()
    }

    fn begin_track_stats(&mut self) -> Result<()> {
        Arc::get_mut(&mut self.weight)
            .context("Failed to get &mut to weight")?
            .begin_track_stats()
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.weight.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.weight.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.weight.unquant_weight_bias()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight =
            self.weight
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        let bias = match &self.bias {
            Some(b) => {
                let (dtype, device) = weight.dtype_and_device();
                Some(b.to_device(&device)?.to_dtype(dtype)?)
            }
            None => None,
        };
        Ok(Arc::new(Self {
            weight,
            bias,
            all_reduce: self.all_reduce.clone(),
        }))
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::RowParallel)
    }
}

impl QuantizedSerde for RowParallelLayer {
    fn isq_serde_supported(&self) -> bool {
        self.weight.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.weight.name()
    }
    fn serialize(&self) -> Result<std::borrow::Cow<[u8]>> {
        self.weight.serialize_with_bias(self.bias.clone())
    }
    fn deserialize(
        data: std::borrow::Cow<[u8]>,
        device: &candle_core::Device,
        comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
        let isq_type = data[crate::UQFF_QUANT_TYPE_OFFSET];
        let (weight, bias) = match QuantizedSerdeType::try_from(isq_type as usize)? {
            QuantizedSerdeType::Gguf => GgufMatMul::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Unquant => {
                UnquantLinear::deserialize_ext_bias(data, device, guard)?
            }
            QuantizedSerdeType::Hqq => HqqLayer::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Fp8 => FP8Linear::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Afq => AfqLayer::deserialize_ext_bias(data, device, guard)?,
        };
        Ok(Arc::new(Self {
            weight,
            bias,
            all_reduce: SumAllReduce::new(comm),
        }))
    }
}

#[derive(Debug)]
/// This layer has a weight that is parallelized along the output dimension,
/// taking the "full" input dimension.
pub struct ColumnParallelLayer {
    weight: Arc<dyn QuantMethod>,
    bias: Option<Tensor>,
}

impl ColumnParallelLayer {
    #[allow(clippy::new_ret_no_self)]
    pub fn new_with_shard(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        shard: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::Gptq { .. }
                    | QuantizedConfig::Bitsandbytes { .. }
                    | QuantizedConfig::Afq { .. }
            ) && comm.world_size() != 1
            {
                candle_core::bail!(
                    "GPTQ and BNB and AFQ quantization types to not support tensor parallelism, but got a world size of {}",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::Gptq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { .. } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    blockwise_fp8_linear_b(in_dim, out_dim, quant_conf, false, shard, vb.clone())?
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            // Handle the case where the layer is dummy (no tensors)
            if !vb.contains_tensor("weight") {
                let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, None),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        Ok(Arc::new(Self { weight, bias }))
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(0, rank, world_size);

        Self::new_with_shard(in_dim, out_dim, config, bias, comm, shard, vb)
    }
}

impl QuantMethod for ColumnParallelLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("ColumnParallelLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let mut xs = self.weight.forward(a)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.weight.add_delta_w(delta)?;
        Ok(Arc::new(Self {
            weight,
            bias: self.bias.clone(),
        }))
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.weight.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.weight.dtype_and_device()
    }

    fn begin_track_stats(&mut self) -> Result<()> {
        Arc::get_mut(&mut self.weight)
            .context("Failed to get &mut to weight")?
            .begin_track_stats()
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.weight.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.weight.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.weight.unquant_weight_bias()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight =
            self.weight
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        let bias = match &self.bias {
            Some(b) => {
                let (dtype, device) = weight.dtype_and_device();
                Some(b.to_device(&device)?.to_dtype(dtype)?)
            }
            None => None,
        };
        Ok(Arc::new(Self { weight, bias }))
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::ColumnParallel)
    }
}

impl QuantizedSerde for ColumnParallelLayer {
    fn isq_serde_supported(&self) -> bool {
        self.weight.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.weight.name()
    }
    fn serialize(&self) -> Result<std::borrow::Cow<[u8]>> {
        self.weight.serialize_with_bias(self.bias.clone())
    }
    fn deserialize(
        data: std::borrow::Cow<[u8]>,
        device: &candle_core::Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
        let isq_type = data[crate::UQFF_QUANT_TYPE_OFFSET];
        let (weight, bias) = match QuantizedSerdeType::try_from(isq_type as usize)? {
            QuantizedSerdeType::Gguf => GgufMatMul::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Unquant => {
                UnquantLinear::deserialize_ext_bias(data, device, guard)?
            }
            QuantizedSerdeType::Hqq => HqqLayer::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Fp8 => FP8Linear::deserialize_ext_bias(data, device, guard)?,
            QuantizedSerdeType::Afq => AfqLayer::deserialize_ext_bias(data, device, guard)?,
        };
        Ok(Arc::new(Self { weight, bias }))
    }
}

#[derive(Debug)]
/// This layer has no parallelization
pub struct ReplicatedLayer(Arc<dyn QuantMethod>);

impl ReplicatedLayer {
    pub fn from_linear(lin: Linear) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(lin),
        )?))
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let layer = if let Some(quant_conf) = &config {
            match quant_conf {
                QuantizedConfig::Gptq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
                QuantizedConfig::Fp8 { .. } => blockwise_fp8_linear_b(
                    in_dim,
                    out_dim,
                    quant_conf,
                    bias,
                    Default::default(),
                    vb,
                )?,
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb)?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            // Handle the case where the layer is dummy (no tensors)
            if !vb.contains_tensor("weight") {
                let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            } else {
                let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
                let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

                let bias = if bias {
                    Some(vb.get_with_hints((out_dim,), "bias", Default::default())?)
                } else {
                    None
                };
                let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(weight, bias),
                ))?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            }
        };

        Ok(Arc::new(Self(layer)))
    }
}

impl QuantMethod for ReplicatedLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("ReplicatedLayer should not be constructed with `QuantMethod::new`")
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward(a)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        self.0.add_delta_w(delta)
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.0.dequantize_w()
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.0.dtype_and_device()
    }

    fn begin_track_stats(&mut self) -> Result<()> {
        Arc::get_mut(&mut self.0)
            .context("Failed to get &mut to weight")?
            .begin_track_stats()
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.0.end_track_stats()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.0.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.0.unquant_weight_bias()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        self.0
            .clone()
            .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        Some(DistributedKind::Replicated)
    }
}

impl QuantizedSerde for ReplicatedLayer {
    fn isq_serde_supported(&self) -> bool {
        self.0.isq_serde_supported()
    }
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn serialize(&self) -> Result<std::borrow::Cow<[u8]>> {
        self.0.serialize()
    }
    fn deserialize(
        data: std::borrow::Cow<[u8]>,
        device: &candle_core::Device,
        comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
        let isq_type = data[crate::UQFF_QUANT_TYPE_OFFSET];
        let deserialized = match QuantizedSerdeType::try_from(isq_type as usize)? {
            QuantizedSerdeType::Gguf => GgufMatMul::deserialize(data, device, comm, guard)?,
            QuantizedSerdeType::Unquant => UnquantLinear::deserialize(data, device, comm, guard)?,
            QuantizedSerdeType::Hqq => HqqLayer::deserialize(data, device, comm, guard)?,
            QuantizedSerdeType::Fp8 => FP8Linear::deserialize(data, device, comm, guard)?,
            QuantizedSerdeType::Afq => AfqLayer::deserialize(data, device, comm, guard)?,
        };
        Ok(Arc::new(Self(deserialized)))
    }
}

#[derive(Debug)]
pub struct PackedExperts {
    pub gate_proj: Vec<Arc<dyn QuantMethod>>,
    pub up_proj: Vec<Arc<dyn QuantMethod>>,
    pub down_proj: Vec<Arc<dyn QuantMethod>>,
}

impl PackedExperts {
    /// Note: we only support AFQ and unquantized here because they are the only ones that support indexed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_local_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        if bias {
            candle_core::bail!("PackedExperts does not support bias.");
        }

        let (gate_proj, up_proj, down_proj) = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if comm.world_size() != 1 {
                candle_core::bail!(
                    "PackedExperts with quantization config does not support distributed (world size {}). Use ISQ.",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::Afq { .. } => {
                    if !vb.contains_tensor("gate_up_proj")
                        || !vb.contains_tensor("gate_up_proj.weight")
                    {
                        candle_core::bail!("PackedExperts with AFQ quantization config does not support `gate_up_proj` format.");
                    }

                    (
                        vec![AfqLayer::afq_packed_linear_b(
                            num_local_experts,
                            hidden_size,
                            intermediate_size,
                            quant_conf,
                            bias,
                            vb.pp("gate_proj"),
                        )?],
                        vec![AfqLayer::afq_packed_linear_b(
                            num_local_experts,
                            hidden_size,
                            intermediate_size,
                            quant_conf,
                            bias,
                            vb.pp("up_proj"),
                        )?],
                        vec![AfqLayer::afq_packed_linear_b(
                            num_local_experts,
                            intermediate_size,
                            hidden_size,
                            quant_conf,
                            bias,
                            vb.pp("down_proj"),
                        )?],
                    )
                }
                _ => candle_core::bail!(
                    "PackedExperts with quantization config only allows AFQ quantization"
                ),
            }
        } else if !vb.contains_tensor("down_proj") {
            // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
            let mut gs = Vec::new();
            let mut us = Vec::new();
            let mut ds = Vec::new();
            for _ in 0..num_local_experts {
                let gate_proj = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                let up_proj = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                let down_proj = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                gs.push(Arc::new(gate_proj) as Arc<dyn QuantMethod>);
                us.push(Arc::new(up_proj) as Arc<dyn QuantMethod>);
                ds.push(Arc::new(down_proj) as Arc<dyn QuantMethod>);
            }
            (gs, us, ds)
        } else {
            // Parallelized like:
            // Each gpu holds all experts.
            // Gate/Up proj is parallelized on dim 2 (column)
            // Down proj is parallelized on dim 1 (row)
            // All reduce at the end.

            // Handle the case where the layer is dummy (no tensors)
            let gate_up_block_size = intermediate_size / comm.world_size();
            let gate_up_start = gate_up_block_size * comm.rank();

            // Gate is right before Up in the gate_up
            let shard_gate = Shard::Offset {
                dim: 2,
                offset: gate_up_start,
                len: gate_up_block_size,
            };
            let shard_up = Shard::Offset {
                dim: 2,
                offset: intermediate_size + gate_up_start,
                len: gate_up_block_size,
            };
            let shard_down = Shard::Simple {
                dim: 1,
                rank: comm.rank(),
                world_size: comm.world_size(),
            };

            let gate_proj = vb
                .get_with_hints(
                    (num_local_experts, hidden_size, intermediate_size * 2),
                    "gate_up_proj",
                    shard_gate,
                )?
                .t()?
                .contiguous()?;
            let up_proj = vb
                .get_with_hints(
                    (num_local_experts, hidden_size, intermediate_size * 2),
                    "gate_up_proj",
                    shard_up,
                )?
                .t()?
                .contiguous()?;
            let down_proj = vb
                .get_with_hints(
                    (num_local_experts, intermediate_size, hidden_size),
                    "down_proj",
                    shard_down,
                )?
                .t()?
                .contiguous()?;

            let mut gs = Vec::new();
            let mut us = Vec::new();
            let mut ds = Vec::new();
            for ((mut gate_proj, mut up_proj), mut down_proj) in gate_proj
                .chunk(num_local_experts, 0)?
                .into_iter()
                .zip(up_proj.chunk(num_local_experts, 0)?)
                .zip(down_proj.chunk(num_local_experts, 0)?)
            {
                gate_proj = gate_proj.squeeze(0)?;
                up_proj = up_proj.squeeze(0)?;
                down_proj = down_proj.squeeze(0)?;
                let gate_proj = merge_lora_weights(
                    &vb,
                    gate_proj,
                    hidden_size,
                    intermediate_size * 2,
                    shard_gate,
                )?;
                let up_proj =
                    merge_lora_weights(&vb, up_proj, hidden_size, intermediate_size * 2, shard_up)?;
                let down_proj =
                    merge_lora_weights(&vb, down_proj, intermediate_size, hidden_size, shard_down)?;

                let gate_proj = <UnquantLinear as QuantMethod>::new(
                    QuantMethodConfig::Unquantized(Linear::new(gate_proj, None)),
                )?;
                let up_proj = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                    Linear::new(up_proj, None),
                ))?;
                let down_proj = <UnquantLinear as QuantMethod>::new(
                    QuantMethodConfig::Unquantized(Linear::new(down_proj, None)),
                )?;
                gs.push(Arc::new(gate_proj) as Arc<dyn QuantMethod>);
                us.push(Arc::new(up_proj) as Arc<dyn QuantMethod>);
                ds.push(Arc::new(down_proj) as Arc<dyn QuantMethod>);
            }
            (gs, us, ds)
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Compute the appropriate KV shard. This handles KV head replication. Be sure to use `compute_n_kv_groups` in tandem.
pub fn compute_kv_shard(total_num_kv_heads: usize, head_dim: usize, comm: &Comm) -> Shard {
    if comm.world_size() == 1 {
        return Shard::default();
    }

    // Tensor parallelism case

    // We may need to replicate the kv heads
    let kv_replicate = if comm.world_size() > total_num_kv_heads {
        comm.world_size() / total_num_kv_heads
    } else {
        return Shard::Simple {
            dim: 0,
            rank: comm.rank(),
            world_size: comm.world_size(),
        };
    };

    let num_kv_heads = (total_num_kv_heads / comm.world_size()).max(1);
    let kv_shard_id = (comm.rank() / kv_replicate) * num_kv_heads;
    Shard::Offset {
        dim: 0,
        offset: kv_shard_id * head_dim,
        len: head_dim,
    }
}

/// Compute the number of KV groups, taking into account KV head replication.
pub fn compute_n_kv_groups(
    total_num_kv_heads: usize,
    num_attention_heads: usize,
    comm: &Comm,
) -> usize {
    let kv_replicate = if comm.world_size() > total_num_kv_heads {
        comm.world_size() / total_num_kv_heads
    } else {
        1
    };
    if kv_replicate != 0 {
        (num_attention_heads / total_num_kv_heads) / kv_replicate
    } else {
        num_attention_heads / total_num_kv_heads
    }
}
