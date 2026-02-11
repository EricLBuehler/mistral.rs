use std::sync::Arc;

use candle_core::{Context, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;

use crate::{
    blockwise_fp8::{blockwise_fp8_linear_b, blockwise_fp8_moe},
    distributed,
    gptq::gptq_linear,
    lora::merge_lora_weights,
    pertensor_fp8::pertensor_fp8_linear_b,
    should_apply_immediate_isq,
    utils::isq::{apply_immediate_isq, apply_immediate_isq_always},
    AfqLayer, BnbLinear, DistributedKind, DummyLayer, F8Q8Linear, FP8Linear, GgufMatMul, HqqLayer,
    MXFP4Layer, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, Shard, ShardedVarBuilder, UnquantLinear,
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

        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::GptqAwq { .. }
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
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
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

        let this_unquant = Arc::new(Self {
            weight,
            bias,
            all_reduce: distributed::SumAllReduce::new(comm),
        });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_matformer(
        in_dim: usize,
        out_dim: usize,
        orig_intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(1, rank, world_size);

        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        if config.is_some() {
            candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
        }

        // Handle the case where the layer is dummy (no tensors)
        let weight = if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb
                .get_with_hints(
                    (out_dim, orig_intermediate_size),
                    "weight",
                    Default::default(),
                )?
                .i((.., ..in_dim))?
                .contiguous()?;

            let weight = shard.apply_to(&weight)?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self {
            weight,
            bias,
            all_reduce: distributed::SumAllReduce::new(comm),
        });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
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
    fn serialize(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
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
            QuantizedSerdeType::F8Q8 => F8Q8Linear::deserialize_ext_bias(data, device, guard)?,
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
        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let weight = if let Some(quant_conf) = &config {
            // GPTQ and BNB do not support tensor parallelism
            if matches!(
                quant_conf,
                QuantizedConfig::GptqAwq { .. }
                    | QuantizedConfig::Bitsandbytes { .. }
                    | QuantizedConfig::Afq { .. }
            ) && comm.world_size() != 1
            {
                candle_core::bail!(
                    "GPTQ/AWQ and BNB and AFQ quantization types to not support tensor parallelism, but got a world size of {}",
                    comm.world_size()
                );
            }

            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    // NOTE: no bias for fp8 as it might be parallelized
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            false,
                            shard,
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
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

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
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

    #[allow(clippy::new_ret_no_self)]
    pub fn new_matformer(
        in_dim: usize,
        out_dim: usize,
        orig_intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shard = shard(0, rank, world_size);

        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        if config.is_some() {
            candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
        }

        // Handle the case where the layer is dummy (no tensors)
        let weight = if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb
                .get_with_hints(
                    (orig_intermediate_size, in_dim),
                    "weight",
                    Default::default(),
                )?
                .i((..out_dim, ..))?
                .contiguous()?;

            let weight = shard.apply_to(&weight)?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, shard)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        };

        // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", shard)?)
        } else {
            None
        };

        let this_unquant = Arc::new(Self { weight, bias });
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
    }

    pub fn new_merged(
        in_dim: usize,
        out_dim: usize,
        chunks: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<crate::Comm>,
        vb: ShardedVarBuilder,
    ) -> Result<Vec<Arc<dyn QuantMethod>>> {
        let mut vec_layers = Vec::<Arc<dyn QuantMethod>>::new();
        for chunk_idx in 0..chunks {
            let layer = ColumnParallelLayer::new_with_shard(
                in_dim,
                out_dim,
                config,
                bias,
                comm,
                shard(
                    0,
                    chunk_idx * comm.world_size() + comm.rank(),
                    chunks * comm.world_size(),
                ),
                vb.clone(),
            )?;
            vec_layers.push(layer);
        }
        Ok(vec_layers)
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
    fn serialize(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
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
            QuantizedSerdeType::F8Q8 => F8Q8Linear::deserialize_ext_bias(data, device, guard)?,
        };
        Ok(Arc::new(Self { weight, bias }))
    }
}

#[derive(Debug)]
/// This layer has no parallelization
pub struct ReplicatedLayer(Arc<dyn QuantMethod>);

impl ReplicatedLayer {
    pub fn from_linear(lin: Linear) -> Result<Arc<dyn QuantMethod>> {
        let dev = lin.weight().device().clone();
        // When immediate ISQ is active, the weight must be on CPU for GGML quantization.
        // Move it there first, then quantize, which will place the result on `dev`.
        let lin = if crate::get_immediate_isq().is_some() && !dev.is_cpu() {
            Linear::new(lin.weight().to_device(&Device::Cpu)?, lin.bias().cloned())
        } else {
            lin
        };
        let this_unquant = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lin))?);
        let this: Arc<dyn QuantMethod> = apply_immediate_isq_always(this_unquant, &dev)?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let layer = if let Some(quant_conf) = &config {
            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
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

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_layers_matformer_indices(
        in_dim: usize,
        out_dim: usize,
        kept_layers_indices: Option<&Tensor>,
        orig_num_hidden_layers: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base_vb = vb.clone();
        let vb = if should_apply_immediate_isq(&vb) {
            vb.set_device(Device::Cpu)
        } else {
            vb
        };

        let layer = if let Some(quant_conf) = &config {
            if kept_layers_indices.is_some() {
                candle_core::bail!("Cannot load a matformer layer with a pre-quantized model.");
            }

            match quant_conf {
                QuantizedConfig::GptqAwq { .. } => {
                    gptq_linear(in_dim, out_dim, quant_conf, vb.clone())?
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    if weight_block_size.is_some() {
                        blockwise_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    } else {
                        pertensor_fp8_linear_b(
                            in_dim,
                            out_dim,
                            quant_conf,
                            bias,
                            Default::default(),
                            vb.clone(),
                        )?
                    }
                }
                QuantizedConfig::Bitsandbytes { .. } => {
                    Arc::new(BnbLinear::linear_b(in_dim, out_dim, bias, vb.clone())?) as Arc<_>
                }
                QuantizedConfig::Afq { .. } => {
                    AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
                QuantizedConfig::MXFP4 {} => {
                    MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, bias, vb.clone())?
                }
            }
        } else {
            // Handle the case where the layer is dummy (no tensors)
            if !vb.contains_tensor("weight") {
                let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
                Arc::new(layer) as Arc<dyn QuantMethod>
            } else {
                let mut weight =
                    vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;

                if let Some(kept_layers_indices) = &kept_layers_indices {
                    let weight_reshaped = weight.reshape((
                        orig_num_hidden_layers,
                        weight.dim(0)? / orig_num_hidden_layers,
                        weight.dim(1)?,
                    ))?;

                    weight = weight_reshaped
                        .index_select(&kept_layers_indices.to_device(weight.device())?, 0)?
                        .reshape(((), weight_reshaped.dim(D::Minus1)?))?
                        .contiguous()?;
                }

                weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

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

        let this_unquant = Arc::new(Self(layer));
        let this: Arc<dyn QuantMethod> = apply_immediate_isq(this_unquant, base_vb)?;
        Ok(this)
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
    fn serialize(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
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
            QuantizedSerdeType::F8Q8 => F8Q8Linear::deserialize(data, device, comm, guard)?,
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

                    let base_vb = vb.clone();

                    let vb_gate_proj = if should_apply_immediate_isq(&vb) {
                        vb.pp("gate_proj").set_device(Device::Cpu)
                    } else {
                        vb.pp("gate_proj")
                    };
                    let vb_up_proj = if should_apply_immediate_isq(&vb) {
                        vb.pp("up_proj").set_device(Device::Cpu)
                    } else {
                        vb.pp("up_proj")
                    };
                    let vb_down_proj = if should_apply_immediate_isq(&vb) {
                        vb.pp("down_proj").set_device(Device::Cpu)
                    } else {
                        vb.pp("down_proj")
                    };
                    let mut gate_proj = AfqLayer::afq_packed_linear_b(
                        num_local_experts,
                        hidden_size,
                        intermediate_size,
                        quant_conf,
                        bias,
                        vb_gate_proj,
                    )?;
                    let mut up_proj = AfqLayer::afq_packed_linear_b(
                        num_local_experts,
                        hidden_size,
                        intermediate_size,
                        quant_conf,
                        bias,
                        vb_up_proj,
                    )?;
                    let mut down_proj = AfqLayer::afq_packed_linear_b(
                        num_local_experts,
                        intermediate_size,
                        hidden_size,
                        quant_conf,
                        bias,
                        vb_down_proj,
                    )?;

                    gate_proj = apply_immediate_isq(gate_proj, base_vb.pp("gate_proj"))?;
                    up_proj = apply_immediate_isq(up_proj, base_vb.pp("up_proj"))?;
                    down_proj = apply_immediate_isq(down_proj, base_vb.pp("down_proj"))?;

                    (vec![gate_proj], vec![up_proj], vec![down_proj])
                }
                QuantizedConfig::Fp8 { weight_block_size } => {
                    // FP8 quantization for PackedExperts
                    // Keep weights as FP8 using BlockwiseFP8Linear to leverage native FP8 GEMM
                    let Some(weight_block_size) = weight_block_size else {
                        candle_core::bail!("Blockwise FP8 for PackedExperts requires weight_block_size to be set.")
                    };
                    if weight_block_size.len() != 2 {
                        candle_core::bail!(
                            "Expected weight_block_size to have length 2, got {weight_block_size:?}"
                        );
                    }

                    // Check if we have stacked format (gate_up_proj) or per-expert format
                    // Note: vb already has the "experts" prefix from the caller (experts.rs)
                    let is_stacked_format = vb.contains_tensor("gate_up_proj");

                    if is_stacked_format {
                        // Stacked format: load FP8 tensors and split
                        let has_fp8_scales = vb.contains_tensor("gate_up_proj.weight_scale_inv");

                        if has_fp8_scales {
                            // Load gate_up_proj FP8 tensor and scale
                            let gate_up_fp8 = vb.get_with_hints_dtype(
                                (num_local_experts, hidden_size, intermediate_size * 2),
                                "gate_up_proj",
                                Default::default(),
                                candle_core::DType::F8E4M3,
                            )?;
                            let gate_up_scale = vb.get_with_hints_dtype(
                                (
                                    num_local_experts,
                                    hidden_size.div_ceil(weight_block_size[0]),
                                    (intermediate_size * 2).div_ceil(weight_block_size[1]),
                                ),
                                "gate_up_proj.weight_scale_inv",
                                Default::default(),
                                candle_core::DType::F32,
                            )?;

                            // Load down_proj FP8 tensor and scale
                            let down_fp8 = vb.get_with_hints_dtype(
                                (num_local_experts, intermediate_size, hidden_size),
                                "down_proj",
                                Default::default(),
                                candle_core::DType::F8E4M3,
                            )?;
                            let down_scale = vb.get_with_hints_dtype(
                                (
                                    num_local_experts,
                                    intermediate_size.div_ceil(weight_block_size[0]),
                                    hidden_size.div_ceil(weight_block_size[1]),
                                ),
                                "down_proj.weight_scale_inv",
                                Default::default(),
                                candle_core::DType::F32,
                            )?;

                            // Split and create individual BlockwiseFP8Linear for each expert
                            let mut gs = Vec::new();
                            let mut us = Vec::new();
                            let mut ds = Vec::new();

                            for i in 0..num_local_experts {
                                // Extract this expert's weights
                                let gate_up_expert =
                                    gate_up_fp8.i(i)?.transpose(0, 1)?.contiguous()?;
                                let gate_up_scale_expert = gate_up_scale.i(i)?.contiguous()?;
                                let down_expert = down_fp8.i(i)?.transpose(0, 1)?.contiguous()?;
                                let down_scale_expert = down_scale.i(i)?.contiguous()?;

                                // Split gate_up into gate and up
                                let gate_expert = gate_up_expert.narrow(0, 0, intermediate_size)?;
                                let up_expert = gate_up_expert.narrow(
                                    0,
                                    intermediate_size,
                                    intermediate_size,
                                )?;

                                // Split scales
                                let gate_scale_expert = gate_up_scale_expert.narrow(
                                    1,
                                    0,
                                    intermediate_size.div_ceil(weight_block_size[1]),
                                )?;
                                let up_scale_expert = gate_up_scale_expert.narrow(
                                    1,
                                    intermediate_size.div_ceil(weight_block_size[1]),
                                    intermediate_size.div_ceil(weight_block_size[1]),
                                )?;

                                // Create BlockwiseFP8Linear for each projection
                                use crate::blockwise_fp8::BlockwiseFP8Linear;
                                use crate::QuantMethodConfig;

                                let gate_layer: Arc<dyn QuantMethod> = Arc::new(
                                    BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                                        weight: gate_expert,
                                        weight_scale_inv: gate_scale_expert.transpose(0, 1)?,
                                        bias: None,
                                        dequant_dtype: vb.dtype(),
                                        weight_block_size: weight_block_size.clone(),
                                    })?,
                                );
                                let up_layer: Arc<dyn QuantMethod> = Arc::new(
                                    BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                                        weight: up_expert,
                                        weight_scale_inv: up_scale_expert.transpose(0, 1)?,
                                        bias: None,
                                        dequant_dtype: vb.dtype(),
                                        weight_block_size: weight_block_size.clone(),
                                    })?,
                                );
                                let down_layer: Arc<dyn QuantMethod> = Arc::new(
                                    BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                                        weight: down_expert,
                                        weight_scale_inv: down_scale_expert.transpose(0, 1)?,
                                        bias: None,
                                        dequant_dtype: vb.dtype(),
                                        weight_block_size: weight_block_size.clone(),
                                    })?,
                                );

                                gs.push(gate_layer);
                                us.push(up_layer);
                                ds.push(down_layer);
                            }

                            (gs, us, ds)
                        } else {
                            candle_core::bail!(
                                "PackedExperts with FP8 requires weight_scale_inv tensors"
                            );
                        }
                    } else {
                        // Per-expert format: load each expert individually
                        let mut gs = Vec::new();
                        let mut us = Vec::new();
                        let mut ds = Vec::new();

                        for i in 0..num_local_experts {
                            let expert_vb = vb.pp(i);

                            // Load FP8 weights and scales for each projection
                            let gate_fp8 = expert_vb.get_with_hints_dtype(
                                (intermediate_size, hidden_size),
                                "gate_proj.weight",
                                Default::default(),
                                candle_core::DType::F8E4M3,
                            )?;
                            let gate_scale = expert_vb.get_with_hints_dtype(
                                (
                                    intermediate_size.div_ceil(weight_block_size[0]),
                                    hidden_size.div_ceil(weight_block_size[1]),
                                ),
                                "gate_proj.weight_scale_inv",
                                Default::default(),
                                candle_core::DType::F32,
                            )?;

                            let up_fp8 = expert_vb.get_with_hints_dtype(
                                (intermediate_size, hidden_size),
                                "up_proj.weight",
                                Default::default(),
                                candle_core::DType::F8E4M3,
                            )?;
                            let up_scale = expert_vb.get_with_hints_dtype(
                                (
                                    intermediate_size.div_ceil(weight_block_size[0]),
                                    hidden_size.div_ceil(weight_block_size[1]),
                                ),
                                "up_proj.weight_scale_inv",
                                Default::default(),
                                candle_core::DType::F32,
                            )?;

                            let down_fp8 = expert_vb.get_with_hints_dtype(
                                (hidden_size, intermediate_size),
                                "down_proj.weight",
                                Default::default(),
                                candle_core::DType::F8E4M3,
                            )?;
                            let down_scale = expert_vb.get_with_hints_dtype(
                                (
                                    hidden_size.div_ceil(weight_block_size[0]),
                                    intermediate_size.div_ceil(weight_block_size[1]),
                                ),
                                "down_proj.weight_scale_inv",
                                Default::default(),
                                candle_core::DType::F32,
                            )?;

                            // Create BlockwiseFP8Linear for each projection
                            use crate::blockwise_fp8::BlockwiseFP8Linear;
                            use crate::QuantMethodConfig;

                            let gate_layer: Arc<dyn QuantMethod> = Arc::new(
                                BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                                    weight: gate_fp8,
                                    weight_scale_inv: gate_scale,
                                    bias: None,
                                    dequant_dtype: vb.dtype(),
                                    weight_block_size: weight_block_size.clone(),
                                })?,
                            );
                            let up_layer: Arc<dyn QuantMethod> = Arc::new(BlockwiseFP8Linear::new(
                                QuantMethodConfig::BlockwiseFP8 {
                                    weight: up_fp8,
                                    weight_scale_inv: up_scale,
                                    bias: None,
                                    dequant_dtype: vb.dtype(),
                                    weight_block_size: weight_block_size.clone(),
                                },
                            )?);
                            let down_layer: Arc<dyn QuantMethod> = Arc::new(
                                BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                                    weight: down_fp8,
                                    weight_scale_inv: down_scale,
                                    bias: None,
                                    dequant_dtype: vb.dtype(),
                                    weight_block_size: weight_block_size.clone(),
                                })?,
                            );

                            gs.push(gate_layer);
                            us.push(up_layer);
                            ds.push(down_layer);
                        }

                        (gs, us, ds)
                    }
                }
                QuantizedConfig::MXFP4 {} => {
                    // MXFP4 quantization for PackedExperts
                    // Keep weights as MXFP4 using MXFP4Layer to leverage native MXFP4 GEMM
                    // Note: MXFP4 models use stacked format, so we load directly as packed experts
                    let gate_proj = MXFP4Layer::packed_linear_b(
                        num_local_experts,
                        hidden_size,
                        intermediate_size,
                        quant_conf,
                        bias,
                        vb.pp("gate_proj"),
                    )?;
                    let up_proj = MXFP4Layer::packed_linear_b(
                        num_local_experts,
                        hidden_size,
                        intermediate_size,
                        quant_conf,
                        bias,
                        vb.pp("up_proj"),
                    )?;
                    let down_proj = MXFP4Layer::packed_linear_b(
                        num_local_experts,
                        intermediate_size,
                        hidden_size,
                        quant_conf,
                        bias,
                        vb.pp("down_proj"),
                    )?;

                    (vec![gate_proj], vec![up_proj], vec![down_proj])
                }
                _ => candle_core::bail!(
                    "PackedExperts with quantization config only allows AFQ, FP8, or MXFP4 quantization"
                ),
            }
        } else if !vb.contains_tensor("gate_up_proj") {
            // Handle the case where the layer is dummy (no tensors) during UQFF loading. Deserialize will handle it.
            let mut gs: Vec<Arc<dyn QuantMethod>> = Vec::new();
            let mut us: Vec<Arc<dyn QuantMethod>> = Vec::new();
            let mut ds: Vec<Arc<dyn QuantMethod>> = Vec::new();
            for _ in 0..num_local_experts {
                gs.push(Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?));
                us.push(Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?));
                ds.push(Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?));
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

            let vb_gate_up_proj = if should_apply_immediate_isq(&vb) {
                vb.pp("gate_up_proj").set_device(Device::Cpu)
            } else {
                vb.pp("gate_up_proj")
            };
            let vb_down_proj = if should_apply_immediate_isq(&vb) {
                vb.pp("down_proj").set_device(Device::Cpu)
            } else {
                vb.pp("down_proj")
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

            let gc = gate_proj.chunk(num_local_experts, 0)?;
            let uc = up_proj.chunk(num_local_experts, 0)?;
            let dc = down_proj.chunk(num_local_experts, 0)?;
            drop((gate_proj, up_proj, down_proj));

            let mut gs = Vec::new();
            let mut us = Vec::new();
            let mut ds = Vec::new();
            for ((mut gate_proj, mut up_proj), mut down_proj) in
                gc.into_iter().zip(uc.into_iter()).zip(dc.into_iter())
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

                let mut gate_proj: Arc<dyn QuantMethod> =
                    Arc::new(<UnquantLinear as QuantMethod>::new(
                        QuantMethodConfig::Unquantized(Linear::new(gate_proj, None)),
                    )?);
                gate_proj = apply_immediate_isq(gate_proj, vb_gate_up_proj.clone())?;
                let mut up_proj: Arc<dyn QuantMethod> =
                    Arc::new(<UnquantLinear as QuantMethod>::new(
                        QuantMethodConfig::Unquantized(Linear::new(up_proj, None)),
                    )?);
                up_proj = apply_immediate_isq(up_proj, vb_gate_up_proj.clone())?;
                let mut down_proj: Arc<dyn QuantMethod> =
                    Arc::new(<UnquantLinear as QuantMethod>::new(
                        QuantMethodConfig::Unquantized(Linear::new(down_proj, None)),
                    )?);
                down_proj = apply_immediate_isq(down_proj, vb_down_proj.clone())?;
                gs.push(gate_proj);
                us.push(up_proj);
                ds.push(down_proj);
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

pub struct FusedExperts {
    pub fused_gate_proj: Arc<dyn QuantMethod>,
    pub fused_up_proj: Arc<dyn QuantMethod>,
    pub fused_down_proj: Arc<dyn QuantMethod>,
}

impl FusedExperts {
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        quantization_config: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Detect if weights are in stacked format (e.g., Qwen3 VL MoE):
        // - experts.gate_up_proj: (num_experts, hidden_size, intermediate_size * 2)
        // - experts.down_proj: (num_experts, intermediate_size, hidden_size)
        // Or per-expert format (e.g., Qwen3 MoE):
        // - experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, experts.{i}.down_proj.weight
        let experts_vb = vb.pp("experts");
        let is_stacked_format = experts_vb.contains_tensor("gate_up_proj");

        let (fused_gate_proj, fused_up_proj, fused_down_proj) = if matches!(
            &quantization_config,
            Some(QuantizedConfig::Afq { .. })
        ) {
            let quantization_config = quantization_config.as_ref().unwrap();

            let fused_gate_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.gate_proj"),
            )?;
            let fused_up_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.up_proj"),
            )?;
            let fused_down_proj = AfqLayer::afq_packed_linear_b(
                num_experts,
                moe_intermediate_size,
                hidden_size,
                quantization_config,
                false,
                vb.pp("switch_mlp.down_proj"),
            )?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if is_stacked_format
            && matches!(&quantization_config, Some(QuantizedConfig::Fp8 { .. }))
        {
            // Stacked format with FP8 quantization
            // Keep weights as FP8 using BlockwiseFP8 to leverage native FP8 GEMM in gather_forward
            let has_fp8_scales = experts_vb.contains_tensor("gate_up_proj.weight_scale_inv");

            if has_fp8_scales {
                let weight_block_size = match quantization_config {
                    Some(QuantizedConfig::Fp8 { weight_block_size }) => weight_block_size.clone(),
                    _ => unreachable!(),
                };

                let Some(weight_block_size) = weight_block_size else {
                    candle_core::bail!(
                        "Blockwise FP8 for stacked experts requires weight_block_size to be set."
                    )
                };
                if weight_block_size.len() != 2 {
                    candle_core::bail!(
                        "Expected weight_block_size to have length 2, got {weight_block_size:?}"
                    );
                }

                // Load gate_up_proj FP8 tensor and scale
                // Shape: [num_experts, hidden_size, intermediate_size * 2]
                let gate_up_fp8 = experts_vb.get_with_hints_dtype(
                    (num_experts, hidden_size, moe_intermediate_size * 2),
                    "gate_up_proj",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let gate_up_scale = experts_vb.get_with_hints_dtype(
                    (
                        num_experts,
                        hidden_size.div_ceil(weight_block_size[0]),
                        (moe_intermediate_size * 2).div_ceil(weight_block_size[1]),
                    ),
                    "gate_up_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                // Load down_proj FP8 tensor and scale
                // Shape: [num_experts, intermediate_size, hidden_size]
                let down_fp8 = experts_vb.get_with_hints_dtype(
                    (num_experts, moe_intermediate_size, hidden_size),
                    "down_proj",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let down_scale = experts_vb.get_with_hints_dtype(
                    (
                        num_experts,
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "down_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                // Split gate_up into gate and up
                let gate_fp8 = gate_up_fp8.narrow(2, 0, moe_intermediate_size)?;
                let up_fp8 = gate_up_fp8.narrow(2, moe_intermediate_size, moe_intermediate_size)?;

                // Split scales similarly
                let gate_scale = gate_up_scale.narrow(
                    2,
                    0,
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                )?;
                let up_scale = gate_up_scale.narrow(
                    2,
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                    moe_intermediate_size.div_ceil(weight_block_size[1]),
                )?;

                // Transpose to match expected format: [num_experts, N, K]
                // gate/up: [num_experts, hidden_size, intermediate_size] -> [num_experts, intermediate_size, hidden_size]
                let gate_fp8 = gate_fp8.transpose(1, 2)?.contiguous()?;
                let up_fp8 = up_fp8.transpose(1, 2)?.contiguous()?;
                // down: [num_experts, intermediate_size, hidden_size] -> [num_experts, hidden_size, intermediate_size]
                let down_fp8 = down_fp8.transpose(1, 2)?.contiguous()?;

                // Transpose scales to match weight layout
                let gate_scale = gate_scale.transpose(1, 2)?.contiguous()?;
                let up_scale = up_scale.transpose(1, 2)?.contiguous()?;
                let down_scale = down_scale.transpose(1, 2)?.contiguous()?;

                // Create BlockwiseFP8Linear for each projection
                let fused_gate_proj =
                    blockwise_fp8_moe(gate_fp8, gate_scale, weight_block_size.clone(), vb.dtype())?;
                let fused_up_proj =
                    blockwise_fp8_moe(up_fp8, up_scale, weight_block_size.clone(), vb.dtype())?;
                let fused_down_proj =
                    blockwise_fp8_moe(down_fp8, down_scale, weight_block_size, vb.dtype())?;

                (fused_gate_proj, fused_up_proj, fused_down_proj)
            } else {
                // FP8 config but no scale tensors - weights are actually unquantized
                tracing::warn!(
                        "FP8 quantization config specified but no scale tensors found for stacked MoE experts. \
                        Loading as unquantized."
                    );
                let gate_up_proj = experts_vb.get(
                    (num_experts, hidden_size, moe_intermediate_size * 2),
                    "gate_up_proj",
                )?;
                let down_proj_packed = experts_vb.get(
                    (num_experts, moe_intermediate_size, hidden_size),
                    "down_proj",
                )?;

                // Split gate_up_proj into gate_proj and up_proj along the last dimension
                let gate_proj = gate_up_proj.narrow(2, 0, moe_intermediate_size)?;
                let up_proj =
                    gate_up_proj.narrow(2, moe_intermediate_size, moe_intermediate_size)?;

                // Transpose dims 1 and 2 to match GGUF format
                let gate_proj = gate_proj.transpose(1, 2)?.contiguous()?;
                let up_proj = up_proj.transpose(1, 2)?.contiguous()?;
                let down_proj = down_proj_packed.transpose(1, 2)?.contiguous()?;

                // When immediate ISQ is active, the weight must be on CPU for GGML quantization.
                // Move it there first, then quantize, which will place the result on `target_device`.
                let target_device = gate_proj.device().clone();
                let (gate_proj, up_proj, down_proj) =
                    if crate::get_immediate_isq().is_some() && !target_device.is_cpu() {
                        (
                            gate_proj.to_device(&Device::Cpu)?,
                            up_proj.to_device(&Device::Cpu)?,
                            down_proj.to_device(&Device::Cpu)?,
                        )
                    } else {
                        (gate_proj, up_proj, down_proj)
                    };

                let mut fused_gate_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(gate_proj, None)),
                )?);
                let mut fused_up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(up_proj, None)),
                )?);
                let mut fused_down_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(down_proj, None)),
                )?);
                // Use apply_immediate_isq_always to ensure ISQ is applied to expert weights
                fused_gate_proj = apply_immediate_isq_always(fused_gate_proj, &target_device)?;
                fused_up_proj = apply_immediate_isq_always(fused_up_proj, &target_device)?;
                fused_down_proj = apply_immediate_isq_always(fused_down_proj, &target_device)?;

                (fused_gate_proj, fused_up_proj, fused_down_proj)
            }
        } else if is_stacked_format
            && matches!(&quantization_config, Some(QuantizedConfig::MXFP4 {}))
        {
            // Stacked format with MXFP4 quantization
            // For MXFP4, weights are stored as packed FP4 (2 values per byte)
            // with E8M0 scales
            let quantization_config = quantization_config.as_ref().unwrap();

            // Load MXFP4 packed experts using MXFP4Layer::packed_linear_b
            // The tensors are expected at:
            //   gate_proj.blocks: [num_experts, intermediate_size, hidden_size/2]
            //   gate_proj.scales: [num_experts, intermediate_size, hidden_size/32]
            let fused_gate_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                experts_vb.pp("gate_proj"),
            )?;
            let fused_up_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                hidden_size,
                moe_intermediate_size,
                quantization_config,
                false,
                experts_vb.pp("up_proj"),
            )?;
            let fused_down_proj = MXFP4Layer::packed_linear_b(
                num_experts,
                moe_intermediate_size,
                hidden_size,
                quantization_config,
                false,
                experts_vb.pp("down_proj"),
            )?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if is_stacked_format {
            // Stacked format from safetensors:
            // - gate_up_proj: [num_experts, hidden_size, intermediate_size * 2] = [128, 2048, 1536]
            // - down_proj: [num_experts, intermediate_size, hidden_size] = [128, 768, 2048]
            //
            // GGUF/indexed_moe_forward expects:
            // - gate/up: [num_experts, intermediate_size, hidden_size] = [128, 768, 2048]
            // - down: [num_experts, hidden_size, intermediate_size] = [128, 2048, 768]
            let gate_up_proj = experts_vb.get(
                (num_experts, hidden_size, moe_intermediate_size * 2),
                "gate_up_proj",
            )?;
            let down_proj_packed = experts_vb.get(
                (num_experts, moe_intermediate_size, hidden_size),
                "down_proj",
            )?;

            // Split gate_up_proj into gate_proj and up_proj along the last dimension
            // gate_proj: [num_experts, hidden_size, intermediate_size]
            // up_proj: [num_experts, hidden_size, intermediate_size]
            let gate_proj = gate_up_proj.narrow(2, 0, moe_intermediate_size)?;
            let up_proj = gate_up_proj.narrow(2, moe_intermediate_size, moe_intermediate_size)?;

            // Transpose dims 1 and 2 to match GGUF format:
            // gate/up: [num_experts, hidden_size, intermediate_size] -> [num_experts, intermediate_size, hidden_size]
            let gate_proj = gate_proj.transpose(1, 2)?.contiguous()?;
            let up_proj = up_proj.transpose(1, 2)?.contiguous()?;
            // down_proj: [num_experts, intermediate_size, hidden_size] -> [num_experts, hidden_size, intermediate_size]
            let down_proj = down_proj_packed.transpose(1, 2)?.contiguous()?;

            // When immediate ISQ is active, the weight must be on CPU for GGML quantization.
            // Move it there first, then quantize, which will place the result on `target_device`.
            let target_device = gate_proj.device().clone();
            let (gate_proj, up_proj, down_proj) =
                if crate::get_immediate_isq().is_some() && !target_device.is_cpu() {
                    (
                        gate_proj.to_device(&Device::Cpu)?,
                        up_proj.to_device(&Device::Cpu)?,
                        down_proj.to_device(&Device::Cpu)?,
                    )
                } else {
                    (gate_proj, up_proj, down_proj)
                };

            let mut fused_gate_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(gate_proj, None)),
            )?);
            let mut fused_up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(up_proj, None)),
            )?);
            let mut fused_down_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(down_proj, None)),
            )?);
            // Use apply_immediate_isq_always to ensure ISQ is applied to expert weights
            fused_gate_proj = apply_immediate_isq_always(fused_gate_proj, &target_device)?;
            fused_up_proj = apply_immediate_isq_always(fused_up_proj, &target_device)?;
            fused_down_proj = apply_immediate_isq_always(fused_down_proj, &target_device)?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if matches!(&quantization_config, Some(QuantizedConfig::Fp8 { .. })) {
            // Per-expert format with FP8 quantization
            // Keep weights as FP8 using BlockwiseFP8 to leverage native FP8 GEMM in gather_forward
            let weight_block_size = match quantization_config {
                Some(QuantizedConfig::Fp8 { weight_block_size }) => weight_block_size.clone(),
                _ => unreachable!(),
            };

            let Some(weight_block_size) = weight_block_size else {
                candle_core::bail!(
                    "Blockwise FP8 for per-expert format requires weight_block_size to be set."
                )
            };
            if weight_block_size.len() != 2 {
                candle_core::bail!(
                    "Expected weight_block_size to have length 2, got {weight_block_size:?}"
                );
            }

            let mut gate_fp8_vec = Vec::new();
            let mut gate_scale_vec = Vec::new();
            let mut up_fp8_vec = Vec::new();
            let mut up_scale_vec = Vec::new();
            let mut down_fp8_vec = Vec::new();
            let mut down_scale_vec = Vec::new();

            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(i);

                // Load FP8 weights and scales for each projection
                let gate_fp8 = expert_vb.get_with_hints_dtype(
                    (moe_intermediate_size, hidden_size),
                    "gate_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let gate_scale = expert_vb.get_with_hints_dtype(
                    (
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "gate_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                let up_fp8 = expert_vb.get_with_hints_dtype(
                    (moe_intermediate_size, hidden_size),
                    "up_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let up_scale = expert_vb.get_with_hints_dtype(
                    (
                        moe_intermediate_size.div_ceil(weight_block_size[0]),
                        hidden_size.div_ceil(weight_block_size[1]),
                    ),
                    "up_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                let down_fp8 = expert_vb.get_with_hints_dtype(
                    (hidden_size, moe_intermediate_size),
                    "down_proj.weight",
                    Default::default(),
                    candle_core::DType::F8E4M3,
                )?;
                let down_scale = expert_vb.get_with_hints_dtype(
                    (
                        hidden_size.div_ceil(weight_block_size[0]),
                        moe_intermediate_size.div_ceil(weight_block_size[1]),
                    ),
                    "down_proj.weight_scale_inv",
                    Default::default(),
                    candle_core::DType::F32,
                )?;

                gate_fp8_vec.push(gate_fp8);
                gate_scale_vec.push(gate_scale);
                up_fp8_vec.push(up_fp8);
                up_scale_vec.push(up_scale);
                down_fp8_vec.push(down_fp8);
                down_scale_vec.push(down_scale);
            }

            // Stack into [num_experts, N, K]
            let gate_fp8 = Tensor::stack(&gate_fp8_vec, 0)?;
            let gate_scale = Tensor::stack(&gate_scale_vec, 0)?;
            let up_fp8 = Tensor::stack(&up_fp8_vec, 0)?;
            let up_scale = Tensor::stack(&up_scale_vec, 0)?;
            let down_fp8 = Tensor::stack(&down_fp8_vec, 0)?;
            let down_scale = Tensor::stack(&down_scale_vec, 0)?;

            // Create BlockwiseFP8Linear for each projection
            let fused_gate_proj =
                blockwise_fp8_moe(gate_fp8, gate_scale, weight_block_size.clone(), vb.dtype())?;
            let fused_up_proj =
                blockwise_fp8_moe(up_fp8, up_scale, weight_block_size.clone(), vb.dtype())?;
            let fused_down_proj =
                blockwise_fp8_moe(down_fp8, down_scale, weight_block_size, vb.dtype())?;

            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else if !experts_vb.pp("0").contains_tensor("gate_proj.weight") {
            // Handle the case where the layer is dummy (no tensors) during UQFF loading.
            // Deserialize will handle it.
            let fused_gate_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            let fused_up_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            let fused_down_proj: Arc<dyn QuantMethod> =
                Arc::new(DummyLayer::new(QuantMethodConfig::Dummy)?);
            (fused_gate_proj, fused_up_proj, fused_down_proj)
        } else {
            // Per-expert format: load each expert individually and stack
            // When immediate ISQ is active, load on CPU for GGML quantization.
            let load_experts_vb =
                if crate::get_immediate_isq().is_some() && !experts_vb.device().is_cpu() {
                    experts_vb.clone().set_device(Device::Cpu)
                } else {
                    experts_vb.clone()
                };
            let mut gate_proj_vec = Vec::new();
            let mut up_proj_vec = Vec::new();
            let mut down_proj_vec = Vec::new();
            for i in 0..num_experts {
                let expert_vb = load_experts_vb.pp(i);
                let gate_proj =
                    expert_vb.get((moe_intermediate_size, hidden_size), "gate_proj.weight")?;
                let up_proj =
                    expert_vb.get((moe_intermediate_size, hidden_size), "up_proj.weight")?;
                let down_proj =
                    expert_vb.get((hidden_size, moe_intermediate_size), "down_proj.weight")?;

                gate_proj_vec.push(gate_proj);
                up_proj_vec.push(up_proj);
                down_proj_vec.push(down_proj);
            }

            let mut gate_proj: Arc<dyn QuantMethod> =
                Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                    Linear::new(Tensor::stack(&gate_proj_vec, 0)?, None),
                ))?);
            let mut up_proj: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(Tensor::stack(&up_proj_vec, 0)?, None)),
            )?);
            let mut down_proj: Arc<dyn QuantMethod> =
                Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                    Linear::new(Tensor::stack(&down_proj_vec, 0)?, None),
                ))?);
            // Use experts.0.{proj} prefix to match the actual weight paths for ISQ predicate matching
            let expert0_vb = experts_vb.pp("0");
            gate_proj = apply_immediate_isq(gate_proj, expert0_vb.pp("gate_proj"))?;
            up_proj = apply_immediate_isq(up_proj, expert0_vb.pp("up_proj"))?;
            down_proj = apply_immediate_isq(down_proj, expert0_vb.pp("down_proj"))?;

            (gate_proj, up_proj, down_proj)
        };

        Ok(Self {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
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
